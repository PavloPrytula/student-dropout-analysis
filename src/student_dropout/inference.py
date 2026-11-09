"""Q2 multinomial associations: training only, Graduate reference."""

import warnings

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import build_design_matrices, dmatrix
from scipy.optimize import minimize
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from .data import OUT, SEED, dump, prepare

FORMULA = "1 + C(Course, Treatment(reference=9500)) + Gender + bs(age, df=4, degree=3, include_intercept=False) + admission + previous"


def variables(t: pd.DataFrame) -> pd.DataFrame:
    """Add the prespecified age and grade-scaled Q2 design variables."""
    return t.assign(
        age=t["Age at enrollment"],
        admission=(t["Admission grade"] - 130) / 10,
        previous=(t["Previous qualification (grade)"] - 130) / 10,
    )


def fit_multinomial(A, B, penalty: float, start=None):
    """Fit the fixed three-class BFGS objective; verify score despite precision warnings."""
    p = A.shape[1]
    mask = np.ones((p, 2))
    mask[0] = 0

    def fg(w):
        W = w.reshape(p, 2)
        z = np.c_[np.zeros(len(A)), A @ W]
        pr = softmax(z, axis=1)
        from scipy.special import logsumexp

        val = np.sum(
            logsumexp(z, axis=1) - np.sum(B * z, axis=1)
        ) + penalty / 2 * np.sum(mask * W**2)
        g = A.T @ (pr[:, 1:] - B[:, 1:]) + penalty * mask * W
        return val, g.ravel()

    r = minimize(
        fg,
        np.zeros(p * 2) if start is None else start,
        jac=True,
        method="BFGS",
        options={"gtol": 1e-6, "maxiter": 2000},
    )
    # BFGS may report precision loss near machine precision; independently check score.
    if not np.isfinite(r.fun) or abs(r.jac).max() > 1e-4:
        raise RuntimeError(str(r.message) + str(abs(r.jac).max()))
    return r


def robust_covariance(X, Y, W, penalty: float) -> tuple[np.ndarray, np.ndarray]:
    """Return HC0 covariance and Hessian in feature-major, two-logit order."""
    n, p = X.shape
    mask = np.ones((p, 2))
    mask[0] = 0
    pr = softmax(np.c_[np.zeros(n), X @ W], axis=1)
    score = (X[:, :, None] * (Y[:, None, 1:] - pr[:, None, 1:])).reshape(n, p * 2)
    # Row-major coefficient ordering: each feature's Dropout and Enrolled terms.
    H = np.zeros((p * 2, p * 2))
    for j in range(2):
        for k in range(2):
            weight = pr[:, j + 1] * ((j == k) - pr[:, k + 1])
            H[j::2, k::2] = X.T @ (weight[:, None] * X)
    H += np.diag((penalty * mask).ravel())
    inv = np.linalg.inv(H)
    centered = score - score.mean(axis=0)
    cov = inv @ (centered.T @ centered) @ inv
    return cov, H


def dropout_probability(design: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Predict Q2 Dropout (column 1), retaining Graduate as reference and Enrolled."""
    return softmax(np.c_[np.zeros(len(design)), design @ coefficients], axis=1)[:, 1]


def contrast_gradient(design: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Differentiate the empirical mean Dropout probability at fixed covariates."""
    probabilities = softmax(np.c_[np.zeros(len(design)), design @ coefficients], axis=1)
    weights = probabilities[:, 1, None] * (
        np.array([1.0, 0.0])[None, :] - probabilities[:, 1:]
    )
    return (design[:, :, None] * weights[:, None, :]).mean(axis=0).ravel()


def run_inference(n_boot: int = 200) -> tuple[dict, pd.DataFrame]:
    """Fit Q2 training-only associations, robust contrasts and bootstrap sensitivity."""
    t, _ = prepare()
    v = variables(t)
    y = t.Target.map({"Graduate": 0, "Dropout": 1, "Enrolled": 2}).to_numpy()
    design = dmatrix(FORMULA, v, return_type="dataframe")
    info = design.design_info
    X = design.to_numpy()
    n, p = X.shape
    cells = pd.crosstab(t.Course, t.Target)
    cells.to_csv(OUT / "Q2_course_cells.csv")
    diagnostics = {
        "n": n,
        "p_per_logit": p,
        "rank": int(np.linalg.matrix_rank(X)),
        "zero_course_cells": int((cells == 0).sum().sum()),
        "cells_below_5": int((cells < 5).sum().sum()),
    }
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        try:
            unreg = sm.MNLogit(y, X).fit(
                method="newton", maxiter=300, tol=1e-9, disp=False
            )
            diagnostics["unpenalized"] = {
                "converged": bool(unreg.mle_retvals["converged"]),
                "iterations": unreg.mle_retvals.get("iterations"),
                "finite": bool(np.isfinite(unreg.params).all()),
                "max_abs_coef": float(np.nanmax(abs(unreg.params))),
            }
        except (np.linalg.LinAlgError, ValueError) as e:
            diagnostics["unpenalized"] = {
                "converged": False,
                "finite": False,
                "error": str(e),
            }
    diagnostics["unpenalized"]["warnings"] = [str(w.message) for w in ws]
    # Empty programme-outcome cells imply a non-finite MLE. Retain all programmes
    # and use a disclosed fixed weak ridge rather than merge unlike programmes.
    regularized = (
        diagnostics["zero_course_cells"] > 0
        or not diagnostics["unpenalized"]["finite"]
        or not diagnostics["unpenalized"]["converged"]
    )
    penalty = 0.5 if regularized else 0.0
    Y = np.eye(3)[y]

    with threadpool_limits(limits=1):
        r = fit_multinomial(X, Y, penalty)
        bfgs_x = r.x.copy()
        if not regularized:
            r.x = np.asarray(unreg.params).ravel()
        W = r.x.reshape(p, 2)
        cov, H = robust_covariance(X, Y, W, penalty)
        se = np.sqrt(np.diag(cov)).reshape(p, 2)
        coefficients = []
        for i, name in enumerate(design.columns):
            for j, label in enumerate(["Dropout", "Enrolled"]):
                b = W[i, j]
                s = se[i, j]
                coefficients.append(
                    {
                        "term": name,
                        "outcome_vs_Graduate": label,
                        "estimate": b,
                        "robust_se": s,
                        "rrr": np.exp(b),
                        "rrr_lo": np.exp(b - 1.96 * s),
                        "rrr_hi": np.exp(b + 1.96 * s),
                    }
                )
        pd.DataFrame(coefficients).to_csv(OUT / "Q2_coefficients.csv", index=False)
        contrasts = []
        matrices = []
        for label, col, low, high in [
            ("Gender: male - female", "Gender", 0, 1),
            ("Age: 25 - 20 years", "age", 20, 25),
            ("Admission grade: 140 - 130", "admission", 0, 1),
            ("Previous grade: 140 - 130", "previous", 0, 1),
        ]:
            lo = v.copy()
            hi = v.copy()
            lo[col] = low
            hi[col] = high
            A = np.asarray(build_design_matrices([info], lo)[0])
            B = np.asarray(build_design_matrices([info], hi)[0])
            matrices.append((A, B))

            g = contrast_gradient(B, W) - contrast_gradient(A, W)
            delta_se = float(np.sqrt(g @ cov @ g))
            value = float(
                np.mean(dropout_probability(B, W) - dropout_probability(A, W))
            )
            contrasts.append(
                {
                    "contrast": label,
                    "dropout_probability_difference": value,
                    "robust_delta_se": delta_se,
                    "normal_ci_low": value - 1.96 * delta_se,
                    "normal_ci_high": value + 1.96 * delta_se,
                }
            )
        rng = np.random.default_rng(SEED + 200)
        boot = []
        boot_diag = []
        for b in range(n_boot):
            ix = rng.integers(n, size=n)
            rr = fit_multinomial(X[ix], Y[ix], penalty, r.x)
            cc = pd.crosstab(
                t.Course.iloc[ix].to_numpy(), t.Target.iloc[ix].to_numpy()
            ).reindex(index=cells.index, columns=cells.columns, fill_value=0)
            boot_diag.append(
                {
                    "replicate": b,
                    "zero_course_cells": int((cc == 0).sum().sum()),
                    "max_abs_coef": float(abs(rr.x).max()),
                    "gradient_inf": float(abs(rr.jac).max()),
                }
            )
            boot.append(
                [
                    float(
                        np.mean(
                            dropout_probability(B[ix], rr.x.reshape(p, 2))
                            - dropout_probability(A[ix], rr.x.reshape(p, 2))
                        )
                    )
                    for A, B in matrices
                ]
            )
            if (b + 1) % 50 == 0:
                print("Q2 bootstrap", b + 1, flush=True)
        boot = np.array(boot)
        for j, row in enumerate(contrasts):
            row.update(
                bootstrap_se=float(boot[:, j].std(ddof=1)),
                ci_low=float(np.quantile(boot[:, j], 0.025)),
                ci_high=float(np.quantile(boot[:, j], 0.975)),
            )
        pd.DataFrame(contrasts).to_csv(OUT / "Q2_contrasts.csv", index=False)
        pd.DataFrame(boot, columns=[r["contrast"] for r in contrasts]).to_csv(
            OUT / "Q2_bootstrap.csv", index=False
        )
    pd.DataFrame(boot_diag).to_csv(OUT / "Q2_bootstrap_diagnostics.csv", index=False)
    diagnostics["newton_bfgs_max_coefficient_difference"] = float(
        abs(r.x - bfgs_x).max()
    )
    diagnostics["newton_gradient_inf"] = (
        float(
            abs(sm.MNLogit(y, X).score(np.asarray(unreg.params).ravel(order="F"))).max()
        )
        if not regularized
        else None
    )
    diagnostics.update(
        penalty_sum_loss=penalty,
        criterion="multinomial NLL + fixed ridge" if penalty else "multinomial NLL",
        optimizer="BFGS",
        start="zero",
        tolerance=1e-6,
        maxiter=2000,
        iterations=r.nit,
        message=str(r.message),
        success=bool(r.success),
        gradient_inf=float(abs(r.jac).max()),
        hessian_min_eigenvalue=float(np.linalg.eigvalsh(H).min()),
        hessian_condition=float(np.linalg.cond(H)),
        max_abs_coefficient=float(abs(W).max()),
        bootstrap_replicates=n_boot,
        bootstrap_seed=SEED + 200,
    )
    dump("Q2_diagnostics.json", diagnostics)
    # Patsy DesignInfo is not pickleable: formula, training frame and exact design
    # are persisted with coefficients, making the learned knots reproducible.
    joblib.dump(
        {
            "coefficients": W,
            "covariance": cov,
            "formula": FORMULA,
            "design_columns": list(design.columns),
            "training_variables": v,
            "design": X,
            "penalty": penalty,
        },
        "models/Q2.joblib",
    )
    return diagnostics, pd.DataFrame(contrasts)
