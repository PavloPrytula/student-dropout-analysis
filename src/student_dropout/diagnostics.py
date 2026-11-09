"""Report tables and checks computed only from saved training / CV results."""

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import check_grad
from scipy.special import expit, logit

from .data import CANDIDATES, FEATURES, OUT, dump, prepare
from .prediction import ProbabilityLogit, metrics, model


def run_diagnostics():
    t, s = prepare()
    o = pd.read_csv(OUT / "oof.csv")
    y = o.y.to_numpy()
    rows = []
    for cid in CANDIDATES:
        p = o[cid].to_numpy()
        row = {"candidate": cid, **metrics(y, p)}
        if cid != "B0":
            cal = sm.GLM(
                y,
                sm.add_constant(logit(np.clip(p, 1e-10, 1 - 1e-10))),
                family=sm.families.Binomial(),
            ).fit()
            row.update(
                calibration_intercept=cal.params[0], calibration_slope=cal.params[1]
            )
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    bins = []
    for cid in ["P1", "P2", "P3", "P4", "P1B"]:
        z = o[["y", cid]].copy()
        z["bin"] = pd.qcut(z[cid], 10, duplicates="drop")
        b = (
            z.groupby("bin", observed=True)
            .agg(
                n=("y", "size"),
                mean_probability=(cid, "mean"),
                observed_dropout=("y", "mean"),
            )
            .reset_index()
        )
        b["candidate"] = cid
        bins.append(b)
    pd.concat(bins).to_csv(OUT / "calibration_bins.csv", index=False)
    diff = o.P1B - o.P1
    dump(
        "criterion_comparison.json",
        {
            "mean_abs_oof_probability_change": float(abs(diff).mean()),
            "max_abs_oof_probability_change": float(abs(diff).max()),
            "rms_oof_probability_change": float(np.sqrt(np.mean(diff**2))),
            "changed_labels_at_half": int(((o.P1 >= 0.5) != (o.P1B >= 0.5)).sum()),
            "paired_log_loss_difference_Brier_minus_log": float(
                metrics(y, o.P1B)["log_loss"] - metrics(y, o.P1)["log_loss"]
            ),
        },
    )
    fig, ax = plt.subplots(figsize=(6.8, 2.7))
    for cid, color in [("P1", "#176B87"), ("P2", "#8B448C"), ("P1B", "#D57925")]:
        r = sorted(
            json.loads((OUT / f"search_{cid}_final.json").read_text()),
            key=lambda r: r["params"]["alpha"],
        )
        a = [x["params"]["alpha"] for x in r]
        v = [x["mean_log_loss"] for x in r]
        ax.plot(a, v, "o-", label=cid, color=color)
        ax.scatter(a[np.argmin(v)], min(v), s=90, facecolors="none", edgecolors=color)
    ax.set(
        xscale="log",
        xlabel="Penalty alpha (mean loss + alpha/2 ||beta||²)",
        ylabel="Mean inner-CV log loss",
    )
    ax.legend(ncol=3)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "cv_curve.png", dpi=200)
    plt.close(fig)
    q1 = t[t.Target.isin(["Dropout", "Graduate"])]
    q1.groupby("Target")[
        [
            "Age at enrollment",
            "Admission grade",
            "Previous qualification (grade)",
            "Gender",
        ]
    ].agg(["size", "mean", "median"]).to_csv(OUT / "Q1_training.csv")
    q1.groupby(["Course", "Target"])[["Age at enrollment", "Admission grade"]].agg(
        ["size", "mean"]
    ).to_csv(OUT / "Q1_training_by_course.csv")
    fitted = []
    for cid in ["P1", "P2", "P1B"]:
        m = joblib.load(f"models/{cid}.joblib")
        z = m.named_steps["pre"].transform(t[FEATURES])
        A = np.c_[np.ones(len(z)), z]
        est = m.named_steps["model"]
        p = expit(A @ np.r_[est.intercept_, est.coef_[0]])
        H = A.T @ ((p * (1 - p))[:, None] * A) / len(A)
        pen = np.eye(A.shape[1]) * est.alpha
        pen[0, 0] = 0
        fitted.append(
            {
                "candidate": cid,
                "log_likelihood_effective_df": float(
                    np.trace(np.linalg.solve(H + pen, H))
                )
                if cid != "P1B"
                else None,
            }
        )
    dump("effective_df.json", fitted)
    checks = {}
    checks["split_disjoint_complete"] = (
        not set(s["train"]) & set(s["test"])
        and len(s["train"]) + len(s["test"]) == 4424
    )
    checks["oof_only_training"] = set(o.row_id) == set(s["train"]) and not set(
        o.row_id
    ) & set(s["test"])
    folds = json.loads((OUT / "folds.json").read_text())
    valid = []
    for k in map(str, range(5)):
        f = folds[k]
        assert not set(f["train"]) & set(f["valid"])
        valid += f["valid"]
        for g in f["inner"]:
            assert set(g["train"] + g["valid"]) == set(f["train"]) and not set(
                g["train"]
            ) & set(g["valid"])
    checks["fold_containment_and_coverage"] = sorted(valid) == sorted(s["train"])
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 4))
    yy = rng.integers(2, size=20)
    w = rng.normal(size=5)
    for criterion in ["log", "brier"]:
        est = ProbabilityLogit(criterion=criterion)
        err = check_grad(
            lambda v: est.loss_grad(v, X, yy)[0],
            lambda v: est.loss_grad(v, X, yy)[1],
            w,
        )
        checks[f"{criterion}_gradient_error"] = err
        assert err < 1e-6
    for cid in ["P1", "P2", "P1B", "P3", "P4"]:
        m = joblib.load(f"models/{cid}.joblib")
        pred = m.predict_proba(t[FEATURES])[:, 1]
        checks[f"{cid}_finite_probabilities"] = bool(
            np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all()
        )
        used = [
            c
            for _, _, cols in m.named_steps["pre"].transformers_
            if isinstance(cols, list)
            for c in cols
        ]
        checks[f"{cid}_allowed_features"] = set(used) == set(FEATURES)
        if cid in ["P1", "P2", "P1B", "P3", "P4"]:
            params = json.loads((OUT / "model_settings.json").read_text())[cid][
                "chosen"
            ]
            mm = model(cid, params).fit(
                t[FEATURES], (t.Target == "Dropout").astype(int)
            )
            checks[f"{cid}_refit_max_probability_difference"] = float(
                abs(mm.predict_proba(t[FEATURES])[:, 1] - pred).max()
            )
            assert checks[f"{cid}_refit_max_probability_difference"] < 1e-10
    # Original EDA is reproduced in-memory by the notebook, with source hashes retained.
    checks["raw_data_sha256"] = hashlib.sha256(
        Path("data/data.csv").read_bytes()
    ).hexdigest()
    checks["main_notebook_sha256"] = hashlib.sha256(
        Path("main.ipynb").read_bytes()
    ).hexdigest()
    assert all(v for v in checks.values() if isinstance(v, bool))
    reference = OUT / "reproduction_reference_oof.csv"
    if reference.exists():
        previous = pd.read_csv(reference)
        assert previous[["row_id", "y"]].equals(o[["row_id", "y"]])
        differences = {
            c: float(abs(previous[c] - o[c]).max())
            for c in ["B0", "P1", "P2", "P3", "P4", "P1B"]
        }
        assert max(differences.values()) < 1e-10
        dump(
            "reproduction.json",
            {
                "description": "Fresh no-cache full search versus previous completed protocol",
                "max_abs_oof_difference": differences,
            },
        )
    stops = []
    for cid in ["P3", "P4"]:
        m = joblib.load(f"models/{cid}.joblib").named_steps["model"]
        trees = np.asarray(m.estimators_, dtype=object).ravel()
        min_leaf = min(
            int(tree.tree_.n_node_samples[tree.tree_.children_left == -1].min())
            for tree in trees
        )
        max_depth = max(tree.tree_.max_depth for tree in trees)
        assert len(trees) == m.n_estimators and min_leaf >= m.min_samples_leaf
        assert m.max_depth is None or max_depth <= m.max_depth
        stops.append(
            {
                "candidate": cid,
                "tree_count": len(trees),
                "max_observed_depth": max_depth,
                "minimum_leaf_training_samples": min_leaf,
                "constraints_satisfied": True,
            }
        )
    dump("tree_stopping_checks.json", stops)
    dump("checks.json", checks)
    # Budget counts include 3 folds/config + 6 outer/final refits per candidate.
    budget = []
    for cid in CANDIDATES:
        paths = list(OUT.glob(f"search_{cid}_*.json"))
        nconf = sum(len(json.loads(p.read_text())) for p in paths)
        budget.append(
            {
                "candidate": cid,
                "evaluated_configurations_across_6_searches": nconf,
                "cv_fits": 3 * nconf,
                "outer_and_final_refits": 6,
                "optimizer_starts_per_fit": 3 if cid == "P1B" else 1,
            }
        )
    pd.DataFrame(budget).to_csv(OUT / "budget.csv", index=False)
    return pd.DataFrame(rows), checks


def reproduce_historical_eda():
    """Execute original cells in memory; never rewrite the original notebook."""
    import warnings

    from IPython.display import display
    from IPython.utils.capture import capture_output

    original = json.loads(Path("main.ipynb").read_text())
    namespace = {"display": display}
    executed = []
    with warnings.catch_warnings(record=True) as ws, capture_output():
        warnings.simplefilter("always")
        for i, cell in enumerate(original["cells"]):
            if cell["cell_type"] == "code":
                exec(
                    compile(
                        "".join(cell["source"]), f"main.ipynb cell {i + 1}", "exec"
                    ),
                    namespace,
                )
                executed.append(i + 1)
        plt.close("all")
    result = {
        "executed_code_cells": executed,
        "warnings": [str(w.message) for w in ws],
        "resolved_n": len(namespace["df"]),
        "age_means": namespace["df"]
        .groupby("Target")["Age at enrollment"]
        .mean()
        .to_dict(),
        "course_cramers_v": namespace["v"],
    }
    dump("historical_eda_reproduction.json", result)
    return result
