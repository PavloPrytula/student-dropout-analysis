"""Q3 candidate estimators and probability metrics; no fitting at import time."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

from .data import SEED
from .preprocessing import preprocessor

OPTIMIZATION_LOG = []


class ProbabilityLogit(ClassifierMixin, BaseEstimator):
    """Mean proper loss + alpha/2 ||beta||², intercept unpenalized."""

    def __init__(self, alpha=0.001, criterion="log", seed=SEED):
        self.alpha = alpha
        self.criterion = criterion
        self.seed = seed

    def loss_grad(self, w, X, y):
        """Return penalized mean loss and analytic gradient for transformed inputs."""
        z = X @ w[1:] + w[0]
        p = expit(z)
        if self.criterion == "log":
            loss = np.mean(np.logaddexp(0, z) - y * z)
            r = p - y
        else:
            loss = np.mean((p - y) ** 2)
            r = 2 * (p - y) * p * (1 - p)
        return loss + self.alpha / 2 * np.dot(w[1:], w[1:]), np.r_[
            r.mean(), X.T @ r / len(y) + self.alpha * w[1:]
        ]

    def fit(self, X, y):
        """Fit numeric inputs with fixed starts and tolerances; verify the final gradient."""
        X = np.asarray(X)
        y = np.asarray(y)
        rng = np.random.default_rng(self.seed)
        starts = [np.zeros(X.shape[1] + 1)]
        if self.criterion == "brier":
            starts += [
                rng.normal(0, 0.1, X.shape[1] + 1),
                rng.normal(0, 0.5, X.shape[1] + 1),
            ]
        results = [
            minimize(
                self.loss_grad,
                w,
                args=(X, y),
                jac=True,
                method="L-BFGS-B",
                options={"gtol": 1e-7, "ftol": 1e-13, "maxiter": 3000, "maxls": 50},
            )
            for w in starts
        ]
        spread_before = float(max(r.fun for r in results) - min(r.fun for r in results))
        second_optimizer = spread_before > 1e-7
        if second_optimizer:
            results += [
                minimize(
                    self.loss_grad,
                    r.x,
                    args=(X, y),
                    jac=True,
                    method="BFGS",
                    options={"gtol": 1e-7, "maxiter": 3000},
                )
                for r in list(results)
            ]
        self.starts_ = [
            {
                "success": bool(r.success),
                "message": r.message,
                "nit": r.nit,
                "objective": r.fun,
                "gradient_inf": float(abs(r.jac).max()),
            }
            for r in results
        ]
        r = min(results, key=lambda r: r.fun)
        self.coef_ = r.x[1:][None, :]
        self.intercept_ = np.array([r.x[0]])
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        if not r.success or abs(r.jac).max() > 1e-5:
            raise RuntimeError(self.starts_)
        OPTIMIZATION_LOG.append(
            {
                "criterion": self.criterion,
                "alpha": self.alpha,
                "n": len(y),
                "p": X.shape[1],
                "second_optimizer": second_optimizer,
                "initial_objective_spread": spread_before,
                "objective_spread": float(
                    max(r.fun for r in results) - min(r.fun for r in results)
                ),
                "best_gradient_inf": float(abs(r.jac).max()),
                "starts": self.starts_,
            }
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return probabilities in class order [non-Dropout, Dropout]."""
        p = expit(np.asarray(X) @ self.coef_[0] + self.intercept_[0])
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def model(candidate_id: str, params: dict, seed: int = SEED, omit_parent: bool = False):
    """Build an unfitted B0/P1–P4 or P1B estimator with the frozen preprocessing policy."""
    if candidate_id == "B0":
        return DummyClassifier(strategy="prior")
    if candidate_id in ["P1", "P2", "P1B"]:
        estimator = ProbabilityLogit(
            criterion="brier" if candidate_id == "P1B" else "log", seed=seed, **params
        )
    elif candidate_id == "P3":
        estimator = RandomForestClassifier(
            n_estimators=500, criterion="gini", random_state=seed, n_jobs=1, **params
        )
    else:
        estimator = GradientBoostingClassifier(
            loss="log_loss", random_state=seed, n_iter_no_change=None, **params
        )
    return Pipeline(
        [("pre", preprocessor(candidate_id == "P2", omit_parent)), ("model", estimator)]
    )


def metrics(y, p) -> dict[str, float]:
    """Summarize binary labels/probabilities; 0.5 classification is descriptive only."""
    pr, re, f, _ = precision_recall_fscore_support(
        y, p >= 0.5, average="binary", zero_division=0
    )
    return dict(
        log_loss=log_loss(y, p),
        brier=brier_score_loss(y, p),
        average_precision=average_precision_score(y, p),
        precision=pr,
        recall=re,
        f1=f,
    )
