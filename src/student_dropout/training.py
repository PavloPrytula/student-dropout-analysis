"""Nested training-only search, final refits and fixed-specification diagnostics."""

import importlib.metadata
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from threadpoolctl import threadpool_limits

from .data import CANDIDATES, FEATURES, OUT, SEED, dump, prepare
from .prediction import OPTIMIZATION_LOG, metrics, model

RESUME = False


def spaces(candidate_id: str) -> list[dict]:
    """Return the original seeded candidate configurations before boundary extension."""
    if candidate_id == "B0":
        return [{}]
    if candidate_id in ["P1", "P2", "P1B"]:
        return [{"alpha": v} for v in [1e-5, 1e-4, 0.001, 0.01, 0.1]]
    if candidate_id == "P3":
        return list(
            ParameterSampler(
                {
                    "max_depth": [4, 8, None],
                    "min_samples_leaf": [5, 15, 30],
                    "max_features": ["sqrt", 0.5, 1.0],
                },
                n_iter=8,
                random_state=SEED,
            )
        )
    return list(
        ParameterSampler(
            {
                "max_depth": [1, 2, 3],
                "min_samples_leaf": [10, 30],
                "learning_rate": [0.03, 0.1],
                "n_estimators": [100, 300],
            },
            n_iter=8,
            random_state=SEED,
        )
    )


def search(
    candidate_id: str, X: pd.DataFrame, y: np.ndarray, folds: list, tag: str
) -> dict:
    """Choose parameters using only the supplied inner folds and fixed extension rules."""
    configs = spaces(candidate_id)
    cache = OUT / f"search_{candidate_id}_{tag}.json"
    rows = json.loads(cache.read_text()) if RESUME and cache.exists() else []

    def evaluate(p):
        if any(r["params"] == p for r in rows):
            return
        losses = []
        for train_indices, valid_indices in folds:
            estimator = model(candidate_id, p).fit(
                X.iloc[train_indices], y[train_indices]
            )
            losses.append(
                log_loss(
                    y[valid_indices],
                    estimator.predict_proba(X.iloc[valid_indices])[:, 1],
                )
            )
        rows.append(
            {
                "candidate": candidate_id,
                "stage": tag,
                "params": p,
                "mean_log_loss": np.mean(losses),
                "fold_losses": losses,
            }
        )

    for p in configs:
        evaluate(p)
    if candidate_id in ["P1", "P2", "P1B"]:
        for _ in range(3):
            best = min(rows, key=lambda r: r["mean_log_loss"])["params"]["alpha"]
            vals = [r["params"]["alpha"] for r in rows]
            if best == min(vals):
                evaluate({"alpha": best / 10})
            elif best == max(vals):
                evaluate({"alpha": best * 10})
            else:
                break
    if candidate_id in ["P3", "P4"]:
        # One pre-specified expansion pass, only on INNER validation losses.
        best0 = min(rows, key=lambda r: r["mean_log_loss"])["params"].copy()
        extension = (
            {"min_samples_leaf": {5: 2, 30: 60}, "max_depth": {4: 2}}
            if candidate_id == "P3"
            else {
                "n_estimators": {100: 50, 300: 600},
                "learning_rate": {0.03: 0.01, 0.1: 0.2},
                "min_samples_leaf": {10: 5, 30: 60},
            }
        )
        for key, mapping in extension.items():
            if best0[key] in mapping:
                p = best0.copy()
                p[key] = mapping[p[key]]
                evaluate(p)
    best = min(rows, key=lambda r: r["mean_log_loss"])
    dump(f"search_{candidate_id}_{tag}.json", rows)
    return best["params"]


def summarize_candidate(candidate_id: str, fitted_model, y, probabilities) -> dict:
    """Describe one final fit and its separate outer-fold predictions; save coefficients."""
    summary = {"candidate": candidate_id, **metrics(y, probabilities)}
    if candidate_id == "B0":
        summary["prevalence"] = fitted_model.class_prior_[1]
    elif candidate_id in ["P1", "P2", "P1B"]:
        est = fitted_model.named_steps["model"]
        summary.update(
            n_coefficients=est.coef_.size,
            intercept=est.intercept_[0],
            coef_norm=float(np.linalg.norm(est.coef_)),
            starts=est.starts_,
        )
        pd.DataFrame(
            {
                "feature": fitted_model.named_steps["pre"].get_feature_names_out(),
                "coefficient": est.coef_[0],
            }
        ).to_csv(OUT / f"{candidate_id}_coefficients.csv", index=False)
    else:
        est = fitted_model.named_steps["model"]
        trees = np.array(est.estimators_, dtype=object).ravel()
        summary.update(
            n_trees=len(trees),
            mean_depth=float(np.mean([r.tree_.max_depth for r in trees])),
            mean_leaves=float(np.mean([r.tree_.n_leaves for r in trees])),
        )
    return summary


def run_sensitivity_checks(X, y, outer, configs, oof) -> None:
    """Run fixed tree seeds and parental-field sensitivity on training folds only."""
    # Fixed-specification diagnostics on one train/validation split, never test.
    train_indices, valid_indices = outer[0]
    checks = []
    for candidate_id in ["P3", "P4"]:
        for seed in [481, 482, 483]:
            estimator = model(candidate_id, configs[candidate_id]["chosen"], seed).fit(
                X.iloc[train_indices], y[train_indices]
            )
            checks.append(
                {
                    "candidate": candidate_id,
                    "seed": seed,
                    **metrics(
                        y[valid_indices],
                        estimator.predict_proba(X.iloc[valid_indices])[:, 1],
                    ),
                }
            )
    dump("tree_stability.json", checks)
    # Unknown-code sensitivity: remove all four parental columns, fixed P1 penalty, same outer folds.
    p = np.zeros(len(y))
    for train_indices, valid_indices in outer:
        estimator = model("P1", configs["P1"]["chosen"], omit_parent=True).fit(
            X.iloc[train_indices], y[train_indices]
        )
        p[valid_indices] = estimator.predict_proba(X.iloc[valid_indices])[:, 1]
    dump(
        "sensitivity.json",
        {
            "description": "Omit all four parental fields; fixed final P1 penalty, diagnostic only (not nested selection).",
            "metrics": metrics(y, p),
            "mean_abs_probability_change": np.mean(abs(p - oof.P1.to_numpy())),
        },
    )


def run_prediction() -> list[dict]:
    """Run nested Q3 selection and save final fits; never predict on reserved rows."""
    training, _ = prepare()
    X = training[FEATURES]
    y = (training.Target == "Dropout").astype(int).to_numpy()
    outer = list(
        StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, training.Target)
    )
    fold_membership = {}
    summaries = []
    outer_rows = []
    oof = pd.DataFrame({"row_id": training.index, "y": y})
    configs = {}
    for candidate_id in CANDIDATES:
        oof_probabilities = np.zeros(len(y))
        print("Training", candidate_id, flush=True)
        for k, (train_indices, valid_indices) in enumerate(outer):
            inner = list(
                StratifiedKFold(3, shuffle=True, random_state=SEED + k + 1).split(
                    X.iloc[train_indices], training.Target.iloc[train_indices]
                )
            )
            fold_membership[str(k)] = {
                "train": training.index[train_indices].tolist(),
                "valid": training.index[valid_indices].tolist(),
                "inner": [
                    {
                        "train": training.index[train_indices[i]].tolist(),
                        "valid": training.index[train_indices[j]].tolist(),
                    }
                    for i, j in inner
                ],
            }
            p = search(
                candidate_id,
                X.iloc[train_indices],
                y[train_indices],
                inner,
                f"outer{k}",
            )
            estimator = model(candidate_id, p).fit(
                X.iloc[train_indices], y[train_indices]
            )
            oof_probabilities[valid_indices] = estimator.predict_proba(
                X.iloc[valid_indices]
            )[:, 1]
            outer_rows.append(
                {
                    "candidate": candidate_id,
                    "fold": k,
                    **metrics(y[valid_indices], oof_probabilities[valid_indices]),
                    "params": json.dumps(p),
                }
            )
            print(
                candidate_id,
                "outer",
                k,
                round(log_loss(y[valid_indices], oof_probabilities[valid_indices]), 5),
                flush=True,
            )
        final_folds = list(
            StratifiedKFold(3, shuffle=True, random_state=SEED + 100).split(
                X, training.Target
            )
        )
        params = search(candidate_id, X, y, final_folds, "final")
        estimator = model(candidate_id, params).fit(X, y)
        joblib.dump(estimator, Path("models") / f"{candidate_id}.joblib")
        configs[candidate_id] = {
            "chosen": params,
            "settings": estimator.get_params(deep=True),
        }
        summary = summarize_candidate(candidate_id, estimator, y, oof_probabilities)
        summaries.append(summary)
        oof[candidate_id] = oof_probabilities
        dump("summary.json", summaries)
        dump("model_settings.json", configs)
        oof.to_csv(OUT / "oof.csv", index=False)
        pd.DataFrame(outer_rows).to_csv(OUT / "outer_scores.csv", index=False)
    fold_membership["final"] = [
        {
            "train": training.index[train_indices].tolist(),
            "valid": training.index[valid_indices].tolist(),
        }
        for train_indices, valid_indices in final_folds
    ]
    dump("folds.json", fold_membership)
    run_sensitivity_checks(X, y, outer, configs, oof)
    return summaries


def run(resume: bool = False) -> list[dict]:
    """Execute Q3 with one numerical thread and persist optimizer/runtime evidence."""
    global RESUME
    RESUME = resume
    OPTIMIZATION_LOG.clear()
    started = time.time()
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        result = run_prediction()
    dump("optimization_log.json", OPTIMIZATION_LOG)
    dump(
        "warnings.json",
        [{"category": w.category.__name__, "message": str(w.message)} for w in ws],
    )
    dump(
        "runtime.json",
        {
            "seconds": time.time() - started,
            "seed": SEED,
            "python": platform.python_version(),
            "versions": {
                p: importlib.metadata.version(p)
                for p in [
                    "numpy",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "statsmodels",
                    "joblib",
                    "threadpoolctl",
                    "matplotlib",
                    "nbformat",
                    "nbclient",
                ]
            },
        },
    )
    return result
