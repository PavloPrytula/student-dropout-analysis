"""Q4 review allocation, integrated with Part II's saved nested-CV predictions.
No model fitting, test predictions, causal effects or CV winner selection.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.model_selection import StratifiedKFold

from .data import CANDIDATES, OUT, SEED, dump, load_data
from .prediction import metrics

CAPACITIES = (0.05, 0.10, 0.20)
TIE_SEED = 1481
BOOT_SEED = 2481


def tie_key(row_id: int) -> str:
    """Hash the supplied record position independently of outcome and score."""
    return hashlib.sha256(f"{TIE_SEED}:{int(row_id)}".encode()).hexdigest()


def select_top(frame: pd.DataFrame, score: str, k: int) -> np.ndarray:
    """Return ordered row IDs using risk, fixed hash ties, then ID; no outcomes enter."""
    if not isinstance(k, (int, np.integer)) or not 0 <= k <= len(frame):
        raise ValueError("Invalid list size")
    if frame.row_id.duplicated().any():
        raise ValueError("Duplicate row IDs")
    if not np.isfinite(frame[score]).all() or not frame[score].between(0, 1).all():
        raise ValueError("Invalid probabilities")
    return (
        frame.assign(_tie=frame.row_id.map(tie_key))
        .sort_values([score, "_tie", "row_id"], ascending=[False, True, True])
        .head(k)
        .row_id.to_numpy()
    )


def allocation_metrics(n: int, d: int, k: int, h: int) -> dict:
    """Summarize cohort/dropout/selected/hit counts against equal-capacity random review."""
    if not (
        0 <= h <= min(k, d) and max(0, k - (n - d)) <= h and 0 <= k <= n and 0 <= d <= n
    ):
        raise ValueError("Inconsistent counts")
    e = k * d / n if n else 0.0
    variance = k * (d / n) * (1 - d / n) * (n - k) / (n - 1) if n > 1 else 0.0
    return dict(
        n=n,
        dropouts=d,
        k=k,
        identified=h,
        precision=h / k if k else np.nan,
        recall=h / d if d else np.nan,
        random_expected=e,
        extra_identified=h - e,
        lift=h / e if e else np.nan,
        random_variance=variance,
    )


def validate_oof() -> tuple[pd.DataFrame, dict]:
    """Load saved OOF scores after checking training-only identity, folds and losses."""
    s = json.loads((OUT / "split.json").read_text())
    folds = json.loads((OUT / "folds.json").read_text())
    o = pd.read_csv(OUT / "oof.csv")
    raw = load_data()
    assert hashlib.sha256(Path("data/data.csv").read_bytes()).hexdigest() == s["sha256"]
    assert (
        not o.row_id.duplicated().any()
        and set(o.row_id) == set(s["train"])
        and not set(o.row_id) & set(s["test"])
    )
    assert np.array_equal(o.y, raw.loc[o.row_id, "Target"].eq("Dropout").astype(int))
    training = raw.iloc[s["train"]]
    expected = list(
        StratifiedKFold(5, shuffle=True, random_state=SEED).split(
            training, training.Target
        )
    )
    outer_scores = pd.read_csv(OUT / "outer_scores.csv")
    assigned = []
    for f, (a, b) in enumerate(expected):
        saved = folds[str(f)]
        assert (
            saved["train"] == training.index[a].tolist()
            and saved["valid"] == training.index[b].tolist()
        )
        assert not set(saved["train"]) & set(saved["valid"])
        assigned += saved["valid"]
        inner = list(
            StratifiedKFold(3, shuffle=True, random_state=SEED + f + 1).split(
                training.iloc[a], training.Target.iloc[a]
            )
        )
        for (i, j), g in zip(inner, saved["inner"], strict=True):
            assert (
                g["train"] == training.index[a[i]].tolist()
                and g["valid"] == training.index[a[j]].tolist()
            )
            assert not set(g["train"] + g["valid"]) & set(saved["valid"])
        v = o[o.row_id.isin(saved["valid"])]
        for cid in CANDIDATES:
            search = json.loads((OUT / f"search_{cid}_outer{f}.json").read_text())
            best = min(search, key=lambda r: r["mean_log_loss"])["params"]
            score = outer_scores[
                (outer_scores.candidate == cid) & (outer_scores.fold == f)
            ].iloc[0]
            assert best == json.loads(score.params)
            assert abs(metrics(v.y, v[cid])["log_loss"] - score.log_loss) < 1e-12
    assert sorted(assigned) == sorted(s["train"])
    return o, folds


def random_reference(fold_rows, h):
    """Exact distribution of H - sum(R_f), R_f uniform-without-replacement counts."""
    pmf = np.array([1.0])
    for r in fold_rows:
        p = hypergeom.pmf(np.arange(r["k"] + 1), r["n"], r["dropouts"], r["k"])
        pmf = np.convolve(pmf, p)
    pmf /= pmf.sum()
    cdf = np.cumsum(pmf)
    lo, hi = np.searchsorted(cdf, [0.025, 0.975])
    return int(h - hi), int(h - lo)


def bootstrap_gains(
    groups: list[pd.DataFrame], cid: str, q: float, n_boot: int
) -> np.ndarray:
    """Return conditional pair-bootstrap gains; frozen scores exclude refit uncertainty."""
    rng = np.random.default_rng(BOOT_SEED)
    deltas = np.zeros(n_boot)
    for v in groups:
        ranked_ids = select_top(v, cid, len(v))
        positions = (
            pd.Series(np.arange(len(v)), index=v.row_id).loc[ranked_ids].to_numpy()
        )
        counts = rng.multinomial(len(v), np.full(len(v), 1 / len(v)), size=n_boot)
        ranked_counts = counts[:, positions]
        kk = math.floor(q * len(v))
        remaining = kk - (np.cumsum(ranked_counts, axis=1) - ranked_counts)
        chosen = np.minimum(np.maximum(remaining, 0), ranked_counts)
        hits = chosen @ v.y.to_numpy()[positions]
        random_mean = kk * (counts @ v.y.to_numpy()) / len(v)
        deltas += hits - random_mean
    return deltas


def run_decision(n_boot: int = 2000) -> pd.DataFrame:
    """Evaluate every fixed rule on saved OOF scores and write conditional Q4 evidence."""
    o, folds = validate_oof()
    fold_rows = []
    selected = []
    summary = []
    bootstrap = []
    for cid in CANDIDATES:
        groups = [o[o.row_id.isin(folds[str(f)]["valid"])].copy() for f in range(5)]
        # Fixed candidate rules; no selection by existing CV scores or Q4 outcomes.
        for q in CAPACITIES:
            local = []
            for f, v in enumerate(groups):
                k = math.floor(q * len(v))
                ids = select_top(v, cid, k)
                h = int(v[v.row_id.isin(ids)].y.sum())
                r = dict(
                    candidate=cid,
                    capacity=q,
                    fold=f,
                    **allocation_metrics(len(v), int(v.y.sum()), k, h),
                )
                local.append(r)
                fold_rows.append(r)
                selected.extend(
                    dict(candidate=cid, capacity=q, fold=f, row_id=int(i)) for i in ids
                )
            n = sum(r["n"] for r in local)
            d = sum(r["dropouts"] for r in local)
            k = sum(r["k"] for r in local)
            h = sum(r["identified"] for r in local)
            e = sum(r["random_expected"] for r in local)
            row = dict(
                candidate=cid,
                capacity=q,
                n=n,
                dropouts=d,
                k=k,
                identified=h,
                precision=h / k,
                recall=h / d,
                random_expected=e,
                extra_identified=h - e,
                lift=h / e,
                random_selection_sd=np.sqrt(sum(r["random_variance"] for r in local)),
            )
            row["random_reference_low"], row["random_reference_high"] = (
                random_reference(local, h)
            )
            # Conditional empirical-pair bootstrap, separately within each fixed validation fold.
            # Treat scores as frozen marks; does NOT refit, correct CV dependence, or account
            # for training/model-selection uncertainty. Report as sensitivity, not full CV CI.
            deltas = bootstrap_gains(groups, cid, q, n_boot)
            row.update(
                conditional_bootstrap_se=float(np.std(deltas, ddof=1)),
                conditional_bootstrap_low=float(np.quantile(deltas, 0.025)),
                conditional_bootstrap_high=float(np.quantile(deltas, 0.975)),
            )
            bootstrap.extend(
                dict(candidate=cid, capacity=q, replicate=b, extra_identified=val)
                for b, val in enumerate(deltas)
            )
            summary.append(row)
        print("Q4 evaluated", cid, flush=True)
    pd.DataFrame(fold_rows).to_csv(OUT / "Q4_fold_metrics.csv", index=False)
    pd.DataFrame(selected).to_csv(OUT / "Q4_selected_rows.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "Q4_summary.csv", index=False)
    pd.DataFrame(bootstrap).to_csv(OUT / "Q4_bootstrap.csv", index=False)
    paths = [
        Path("data/data.csv"),
        OUT / "oof.csv",
        OUT / "folds.json",
        OUT / "split.json",
        Path("src/student_dropout/data.py"),
        Path("src/student_dropout/preprocessing.py"),
        Path("src/student_dropout/prediction.py"),
        Path("src/student_dropout/training.py"),
        Path("src/student_dropout/decision.py"),
    ]
    dump(
        "Q4_protocol.json",
        {
            "capacities": CAPACITIES,
            "candidates": CANDIDATES,
            "rule": "floor(capacity * validation-fold N), descending risk then SHA256(1481:row_id), then row_id",
            "tie_seed": TIE_SEED,
            "bootstrap_seed": BOOT_SEED,
            "bootstrap_replicates": n_boot,
            "pooling": "sum counts and random expectations across folds; precision=H/K, recall=H/D, lift=H/sum(k_f D_f/N_f); never pool ranks",
            "uncertainty": "Exact hypergeometric random-reference interval conditional on fixed folds and observed outcomes; bootstrap is empirical composition sensitivity conditional on frozen OOF pairs, not full training/selection/CV uncertainty",
            "candidate_selection": "None: all six pre-existing rules reported, no independent winner claim",
            "provenance_checks": "dataset hash, row identity/event mapping, recreated outer/inner splits, chosen inner parameters, OOF-to-outer-score agreement; fitting/preprocessing code reviewed",
            "input_sha256": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
            },
        },
    )
    return pd.DataFrame(summary)
