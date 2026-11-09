"""Post-review descriptive diagnostics; no retuning, test scoring or winner selection."""

import hashlib
import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, precision_recall_curve
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import FEATURES, OUT, dump, load_data
from .decision import CAPACITIES, validate_oof
from .prediction import model

COLORS = {"P1": "#216C91", "P2": "#8A4F98", "P3": "#7C8357", "P4": "#D66A2C"}


def paired_and_groups():
    o, folds = validate_oof()
    selected = pd.read_csv(OUT / "Q4_selected_rows.csv")
    boot = pd.read_csv(OUT / "Q4_bootstrap.csv")
    raw = load_data()
    rows = []
    for q in CAPACITIES:
        for fold in range(5):
            sets = {
                cid: set(
                    selected[
                        (selected.candidate == cid)
                        & (selected.capacity == q)
                        & (selected.fold == fold)
                    ].row_id
                )
                for cid in ["P2", "P4"]
            }
            both = sets["P2"] & sets["P4"]
            only2 = sets["P2"] - sets["P4"]
            only4 = sets["P4"] - sets["P2"]
            lookup = o.set_index("row_id").y
            rows.append(
                dict(
                    capacity=q,
                    fold=fold,
                    k=len(sets["P2"]),
                    both=len(both),
                    P2_only=len(only2),
                    P4_only=len(only4),
                    both_dropouts=int(lookup.loc[list(both)].sum()),
                    P2_only_dropouts=int(lookup.loc[list(only2)].sum()),
                    P4_only_dropouts=int(lookup.loc[list(only4)].sum()),
                )
            )
    paired = pd.DataFrame(rows)
    paired["difference_P2_minus_P4"] = paired.P2_only_dropouts - paired.P4_only_dropouts
    paired.to_csv(OUT / "review_paired_folds.csv", index=False)
    pooled = []
    for q, g in paired.groupby("capacity"):
        r = g.drop(columns=["capacity", "fold"]).sum().to_dict()
        r["capacity"] = q
        r["overlap_fraction"] = r["both"] / r["k"]
        # decision.run_decision resets the same RNG for each rule, using identical
        # multinomial draws in the same original-row order: these replicates are paired.
        b2 = (
            boot[(boot.candidate == "P2") & (boot.capacity == q)]
            .set_index("replicate")
            .extra_identified
        )
        b4 = (
            boot[(boot.candidate == "P4") & (boot.capacity == q)]
            .set_index("replicate")
            .extra_identified
        )
        diff = b2 - b4
        r.update(
            conditional_paired_se=diff.std(ddof=1),
            conditional_paired_low=diff.quantile(0.025),
            conditional_paired_high=diff.quantile(0.975),
        )
        pooled.append(r)
    pd.DataFrame(pooled).to_csv(OUT / "review_paired_summary.csv", index=False)
    # No re-ranking or quotas within subgroups: evaluate the same original fold lists.
    frame = o[["row_id", "y"]].copy()
    frame["Course"] = raw.loc[frame.row_id, "Course"].to_numpy()
    frame["Gender"] = (
        raw.loc[frame.row_id, "Gender"].map({0: "Female", 1: "Male"}).to_numpy()
    )
    frame["Age group"] = pd.cut(
        raw.loc[frame.row_id, "Age at enrollment"].to_numpy(),
        [0, 20, 25, 30, np.inf],
        labels=["<=20", "21-25", "26-30", "31+"],
    )
    group_rows = []
    for cid in ["P1", "P2", "P3", "P4"]:
        for q in CAPACITIES:
            ids = set(
                selected[(selected.candidate == cid) & (selected.capacity == q)].row_id
            )
            frame["selected"] = frame.row_id.isin(ids)
            for dimension in ["Course", "Gender", "Age group"]:
                for label, g in frame.groupby(dimension, observed=True):
                    n = len(g)
                    d = int(g.y.sum())
                    k = int(g.selected.sum())
                    h = int(g.loc[g.selected, "y"].sum())
                    group_rows.append(
                        dict(
                            candidate=cid,
                            capacity=q,
                            dimension=dimension,
                            group=str(label),
                            n=n,
                            dropouts=d,
                            selected=k,
                            identified=h,
                            selection_rate=k / n,
                            precision=h / k if k else np.nan,
                            recall=h / d if d else np.nan,
                            sparse=(n < 50 or d < 20 or k < 20),
                        )
                    )
    pd.DataFrame(group_rows).to_csv(OUT / "review_subgroups.csv", index=False)

    def loss(p):
        return -(
            o.y * np.log(np.clip(p, 1e-15, 1 - 1e-15))
            + (1 - o.y) * np.log1p(-np.clip(p, 1e-15, 1 - 1e-15))
        )

    delta = loss(o.P2) - loss(o.P4)
    dump(
        "review_paired_logloss.json",
        {
            "P2_minus_P4": float(delta.mean()),
            "fold_differences": [
                float(delta[o.row_id.isin(folds[str(f)]["valid"])].mean())
                for f in range(5)
            ],
            "interpretation": "Paired descriptive losses; folds are not independent inference replicates.",
        },
    )
    return pd.DataFrame(pooled), pd.DataFrame(group_rows)


def interpretation(n_repeats=10):
    o, folds = validate_oof()
    raw = load_data()
    importance = []
    pdp = []
    fit_checks = []
    start = time.time()
    destination = OUT / "inspection_models"
    destination.mkdir(exist_ok=True)
    with threadpool_limits(limits=1):
        for cid in ["P2", "P4"]:
            for f in range(5):
                a = folds[str(f)]["train"]
                b = folds[str(f)]["valid"]
                tr = raw.loc[a]
                va = raw.loc[b]
                y = va.Target.eq("Dropout").to_numpy()
                X = va[FEATURES]
                searches = json.loads((OUT / f"search_{cid}_outer{f}.json").read_text())
                params = min(searches, key=lambda r: r["mean_log_loss"])["params"]
                signature = hashlib.sha256(
                    json.dumps(
                        {
                            "params": params,
                            "train": a,
                            "valid": b,
                            "data": hashlib.sha256(
                                Path("data/data.csv").read_bytes()
                            ).hexdigest(),
                            "code": hashlib.sha256(
                                b"".join(
                                    (Path(__file__).parent / name).read_bytes()
                                    for name in (
                                        "data.py",
                                        "preprocessing.py",
                                        "prediction.py",
                                        "training.py",
                                    )
                                )
                            ).hexdigest(),
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest()
                path = destination / f"{cid}_outer{f}.joblib"
                meta = destination / f"{cid}_outer{f}.json"
                cached = (
                    path.exists()
                    and meta.exists()
                    and json.loads(meta.read_text())["signature"] == signature
                )
                if cached:
                    m = joblib.load(path)
                else:
                    m = model(cid, params).fit(
                        tr[FEATURES], tr.Target.eq("Dropout").astype(int)
                    )
                    joblib.dump(m, path)
                    meta.write_text(
                        json.dumps({"signature": signature, "params": params})
                    )
                p = m.predict_proba(X)[:, 1]
                expected = o.set_index("row_id").loc[b, cid].to_numpy()
                error = float(abs(p - expected).max())
                assert error < 1e-10
                fit_checks.append(
                    dict(
                        candidate=cid,
                        fold=f,
                        cached=cached,
                        n_train=len(a),
                        n_valid=len(b),
                        max_oof_difference=error,
                        params=params,
                    )
                )
                base = log_loss(y, p)
                for j, feature in enumerate(FEATURES):
                    rng = np.random.default_rng(3481 + 100 * f + j)
                    for repeat in range(n_repeats):
                        altered = X.copy()
                        altered[feature] = rng.permutation(X[feature].to_numpy())
                        change = log_loss(y, m.predict_proba(altered)[:, 1]) - base
                        importance.append(
                            dict(
                                candidate=cid,
                                fold=f,
                                n=len(b),
                                feature=feature,
                                repeat=repeat,
                                logloss_increase=change,
                            )
                        )
                for feature, grid in [
                    ("Age at enrollment", [18, 20, 25, 30, 40, 50]),
                    ("Admission grade", [100, 110, 120, 130, 140, 150, 160, 170]),
                ]:
                    for value in grid:
                        altered = X.copy()
                        altered[feature] = value
                        pdp.append(
                            dict(
                                candidate=cid,
                                fold=f,
                                n=len(b),
                                feature=feature,
                                value=value,
                                mean_probability=float(
                                    m.predict_proba(altered)[:, 1].mean()
                                ),
                                observed_within_5=int(
                                    (abs(X[feature] - value) <= 5).sum()
                                ),
                            )
                        )
                print("Interpretation", cid, "fold", f, "OOF parity", error, flush=True)
    imp = pd.DataFrame(importance)
    imp.to_csv(OUT / "review_permutation_raw.csv", index=False)
    pooled = []
    for (cid, feature), g in imp.groupby(["candidate", "feature"]):
        vals = g.groupby("repeat").apply(
            lambda z: np.average(z.logloss_increase, weights=z.n), include_groups=False
        )
        pooled.append(
            dict(
                candidate=cid,
                feature=feature,
                mean_logloss_increase=vals.mean(),
                permutation_sd=vals.std(ddof=1),
            )
        )
    pd.DataFrame(pooled).to_csv(OUT / "review_permutation_summary.csv", index=False)
    pd.DataFrame(pdp).to_csv(OUT / "review_pdp.csv", index=False)
    dump(
        "review_interpretation_checks.json",
        {
            "fits": fit_checks,
            "seconds": time.time() - start,
            "permutations_per_feature_fold": n_repeats,
            "permutation_seed_formula": "3481 + 100*fold + feature_index; same permutations for P2/P4",
            "scope": "10 fixed-parameter outer-training refits on first execution, no new search; original final models not overwritten. Importance computed on outer validation folds. PDP is descriptive model response, not a causal effect; correlated inputs can yield unsupported combinations.",
        },
    )


def figures():
    q = pd.read_csv(OUT / "Q4_summary.csv")
    o = pd.read_csv(OUT / "oof.csv")
    cal = pd.read_csv(OUT / "calibration_bins.csv")
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    for cid in ["P1", "P2", "P4"]:
        t = q[q.candidate == cid]
        ax.plot(100 * t.capacity, t.identified, "o-", label=cid, color=COLORS[cid])
    b = q[q.candidate == "B0"]
    ax.plot(
        100 * b.capacity,
        b.random_expected,
        "--",
        label="Random expectation",
        color="#555555",
    )
    ax.set(
        xlabel="Hypothetical review capacity (%)",
        ylabel="Recorded Dropout identified",
        xticks=[5, 10, 20],
    )
    ax.grid(alpha=0.2)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "review_capacity.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    for cid in ["P1", "P2", "P4"]:
        precision, recall, _ = precision_recall_curve(o.y, o[cid])
        axes[0].plot(recall, precision, label=cid, color=COLORS[cid])
        t = cal[cal.candidate == cid]
        axes[1].plot(
            t.mean_probability,
            t.observed_dropout,
            "o-",
            label=cid,
            color=COLORS[cid],
            markersize=3,
        )
    axes[0].axhline(o.y.mean(), color="gray", ls="--", lw=0.8)
    axes[0].set(
        xlabel="Recall", ylabel="Precision", title="OOF PR (descriptive pooled scores)"
    )
    axes[1].plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
    axes[1].set(
        xlabel="Mean predicted probability",
        ylabel="Observed Dropout fraction",
        title="OOF calibration: decile bins",
    )
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "review_pr_calibration.png", dpi=180)
    plt.close(fig)
    imp = pd.read_csv(OUT / "review_permutation_summary.csv")
    pdp = pd.read_csv(OUT / "review_pdp.csv")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    features = (
        imp.groupby("feature").mean_logloss_increase.mean().nlargest(6).index[::-1]
    )
    for j, cid in enumerate(["P2", "P4"]):
        t = imp[imp.candidate == cid].set_index("feature").loc[features]
        axes[0].barh(
            np.arange(6) + (j - 0.5) * 0.32,
            t.mean_logloss_increase,
            height=0.32,
            label=cid,
            color=COLORS[cid],
        )
        for ax, feature in zip(axes[1:], ["Age at enrollment", "Admission grade"]):
            t = (
                pdp[(pdp.candidate == cid) & (pdp.feature == feature)]
                .groupby("value")
                .apply(
                    lambda g: np.average(g.mean_probability, weights=g.n),
                    include_groups=False,
                )
            )
            ax.plot(t.index, t.values, "o-", label=cid, color=COLORS[cid], markersize=3)
            ax.set(xlabel=feature, ylabel="Mean predicted Dropout")
            ax.grid(alpha=0.2)
    axes[0].set(
        yticks=np.arange(6),
        yticklabels=[
            f.replace(" at enrollment", "").replace(
                "Previous qualification (grade)", "Previous grade"
            )
            for f in features
        ],
        xlabel="Validation log-loss increase",
    )
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "review_interpretation.png", dpi=180)
    plt.close(fig)


def run_review_diagnostics():
    paired, groups = paired_and_groups()
    interpretation()
    figures()
    return paired, groups
