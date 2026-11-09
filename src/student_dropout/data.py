"""Frozen enrolment schema, raw-data checks and training/test split persistence.

Run analysis from the project root. Reading the full CSV here does not score test rows.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

OUT = Path("outputs/part2")
SEED = 481
CANDIDATES = ("B0", "P1", "P2", "P3", "P4", "P1B")
NUMERIC_FEATURES = [
    "Age at enrollment",
    "Admission grade",
    "Previous qualification (grade)",
    "Application order",
]
CATEGORICAL_FEATURES = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Gender",
    "International",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def dump(name: str, obj: object) -> None:
    """Write a named Part II JSON artifact, including NumPy scalar/array values."""
    (OUT / name).write_text(
        json.dumps(
            obj,
            indent=2,
            default=lambda x: (
                x.tolist() if isinstance(x, (np.ndarray, np.generic)) else str(x)
            ),
        )
    )


def load_data() -> pd.DataFrame:
    """Read the immutable CSV, normalize headers and check the recorded three-class target."""
    data = pd.read_csv("data/data.csv", sep=";", encoding="utf-8-sig")
    data.columns = data.columns.str.strip()
    assert data.columns.is_unique and set(data.Target) == {
        "Dropout",
        "Graduate",
        "Enrolled",
    }
    assert not data.isna().any().any() and not data.duplicated().any()
    return data


def prepare() -> tuple[pd.DataFrame, dict]:
    """Audit data and return training rows with the persisted, hash-checked split."""
    OUT.mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    data = load_data()
    dataset_hash = hashlib.sha256(Path("data/data.csv").read_bytes()).hexdigest()
    audit = {
        "shape": data.shape,
        "sha256": dataset_hash,
        "missing": int(data.isna().sum().sum()),
        "duplicates": int(data.duplicated().sum()),
        "target": data.Target.value_counts().to_dict(),
        "features": FEATURES,
        "excluded": [column for column in data if column not in FEATURES + ["Target"]],
        "categories": {
            column: data[column].value_counts().to_dict()
            for column in CATEGORICAL_FEATURES
        },
        "numeric_ranges": {
            column: [data[column].min(), data[column].max()]
            for column in NUMERIC_FEATURES
        },
        "special_codes": {
            column: int((data[column] == code).sum())
            for column, code in [
                ("Mother's qualification", 34),
                ("Father's qualification", 34),
                ("Mother's occupation", 99),
                ("Father's occupation", 99),
            ]
        },
    }
    dump("audit.json", audit)
    split_path = OUT / "split.json"
    if split_path.exists():
        split = json.loads(split_path.read_text())
        assert split["sha256"] == dataset_hash
    else:
        train_indices, test_indices = train_test_split(
            np.arange(len(data)), test_size=0.2, random_state=SEED, stratify=data.Target
        )
        split = {
            "sha256": dataset_hash,
            "seed": SEED,
            "train": sorted(train_indices.tolist()),
            "test": sorted(test_indices.tolist()),
        }
        dump("split.json", split)
    assert not set(split["train"]) & set(split["test"]) and sorted(
        split["train"] + split["test"]
    ) == list(range(len(data)))
    training = data.iloc[split["train"]].copy()
    return training, split
