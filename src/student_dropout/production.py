"""Validated serving of the trusted frozen Part II pipeline. No training or outcomes."""

import csv
import hashlib
import io
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .decision import select_top

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "assets/model_manifest.json").read_text())
LABELS = json.loads((ROOT / "assets/category_labels.json").read_text())["labels"]
FEATURES = MANIFEST["features"]
MAX_ROWS = 10000


@lru_cache(maxsize=1)
def load_pipeline() -> Pipeline:
    """Only the fixed project artifact; never accept a user-supplied pickle path."""
    path = ROOT / MANIFEST["artifact"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != MANIFEST["sha256"]:
        raise ValueError(
            "Model checksum mismatch. Restore the trusted project artifact."
        )
    pipeline = joblib.load(path)
    if (
        list(pipeline.classes_) != MANIFEST["classes"]
        or list(pipeline.feature_names_in_) != FEATURES
    ):
        raise ValueError("Model class order or feature schema differs from manifest.")
    return pipeline


def read_csv(payload: bytes) -> pd.DataFrame:
    """Parse bounded UTF-8 records, rejecting malformed rows and duplicate headers."""
    if not payload or len(payload) > 5_000_000:
        raise ValueError("CSV must be nonempty and no larger than 5 MB.")
    try:
        text = payload.decode("utf-8-sig")
        first = text.splitlines()[0]
        delimiter = ";" if first.count(";") > first.count(",") else ","
        records = list(csv.reader(io.StringIO(text), delimiter=delimiter, strict=True))
    except (UnicodeError, csv.Error, IndexError) as exc:
        raise ValueError(
            "Use a valid UTF-8 comma- or semicolon-delimited CSV."
        ) from exc
    if not records:
        raise ValueError("Empty CSV.")
    header = [s.strip() for s in records[0]]
    if not all(header) or len(header) != len(set(header)):
        raise ValueError(
            "CSV headers must be nonempty and unique after trimming whitespace."
        )
    if len(records) < 2 or len(records) - 1 > MAX_ROWS:
        raise ValueError(f"Provide 1 to {MAX_ROWS} data records.")
    for i, row in enumerate(records[1:], 1):
        if len(row) != len(header):
            raise ValueError(
                f"Record {i}: wrong number of fields (including blank records). Repair the file."
            )
    return pd.DataFrame(records[1:], columns=header)


@dataclass
class Validation:
    inputs: pd.DataFrame
    rows: pd.DataFrame
    ignored: list[str]


def validate_value(col: str, raw: object) -> tuple[float | None, list[str], list[str]]:
    """Parse one enrolment value; return its number, errors and extrapolation warnings."""
    errors: list[str] = []
    warnings: list[str] = []
    val = str(raw).strip()
    if pd.isna(raw) or not val:
        errors.append(f"{col}: missing; no imputation is fitted")
        return None, errors, warnings
    if col in LABELS:
        reverse = {v: k for k, v in LABELS[col].items()}
        val = reverse.get(val, val)
    try:
        number = float(val)
        if not math.isfinite(number):
            raise ValueError()
    except ValueError, TypeError:
        errors.append(
            f"{col}: expected a finite number or exact documented category label"
        )
        return None, errors, warnings
    if col in LABELS:
        if not number.is_integer() or str(int(number)) not in LABELS[col]:
            errors.append(f"{col}: undocumented category; record rejected")
            return None, errors, warnings
        if int(number) not in MANIFEST["trained_categories"][col]:
            warnings.append(
                f"{col}: documented category absent from training; fitted encoder uses an all-zero block"
            )
    else:
        low, high = MANIFEST["bounds"][col]
        if not low <= number <= high:
            errors.append(f"{col}: outside accepted bounds [{low}, {high}]")
            return None, errors, warnings
        if (
            col in ("Age at enrollment", "Application order")
            and not number.is_integer()
        ):
            errors.append(f"{col}: whole number required")
            return None, errors, warnings
        a, b = MANIFEST["ranges"][col]
        if not a <= number <= b:
            warnings.append(
                f"{col}: outside training range [{a:g}, {b:g}]; extrapolation is unvalidated"
            )
    return number, errors, warnings


def validate(frame: pd.DataFrame) -> Validation:
    """Return valid numeric inputs and diagnostics for every original input position."""
    frame = frame.copy().reset_index(drop=True)
    frame.columns = [str(c).strip() for c in frame.columns]
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names after trimming whitespace.")
    missing = [c for c in FEATURES if c not in frame]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if not 1 <= len(frame) <= MAX_ROWS:
        raise ValueError(f"Provide 1 to {MAX_ROWS} data records.")
    ignored = [c for c in frame if c not in FEATURES]
    clean = pd.DataFrame(index=frame.index, columns=FEATURES, dtype=float)
    errors = [[] for _ in range(len(frame))]
    warnings = [[] for _ in range(len(frame))]
    for col in FEATURES:
        for i, raw in enumerate(frame[col]):
            number, value_errors, value_warnings = validate_value(col, raw)
            errors[i].extend(value_errors)
            warnings[i].extend(value_warnings)
            if number is not None:
                clean.loc[i, col] = number
    rows = pd.DataFrame(
        {
            "row_id": np.arange(1, len(frame) + 1),
            "status": ["rejected" if e else "valid" for e in errors],
            "errors": ["; ".join(e) for e in errors],
            "warnings": ["; ".join(w) for w in warnings],
        }
    )
    clean.index = rows.row_id
    return Validation(
        clean.loc[rows.loc[rows.status.eq("valid"), "row_id"]], rows, ignored
    )


def predict(inputs: pd.DataFrame) -> np.ndarray:
    """Return frozen class-1 probabilities; reject any invalid input record."""
    checked = validate(inputs)
    if len(checked.inputs) != len(inputs):
        raise ValueError("Prediction inputs contain invalid records; validate first.")
    model = load_pipeline()
    return model.predict_proba(checked.inputs[FEATURES])[
        :, list(model.classes_).index(MANIFEST["dropout_class"])
    ]


def capacity_size(
    n: int, *, share: float | None = None, count: int | None = None
) -> int:
    """Convert one capacity specification to a count using the frozen floor rule."""
    if (share is None) == (count is None):
        raise ValueError("Specify either share or count.")
    if share is not None:
        if not math.isfinite(share) or not 0 <= share <= 1:
            raise ValueError("Share must be between 0 and 1.")
        return math.floor(share * n)
    if (
        isinstance(count, bool)
        or not isinstance(count, (int, np.integer))
        or not 0 <= count <= n
    ):
        raise ValueError(
            "Count must be an integer between 0 and the valid cohort size."
        )
    return int(count)


def review(
    validation: Validation,
    *,
    share: float | None = None,
    count: int | None = None,
    accept_partial: bool = False,
) -> pd.DataFrame:
    """Invalid rows remain in export; partial cohort reduction requires explicit consent."""
    if validation.rows.status.eq("rejected").any() and not accept_partial:
        raise ValueError(
            "Explicitly acknowledge rejected rows before ranking the valid subset."
        )
    n = len(validation.inputs)
    if not n:
        raise ValueError("No valid records to score.")
    k = capacity_size(n, share=share, count=count)
    scored = pd.DataFrame(
        {
            "row_id": validation.inputs.index,
            "dropout_probability": predict(validation.inputs),
        }
    )
    ordered = select_top(scored, "dropout_probability", n)
    scored["selected_for_review"] = scored.row_id.isin(ordered[:k])
    scored["rank"] = scored.row_id.map({rid: i + 1 for i, rid in enumerate(ordered)})
    result = validation.rows.merge(scored, on="row_id", how="left")
    result["model_version"] = MANIFEST["version"]
    return result.sort_values(["rank", "row_id"], na_position="last").reset_index(
        drop=True
    )


def export_csv(result: pd.DataFrame) -> bytes:
    """Encode a generated review table as a downloadable UTF-8 CSV."""
    # Export only generated fields, never untrusted input text or identifiers.
    return result.to_csv(index=False).encode("utf-8")
