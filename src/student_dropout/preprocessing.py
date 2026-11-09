"""Build unfitted fold-local encoders, scalers and optional spline bases."""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def preprocessor(gam: bool = False, omit_parent: bool = False) -> ColumnTransformer:
    """Return an unfitted transformer; fit it only inside the relevant training fold."""
    cats = [
        c
        for c in CATEGORICAL_FEATURES
        if not (omit_parent and ("Mother" in c or "Father" in c))
    ]
    transforms = [
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats)
    ]
    if gam:
        transforms += [
            (
                "smooth",
                Pipeline(
                    [
                        (
                            "basis",
                            SplineTransformer(
                                n_knots=4,
                                degree=3,
                                include_bias=False,
                                knots="uniform",
                                extrapolation="linear",
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES[:3],
            ),
            ("linear", StandardScaler(), NUMERIC_FEATURES[3:]),
        ]
    else:
        transforms += [("num", StandardScaler(), NUMERIC_FEATURES)]
    return ColumnTransformer(
        transforms, remainder="drop", verbose_feature_names_out=True
    )
