"""Rank-based normalization of metrics into Safety, Efficiency, Interaction."""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

DIMENSIONS = {
    "Safety": ["maximum_risk", "pet"],
    "Efficiency": ["headway", "gain"],
    "Interaction": ["abs_jerk", "decel"],
}
OBJECTIVES = list(DIMENSIONS)
HIGHER_IS_BETTER = {"pet"}


def qualify(df: pd.DataFrame, min_abs_accel: float = 0.3, min_decel: float = 0.3) -> pd.DataFrame:
    """Drop near-zero jerk and decel readings and clip negative risk to zero."""
    df = df.copy()
    df.loc[df["accel"].abs() < min_abs_accel, "abs_jerk"] = np.nan
    df.loc[df["decel"].notna() & (df["decel"] < min_decel), "decel"] = np.nan
    df["maximum_risk"] = df["maximum_risk"].clip(lower=0)
    return df


def rank_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank of each metric, oriented so 1 is best."""
    scores = pd.DataFrame(index=df.index)
    for metrics in DIMENSIONS.values():
        for m in metrics:
            r = df[m].rank(pct=True)
            scores[m] = r if m in HIGHER_IS_BETTER else 1 - r
    return scores


def composites(df: pd.DataFrame, min_valid: int = 2, knn: dict | None = None) -> pd.DataFrame:
    """Add Safety, Efficiency, Interaction scores and impute missing ones.

    Each dimension is the mean of its available metric scores. Rows with
    fewer than min_valid dimensions are dropped. A single missing dimension
    is imputed by distance-weighted kNN within each dataset.
    """
    knn = knn or {"n_neighbors": 10, "weights": "distance"}
    df = df.copy()
    scores = rank_scores(df)
    for dim, metrics in DIMENSIONS.items():
        df[dim] = scores[metrics].mean(axis=1, skipna=True)

    df["imputed"] = df[OBJECTIVES].isna().any(axis=1)
    df = df[df[OBJECTIVES].notna().sum(axis=1) >= min_valid].copy()

    for _, idx in df.groupby("dataset").groups.items():
        filled = KNNImputer(**knn).fit_transform(df.loc[idx, OBJECTIVES])
        df.loc[idx, OBJECTIVES] = np.clip(filled, 0, 1)
    return df
