"""Co-occurrence penalty, Pareto front, and convex hull frontier."""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from .normalize import DIMENSIONS, HIGHER_IS_BETTER, OBJECTIVES

UTOPIA = np.ones(3)


def violations(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Boolean violation flag per dimension. A dimension is violated if any
    of its metrics crosses its threshold."""
    out = pd.DataFrame(index=df.index)
    for dim, metrics in DIMENSIONS.items():
        flag = pd.Series(False, index=df.index)
        for m in metrics:
            bad = df[m] < thresholds[m] if m in HIGHER_IS_BETTER else df[m] > thresholds[m]
            flag |= df[m].notna() & bad
        out[dim] = flag
    return out


def apply_penalty(df: pd.DataFrame, thresholds: dict, coef: float = 0.1) -> pd.DataFrame:
    """Subtract coef * n_violated^2 from all three dimension scores."""
    df = df.copy()
    df["n_violated"] = violations(df, thresholds).sum(axis=1)
    penalty = coef * df["n_violated"] ** 2
    for dim in OBJECTIVES:
        df[dim] = (df[dim] - penalty).clip(0, 1)
    return df


def non_dominated(X: np.ndarray) -> np.ndarray:
    """Mask of points not dominated by any other (maximization)."""
    dominated = np.zeros(len(X), dtype=bool)
    for i in range(len(X)):
        if not dominated[i]:
            dominated[i] = ((X >= X[i]).all(axis=1) & (X > X[i]).any(axis=1)).any()
    return ~dominated


def pareto_front(df: pd.DataFrame, floor: float = 0.5) -> pd.Series:
    """Pareto flag: non-dominated and at least floor in every dimension."""
    X = df[OBJECTIVES].to_numpy(float)
    return pd.Series(non_dominated(X) & (X >= floor).all(axis=1), index=df.index)


def upper_facets(hull: ConvexHull) -> np.ndarray:
    """Mask of hull facets whose outward normal points toward utopia (1, 1, 1)."""
    return hull.equations[:, :3] @ UTOPIA > 0


def build_surface(points: np.ndarray, thresholds: dict, n_total: int) -> dict:
    """Fit the convex hull to Pareto points and package it for training."""
    hull = ConvexHull(points)
    return {
        "c_hull_model": hull,
        "pareto_points": points,
        "thresholds": thresholds,
        "normalization": "rank-based (empirical CDF)",
        "pareto_stats": {
            "n_points": len(points),
            "n_total": n_total,
            "n_eligible": n_total,
            "n_boundary": len(hull.vertices),
            "n_facets": len(hull.simplices),
            "n_upper_facets": int(upper_facets(hull).sum()),
        },
    }
