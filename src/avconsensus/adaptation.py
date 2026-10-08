"""Quantile mapping between the TGSIM and simulation domains.

A simulated metric is scored by its rank in the baseline simulation
distribution, so a value at the 60th simulation percentile lands where a
TGSIM value at its 60th percentile does. TGSIM thresholds are translated
the same way: threshold -> TGSIM percentile -> simulation value.
"""

import numpy as np
import pandas as pd

METRICS = ["maximum_risk", "pet", "headway", "gain", "abs_jerk", "decel"]
HIGHER_IS_BETTER = {"pet"}


def build_ecdf(values: np.ndarray, n_points: int = 1001, min_n: int = 10) -> dict | None:
    """Percentile lookup table at n_points evenly spaced percentiles."""
    v = values[np.isfinite(values)]
    if len(v) < min_n:
        return None
    pct = np.linspace(0, 100, n_points)
    return {"values": np.percentile(v, pct), "percentiles": pct, "n": len(v)}


def tgsim_reference(df: pd.DataFrame, min_abs_accel: float = 0.3, min_decel: float = 0.3,
                    n_points: int = 1001) -> dict:
    """ECDF tables of the TGSIM AV metrics, qualified as in the Pareto step."""
    df = df.copy()
    df["maximum_risk"] = df["maximum_risk"].clip(lower=0)
    df.loc[df["accel"].abs() < min_abs_accel, "abs_jerk"] = np.nan
    df.loc[df["decel"].notna() & (df["decel"] < min_decel), "decel"] = np.nan

    ref = {}
    for m in METRICS:
        v = df[m].dropna().to_numpy()
        v = v[np.isfinite(v)]
        if len(v):
            e = build_ecdf(v, n_points, min_n=1)
            e.update(mean=float(v.mean()), std=float(v.std()),
                     min=float(v.min()), max=float(v.max()))
            ref[m] = e
    return ref


def translate_threshold(value: float, tgsim: dict, sim: dict) -> tuple[float, float]:
    """TGSIM threshold -> (simulation threshold, TGSIM percentile)."""
    pct = float(np.interp(value, tgsim["values"], tgsim["percentiles"]))
    return float(np.interp(pct, sim["percentiles"], sim["values"])), pct


def adapt(baseline: pd.DataFrame, reference: dict, surface: dict, thresholds: dict,
          n_points: int = 1001, min_n: int = 10) -> dict:
    """Build the adapted surface used by the training reward.

    Args:
        baseline: baseline simulation metrics (one row per AV step).
        reference: TGSIM ECDF tables (see tgsim_reference).
        surface: empirical Pareto surface from Step 3.
        thresholds: TGSIM consensus thresholds.

    Returns:
        The Pareto hull and points plus simulation ECDFs, the scoring spec,
        translated thresholds, and boundary saturation diagnostics.
    """
    sim = baseline.copy()
    for m in METRICS:
        sim[m] = pd.to_numeric(sim[m], errors="coerce")
    sim["maximum_risk"] = sim["maximum_risk"].clip(lower=0)

    ecdfs = {}
    for m in METRICS:
        e = build_ecdf(sim[m].dropna().to_numpy(), n_points, min_n)
        if e is not None:
            ecdfs[m] = e

    th_sim, th_pct, th_score, diag = {}, {}, {}, {}
    for m in METRICS:
        if m in ecdfs and m in reference:
            th_sim[m], th_pct[m] = translate_threshold(thresholds[m], reference[m], ecdfs[m])
        else:
            th_sim[m] = thresholds[m]
        if m in reference:
            p = float(np.interp(thresholds[m], reference[m]["values"],
                                reference[m]["percentiles"])) / 100
            th_score[m] = p if m in HIGHER_IS_BETTER else 1.0 - p
        if m in ecdfs:
            v = sim[m].dropna().to_numpy()
            diag[m] = {"lower_saturation_pct": round(100 * float(np.mean(v <= ecdfs[m]["values"][0])), 2),
                       "upper_saturation_pct": round(100 * float(np.mean(v >= ecdfs[m]["values"][-1])), 2)}

    spec = {m: {"values": e["values"], "percentiles": e["percentiles"],
                "direction": "direct" if m in HIGHER_IS_BETTER else "invert"}
            for m, e in ecdfs.items()}

    return {
        "c_hull_model": surface["c_hull_model"],
        "pareto_points": surface["pareto_points"],
        "pareto_stats": surface["pareto_stats"],
        "normalization_spec": spec,
        "tgsim_reference": reference,
        "sim_ecdfs": ecdfs,
        "thresholds_tgsim": dict(thresholds),
        "thresholds_tgsim_percentile": th_pct,
        "thresholds_sim": th_sim,
        "thresholds_score": th_score,
        "higher_is_worse": [m for m in METRICS if m not in HIGHER_IS_BETTER],
        "higher_is_better": sorted(HIGHER_IS_BETTER),
        "domain_diagnostics": diag,
        "normalization": "QQ mapping: simulation ECDF for scoring, "
                         "TGSIM percentiles for threshold translation",
    }
