"""Merge all six metrics into one AV-timestep table."""

import pandas as pd

from .comfort import add_decel, add_jerk
from .gain import compute_gain
from .headway import compute_headway
from .pet import compute_pet

METRICS = ["maximum_risk", "pet", "headway", "gain", "abs_jerk", "decel"]


def _merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Left-join a metric frame on (id, time) with time rounded to 0.01 s."""
    right = right.assign(time=right["time"].round(2))
    return left.merge(right, on=["id", "time"], how="left")


def build_unified(df: pd.DataFrame, gssm_risk: pd.DataFrame, dataset: str,
                  cfg: dict) -> pd.DataFrame:
    """Compute PET, headway, gain, jerk, and decel, and join GSSM risk.

    Args:
        df: prepared frame for one dataset, all agents.
        gssm_risk: max risk per AV timestep (id, time, maximum_risk).
        dataset: dataset label.
        cfg: metrics config.

    Returns:
        One row per AV timestep with id, time, type, speed, accel, jerk,
        and the six metrics. Missing metrics are NaN.
    """
    nb, p, h, g = cfg["neighbors"], cfg["pet"], cfg["headway"], cfg["gain"]

    av = df[df["type"] == "av"].copy()
    av["time"] = av["time"].round(2)
    av = add_decel(add_jerk(av))
    out = av[["id", "time", "type", "speed", "accel", "jerk", "abs_jerk", "decel"]]

    out = _merge(out, gssm_risk[["id", "time", "maximum_risk"]])

    print("  PET")
    out = _merge(out, compute_pet(df, p["box_half_size"], p["history_s"], p["upper_limit_s"]))
    print("  headway")
    out = _merge(out, compute_headway(df, nb, h["min_speed"], h["upper_limit_s"]))
    print("  gain")
    out = _merge(out, compute_gain(df, nb, g["window_s"], g["min_window_steps"], g["min_rms"]))

    out["dataset"] = dataset
    return out
