"""Interaction metrics: jerk and deceleration intensity."""

import numpy as np
import pandas as pd


def add_jerk(df: pd.DataFrame) -> pd.DataFrame:
    """Add signed and absolute jerk, the time derivative of accel, per agent."""
    df = df.copy()
    df["jerk"] = np.nan
    for _, traj in df.groupby("id"):
        traj = traj.sort_values("time").drop_duplicates(subset="time", keep="first")
        if len(traj) < 3:
            continue
        df.loc[traj.index, "jerk"] = np.gradient(traj["accel"].to_numpy(),
                                                 traj["time"].to_numpy())
    df["abs_jerk"] = df["jerk"].abs()
    return df


def add_decel(df: pd.DataFrame) -> pd.DataFrame:
    """Add deceleration magnitude when braking, NaN otherwise."""
    df = df.copy()
    df["decel"] = np.where(df["accel"] < 0, df["accel"].abs(), np.nan)
    return df
