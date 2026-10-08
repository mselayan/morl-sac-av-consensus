"""Trajectory smoothing and kinematics derivation.

Positions are smoothed with a Gaussian filter. Velocity and acceleration
are then derived from the smoothed positions by numerical gradients.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

LOW_SPEED = 0.1  # m/s, below which heading is unreliable


def smooth_trajectory(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                      sigma: float = 1.0) -> dict:
    """Smooth one trajectory and derive its kinematics.

    Returns a dict with smoothed positions, velocity and acceleration
    components, scalar speed, and signed longitudinal acceleration.
    """
    xs = gaussian_filter1d(x, sigma=sigma, mode="nearest")
    ys = gaussian_filter1d(y, sigma=sigma, mode="nearest")

    # Gradients use the time array, so uneven sampling is handled.
    vx, vy = np.gradient(xs, t), np.gradient(ys, t)
    ax, ay = np.gradient(vx, t), np.gradient(vy, t)
    speed = np.hypot(vx, vy)

    # Longitudinal acceleration: projection of acceleration on heading.
    s = np.maximum(speed, 1e-6)
    accel = (ax * vx + ay * vy) / s

    # Near standstill, use the time derivative of speed instead.
    low = speed < LOW_SPEED
    if low.any():
        accel[low] = np.gradient(speed, t)[low]

    return dict(x=xs, y=ys, vx=vx, vy=vy, ax=ax, ay=ay,
                speed=speed, accel=accel)


def smooth_dataset(df: pd.DataFrame, type_map: dict, dataset_name: str,
                   sigma: float = 1.0, min_len: int = 5,
                   verbose: bool = True) -> pd.DataFrame:
    """Smooth every agent in a transformed TGSIM frame.

    Agents shorter than min_len samples or with missing positions are
    dropped. Output has one row per agent-timestep with columns:
    id, time, x, y, vx, vy, ax, ay, speed, accel, type, dataset,
    lane, length, width.
    """
    df = df.sort_values(["id", "time"]).reset_index(drop=True)
    df["type"] = df["type_most_common"].map(type_map)

    unmapped = df["type"].isna()
    if unmapped.any() and verbose:
        codes = df.loc[unmapped, "type_most_common"].unique()
        print(f"  warning: {unmapped.sum()} rows with unmapped type codes {codes}")

    parts, n_short, n_nan = [], 0, 0
    for agent_id, g in df.groupby("id", sort=False):
        if len(g) < min_len:
            n_short += 1
            continue
        x, y = g["xloc_kf"].to_numpy(), g["yloc_kf"].to_numpy()
        if np.isnan(x).any() or np.isnan(y).any():
            n_nan += 1
            continue

        t = g["time"].to_numpy()
        k = smooth_trajectory(t, x, y, sigma)
        parts.append(pd.DataFrame({
            "id": agent_id, "time": t, **k,
            "type": g["type"].iloc[0],
            "dataset": dataset_name,
            "lane": g["lane_kf"].to_numpy(),
            "length": g["length_smoothed"].to_numpy(),
            "width": g["width_smoothed"].to_numpy(),
        }))

    out = pd.concat(parts, ignore_index=True)
    if verbose:
        _report(out, len(parts), n_short, n_nan)
    return out


def _report(out: pd.DataFrame, n_ok: int, n_short: int, n_nan: int) -> None:
    """Print a short summary and sanity checks."""
    print(f"  agents kept {n_ok}, too short {n_short}, missing positions {n_nan}")
    print(f"  rows {len(out):,} | speed [{out.speed.min():.2f}, {out.speed.max():.2f}] m/s"
          f" | accel [{out.accel.min():.2f}, {out.accel.max():.2f}] m/s^2")
    if (n := (out.speed > 40).sum()):
        print(f"  warning: {n} rows with speed > 40 m/s")
    if (n := (out.accel.abs() > 10).sum()):
        print(f"  warning: {n} rows with |accel| > 10 m/s^2")
    counts = out.groupby("type").agg(rows=("id", "size"), agents=("id", "nunique"))
    for name, r in counts.iterrows():
        print(f"    {name:11s} {r.rows:>9,} rows  {r.agents:>5} agents")
