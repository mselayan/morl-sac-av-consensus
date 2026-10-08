"""Frame preparation and same-lane leader/follower search."""

import numpy as np
import pandas as pd


def prepare_frame(df: pd.DataFrame, default_length: float = 4.5) -> pd.DataFrame:
    """Normalize type labels and fill missing vehicle lengths."""
    df = df.copy()
    df["type"] = df["type"].astype(str).str.lower().str.strip()
    if "length" not in df.columns:
        df["length"] = default_length
    df["length"] = pd.to_numeric(df["length"], errors="coerce").fillna(default_length)
    return df


def nearest_in_lane(dx, dy, vx, vy, hx, hy, ahead: bool, cfg: dict):
    """Find the nearest same-lane, same-direction vehicle ahead or behind ego.

    Args:
        dx, dy: candidate positions relative to ego (m).
        vx, vy: candidate velocities (m/s).
        hx, hy: ego unit heading.
        ahead: True for leader, False for follower.
        cfg: neighbors config.

    Returns:
        (index into candidates, forward distance) or (None, None).
    """
    fwd = dx * hx + dy * hy
    lat = -dx * hy + dy * hx
    f_speed = vx * hx + vy * hy
    speed = np.hypot(vx, vy)

    cos = np.divide(f_speed, speed, out=np.ones_like(f_speed), where=speed > 0.01)
    same_dir = (speed < cfg["min_speed_heading"]) | (cos >= cfg["heading_alignment"])
    valid = (np.abs(lat) < cfg["lane_half_width"]) & same_dir & (f_speed >= -0.5)
    valid &= (fwd > 0) if ahead else (fwd < 0)

    if not valid.any():
        return None, None
    idx = np.flatnonzero(valid)
    k = idx[np.argmin(np.abs(fwd[idx]))]
    return k, fwd[k]


def candidates(dx, dy, is_vehicle, ego: int, max_range: float) -> np.ndarray:
    """Indices of vehicles within range of ego, excluding ego."""
    mask = (dx * dx + dy * dy < max_range ** 2) & is_vehicle
    mask[ego] = False
    return np.flatnonzero(mask)
