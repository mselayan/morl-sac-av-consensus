"""String stability gain: RMS acceleration ratio of follower to AV.

G = RMS(a_follower) / RMS(a_AV) over a trailing window. G > 1 means the
follower amplifies the AV's acceleration variability.
"""

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .neighbors import candidates, nearest_in_lane


def compute_gain(df: pd.DataFrame, nb: dict, window_s: float = 5.0,
                 min_steps: int = 30, min_rms: float = 0.05) -> pd.DataFrame:
    """String stability gain for every AV timestep with a valid follower.

    Args:
        df: prepared frame (see prepare_frame).
        nb: neighbors config.
        window_s: trailing RMS window (s).
        min_steps: minimum samples in both windows.
        min_rms: minimum AV acceleration RMS for a defined gain (m/s^2).

    Returns:
        Frame with columns id, time, gain.
    """
    history = defaultdict(deque)  # id -> (time, accel)
    rows = []

    for t, f in df.groupby("time"):
        x, y = f["x"].to_numpy(), f["y"].to_numpy()
        vx, vy = f["vx"].to_numpy(), f["vy"].to_numpy()
        accel, types, ids = f["accel"].to_numpy(), f["type"].to_numpy(), f["id"].to_numpy()
        is_vehicle = np.isin(types, nb["vehicle_types"])

        # Update trailing acceleration windows for all agents present.
        for vid, a in zip(ids, accel):
            h = history[vid]
            h.append((t, a))
            while h and t - h[0][0] > window_s:
                h.popleft()

        for i in np.flatnonzero(types == "av"):
            speed = np.hypot(vx[i], vy[i])
            if speed < nb["min_speed_heading"]:
                continue
            dx, dy = x - x[i], y - y[i]
            c = candidates(dx, dy, is_vehicle, i, nb["max_range"])
            if c.size == 0:
                continue
            k, _ = nearest_in_lane(dx[c], dy[c], vx[c], vy[c],
                                   vx[i] / speed, vy[i] / speed, False, nb)
            if k is None:
                continue

            h_av, h_fol = history[ids[i]], history[ids[c[k]]]
            if len(h_av) < min_steps or len(h_fol) < min_steps:
                continue
            rms_av = np.sqrt(np.mean(np.square([a for _, a in h_av])))
            if rms_av < min_rms:
                continue
            rms_fol = np.sqrt(np.mean(np.square([a for _, a in h_fol])))
            rows.append((ids[i], t, rms_fol / rms_av))

    return pd.DataFrame(rows, columns=["id", "time", "gain"])
