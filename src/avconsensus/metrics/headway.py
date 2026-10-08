"""Time headway: bumper-to-bumper gap to the nearest leader over ego speed."""

import numpy as np
import pandas as pd

from .neighbors import candidates, nearest_in_lane


def compute_headway(df: pd.DataFrame, nb: dict, min_speed: float = 0.5,
                    upper_limit: float = 12.0) -> pd.DataFrame:
    """Time headway for every AV timestep with a valid leader.

    Args:
        df: prepared frame (see prepare_frame).
        nb: neighbors config.
        min_speed: ego speed below which headway is undefined (m/s).
        upper_limit: headways above this are treated as free flow (s).

    Returns:
        Frame with columns id, time, headway.
    """
    rows = []
    for t, f in df.groupby("time"):
        x, y = f["x"].to_numpy(), f["y"].to_numpy()
        vx, vy = f["vx"].to_numpy(), f["vy"].to_numpy()
        length, types, ids = f["length"].to_numpy(), f["type"].to_numpy(), f["id"].to_numpy()
        is_vehicle = np.isin(types, nb["vehicle_types"])

        for i in np.flatnonzero(types == "av"):
            speed = np.hypot(vx[i], vy[i])
            if speed < min_speed:
                continue
            dx, dy = x - x[i], y - y[i]
            c = candidates(dx, dy, is_vehicle, i, nb["max_range"])
            if c.size == 0:
                continue
            k, fwd = nearest_in_lane(dx[c], dy[c], vx[c], vy[c],
                                     vx[i] / speed, vy[i] / speed, True, nb)
            if k is None:
                continue
            gap = max(fwd - 0.5 * (length[i] + length[c[k]]), 0.0)
            hw = gap / speed
            if hw <= upper_limit:
                rows.append((ids[i], t, hw))

    return pd.DataFrame(rows, columns=["id", "time", "headway"])
