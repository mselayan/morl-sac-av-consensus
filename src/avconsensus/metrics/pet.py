"""Post-encroachment time (PET).

For each AV timestep, PET is the smallest elapsed time between the AV
occupying a position and any other road user now occupying it.
"""

from collections import defaultdict, deque

import numpy as np
import pandas as pd


def compute_pet(df: pd.DataFrame, box_half_size: float = 2.0,
                history_s: float = 10.0, upper_limit: float = 10.0) -> pd.DataFrame:
    """PET for every AV timestep with a detected encroachment.

    Args:
        df: prepared frame (see prepare_frame).
        box_half_size: half side of the square zone around each past AV position (m).
        history_s: how far back past AV positions are kept (s).
        upper_limit: PET values above this are discarded (s).

    Returns:
        Frame with columns id, time, pet.
    """
    history = defaultdict(deque)  # AV id -> (time, x, y)
    rows = []

    for t, f in df.groupby("time"):
        x, y = f["x"].to_numpy(), f["y"].to_numpy()
        types, ids = f["type"].to_numpy(), f["id"].to_numpy()
        av_rows = np.flatnonzero(types == "av")

        for i in av_rows:
            h = history[ids[i]]
            h.append((t, x[i], y[i]))
            while h and t - h[0][0] > history_s:
                h.popleft()

        for i in av_rows:
            h = history[ids[i]]
            if len(h) < 2:
                continue
            others = ids != ids[i]
            if not others.any():
                continue

            past = np.array([p for p in h if p[0] != t])
            if past.size == 0:
                continue
            ox, oy = x[others], y[others]
            occupied = ((np.abs(ox[None, :] - past[:, 1:2]) < box_half_size) &
                        (np.abs(oy[None, :] - past[:, 2:3]) < box_half_size)).any(axis=1)
            if occupied.any():
                pet = t - past[occupied, 0].max()
                if pet <= upper_limit:
                    rows.append((ids[i], t, pet))

    return pd.DataFrame(rows, columns=["id", "time", "pet"])
