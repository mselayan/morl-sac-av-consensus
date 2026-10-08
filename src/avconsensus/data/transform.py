"""Coordinate transformation for raw TGSIM trajectories.

TGSIM places the origin at the top-left of the reference image, with y
pointing down. This module flips y so it points up (standard Cartesian).
"""

import pandas as pd


def flip_y_axis(df: pd.DataFrame, y_extent_px: float, px_to_m: float,
                flip_kinematics: bool = False) -> pd.DataFrame:
    """Flip the y axis of a raw TGSIM frame.

    Args:
        df: raw TGSIM trajectories.
        y_extent_px: image height spanned by y, in pixels.
        px_to_m: pixel-to-meter conversion factor.
        flip_kinematics: also negate y speed and acceleration columns.

    Returns:
        Copy of df with transformed y columns.
    """
    out = df.copy()
    out["yloc_kf"] = y_extent_px * px_to_m - out["yloc_kf"]
    if flip_kinematics:
        out["speed_kf_y"] = -out["speed_kf_y"]
        out["acceleration_kf_y"] = -out["acceleration_kf_y"]
    return out
