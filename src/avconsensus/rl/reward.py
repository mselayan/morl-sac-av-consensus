"""Pareto-guided reward: proximity to the empirical frontier plus balance.

r = -d(x, F)^p + w * min(S, E, I)

x is the penalized (S, E, I) point, F the upper facets of the convex hull.
"""

import joblib
import numpy as np

from ..pareto.surface import upper_facets

DIMENSIONS = {
    "S": ["maximum_risk", "pet"],
    "E": ["headway", "gain"],
    "I": ["abs_jerk", "decel"],
}


def point_triangle_distance(p: np.ndarray, tri: np.ndarray) -> float:
    """Euclidean distance from point p to triangle tri (3 x 3)."""
    a, b, c = tri
    ab, ac = b - a, c - a
    n = np.cross(ab, ac)
    n_len = np.linalg.norm(n)
    if n_len < 1e-12:
        return min(np.linalg.norm(p - v) for v in tri)
    n = n / n_len
    h = np.dot(p - a, n)
    ap = p - h * n - a

    d00, d01, d11 = ab @ ab, ab @ ac, ac @ ac
    d20, d21 = ap @ ab, ap @ ac
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return min(np.linalg.norm(p - v) for v in tri)
    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom
    if u >= 0 and v >= 0 and u + v <= 1:
        return abs(h)

    def seg(s0, s1):
        d = s1 - s0
        L = d @ d
        if L < 1e-12:
            return np.linalg.norm(p - s0)
        return np.linalg.norm(p - (s0 + np.clip((p - s0) @ d / L, 0, 1) * d))

    return min(seg(a, b), seg(b, c), seg(c, a))


class ParetoReward:
    """Scores raw simulation metrics against the domain-adapted surface.

    Args:
        surface_path: adapted surface from Step 4.
        exponent: p in -d^p.
        quality_weight: w on min(S, E, I).
        penalty_coef: co-occurrence penalty coef * n_violated^2.
    """

    def __init__(self, surface_path, exponent=0.5, quality_weight=0.5, penalty_coef=0.1):
        s = joblib.load(surface_path)
        self.spec = s["normalization_spec"]
        self.thresholds = s["thresholds_sim"]
        hull, pts = s["c_hull_model"], s["pareto_points"]
        self.facets = pts[hull.simplices[upper_facets(hull)]]
        self.exponent, self.weight, self.coef = exponent, quality_weight, penalty_coef

    def score(self, value, metric):
        """Simulation-ECDF rank of value, oriented so 1 is best. NaN if missing."""
        if value is None or not np.isfinite(value) or metric not in self.spec:
            return np.nan
        sp = self.spec[metric]
        rank = float(np.interp(value, sp["values"], sp["percentiles"] / 100))
        return 1.0 - rank if sp["direction"] == "invert" else rank

    def violated(self, value, metric):
        """True if value crosses the simulation-domain threshold."""
        if value is None or not np.isfinite(value) or metric not in self.thresholds:
            return False
        t = self.thresholds[metric]
        return value < t if metric == "pet" else value > t

    def distance(self, x: np.ndarray) -> float:
        """Distance from x to the nearest upper facet of the frontier."""
        return min(point_triangle_distance(x, tri) for tri in self.facets)

    def __call__(self, metrics: dict, last_sei=None):
        """Reward for one AV step.

        Args:
            metrics: raw values keyed by maximum_risk, pet, headway, gain,
                abs_jerk, decel (None when unavailable).
            last_sei: (S, E, I) from this AV's last scored step.

        Returns:
            (reward, (S, E, I)). Reward is None when fewer than two
            dimensions are available; one missing dimension is filled with
            its last value, or the mean of the other two if none exists.
        """
        sei = []
        for ms in DIMENSIONS.values():
            vals = [v for v in (self.score(metrics[m], m) for m in ms) if not np.isnan(v)]
            sei.append(np.mean(vals) if vals else np.nan)
        sei = np.array(sei, dtype=float)

        missing = np.isnan(sei)
        if missing.sum() > 1:
            return None, last_sei
        if missing.any():
            sei[missing] = last_sei[int(np.flatnonzero(missing)[0])] if last_sei is not None \
                else np.nanmean(sei)

        n_viol = sum(any(self.violated(metrics[m], m) for m in ms) for ms in DIMENSIONS.values())
        sei = np.maximum(sei - self.coef * n_viol ** 2, 0.0)

        reward = -self.distance(sei) ** self.exponent + self.weight * sei.min()
        return float(reward), tuple(float(v) for v in sei)
