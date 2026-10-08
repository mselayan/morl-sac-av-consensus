"""GSSM risk for one AV per simulation step, from the shared SimState."""

from pathlib import Path

import joblib
import numpy as np
import torch

from ..metrics.gssm import GSSM, risk_score

VRU_KEYS = ("pedestrian", "bicycle", "scooter", "ped")
HEAVY_KEYS = ("bus", "truck")
MIN_SPEED_HEADING = 0.3
RANGE_SECONDS, RANGE_MIN, RANGE_MAX = 3.0, 20.0, 150.0


def _category(vtype: str) -> str:
    """Map a SUMO type id to vru, heavy, or car by substring."""
    if any(k in vtype for k in VRU_KEYS):
        return "vru"
    if any(k in vtype for k in HEAVY_KEYS):
        return "heavy"
    return "car"


def _entropy(values: np.ndarray, n_bins: int = 10) -> float:
    """Shannon entropy (bits) of a count histogram with at most n_bins bins."""
    values = values[np.isfinite(values)]
    if len(values) < 2 or values.max() == values.min():
        return 0.0
    n_bins = min(n_bins, max(2, len(np.unique(values)) - 1))
    hist, _ = np.histogram(values, bins=n_bins)
    p = hist[hist > 0] / hist.sum()
    return float(-np.sum(p * np.log2(p)))


class OnlineGSSM:
    """Pretrained GSSM applied to live SUMO state."""

    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.scaler = joblib.load(model_dir / "gssm_preprocessor.pkl")["scaler"]
        self.model = GSSM()
        self.model.load_state_dict(torch.load(model_dir / "gssm_model.pth",
                                              map_location="cpu", weights_only=True))
        self.model.eval()

    def max_risk(self, av: str, st) -> float:
        """Maximum M over all agents within detection range of the AV.

        Returns 0 when the AV is slower than 1 m/s or no agent is in range.
        Negative M is clipped to 0.
        """
        speed = st.veh_speed[av]
        if speed < 1.0:
            return 0.0
        ax, ay = st.veh_pos[av]
        avx, avy = st.veh_vx[av], st.veh_vy[av]
        av_mag = np.hypot(avx, avy)
        has_heading = av_mag >= MIN_SPEED_HEADING
        hx, hy = (avx / av_mag, avy / av_mag) if has_heading else (0.0, 1.0)

        r = np.clip(speed * RANGE_SECONDS, RANGE_MIN, RANGE_MAX)
        agents = []  # x, y, vx, vy, speed, accel, category
        for vid in st.veh_ids:
            if vid == av:
                continue
            x, y = st.veh_pos[vid]
            if (x - ax) ** 2 + (y - ay) ** 2 <= r ** 2:
                agents.append((x, y, st.veh_vx[vid], st.veh_vy[vid], st.veh_speed[vid],
                               st.veh_accel[vid], _category(st.veh_type[vid])))
        for pid in st.ped_ids:
            x, y = st.ped_pos[pid]
            if (x - ax) ** 2 + (y - ay) ** 2 <= r ** 2:
                s, a = st.ped_speed[pid], np.deg2rad(st.ped_angle[pid])
                agents.append((x, y, s * np.sin(a), s * np.cos(a), s, 0.0, "vru"))
        if not agents:
            return 0.0

        x, y, vx, vy, spd, acc = (np.array(c, dtype=float) for c in list(zip(*agents))[:6])
        cat = np.array([a[6] for a in agents])
        dist = np.hypot(x - ax, y - ay)

        density = int(np.sum(dist <= r))
        order = np.argsort(dist)
        rank = np.empty(len(dist), dtype=int)
        rank[order] = np.arange(1, len(dist) + 1)
        rank = np.clip(rank, 1, 10)
        speed_ent = _entropy(spd)
        dir_ent = _entropy(np.arctan2(vx, vy), n_bins=8)
        has_vru = float((cat == "vru").any())

        # Spacing along the relative velocity axis; agent velocity if that vanishes,
        # straight-line distance if both vanish.
        rx, ry = vx - avx, vy - avy
        norm = np.hypot(rx, ry)
        fb = norm < 1e-6
        rx, ry = np.where(fb, vx, rx), np.where(fb, vy, ry)
        norm = np.hypot(rx, ry)
        zero = norm < 1e-6
        safe = np.where(zero, 1.0, norm)
        ddx, ddy = ax - x, ay - y
        rho = np.where(zero, 0.0, (ry * ddx - rx * ddy) / safe)
        s_ij = np.where(zero, dist, (rx * ddx + ry * ddy) / safe)
        spacing = np.abs(s_ij) + 1e-6

        dx, dy = x - ax, y - ay
        if has_heading:
            fwd, lat = dx * hx + dy * hy, -dx * hy + dy * hx
        else:
            fwd, lat = np.zeros_like(dist), dist
        ag_mag = np.hypot(vx, vy)
        moving = has_heading & (ag_mag >= MIN_SPEED_HEADING)
        hdiff = np.where(moving, np.clip((avx * vx + avy * vy) /
                                         np.where(moving, av_mag * ag_mag, 1.0), -1, 1), 0.0)
        abs_lat = np.abs(lat)

        X = np.column_stack([
            rho, np.hypot(avx - vx, avy - vy), spd, acc,
            np.full_like(dist, speed), np.full_like(dist, st.veh_accel[av]),
            np.full_like(dist, density), hdiff, rank, abs_lat, np.arctan2(lat, fwd),
            ((abs_lat < 2.0) & (hdiff > 0.7)).astype(float),
            np.full_like(dist, speed_ent), np.full_like(dist, dir_ent),
            (cat == "vru").astype(float), np.full_like(dist, has_vru),
            (cat == "heavy").astype(float), spd / max(speed, 0.5),
        ]).astype(np.float32)
        spacing = spacing.astype(np.float32)

        valid = np.isfinite(X).all(axis=1) & (spacing > 0.5)
        if not valid.any():
            return 0.0
        with torch.no_grad():
            mu, log_sigma = self.model(torch.tensor(self.scaler.transform(X[valid])))
            sigma = torch.exp(log_sigma)
        M = risk_score(spacing[valid], mu.numpy().ravel(), sigma.numpy().ravel())
        M = np.clip(np.where(np.isfinite(M), M, 0.0), 0.0, None)
        return float(M.max())
