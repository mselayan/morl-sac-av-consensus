"""Generalized Surrogate Safety Measure (GSSM), kinematic-only variant.

Spacing between an AV and each nearby agent is modeled as lognormal,
conditioned on 18 kinematic features. Risk M measures how much smaller
the observed spacing is than the context-typical spacing.
"""

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import erf
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

FEATURES = [
    "rho_ij", "rel_speed", "speed_agent", "acc_agent",
    "speed_av", "acc_av", "dynamic_density", "heading_diff",
    "rank_agent", "abs_lateral", "bearing_angle", "same_lane_proxy",
    "speed_entropy", "direction_entropy", "is_vru",
    "has_vru_nearby", "is_heavy", "speed_ratio",
]


# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------

def shannon_entropy(values: np.ndarray, n_bins: int = 10) -> float:
    """Histogram-based Shannon entropy (bits). Zero for degenerate input."""
    values = values[np.isfinite(values)]
    if len(values) < 2 or values.max() == values.min():
        return 0.0
    hist, _ = np.histogram(values, bins=n_bins, density=True)
    p = hist[hist > 0]
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


def rotated_spacing(av_v, ag_v, av_p, ag_p):
    """Project AV-agent offset onto the relative velocity frame.

    Returns (rho, s): lateral and longitudinal offset. Falls back to the
    agent's own velocity when relative velocity vanishes, and to zero
    when both vanish.
    """
    ax = ag_v[0] - av_v[0]
    ay = ag_v[1] - av_v[1]
    norm = np.hypot(ax, ay)

    fallback = norm < 1e-6
    ax = np.where(fallback, ag_v[0], ax)
    ay = np.where(fallback, ag_v[1], ay)
    norm = np.hypot(ax, ay)
    zero = norm < 1e-6
    norm = np.where(zero, 1.0, norm)

    dx, dy = av_p[0] - ag_p[0], av_p[1] - ag_p[1]
    rho = np.where(zero, 0.0, (ay * dx - ax * dy) / norm)
    s = np.where(zero, 0.0, (ax * dx + ay * dy) / norm)
    return rho, s


def build_interactions(df: pd.DataFrame, dataset: str, cfg: dict) -> pd.DataFrame:
    """Build one row per AV-agent pair per timestep with the 18 features.

    Args:
        df: prepared frame (see prepare_frame).
        dataset: dataset label stored in the output.
        cfg: gssm config.

    Returns:
        Frame with time, av_id, agent_id, dataset, target s_ij, and FEATURES.
    """
    vru, heavy = cfg["vru_types"], cfg["heavy_types"]
    min_hd = cfg["min_speed_heading"]
    lane = cfg["same_lane"]
    parts = []

    for t, f in df.groupby("time"):
        if len(f) < 2:
            continue
        x, y = f["x"].to_numpy(), f["y"].to_numpy()
        vx, vy = f["vx"].to_numpy(), f["vy"].to_numpy()
        speed, accel = f["speed"].to_numpy(), f["accel"].to_numpy()
        types, ids = f["type"].to_numpy(), f["id"].to_numpy()

        for i in np.flatnonzero(types == "av"):
            dx, dy = x - x[i], y - y[i]
            d2 = dx ** 2 + dy ** 2
            r = np.clip(speed[i] * cfg["range_seconds"], cfg["range_min"], cfg["range_max"])
            near = d2 < r ** 2
            near[i] = False
            j = np.flatnonzero(near)
            if j.size == 0:
                continue
            dist = np.sqrt(d2[j])

            # Context shared by all agents around this AV.
            density = int(np.sum(dist <= max(20.0, speed[i] * 3.0)))
            order = np.argsort(dist)
            rank = np.empty_like(order)
            rank[order] = np.arange(1, len(order) + 1)
            rank = np.clip(rank, 1, 10)
            speed_ent = shannon_entropy(speed[j])
            dir_ent = shannon_entropy(np.arctan2(vy[j], vx[j]), n_bins=8)
            has_vru = float(np.isin(types[j], vru).any())

            # Body-frame position of each agent.
            av_spd = np.hypot(vx[i], vy[i])
            if av_spd >= min_hd:
                hx, hy = vx[i] / av_spd, vy[i] / av_spd
                fwd = dx[j] * hx + dy[j] * hy
                lat = -dx[j] * hy + dy[j] * hx
            else:
                fwd, lat = np.zeros_like(dist), dist

            ag_spd = np.hypot(vx[j], vy[j])
            both_moving = (av_spd >= min_hd) & (ag_spd >= min_hd)
            cos = (vx[i] * vx[j] + vy[i] * vy[j]) / np.where(both_moving, av_spd * ag_spd, 1.0)
            hdiff = np.where(both_moving, np.clip(cos, -1, 1), 0.0)

            rho, s = rotated_spacing((vx[i], vy[i]), (vx[j], vy[j]), (x[i], y[i]), (x[j], y[j]))
            abs_lat = np.abs(lat)

            parts.append(pd.DataFrame({
                "time": t, "av_id": ids[i], "agent_id": ids[j], "dataset": dataset,
                "s_ij": s,
                "rho_ij": rho,
                "rel_speed": np.hypot(vx[i] - vx[j], vy[i] - vy[j]),
                "speed_agent": speed[j],
                "acc_agent": accel[j],
                "speed_av": speed[i],
                "acc_av": accel[i],
                "dynamic_density": density,
                "heading_diff": hdiff,
                "rank_agent": rank,
                "abs_lateral": abs_lat,
                "bearing_angle": np.arctan2(lat, fwd),
                "same_lane_proxy": ((abs_lat < lane["max_lateral"]) &
                                    (hdiff > lane["min_heading_cos"])).astype(float),
                "speed_entropy": speed_ent,
                "direction_entropy": dir_ent,
                "is_vru": np.isin(types[j], vru).astype(float),
                "has_vru_nearby": has_vru,
                "is_heavy": np.isin(types[j], heavy).astype(float),
                "speed_ratio": speed[j] / max(speed[i], 0.5),
            }))

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def filter_interactions(df: pd.DataFrame, min_abs_spacing: float = 0.5,
                        max_abs_accel: float = 10.0) -> pd.DataFrame:
    """Drop degenerate pairs: near-zero spacing, non-finite values, extreme accel."""
    keep = (
        (df["s_ij"].abs() > min_abs_spacing)
        & np.isfinite(df["rho_ij"]) & np.isfinite(df["s_ij"])
        & np.isfinite(df["rel_speed"]) & np.isfinite(df["speed_agent"])
        & (df["acc_agent"].abs() < max_abs_accel)
        & (df["acc_av"].abs() < max_abs_accel)
    )
    return df[keep].reset_index(drop=True)


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

class GSSM(nn.Module):
    """MLP predicting lognormal spacing parameters (mu, log sigma)."""

    def __init__(self, input_dim: int = len(FEATURES)):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 1)
        self.fc_log_sigma = nn.Linear(64, 1)

    def forward(self, x):
        h = F.relu(self.fc2(F.relu(self.fc1(x))))
        return self.fc_mu(h), torch.clamp(self.fc_log_sigma(h), -10, 3)


def nll(log_s, mu, log_sigma):
    """Gaussian NLL on log spacing, i.e. lognormal NLL up to a constant."""
    return torch.mean(0.5 * ((log_s - mu) / torch.exp(log_sigma)) ** 2 + log_sigma)


def risk_score(s, mu, sigma):
    """GSSM risk M = log10(ln 0.5 / ln P(S > s)). Higher is riskier."""
    s = np.clip(s, 1e-6, None)
    cdf = 0.5 * (1 + erf((np.log(s) - mu) / (np.sqrt(2) * sigma)))
    p_exceed = np.clip(1 - cdf, 1e-12, 1 - 1e-12)
    return np.log10(np.abs(np.log(0.5)) / np.abs(np.log(p_exceed)))


def features_and_target(df: pd.DataFrame):
    """Feature matrix (float32, NaN/inf set to 0) and absolute spacing target."""
    X = np.nan_to_num(df[FEATURES].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.abs(df["s_ij"].to_numpy()) + 1e-6
    return X, y


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def split_by_av(X, y, groups, held_out=0.3, test_of_held_out=0.5, seed=42):
    """Split into train/val/test with no AV shared across splits."""
    tr, ho = next(GroupShuffleSplit(1, test_size=held_out, random_state=seed)
                  .split(X, y, groups))
    va_rel, te_rel = next(GroupShuffleSplit(1, test_size=test_of_held_out, random_state=seed)
                          .split(X[ho], y[ho], groups[ho]))
    return tr, ho[va_rel], ho[te_rel]


def train_gssm(df: pd.DataFrame, cfg: dict, seed: int = 42):
    """Fit scaler and GSSM on train AVs, keep the best epoch on validation AVs.

    Returns:
        (model, scaler, test_stats) where test_stats has nll, mae, median_ae.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    tc = cfg["train"]

    X, y = features_and_target(df)
    groups = df["av_id"].astype(str).to_numpy()
    tr, va, te = split_by_av(X, y, groups, tc["held_out"], tc["test_of_held_out"], seed)
    print(f"  split: {len(tr):,} train, {len(va):,} val, {len(te):,} test | AVs "
          f"{len(set(groups[tr]))}/{len(set(groups[va]))}/{len(set(groups[te]))}")

    scaler = StandardScaler().fit(X[tr])

    def tensors(idx):
        return (torch.tensor(scaler.transform(X[idx])),
                torch.tensor(np.log(y[idx]).astype(np.float32)).unsqueeze(1))

    train_ds, val_ds = TensorDataset(*tensors(tr)), TensorDataset(*tensors(va))
    train_dl = DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=tc["batch_size"])

    model = GSSM()
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    best, best_state = np.inf, None

    for epoch in range(tc["epochs"]):
        model.train()
        for xb, yb in train_dl:
            loss = nll(yb, *model(xb))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val = sum(nll(yb, *model(xb)).item() * len(xb) for xb, yb in val_dl) / len(val_ds)
        if val < best:
            best, best_state = val, copy.deepcopy(model.state_dict())
        if epoch % 10 == 0 or epoch == tc["epochs"] - 1:
            print(f"    epoch {epoch:3d}  val NLL {val:.4f}")

    model.load_state_dict(best_state)

    X_te, y_te = tensors(te)
    with torch.no_grad():
        mu, log_sigma = model(X_te)
        test_nll = nll(y_te, mu, log_sigma).item()
    err = np.abs(np.exp(mu.numpy().ravel()) - y[te])
    stats = dict(nll=test_nll, mae=float(err.mean()), median_ae=float(np.median(err)))
    print(f"  test NLL {stats['nll']:.4f}, MAE {stats['mae']:.2f} m, "
          f"median AE {stats['median_ae']:.2f} m")
    return model, scaler, stats


# ------------------------------------------------------------------
# Scoring and threshold
# ------------------------------------------------------------------

@torch.no_grad()
def score(df: pd.DataFrame, model: GSSM, scaler, batch_size: int = 4096) -> np.ndarray:
    """Risk M for every interaction. Non-finite scores are set to 0."""
    model.eval()
    X, y = features_and_target(df)
    X = torch.tensor(scaler.transform(X), dtype=torch.float32)
    mu, sigma = [], []
    for i in range(0, len(X), batch_size):
        m, ls = model(X[i:i + batch_size])
        mu.append(m.numpy().ravel())
        sigma.append(np.exp(ls.numpy().ravel()))
    M = risk_score(y, np.concatenate(mu), np.concatenate(sigma))
    return np.where(np.isfinite(M), M, 0.0)


def fit_threshold(M: np.ndarray, n_components: int = 3, seed: int = 42):
    """Fit a GMM to positive M and return the elevated-risk crossover.

    The threshold is the smallest M at which the highest-mean component
    has posterior probability above 0.5.

    Returns:
        (threshold, fitted GaussianMixture)
    """
    pos = M[np.isfinite(M) & (M > 0)].reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=seed).fit(pos)
    high = int(np.argmax(gmm.means_.ravel()))

    grid = np.linspace(pos.min(), pos.max(), 10000).reshape(-1, 1)
    above = gmm.predict_proba(grid)[:, high] > 0.5
    thr = float(grid[np.argmax(above), 0]) if above.any() else float(gmm.means_.mean())
    return thr, gmm


def max_risk_per_timestep(df: pd.DataFrame, M: np.ndarray) -> pd.DataFrame:
    """Maximum M over all agents for each AV timestep (the AV safety score)."""
    out = df[["av_id", "time", "dataset"]].assign(maximum_risk=M)
    return (out.groupby(["av_id", "time", "dataset"], as_index=False)["maximum_risk"].max()
               .rename(columns={"av_id": "id"}))
