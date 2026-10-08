"""Baseline vs trained comparison of evaluation records."""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm

METRICS = ["maximum_risk", "pet", "headway", "gain", "abs_jerk", "decel"]
SCORES = {"score_S": "Safety", "score_E": "Efficiency", "score_I": "Interaction"}
VIOLATIONS = {"safety_violated": "Safety", "efficiency_violated": "Efficiency",
              "interaction_violated": "Interaction"}
NUMERIC = METRICS + list(SCORES) + ["reward", "alpha_tau", "alpha_b", "app_tau", "app_b", "speed"]


def load(path) -> pd.DataFrame:
    """Read an evaluation CSV, coerce numeric columns, and add the route group
    (AV id without its trailing vehicle index)."""
    df = pd.read_csv(path)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["group"] = df["av_id"].astype(str).str.rsplit(".", n=1).str[0]
    df["n_violated"] = df[list(VIOLATIONS)].sum(axis=1)
    return df


def pct_change(b: float, t: float) -> float:
    return 100 * (t - b) / abs(b) if b else np.nan


def cohens_d(b: pd.Series, t: pd.Series) -> float:
    """Standardized mean difference (trained minus baseline), pooled SD."""
    nb, nt = len(b), len(t)
    sp = np.sqrt(((nb - 1) * b.var() + (nt - 1) * t.var()) / (nb + nt - 2))
    return (t.mean() - b.mean()) / sp if sp > 0 else np.nan


def two_proportion_p(b: pd.Series, t: pd.Series) -> float:
    """Two-sided z-test p-value for a difference in rates."""
    pb, pt, nb, nt = b.mean(), t.mean(), len(b), len(t)
    p = (b.sum() + t.sum()) / (nb + nt)
    se = np.sqrt(p * (1 - p) * (1 / nb + 1 / nt))
    return 2 * norm.sf(abs(pt - pb) / se) if se > 0 else np.nan


def key_results(base: pd.DataFrame, trained: pd.DataFrame) -> dict:
    """Headline changes: median scores, 2+ co-occurrence, and mean reward."""
    out = {}
    for c, name in SCORES.items():
        out[f"delta_{name[0]}_pct"] = pct_change(base[c].median(), trained[c].median())
    out["delta_cooccurrence_2plus_pct"] = pct_change((base["n_violated"] >= 2).mean(),
                                                     (trained["n_violated"] >= 2).mean())
    out["delta_reward_pct"] = pct_change(base["reward"].mean(), trained["reward"].mean())
    return out


def summarize(base: pd.DataFrame, trained: pd.DataFrame, title: str) -> str:
    """Text report comparing baseline and trained records."""
    L = ["=" * 72, title, "=" * 72,
         f"Baseline: {len(base):,} AV steps, {base['av_id'].nunique()} AVs",
         f"Trained:  {len(trained):,} AV steps, {trained['av_id'].nunique()} AVs"]

    L += ["", "Dimension scores (median)",
          f"  {'':13s}{'baseline':>10s}{'trained':>10s}{'change':>9s}{'MWU p':>11s}{'d':>8s}"]
    for c, name in SCORES.items():
        b, t = base[c].dropna(), trained[c].dropna()
        p = mannwhitneyu(t, b).pvalue if len(b) and len(t) else np.nan
        L.append(f"  {name:13s}{b.median():10.3f}{t.median():10.3f}"
                 f"{pct_change(b.median(), t.median()):+8.1f}%{p:11.2e}{cohens_d(b, t):8.2f}")

    L += ["", "Compliance rate (1 - violation rate)",
          f"  {'':13s}{'baseline':>10s}{'trained':>10s}{'p':>11s}"]
    for c, name in VIOLATIONS.items():
        L.append(f"  {name:13s}{1 - base[c].mean():9.1%} {1 - trained[c].mean():9.1%}"
                 f"{two_proportion_p(base[c], trained[c]):11.2e}")

    L += ["", "Violated dimensions per step",
          f"  {'':13s}{'baseline':>10s}{'trained':>10s}{'p':>11s}"]
    for k, name in [(0, "none"), (1, "exactly 1"), (2, "exactly 2"), (3, "all 3")]:
        L.append(f"  {name:13s}{(base['n_violated'] == k).mean():9.1%} "
                 f"{(trained['n_violated'] == k).mean():9.1%}")
    for k in (2, 3):
        b, t = base["n_violated"] >= k, trained["n_violated"] >= k
        L.append(f"  {f'{k}+':13s}{b.mean():9.1%} {t.mean():9.1%}{two_proportion_p(b, t):11.2e}")

    L += ["", "Raw metric medians",
          f"  {'':13s}{'baseline':>10s}{'trained':>10s}{'change':>9s}"]
    for c in METRICS:
        b, t = base[c].median(), trained[c].median()
        L.append(f"  {c:13s}{b:10.3f}{t:10.3f}{pct_change(b, t):+8.1f}%")

    b_r, t_r = base["reward"].mean(), trained["reward"].mean()
    L += ["", f"Mean per-step reward: {b_r:.4f} -> {t_r:.4f} ({pct_change(b_r, t_r):+.1f}%)"]

    L += ["", "Applied modulation factors (trained)"]
    for c in ("app_tau", "app_b"):
        v = trained[c].dropna()
        L.append(f"  {c:8s} mean {v.mean():.3f}, sd {v.std():.3f}, "
                 f"P10 {np.percentile(v, 10):.3f}, P90 {np.percentile(v, 90):.3f}")

    L += ["", "By route group (median scores; trained factors are means)",
          f"  {'group':14s}{'n_b':>7s}{'n_t':>7s}{'S_b':>7s}{'S_t':>7s}{'E_b':>7s}{'E_t':>7s}"
          f"{'I_b':>7s}{'I_t':>7s}{'tau':>7s}{'b':>7s}"]
    for g in sorted(set(base["group"]) & set(trained["group"])):
        b, t = base[base["group"] == g], trained[trained["group"] == g]
        L.append(f"  {g:14s}{len(b):7,}{len(t):7,}"
                 f"{b['score_S'].median():7.3f}{t['score_S'].median():7.3f}"
                 f"{b['score_E'].median():7.3f}{t['score_E'].median():7.3f}"
                 f"{b['score_I'].median():7.3f}{t['score_I'].median():7.3f}"
                 f"{t['app_tau'].mean():7.2f}{t['app_b'].mean():7.2f}")
    return "\n".join(L)
