"""Step 3: empirical Pareto surface from the TGSIM AV metrics.

Normalizes the six metrics by percentile rank, builds Safety, Efficiency,
and Interaction scores, applies the co-occurrence penalty, finds the
Pareto front, and fits a convex hull. Writes the surface, the scored
composites, and the surface figure to outputs/pareto.

The surface shipped in models/pareto is the one used in the paper.

Usage:
    python scripts/03_pareto.py [--config configs/pareto.yaml]
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
import yaml

from avconsensus.pareto import (OBJECTIVES, apply_penalty, build_surface, composites,
                                pareto_front, qualify)
from avconsensus.plotting import plot_pareto_surface


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    thresholds = yaml.safe_load(Path(cfg["metrics_config"]).read_text())["thresholds"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg["input"])
    print(f"Loaded {len(df):,} AV timesteps")

    q = cfg["qualify"]
    df = qualify(df, q["min_abs_accel_for_jerk"], q["min_decel"])
    df = composites(df, cfg["min_valid_dimensions"], cfg["knn"])
    print(f"  scored: {len(df):,} (imputed {df['imputed'].mean():.1%})")

    df = apply_penalty(df, thresholds, cfg["penalty_coef"])
    print(f"  with any violation: {(df['n_violated'] > 0).mean():.1%}")

    df["pareto_flag"] = pareto_front(df, cfg["floor"])
    front = df[df["pareto_flag"]]
    print(f"  Pareto-optimal: {len(front)} ({len(front) / len(df):.2%}), "
          f"by dataset: {front['dataset'].value_counts().to_dict()}")
    print(df.groupby("pareto_flag")[OBJECTIVES].mean().round(3).to_string())

    surface = build_surface(front[OBJECTIVES].to_numpy(float), thresholds, len(df))
    s = surface["pareto_stats"]
    print(f"  hull: {s['n_facets']} facets, {s['n_upper_facets']} upper, "
          f"volume {surface['c_hull_model'].volume:.4f}")

    joblib.dump(surface, out_dir / "TGSIM_pareto_surface.joblib")
    df.to_csv(out_dir / "TGSIM_pareto_composites.csv", index=False)
    plot_pareto_surface(surface, df, out_dir / "pareto_surface.png", **cfg["plot"])
    print(f"saved surface, composites, and figure to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/pareto.yaml")
    main(p.parse_args().config)
