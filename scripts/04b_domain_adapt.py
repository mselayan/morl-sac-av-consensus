"""Step 4b: map the Pareto surface into the simulation domain.

Builds simulation ECDFs from the baseline run, translates the TGSIM
thresholds through matching percentiles, and saves the adapted surface
used by the training reward. The adapted surface shipped in
models/adaptation is the one used in the paper.

Usage:
    python scripts/04b_domain_adapt.py                       # shipped TGSIM reference
    python scripts/04b_domain_adapt.py --rebuild-reference   # from AV_Unified_Metrics.csv
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
import yaml

from avconsensus.adaptation import METRICS, adapt, tgsim_reference


def load(path):
    return yaml.safe_load(Path(path).read_text())


def main(args):
    cfg = load(args.config)
    thresholds = load(cfg["metrics_config"])["thresholds"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.rebuild_reference:
        q = load(cfg["pareto_config"])["qualify"]
        ref = tgsim_reference(pd.read_csv(cfg["unified_metrics"]), q["min_abs_accel_for_jerk"],
                              q["min_decel"], cfg["ecdf_points"])
        joblib.dump(ref, out_dir / "tgsim_reference.joblib")
        print(f"rebuilt TGSIM reference from {cfg['unified_metrics']}")
    else:
        ref = joblib.load(cfg["tgsim_reference"])

    baseline = pd.read_csv(cfg["baseline_csv"])
    print(f"baseline: {len(baseline):,} AV steps")
    adapted = adapt(baseline, ref, joblib.load(cfg["pareto_surface"]), thresholds,
                    cfg["ecdf_points"], cfg["min_samples"])

    print(f"\n  {'metric':13s}{'TGSIM':>8s}{'pct':>7s}{'sim':>9s}{'violating':>11s}")
    for m in METRICS:
        t = adapted["thresholds_sim"][m]
        v = pd.to_numeric(baseline[m], errors="coerce").dropna()
        if m == "maximum_risk":
            v = v.clip(lower=0)
        bad = (v < t) if m == "pet" else (v > t)
        pct = adapted["thresholds_tgsim_percentile"].get(m, float("nan"))
        print(f"  {m:13s}{thresholds[m]:>8.2f}{pct:>7.1f}{t:>9.3f}{bad.mean():>11.1%}")

    dst = out_dir / "adapted_surface.joblib"
    joblib.dump(adapted, dst)
    print(f"\nsaved {dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/adaptation.yaml")
    p.add_argument("--rebuild-reference", action="store_true",
                   help="recompute TGSIM ECDFs from the unified metrics")
    main(p.parse_args())
