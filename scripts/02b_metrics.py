"""Step 2b: remaining metrics and the unified AV metric table.

Computes PET, time headway, string stability gain, jerk, and
deceleration for every AV timestep, joins the GSSM risk from Step 2a,
and writes data/processed/AV_Unified_Metrics.csv.

Usage:
    python scripts/02b_metrics.py [--config configs/metrics.yaml]
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from avconsensus.metrics import METRICS, build_unified, prepare_frame


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    proc = Path(cfg["processed_dir"])

    risk_file = proc / "gssm_max_risk.csv"
    if not risk_file.exists():
        raise FileNotFoundError(f"{risk_file} not found. Run scripts/02a_gssm.py first.")
    risk = pd.read_csv(risk_file)
    risk["time"] = risk["time"].round(2)

    parts = []
    for name, file in cfg["datasets"].items():
        print(f"[{name}]")
        df = prepare_frame(pd.read_csv(proc / file), cfg["neighbors"]["default_length"])
        parts.append(build_unified(df, risk[risk["dataset"] == name], name, cfg))

    out = pd.concat(parts, ignore_index=True)
    dst = proc / "AV_Unified_Metrics.csv"
    out.to_csv(dst, index=False)

    thr = cfg["thresholds"]
    print(f"\n{len(out):,} AV timesteps, {out['id'].nunique()} AVs")
    print(f"  {'metric':13s}{'available':>12s}{'median':>10s}{'violating':>11s}")
    for m in METRICS:
        v = out[m].dropna()
        bad = (v < thr[m]) if m == "pet" else (v > thr[m])
        print(f"  {m:13s}{len(v):>12,}{v.median():>10.3f}{bad.mean():>11.1%}")
    print(f"saved {dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/metrics.yaml")
    main(p.parse_args().config)
