"""Step 1: transform and smooth raw TGSIM trajectories.

Reads raw CSVs from data/raw, flips the y axis, smooths positions,
derives kinematics, and writes one processed CSV per dataset to
data/processed.

Usage:
    python scripts/01_prepare_data.py [--config configs/data.yaml]
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from avconsensus.data import flip_y_axis, smooth_dataset


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    raw_dir, out_dir = Path(cfg["raw_dir"]), Path(cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    sm = cfg["smoothing"]

    av_rows = {}
    for name, d in cfg["datasets"].items():
        src = raw_dir / d["raw_file"]
        if not src.exists():
            raise FileNotFoundError(f"{src} not found. See data/README.md.")

        print(f"\n[{name}] loading {src.name}")
        df = pd.read_csv(src)
        print(f"  rows {len(df):,}, agents {df['id'].nunique()}")

        df = flip_y_axis(df, d["y_extent_px"], d["px_to_m"], d["flip_y_kinematics"])
        out = smooth_dataset(df, d["types"], name,
                             sigma=sm["sigma"], min_len=sm["min_traj_len"])

        dst = out_dir / d["out_file"]
        out.to_csv(dst, index=False)
        print(f"  saved {dst}")
        av_rows[name] = int((out["type"] == "av").sum())

    print("\nAV timesteps:", ", ".join(f"{k} {v:,}" for k, v in av_rows.items()),
          f"| total {sum(av_rows.values()):,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/data.yaml")
    main(p.parse_args().config)
