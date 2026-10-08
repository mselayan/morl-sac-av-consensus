"""Step 4a: baseline simulation on the training network.

Runs one SUMO episode with EAB-modulated IDM AVs and heterogeneous HDVs,
no RL, and logs the six metrics for every AV step. Its distributions
define the simulation side of the quantile mapping.

Usage:
    python scripts/04a_baseline_sim.py [--sim-config configs/sim.yaml]
"""

import argparse
from pathlib import Path

import yaml

from avconsensus.sim.baseline import run_baseline


def main(args):
    sim_cfg = yaml.safe_load(Path(args.sim_config).read_text())
    out = Path(yaml.safe_load(Path(args.config).read_text())["baseline_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baseline simulation: {sim_cfg['sumo']['config']}")
    n = run_baseline(sim_cfg, str(out))
    print(f"saved {out} ({n:,} AV steps)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/adaptation.yaml")
    p.add_argument("--sim-config", default="configs/sim.yaml")
    main(p.parse_args())
