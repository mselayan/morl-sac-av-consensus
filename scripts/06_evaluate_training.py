"""Step 6: evaluate the trained policy on the training network.

Runs the baseline (EAB-IDM, no RL) and the deterministic trained policy
under the same seed, demand, and HDV draws, then compares dimension
scores, compliance, violation co-occurrence, and raw metrics.
Outputs go to outputs/evaluation/training.

Usage:
    python scripts/06_evaluate_training.py [--summary-only]
"""

import argparse
from pathlib import Path

import yaml

from avconsensus.evaluation.run import evaluate_network


def main(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    sim_cfg = yaml.safe_load(Path(cfg["sim_config"]).read_text())
    evaluate_network(cfg, sim_cfg["sumo"]["config"], "training",
                     "TRAINING NETWORK: BASELINE vs TRAINED", args.summary_only)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--summary-only", action="store_true",
                   help="rebuild the summary from existing CSVs")
    main(p.parse_args())
