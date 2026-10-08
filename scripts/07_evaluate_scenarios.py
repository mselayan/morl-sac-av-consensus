"""Step 7: zero-shot evaluation on the five unseen networks.

The trained actor is applied unchanged, with no retraining or adaptation.
Each scenario runs baseline and trained episodes and writes a summary to
outputs/evaluation/<scenario>. A table of headline changes across all
evaluated scenarios is written to outputs/evaluation/scenario_results.csv.

Usage:
    python scripts/07_evaluate_scenarios.py              # all scenarios
    python scripts/07_evaluate_scenarios.py S1 S4        # selected
    python scripts/07_evaluate_scenarios.py --summary-only
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from avconsensus.evaluation.run import evaluate_network


def main(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    ids = [s.upper() for s in args.scenarios] or list(cfg["scenarios"])
    unknown = set(ids) - set(cfg["scenarios"])
    if unknown:
        raise SystemExit(f"unknown scenarios: {sorted(unknown)}")

    rows = []
    for sid in ids:
        sc = cfg["scenarios"][sid]
        res = evaluate_network(cfg, sc["config"], sid, f"{sid}: {sc['facility'].upper()}",
                               args.summary_only)
        rows.append({"scenario": sid, "facility": sc["facility"], **res})

    table = pd.DataFrame(rows)
    dst = Path(cfg["output_dir"]) / "scenario_results.csv"
    table.to_csv(dst, index=False, float_format="%.1f")
    print("\n" + table.to_string(index=False, float_format=lambda x: f"{x:+.1f}"))
    print(f"\nsaved {dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("scenarios", nargs="*", help="scenario ids (default: all)")
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--summary-only", action="store_true",
                   help="rebuild summaries from existing CSVs")
    main(p.parse_args())
