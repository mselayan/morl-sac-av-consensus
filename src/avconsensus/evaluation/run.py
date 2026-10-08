"""Run baseline and trained episodes on one network and write the comparison."""

from pathlib import Path

import yaml

from .episode import load_actor, run_eval_episode
from .report import key_results, load, summarize


def evaluate_network(eval_cfg: dict, sumocfg: str, name: str, title: str,
                     summary_only: bool = False) -> dict:
    """Evaluate one network. Outputs go to output_dir/name.

    Returns:
        Headline changes (see key_results).
    """
    sim_cfg = yaml.safe_load(Path(eval_cfg["sim_config"]).read_text())
    sim_cfg["sumo"]["config"] = sumocfg
    train_cfg = yaml.safe_load(Path(eval_cfg["train_config"]).read_text())
    out = Path(eval_cfg["output_dir"]) / name
    out.mkdir(parents=True, exist_ok=True)
    base_csv, trained_csv = out / "baseline.csv", out / "trained.csv"

    if not summary_only:
        actor = load_actor(eval_cfg["actor"], train_cfg["sac"]["hidden"])
        for mode, path, policy in [("baseline", base_csv, None), ("trained", trained_csv, actor)]:
            print(f"  [{name}] {mode}", flush=True)
            n = run_eval_episode(sim_cfg, train_cfg, eval_cfg["adapted_surface"], str(path),
                                 policy, eval_cfg["seed"])
            print(f"  [{name}] {mode}: {n:,} AV steps", flush=True)

    base, trained = load(base_csv), load(trained_csv)
    text = summarize(base, trained, title)
    (out / "summary.txt").write_text(text + "\n")
    print(text)
    return key_results(base, trained)
