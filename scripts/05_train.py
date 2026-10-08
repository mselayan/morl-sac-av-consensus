"""Step 5: train the Pareto-guided SAC policy.

Each batch runs `workers` SUMO episodes in parallel with the current
actor, adds their transitions to the replay buffer, and performs up to
`max_updates` gradient steps. Checkpoints and a progress log are written
to output_dir after every batch; training resumes from the latest full
checkpoint if one exists.

The trained actor used in the paper is shipped as models/sac/sac_actor.pth.

Usage:
    python scripts/05_train.py [--config configs/train.yaml]
"""

import argparse
import csv
import multiprocessing as mp
import re
from pathlib import Path

import numpy as np
import yaml

from avconsensus.rl.env import run_training_episode
from avconsensus.rl.sac import SACAgent
from avconsensus.sim.sumo import check_config

LOG_FIELDS = ["batch_id", "episode_start", "episode_end", "mean_reward", "std_reward",
              "mean_alpha_tau", "std_alpha_tau", "mean_alpha_b", "std_alpha_b",
              "mean_collisions"]


def worker(args):
    worker_id, actor_state, sim_cfg, cfg = args
    return run_training_episode(worker_id, actor_state, cfg["base_port"] + worker_id,
                                sim_cfg, cfg)


def latest_checkpoint(out_dir: Path):
    """Most recent full checkpoint and its episode number, or (None, 0)."""
    cps = [(int(re.search(r"ep(\d+)", p.name).group(1)), p)
           for p in out_dir.glob("sac_full_ep*.pth")]
    return max(cps)[::-1] if cps else (None, 0)


def main(config_path: str) -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    sim_cfg = yaml.safe_load(Path(cfg["sim_config"]).read_text())
    check_config(sim_cfg["sumo"]["config"])
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "sac_training_progress.csv"

    s = cfg["sac"]
    agent = SACAgent(10, 2, s["hidden"], s["lr"], s["gamma"], s["tau"], s["buffer_size"])
    path, last_ep = latest_checkpoint(out_dir)
    if path:
        agent.load(str(path))
        print(f"resumed from {path.name}")
    if not log.exists():
        with open(log, "w", newline="") as f:
            csv.writer(f).writerow(LOG_FIELDS)

    n_workers, total = cfg["workers"], cfg["episodes"]
    episode, batch_id = last_ep + 1, last_ep // n_workers

    mp.set_start_method("spawn", force=True)
    with mp.Pool(n_workers) as pool:
        while episode <= total:
            eps = list(range(episode, min(episode + n_workers, total + 1)))
            actor_state = agent.actor.state_dict()
            results = pool.map(worker, [(i, actor_state, sim_cfg, cfg) for i in range(len(eps))])

            for res in results:
                for tr in res["transitions"]:
                    agent.replay_buffer.push(*tr)
            if len(agent.replay_buffer) > cfg["min_buffer"]:
                for _ in range(min(len(agent.replay_buffer) // cfg["batch_size"],
                                   cfg["max_updates"])):
                    agent.update(cfg["batch_size"])

            col = {k: np.array([r[k] for r in results], dtype=float)
                   for k in ("reward", "avg_alpha_tau", "avg_alpha_b", "collisions")}
            row = [batch_id, eps[0], eps[-1],
                   col["reward"].mean(), col["reward"].std(ddof=1) if len(eps) > 1 else 0.0,
                   col["avg_alpha_tau"].mean(), col["avg_alpha_tau"].std(ddof=1) if len(eps) > 1 else 0.0,
                   col["avg_alpha_b"].mean(), col["avg_alpha_b"].std(ddof=1) if len(eps) > 1 else 0.0,
                   col["collisions"].mean()]
            with open(log, "a", newline="") as f:
                csv.writer(f).writerow(row)
            agent.save(eps[-1], str(out_dir))
            print(f"batch {batch_id:3d} | episodes {eps[0]}-{eps[-1]} | "
                  f"reward {row[3]:9.1f} | alpha_tau {row[5]:.2f} | alpha_b {row[7]:.2f} | "
                  f"entropy coef {agent.alpha:.3f}", flush=True)

            batch_id += 1
            episode = eps[-1] + 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/train.yaml")
    main(p.parse_args().config)
