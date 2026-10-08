"""Evaluation episode: baseline EAB-IDM or the deterministic trained policy.

Both modes compute the metrics at every AV step and log the AV steps at or
above min_speed with raw metrics, (S, E, I) scores, reward, and per-dimension
violation flags.
"""

import csv

import numpy as np
import torch
import traci

from ..rl.env import M_MAX, build_state, to_alpha
from ..rl.reward import DIMENSIONS, ParetoReward
from ..rl.sac import Actor
from ..sim import sumo
from ..sim.gssm_online import OnlineGSSM
from ..sim.metrics_online import OnlineMetrics
from ..sim.state import SimState
from ..sim.traffic import Traffic

FIELDS = ["av_id", "time", "speed", "accel",
          "maximum_risk", "pet", "headway", "gain", "abs_jerk", "decel",
          "eta", "alpha_tau", "alpha_b", "app_tau", "app_b",
          "score_S", "score_E", "score_I", "reward",
          "safety_violated", "efficiency_violated", "interaction_violated"]


def _r(x):
    return round(float(x), 4) if x is not None else ""


def load_actor(path: str, hidden: int = 256) -> Actor:
    actor = Actor(10, 2, hidden)
    actor.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    actor.eval()
    return actor


def run_eval_episode(sim_cfg: dict, train_cfg: dict, surface_path: str, out_csv: str,
                     actor: Actor | None = None, seed: int = 42) -> int:
    """Run one evaluation episode and write AV step records to out_csv.

    Args:
        sim_cfg: sim config; sim_cfg["sumo"]["config"] selects the network.
        train_cfg: action range, smoothing, and reward settings.
        surface_path: adapted surface for scores, reward, and thresholds.
        out_csv: output path.
        actor: trained actor, or None for the baseline.
        seed: seed for SUMO and for HDV and EAB sampling.

    Returns:
        Number of rows written.
    """
    np.random.seed(seed)
    cfg = {**sim_cfg, "sumo": {**sim_cfg["sumo"],
                               "extra_args": [*sim_cfg["sumo"]["extra_args"], "--seed", str(seed)]}}
    low, high = train_cfg["action"]["low"], train_cfg["action"]["high"]
    k_tau, k_b = train_cfg["smoothing"]["tau"], train_cfg["smoothing"]["b"]
    idm, dt = cfg["av_idm"], cfg["sumo"]["step_length"]
    rw = train_cfg["reward"]

    sumo.start(cfg)
    st, traffic = SimState(), Traffic(cfg)
    gssm, metrics = OnlineGSSM(cfg["gssm_dir"]), OnlineMetrics(dt)
    reward_fn = ParetoReward(surface_path, rw["exponent"], rw["quality_weight"], rw["penalty_coef"])

    raw_alpha, applied, last_sei = {}, {}, {}
    n_rows = 0
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for step in range(int(cfg["sumo"]["duration_s"] / dt)):
            traci.simulationStep()
            st.update()
            for v in traffic.handle_departures():
                raw_alpha[v] = applied[v] = (1.0, 1.0)
            if step % 10000 == 0:
                print(f"    t={st.time:7.1f} s, rows {n_rows:,}", flush=True)

            for av in st.veh_ids:
                if av not in traffic.profiles or not traffic.is_av(st.veh_type.get(av, "")):
                    continue
                eta = traffic.eta(av, st.time)
                m = min(gssm.max_risk(av, st), M_MAX)
                o = metrics.compute_all(av, st)

                if actor is not None:
                    state = build_state(av, st, eta, *raw_alpha[av], m, o["pet"], low, high)
                    with torch.no_grad():
                        a = torch.tanh(actor(torch.as_tensor(state).unsqueeze(0))[0]).numpy()[0]
                    a_tau, a_b = to_alpha(a, low, high)
                    app_tau = k_tau * applied[av][0] + (1 - k_tau) * a_tau
                    app_b = k_b * applied[av][1] + (1 - k_b) * a_b
                    traci.vehicle.setTau(av, idm["tau"] * eta * app_tau)
                    traci.vehicle.setMinGap(av, idm["min_gap"] * eta)
                    traci.vehicle.setDecel(av, idm["decel"] * app_b)
                    raw_alpha[av], applied[av] = (a_tau, a_b), (app_tau, app_b)
                else:
                    a_tau = a_b = app_tau = app_b = 1.0
                    traci.vehicle.setTau(av, idm["tau"] * eta)
                    traci.vehicle.setMinGap(av, idm["min_gap"] * eta)

                if st.veh_speed[av] < cfg["min_speed"]:
                    continue

                values = {"maximum_risk": m, "pet": o["pet"], "headway": o["headway"],
                          "gain": o["gain"], "abs_jerk": o["jerk"], "decel": o["decel"]}
                r, sei = reward_fn(values, last_sei.get(av))
                if sei is not None:
                    last_sei[av] = sei
                viol = [any(reward_fn.violated(values[k], k) for k in ks)
                        for ks in DIMENSIONS.values()]

                writer.writerow({
                    "av_id": av, "time": round(st.time, 2),
                    "speed": _r(st.veh_speed[av]), "accel": _r(st.veh_accel[av]),
                    "maximum_risk": _r(m) if m > 0 else "",
                    "pet": _r(o["pet"]), "headway": _r(o["headway"]), "gain": _r(o["gain"]),
                    "abs_jerk": _r(o["jerk"]), "decel": _r(o["decel"]), "eta": _r(eta),
                    "alpha_tau": _r(a_tau), "alpha_b": _r(a_b),
                    "app_tau": _r(app_tau), "app_b": _r(app_b),
                    "score_S": _r(sei[0]) if sei else "", "score_E": _r(sei[1]) if sei else "",
                    "score_I": _r(sei[2]) if sei else "", "reward": _r(r),
                    "safety_violated": int(viol[0]), "efficiency_violated": int(viol[1]),
                    "interaction_violated": int(viol[2]),
                })
                n_rows += 1

            traffic.handle_arrivals()

    traci.close()
    return n_rows
