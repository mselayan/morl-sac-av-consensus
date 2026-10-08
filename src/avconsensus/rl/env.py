"""Training episode: SAC modulates AV headway and braking on top of EAB-IDM.

Each AV step:
  1. EAB multiplier eta from the leader disturbance profile.
  2. GSSM risk and the five other metrics from the live state.
  3. 10-D state -> action in [-1, 1]^2 -> modulation factors in [0.3, 2.0].
  4. Exponentially smoothed factors set tau = 1.55 eta a_tau, decel = 1.02 a_b.
  5. Reward from proximity to the adapted Pareto frontier.
"""

import os
import time

import numpy as np
import torch
import traci

from ..sim import sumo
from ..sim.gssm_online import OnlineGSSM
from ..sim.metrics_online import OnlineMetrics
from ..sim.state import SimState
from ..sim.traffic import Traffic
from .reward import ParetoReward
from .sac import Actor

# Fixed scaling ranges for the state vector.
M_MAX, PET_MAX, V_MAX, GAP_MAX = 10.0, 10.0, 30.0, 120.0
REL_SPEED_RANGE, LEADER_ACCEL_RANGE, DENSITY_MAX = 15.0, 8.0, 30.0


def to_alpha(a: np.ndarray, low: float, high: float) -> np.ndarray:
    """Map tanh output in [-1, 1] to a modulation factor in [low, high]."""
    scale = (high - low) / 2
    return low + (a + 1) * scale


def build_state(av, st, eta, alpha_tau, alpha_b, m, pet, low, high) -> np.ndarray:
    """10-D state in [0, 1]: risk, PET, speed, gap, relative speed, leader
    accel, density, eta, and the previous raw modulation factors."""
    v = st.veh_speed.get(av, 0.0)

    leader = traci.vehicle.getLeader(av, GAP_MAX)
    gap, rel, l_acc = GAP_MAX, 0.0, 0.0
    if leader and np.isfinite(leader[1]):
        gap = leader[1]
        try:
            rel = v - traci.vehicle.getSpeed(leader[0])
            l_acc = traci.vehicle.getAcceleration(leader[0])
            l_acc = l_acc if np.isfinite(l_acc) else 0.0
        except traci.TraCIException:
            rel, l_acc = 0.0, 0.0

    ax, ay = st.veh_pos[av]
    r = max(20.0, v * 3.0)
    pts = [st.veh_pos[o] for o in st.veh_ids if o != av] + [st.ped_pos[p] for p in st.ped_ids]
    density = sum(np.hypot(x - ax, y - ay) <= r for x, y in pts)

    raw = [
        m / M_MAX,
        (pet if pet is not None else PET_MAX) / PET_MAX,
        v / V_MAX,
        gap / GAP_MAX,
        (rel + REL_SPEED_RANGE) / (2 * REL_SPEED_RANGE),
        (l_acc + LEADER_ACCEL_RANGE) / (2 * LEADER_ACCEL_RANGE),
        density / DENSITY_MAX,
        (eta - 0.5) / 1.0,
        (alpha_tau - low) / (high - low),
        (alpha_b - low) / (high - low),
    ]
    return np.clip(np.array(raw, dtype=np.float32), 0.0, 1.0)


def run_training_episode(worker_id: int, actor_state: dict, port: int, sim_cfg: dict,
                         train_cfg: dict) -> dict:
    """Run one stochastic-policy episode and collect transitions.

    Returns:
        dict with total reward, collisions, mean applied factors, and the
        list of (s, a, r, s', done) transitions.
    """
    act = train_cfg["action"]
    low, high = act["low"], act["high"]
    k_tau, k_b = train_cfg["smoothing"]["tau"], train_cfg["smoothing"]["b"]
    idm, dt = sim_cfg["av_idm"], sim_cfg["sumo"]["step_length"]

    actor = Actor(10, 2, train_cfg["sac"]["hidden"])
    actor.load_state_dict(actor_state)
    actor.eval()

    time.sleep(worker_id * 0.5)  # stagger SUMO launches
    tmp = os.path.join(train_cfg["output_dir"], "tmp", f"worker_{worker_id}")
    os.makedirs(tmp, exist_ok=True)
    os.environ["TMPDIR"] = tmp

    result = {"reward": 0.0, "collisions": 0, "avg_alpha_tau": 1.0,
              "avg_alpha_b": 1.0, "transitions": []}
    try:
        sumo.start(sim_cfg, port=port)
        st, traffic = SimState(), Traffic(sim_cfg)
        gssm, metrics = OnlineGSSM(sim_cfg["gssm_dir"]), OnlineMetrics(dt)
        rw = train_cfg["reward"]
        reward_fn = ParetoReward(train_cfg["adapted_surface"], rw["exponent"],
                                 rw["quality_weight"], rw["penalty_coef"])

        transitions, applied_log = [], []
        raw_alpha, applied, prev, last_sei, last_idx = {}, {}, {}, {}, {}

        for _ in range(int(sim_cfg["sumo"]["duration_s"] / dt)):
            traci.simulationStep()
            st.update()
            result["collisions"] += traci.simulation.getCollidingVehiclesNumber()
            for v in traffic.handle_departures():
                raw_alpha[v] = applied[v] = (1.0, 1.0)

            for av in st.veh_ids:
                if av not in traffic.profiles or not traffic.is_av(st.veh_type.get(av, "")):
                    continue
                eta = traffic.eta(av, st.time)

                m = min(gssm.max_risk(av, st), M_MAX)
                o = metrics.compute_all(av, st)
                state = build_state(av, st, eta, *raw_alpha[av], m, o["pet"], low, high)

                with torch.no_grad():
                    action = actor.sample(torch.as_tensor(state).unsqueeze(0))[0].numpy()[0]
                a_tau, a_b = to_alpha(action, low, high)
                app_tau = k_tau * applied[av][0] + (1 - k_tau) * a_tau
                app_b = k_b * applied[av][1] + (1 - k_b) * a_b
                applied_log.append((app_tau, app_b))

                traci.vehicle.setTau(av, idm["tau"] * eta * app_tau)
                traci.vehicle.setMinGap(av, idm["min_gap"] * eta)
                traci.vehicle.setDecel(av, idm["decel"] * app_b)

                if st.veh_speed[av] >= sim_cfg["min_speed"]:
                    r, sei = reward_fn({"maximum_risk": m, "pet": o["pet"],
                                        "headway": o["headway"], "gain": o["gain"],
                                        "abs_jerk": o["jerk"], "decel": o["decel"]},
                                       last_sei.get(av))
                    if sei is not None:
                        last_sei[av] = sei
                    if av in prev and r is not None:
                        transitions.append((*prev[av], r, state, False))
                        last_idx[av] = len(transitions) - 1
                        result["reward"] += r

                prev[av] = (state, action)
                raw_alpha[av] = (a_tau, a_b)
                applied[av] = (app_tau, app_b)

            for v in traffic.handle_arrivals():
                if v in last_idx:
                    _terminal(transitions, last_idx, v)

        for v in list(last_idx):
            _terminal(transitions, last_idx, v)
        traci.close()

        if applied_log:
            result["avg_alpha_tau"], result["avg_alpha_b"] = map(float, np.mean(applied_log, 0))
        result["transitions"] = transitions
    except Exception as e:  # keep the batch alive if one worker fails
        try:
            traci.close()
        except Exception:
            pass
        print(f"[worker {worker_id}] failed: {e}", flush=True)
    return result


def _terminal(transitions: list, last_idx: dict, v: str) -> None:
    """Mark the last transition of vehicle v as terminal."""
    i = last_idx.pop(v)
    s, a, r, s2, _ = transitions[i]
    transitions[i] = (s, a, r, s2, True)
