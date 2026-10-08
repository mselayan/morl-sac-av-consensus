"""Baseline episode: EAB-modulated IDM AVs, no RL, metrics logged per AV step."""

import csv

import numpy as np
import traci

from . import sumo
from .gssm_online import OnlineGSSM
from .metrics_online import OnlineMetrics
from .state import SimState
from .traffic import Traffic

FIELDS = ["av_id", "time", "speed", "accel", "maximum_risk", "pet",
          "headway", "gain", "abs_jerk", "decel", "eta"]


def _fmt(x):
    return round(x, 4) if x is not None else ""


def run_baseline(cfg: dict, out_csv: str) -> int:
    """Run one baseline episode and write AV metrics to out_csv.

    Each AV's time headway and minimum gap are scaled by its EAB multiplier.
    Metrics are recorded only while the AV moves at min_speed or faster.

    Returns:
        Number of rows written.
    """
    dt = cfg["sumo"]["step_length"]
    idm = cfg["av_idm"]
    sumo.start(cfg)
    st, traffic = SimState(), Traffic(cfg)
    gssm, metrics = OnlineGSSM(cfg["gssm_dir"]), OnlineMetrics(dt)

    n_rows = 0
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for step in range(int(cfg["sumo"]["duration_s"] / dt)):
            traci.simulationStep()
            st.update()
            traffic.handle_departures()
            if step % 10000 == 0:
                print(f"  t={st.time:7.1f} s, rows {n_rows:,}", flush=True)

            for av in st.veh_ids:
                if av not in traffic.profiles or not traffic.is_av(st.veh_type.get(av, "")):
                    continue
                eta = traffic.eta(av, st.time)
                traci.vehicle.setTau(av, idm["tau"] * eta)
                traci.vehicle.setMinGap(av, idm["min_gap"] * eta)

                if st.veh_speed[av] < cfg["min_speed"]:
                    continue
                m = gssm.max_risk(av, st)
                o = metrics.compute_all(av, st)
                writer.writerow({
                    "av_id": av, "time": round(st.time, 2),
                    "speed": round(st.veh_speed[av], 4), "accel": round(st.veh_accel[av], 4),
                    "maximum_risk": round(m, 4) if m > 0 else "",
                    "pet": _fmt(o["pet"]), "headway": _fmt(o["headway"]),
                    "gain": _fmt(o["gain"]), "abs_jerk": _fmt(o["jerk"]),
                    "decel": _fmt(o["decel"]), "eta": round(eta, 4),
                })
                n_rows += 1

            traffic.handle_arrivals()

    traci.close()
    return n_rows
