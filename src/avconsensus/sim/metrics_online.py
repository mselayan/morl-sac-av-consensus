"""Headway, gain, jerk, deceleration, and PET for one AV per simulation step.

Leader and follower come from SUMO's lane topology. Per-AV histories are
kept between calls, so call compute_all once per AV per step.
"""

from collections import defaultdict, deque

import numpy as np
import traci

GAIN_WINDOW_S = 5.0
PET_WINDOW_S = 10.0
PET_BOX = 2.0           # m, half side of the zone around past AV positions
PET_PREFILTER = 50.0    # m
PET_MAX = 10.0          # s
HEADWAY_MAX = 12.0      # s
MIN_RMS = 0.05          # m/s^2
MIN_GAIN_STEPS = 30
LOOKAHEAD = 120.0       # m


class OnlineMetrics:

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.accel_history = defaultdict(lambda: deque(maxlen=int(GAIN_WINDOW_S / dt)))
        self.accel_recent = defaultdict(lambda: deque(maxlen=5))
        self.prev_smoothed = {}
        self.pet_history = defaultdict(lambda: deque(maxlen=int(PET_WINDOW_S / dt)))

    def reset(self) -> None:
        for d in (self.accel_history, self.accel_recent, self.prev_smoothed, self.pet_history):
            d.clear()

    @staticmethod
    def neighbors(av: str):
        """(leader id, gap to leader, follower id) from SUMO, None if absent.

        SUMO measures the leader gap from the ego front bumper plus minGap.
        """
        info = traci.vehicle.getLeader(av, LOOKAHEAD)
        leader, gap = (info[0], info[1]) if info and np.isfinite(info[1]) else (None, None)
        try:
            f = traci.vehicle.getFollower(av, LOOKAHEAD)
            follower = f[0] if f and f[0] != "" and np.isfinite(f[1]) else None
        except traci.TraCIException:
            follower = None
        return leader, gap, follower

    @staticmethod
    def headway(av, leader, gap, st):
        """Leader gap over ego speed; None without a leader, below 0.5 m/s, or above 12 s."""
        if leader is None or gap is None or st.veh_speed[av] < 0.5:
            return None
        hw = max(gap, 0.0) / st.veh_speed[av]
        return hw if hw <= HEADWAY_MAX else None

    def gain(self, av, follower, st):
        """RMS(follower accel) / RMS(AV accel) over the trailing window.

        Both histories are stored per AV; the follower history holds the
        accel of whichever vehicle followed the AV at each step.
        """
        if follower is None:
            return None
        h_av, h_f = self.accel_history[("av", av)], self.accel_history[("follower", av)]
        h_av.append(st.veh_accel[av])
        h_f.append(st.veh_accel.get(follower, 0.0))
        if len(h_av) < MIN_GAIN_STEPS or len(h_f) < MIN_GAIN_STEPS:
            return None
        rms_av = np.sqrt(np.mean(np.square(h_av)))
        if rms_av < MIN_RMS:
            return None
        return np.sqrt(np.mean(np.square(h_f))) / rms_av

    def jerk(self, av, st):
        """|Jerk| from a 3-point moving average of acceleration."""
        q = self.accel_recent[av]
        q.append(st.veh_accel[av])
        if len(q) < 3:
            return None
        smoothed = np.mean(list(q)[-3:])
        prev = self.prev_smoothed.get(av)
        self.prev_smoothed[av] = smoothed
        if prev is None:
            return None
        return abs((smoothed - prev) / self.dt)

    @staticmethod
    def decel(av, st):
        """Deceleration magnitude when braking, None otherwise."""
        a = st.veh_accel[av]
        return abs(a) if a < 0 else None

    def pet(self, av, st):
        """Post-encroachment time over the trailing 10 s of AV positions.

        For each nearby agent, the elapsed time to the first stored AV
        position (oldest first) whose zone it occupies; PET is the minimum
        across agents, kept if at most 10 s.
        """
        ax, ay = st.veh_pos[av]
        t = st.time
        hist = self.pet_history[av]
        hist.append((t, ax, ay))

        others = [st.veh_pos[v] for v in st.veh_ids if v != av]
        others += [st.ped_pos[p] for p in st.ped_ids]

        best = None
        for ox, oy in others:
            if abs(ox - ax) > PET_PREFILTER and abs(oy - ay) > PET_PREFILTER:
                continue
            for pt, px, py in hist:
                if pt == t:
                    continue
                dt = t - pt
                if best is not None and dt >= best:
                    continue
                if abs(ox - px) < PET_BOX and abs(oy - py) < PET_BOX:
                    best = dt
                    break
        return best if best is not None and best <= PET_MAX else None

    def compute_all(self, av, st) -> dict:
        leader, gap, follower = self.neighbors(av)
        return {
            "headway": self.headway(av, leader, gap, st),
            "gain": self.gain(av, follower, st),
            "jerk": self.jerk(av, st),
            "decel": self.decel(av, st),
            "pet": self.pet(av, st),
        }
