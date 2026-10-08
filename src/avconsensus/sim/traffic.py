"""Traffic heterogeneity: HDV IDM sampling and the AV EAB profile."""

import numpy as np
import traci
from scipy.stats import truncnorm


def truncated_normals(spec: dict) -> dict:
    """Frozen truncated normals from {param: [mean, sd, low, high]}."""
    return {k: truncnorm((lo - m) / sd, (hi - m) / sd, loc=m, scale=sd)
            for k, (m, sd, lo, hi) in spec.items()}


class Traffic:
    """Assigns HDV parameters at departure and tracks each AV's EAB multiplier.

    The EAB multiplier eta(t) follows a piecewise profile: four plateau levels
    joined by three linear ramps, starting t1 seconds after the AV's leader
    first decelerates beyond the trigger. The profile resets when the leader
    changes or disappears.
    """

    def __init__(self, cfg: dict):
        self.av_prefix = cfg["vehicles"]["av_prefix"]
        self.hdv_type = cfg["vehicles"]["hdv_type"]
        self.hdv_dist = truncated_normals(cfg["hdv_idm"])
        self.eab = cfg["eab"]
        self.profiles, self.disturbances, self.last_leader = {}, {}, {}
        self.hdvs = set()

    def is_av(self, vtype: str) -> bool:
        return vtype.startswith(self.av_prefix)

    def handle_departures(self) -> list:
        """Initialize newly departed vehicles. Returns the new AV ids."""
        new_avs = []
        for v in traci.simulation.getDepartedIDList():
            vtype = traci.vehicle.getTypeID(v)
            if self.is_av(vtype):
                self.profiles[v] = self._sample_profile()
                new_avs.append(v)
            elif vtype == self.hdv_type and v not in self.hdvs:
                p = {k: float(d.rvs()) for k, d in self.hdv_dist.items()}
                traci.vehicle.setTau(v, p["tau"])
                traci.vehicle.setMinGap(v, p["minGap"])
                traci.vehicle.setAccel(v, p["accel"])
                traci.vehicle.setDecel(v, p["decel"])
                self.hdvs.add(v)
        return new_avs

    def handle_arrivals(self) -> list:
        """Forget arrived vehicles. Returns the arrived ids."""
        arrived = list(traci.simulation.getArrivedIDList())
        for v in arrived:
            self.hdvs.discard(v)
            self.profiles.pop(v, None)
            self.disturbances.pop(v, None)
            self.last_leader.pop(v, None)
        return arrived

    def eta(self, av: str, t: float) -> float:
        """Current EAB multiplier for an AV."""
        leader = traci.vehicle.getLeader(av, self.eab["leader_range"])
        if leader:
            if (av not in self.disturbances and
                    traci.vehicle.getAcceleration(leader[0]) < self.eab["trigger_decel"]):
                self.disturbances[av] = t
            if av in self.disturbances and leader[0] != self.last_leader.get(av):
                del self.disturbances[av]
            self.last_leader[av] = leader[0]
        else:
            self.disturbances.pop(av, None)

        rel_t = t - self.disturbances.get(av, t)
        p = self.profiles[av]
        tk, ek, nk = p["t"], p["eps"], p["eta"]
        if rel_t < tk[0]:
            return nk[0]
        for k in range(3):
            if rel_t < tk[k + 1]:
                return nk[k] + ek[k] * (rel_t - tk[k])
        return nk[3]

    def _sample_profile(self) -> dict:
        """Four eta levels, three ramp slopes, and the ramp start times."""
        e = self.eab
        eta = np.random.uniform(*e["eta"], 4)
        eps = np.random.uniform(*e["eps"], 3)
        t = [np.random.uniform(*e["t1"])]
        for k in range(3):
            t.append(t[-1] + (eta[k + 1] - eta[k]) / eps[k] if eps[k] != 0 else t[-1])
        return {"eta": eta, "eps": eps, "t": t}
