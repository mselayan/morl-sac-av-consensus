"""Per-timestep snapshot of all vehicles and pedestrians from TraCI."""

import numpy as np
import traci


class SimState:
    """Query TraCI once per step and share the result across modules."""

    def __init__(self):
        self.time = 0.0
        self.veh_ids, self.ped_ids = [], []
        self.veh_pos, self.veh_speed, self.veh_angle, self.veh_accel = {}, {}, {}, {}
        self.veh_type, self.veh_length, self.veh_vx, self.veh_vy = {}, {}, {}, {}
        self.ped_pos, self.ped_speed, self.ped_angle = {}, {}, {}

    def update(self) -> None:
        self.time = traci.simulation.getTime()
        self.veh_ids = list(traci.vehicle.getIDList())
        for d in (self.veh_pos, self.veh_speed, self.veh_angle, self.veh_accel,
                  self.veh_type, self.veh_length, self.veh_vx, self.veh_vy):
            d.clear()

        for vid in self.veh_ids:
            speed = traci.vehicle.getSpeed(vid)
            angle = traci.vehicle.getAngle(vid)
            accel = traci.vehicle.getAcceleration(vid)
            self.veh_pos[vid] = traci.vehicle.getPosition(vid)
            self.veh_speed[vid] = speed
            self.veh_angle[vid] = angle
            self.veh_accel[vid] = accel if np.isfinite(accel) else 0.0
            self.veh_type[vid] = traci.vehicle.getTypeID(vid).lower()
            self.veh_length[vid] = traci.vehicle.getLength(vid)
            # SUMO angle is degrees clockwise from north.
            self.veh_vx[vid] = speed * np.sin(np.deg2rad(angle))
            self.veh_vy[vid] = speed * np.cos(np.deg2rad(angle))

        self.ped_ids = []
        for d in (self.ped_pos, self.ped_speed, self.ped_angle):
            d.clear()
        try:
            self.ped_ids = list(traci.person.getIDList())
            for pid in self.ped_ids:
                self.ped_pos[pid] = traci.person.getPosition(pid)
                self.ped_speed[pid] = traci.person.getSpeed(pid)
                self.ped_angle[pid] = traci.person.getAngle(pid)
        except traci.TraCIException:
            pass
