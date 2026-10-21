from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpisodeMetrics:
    travel_times: list = field(default_factory=list)
    waiting: dict = field(default_factory=dict)
    inserted: int = 0
    pending: list = field(default_factory=list)
    co2_mg: float = 0.0
    ev_times: list = field(default_factory=list)
    violations: int = 0
    episode_s: float = 3600.0
    actuations: int = 0

    def att(self):
        return float(np.mean(self.travel_times)) if self.travel_times else float("nan")

    def att_plus(self):
        extra = [e + r for e, r in self.pending]
        allv = self.travel_times + extra
        return float(np.mean(allv)) if allv else float("nan")

    def awt(self):
        return float(np.mean(list(self.waiting.values()))) if self.waiting else float("nan")

    def completion(self):
        total = max(self.inserted + len(self.pending_waiting()), 1)
        return len(self.travel_times) / total

    def pending_waiting(self):
        return [p for p in self.pending if p[0] == 0.0]

    def throughput(self):
        return len(self.travel_times) * 3600.0 / self.episode_s

    def co2_kg_per_h(self):
        return self.co2_mg * 1e-6 * 3600.0 / self.episode_s

    def ev_time(self):
        return float(np.mean(self.ev_times)) if self.ev_times else float("nan")

    def violations_per_h(self):
        return self.violations * 3600.0 / self.episode_s

    def as_row(self):
        return {"att": self.att(), "att_plus": self.att_plus(), "awt": self.awt(),
                "completion": self.completion(), "throughput": self.throughput(),
                "co2": self.co2_kg_per_h(), "ev_time": self.ev_time(),
                "violations": self.violations_per_h()}


def collect_pending(traci, t_now, free_flow_remaining):
    pend = []
    for vid in traci.vehicle.getIDList():
        elapsed = t_now - traci.vehicle.getDeparture(vid) if hasattr(traci.vehicle, "getDeparture") else 0.0
        pend.append((elapsed, free_flow_remaining(vid)))
    for vid in traci.simulation.getPendingVehicles():
        pend.append((0.0, free_flow_remaining(vid)))
    return pend
