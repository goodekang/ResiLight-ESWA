from dataclasses import dataclass, field

import numpy as np

from .shield import Shield

TURN_SLOT = {"l": 0, "L": 0, "t": 0, "s": 1, "r": 2, "R": 2}


@dataclass
class Intersection:
    id: str
    in_lanes: list
    out_lanes: list
    links: list
    movements: list
    slot_of_link: list
    phases: list
    phase_states: list
    successors: list
    receiving: list
    capacity: np.ndarray
    neighbors: list = field(default_factory=list)

    @property
    def num_phases(self):
        return len(self.phases)


def _green_phases(logic):
    out = []
    for i, ph in enumerate(logic.phases):
        s = ph.state
        if any(c in "Gg" for c in s) and "y" not in s.lower().replace("g", ""):
            out.append((i, s))
    return out


class SumoNetwork:
    def __init__(self, traci, sumolib_net, spacing_m=7.5, slots=12):
        self.traci = traci
        self.net = sumolib_net
        self.spacing = spacing_m
        self.slots = slots
        self.intersections = {}
        self.order = []
        self._build()

    def _build(self):
        tr = self.traci
        for tid in tr.trafficlight.getIDList():
            logic = tr.trafficlight.getAllProgramLogics(tid)[0]
            links = tr.trafficlight.getControlledLinks(tid)
            greens = _green_phases(logic)
            if not greens:
                continue
            in_lanes, out_lanes, link_list = [], [], []
            for idx, group in enumerate(links):
                if not group:
                    continue
                a, b, _ = group[0]
                if a not in in_lanes:
                    in_lanes.append(a)
                if b not in out_lanes:
                    out_lanes.append(b)
                link_list.append((idx, in_lanes.index(a), out_lanes.index(b), a, b))
            edges = []
            for l in in_lanes:
                e = tr.lane.getEdgeID(l)
                if e not in edges:
                    edges.append(e)
            slot_of = []
            for idx, ia, ob, a, b in link_list:
                appr = min(edges.index(tr.lane.getEdgeID(a)), 3)
                turn = self._turn(a, b)
                slot_of.append(appr * 3 + turn)
            phases, states, moves = [], [], []
            for pi, s in greens:
                states.append(s)
                phases.append(pi)
                moves.append([(ia, ob) for (idx, ia, ob, a, b) in link_list if s[idx] in "Gg"])
            P = len(phases)
            succ = [sorted({(p + 1) % P, (p + 2) % P} - {p}) for p in range(P)]
            recv = [[ob for (idx, ia, ob, a, b) in link_list if s[idx] in "Gg"] for s in states]
            cap = np.array([max(1.0, tr.lane.getLength(l) / self.spacing) for l in in_lanes + out_lanes], dtype=np.float32)
            it = Intersection(tid, in_lanes, out_lanes, link_list, moves, slot_of, phases, states, succ, recv, cap)
            self.intersections[tid] = it
        self.order = sorted(self.intersections)
        self._neighbors()

    def _turn(self, a, b):
        try:
            conn = [c for c in self.net.getLane(a).getOutgoing() if c.getToLane().getID() == b]
            d = conn[0].getDirection() if conn else "s"
        except Exception:
            d = "s"
        return TURN_SLOT.get(d, 1)

    def _neighbors(self):
        tl_of_edge = {}
        for tid, it in self.intersections.items():
            for l in it.in_lanes:
                tl_of_edge[self.traci.lane.getEdgeID(l)] = tid
        for tid, it in self.intersections.items():
            nb = set()
            for l in it.out_lanes:
                e = self.traci.lane.getEdgeID(l)
                if e in tl_of_edge and tl_of_edge[e] != tid:
                    nb.add(tl_of_edge[e])
            for other, o in self.intersections.items():
                for l in o.out_lanes:
                    if self.traci.lane.getEdgeID(l) in {self.traci.lane.getEdgeID(x) for x in it.in_lanes} and other != tid:
                        nb.add(other)
            it.neighbors = sorted(nb)

    def adjacency(self, max_neighbors=4):
        n = len(self.order)
        idx = {t: i for i, t in enumerate(self.order)}
        a = np.eye(n, dtype=bool)
        for t in self.order:
            for o in self.intersections[t].neighbors[:max_neighbors]:
                a[idx[t], idx[o]] = a[idx[o], idx[t]] = True
        return a

    def make_shield(self, it, scfg):
        return Shield(it.num_phases, it.successors, scfg["g_min_s"], scfg["g_max_s"], scfg["yellow_s"],
                      scfg["all_red_s"], scfg["spillback_threshold"], scfg["spillback_enabled"],
                      scfg["hard_constraints_enabled"], it.receiving)

    def central_intersection(self):
        deg = {t: len(self.intersections[t].neighbors) for t in self.order}
        return max(deg, key=deg.get)

    def top_intersections(self, k):
        return sorted(self.order, key=lambda t: -len(self.intersections[t].neighbors))[:k]

    def highest_flow_lanes(self, tid, k):
        it = self.intersections[tid]
        flow = [(self.traci.lane.getLastStepVehicleNumber(l), l) for l in it.in_lanes]
        return [l for _, l in sorted(flow, reverse=True)[:k]]

    def detectors(self):
        return [l for t in self.order for l in self.intersections[t].in_lanes + self.intersections[t].out_lanes]
