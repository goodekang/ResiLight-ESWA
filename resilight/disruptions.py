from dataclasses import dataclass, field

import numpy as np

from .config import load_yaml
from .modes import EMG, DEG, INC, SRG, MODE_INDEX

CLASS_MODE = {"D1": INC, "D2": SRG, "D3": DEG, "D4": EMG}


@dataclass
class Event:
    kind: str
    start: float
    end: float
    params: dict = field(default_factory=dict)

    @property
    def mode(self):
        return CLASS_MODE.get(self.kind, MODE_INDEX.get(self.params.get("mode", "NORM"), 0))

    def active(self, t):
        return self.start <= t < self.end


def evaluation_events(names, cfg=None):
    cfg = cfg or load_yaml("disruptions.yaml")["evaluation"]
    lo, hi = cfg["window_s"]
    out = []
    for n in names:
        p = dict(cfg[n])
        out.append(Event(n, lo, hi, p))
    return out


def sample_training_events(rng, network_intersections, od_pool, cfg=None):
    cfg = cfg or load_yaml("disruptions.yaml")["training"]
    events = []
    for n in ("D1", "D2", "D3", "D4"):
        if rng.random() >= cfg["p_event"]:
            continue
        start = rng.uniform(*cfg["onset_s"])
        dur = rng.uniform(*cfg["duration_s"])
        c = cfg[n]
        if n == "D1":
            p = {"lanes": int(rng.integers(c["lanes_closed"][0], c["lanes_closed"][1] + 1)),
                 "site": network_intersections[int(rng.integers(len(network_intersections)))]}
        elif n == "D2":
            k = int(rng.integers(c["od_relations"][0], c["od_relations"][1] + 1))
            sel = rng.choice(len(od_pool), size=min(k, len(od_pool)), replace=False)
            p = {"od": [od_pool[i] for i in sel], "factor": float(rng.uniform(*c["factor"]))}
        elif n == "D3":
            p = {"fraction": float(rng.uniform(*c["fraction"]))}
        else:
            p = {"count": int(rng.integers(c["count"][0], c["count"][1] + 1)), "random_route": True}
        events.append(Event(n, start, start + dur, p))
    return events


def active_modes(events, t):
    return [e.mode for e in events if e.active(t)]


def scripted_label(events, t):
    from .modes import priority_label
    act = active_modes(events, t)
    return priority_label(act) if act else 0


def overlap_share(rng, n=20000, cfg=None):
    cfg = cfg or load_yaml("disruptions.yaml")["training"]
    k = (rng.random((n, 4)) < cfg["p_event"]).sum(1)
    return float((k >= 2).mean())


class Injector:
    def __init__(self, traci, net, rng):
        self.traci, self.net, self.rng = traci, net, rng
        self.closed = {}
        self.frozen = {}
        self.base_scale = None

    def apply(self, ev, t):
        fn = getattr(self, "_" + ev.kind.lower(), None)
        if fn is None:
            fn = getattr(self, "_" + ev.params.get("type", ""), None)
        if fn and ev.active(t) and not ev.params.get("_on"):
            fn(ev)
            ev.params["_on"] = True
        elif fn and not ev.active(t) and ev.params.get("_on"):
            self.release(ev)
            ev.params["_on"] = False

    def _d1(self, ev):
        site = ev.params.get("site") or self.net.central_intersection()
        lanes = self.net.highest_flow_lanes(site, ev.params.get("lanes", ev.params.get("lanes_closed", 2)))
        for l in lanes:
            self.traci.lane.setDisallowed(l, ["all"])
        self.closed[id(ev)] = lanes

    def _d2(self, ev):
        od = ev.params.get("od") or self.net.heaviest_od(ev.params.get("od_relations", 3))
        ev.params["od"] = od
        self.net.scale_demand(od, ev.params["factor"] if "factor" in ev.params else 1.5)

    def _d3(self, ev):
        frac = ev.params.get("fraction", 0.3)
        dets = self.net.detectors()
        sel = self.rng.choice(len(dets), size=int(round(frac * len(dets))), replace=False)
        self.frozen[id(ev)] = [dets[i] for i in sel]
        self.net.freeze_detectors(self.frozen[id(ev)])

    def _d4(self, ev):
        n = ev.params.get("count", 3)
        gap = ev.params.get("headway_s", 480)
        self.net.insert_emergency(n, ev.start, gap, ev.params.get("random_route", False), self.rng)

    def _detector_noise(self, ev):
        self.net.set_detector_distortion(ev.params["fraction"], "noise", ev.params["sigma"], self.rng)

    def _detector_bias(self, ev):
        self.net.set_detector_distortion(ev.params["fraction"], "bias", ev.params["bias"], self.rng)

    def _detector_dropout(self, ev):
        self.net.set_detector_distortion(ev.params["fraction"], "dropout", ev.params["duty_cycle"], self.rng)

    def _lane_closure(self, ev):
        site = self.net.central_intersection() if ev.params.get("intersections", 1) == 1 else None
        sites = [site] if site else self.net.top_intersections(ev.params["intersections"])
        lanes = []
        for s in sites:
            lanes += self.net.highest_flow_lanes(s, ev.params["lanes_closed"])
        for l in lanes:
            self.traci.lane.setDisallowed(l, ["all"])
        self.closed[id(ev)] = lanes

    def _demand_surge(self, ev):
        od = self.net.heaviest_od(ev.params["od_relations"])
        self.net.scale_demand(od, ev.params["factor"])

    def release(self, ev):
        for l in self.closed.pop(id(ev), []):
            self.traci.lane.setAllowed(l, [])
        if id(ev) in self.frozen:
            self.net.unfreeze_detectors(self.frozen.pop(id(ev)))
        self.net.clear_detector_distortion()
