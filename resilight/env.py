import os
from collections import defaultdict

import numpy as np

from .disruptions import Injector, scripted_label
from .imputation import Imputer
from .metrics import EpisodeMetrics, collect_pending
from .network import SumoNetwork
from .pressure import lane_capacity, phase_pressure
from .reward import mode_reward
from .shield import ShieldState, violations


class TrafficEnv:
    def __init__(self, cfg, net_cfg, gui=False, label="resilight"):
        import sumolib
        import traci
        self.traci, self.sumolib = traci, sumolib
        self.cfg, self.net_cfg = cfg, net_cfg
        self.dt = cfg["control"]["decision_interval_s"]
        self.episode_s = net_cfg.get("episode_s", cfg["control"]["episode_s"])
        self.gui, self.label = gui, label
        self.slots = cfg["model"]["movement_slots"]
        self.rng = np.random.default_rng(0)
        self.net = None
        self.events = []
        self.frozen = set()
        self.last_read = {}
        self.distort = None
        self.od_rates = defaultdict(float)
        self.extra_od = {}
        self.ev_ids = set()
        self.ev_start = {}
        self.metrics = None
        self.t = 0.0

    def reset(self, seed, events=(), preload_net=True):
        if self.net is not None:
            self.traci.close()
        binary = "sumo-gui" if self.gui else "sumo"
        cmd = [binary, "-c", self.net_cfg["sumocfg"], "--seed", str(seed), "--no-step-log", "true",
               "--no-warnings", "true", "--time-to-teleport", "-1", "--step-length", "1"]
        self.traci.start(cmd, label=self.label)
        sb = self.sumolib.net.readNet(self._net_file(), withPrograms=True)
        self.net = SumoNetwork(self.traci, sb, self.cfg["physics"]["spacing_m"], self.slots)
        self.rng = np.random.default_rng(seed)
        self.events = list(events)
        self.injector = Injector(self.traci, self.net, self.rng)
        self.frozen, self.last_read, self.distort = set(), {}, None
        self.extra_od, self.ev_ids, self.ev_start = {}, set(), {}
        self.metrics = EpisodeMetrics(episode_s=self.episode_s)
        self.depart = {}
        self.t = 0.0
        self.shields = {t: self.net.make_shield(self.net.intersections[t], self.cfg["shield"]) for t in self.net.order}
        self.sstate = {t: ShieldState() for t in self.net.order}
        self.imputers = {t: Imputer(len(self.net.intersections[t].in_lanes) * 3, self.cfg["imputation"]["ewm_alpha"],
                                    self.cfg["imputation"]["health_persistence_s"]) for t in self.net.order}
        self.adj = self.net.adjacency(self.cfg["model"]["max_neighbors"])
        self._load_od()
        self.violation_prev = {t: ShieldState() for t in self.net.order}
        for t in self.net.order:
            self._apply_phase(t, 0)
        return self.observe()

    def _net_file(self):
        cfg = self.sumolib.xml
        d = os.path.dirname(self.net_cfg["sumocfg"])
        for f in os.listdir(d):
            if f.endswith(".net.xml"):
                return os.path.join(d, f)
        raise FileNotFoundError(d)

    def _load_od(self):
        self.od_rates.clear()
        for r in self.traci.route.getIDList():
            edges = self.traci.route.getEdges(r)
            if edges:
                self.od_rates[(edges[0], edges[-1])] += 1.0

    def heaviest_od(self, k):
        return [od for od, _ in sorted(self.od_rates.items(), key=lambda kv: -kv[1])[:k]]

    def scale_demand(self, od, factor):
        for o in od:
            self.extra_od[tuple(o)] = factor - 1.0

    def detectors(self):
        return self.net.detectors()

    def freeze_detectors(self, lanes):
        self.frozen |= set(lanes)

    def unfreeze_detectors(self, lanes):
        self.frozen -= set(lanes)

    def set_detector_distortion(self, fraction, kind, param, rng):
        dets = self.detectors()
        sel = rng.choice(len(dets), size=int(round(fraction * len(dets))), replace=False)
        self.distort = {"lanes": {dets[i] for i in sel}, "kind": kind, "param": param}

    def clear_detector_distortion(self):
        self.distort = None

    def insert_emergency(self, n, start, gap, random_route, rng):
        routes = list(self.traci.route.getIDList())
        for k in range(n):
            vid = f"ev_{k}"
            self.ev_start[vid] = start + k * gap
        self._pending_ev = [(vid, t0, routes[int(rng.integers(len(routes)))]) for vid, t0 in self.ev_start.items()]

    def _spawn_ev(self):
        for vid, t0, route in getattr(self, "_pending_ev", []):
            if vid not in self.ev_ids and self.t >= t0:
                self.traci.vehicle.add(vid, route, typeID="emergency" if "emergency" in self.traci.vehicletype.getIDList() else "DEFAULT_VEHTYPE")
                self.traci.vehicle.setParameter(vid, "flagged", "1")
                self.ev_ids.add(vid)

    def _extra_demand(self):
        for od, extra in self.extra_od.items():
            base = self.od_rates.get(od, 0.0)
            lam = base * extra * self.dt / self.episode_s
            for _ in range(int(self.rng.poisson(lam))):
                vid = f"x_{od[0]}_{od[1]}_{self.t}_{self.rng.integers(1e9)}"
                try:
                    rid = f"r_{vid}"
                    self.traci.route.add(rid, [od[0], od[1]])
                    self.traci.vehicle.add(vid, rid)
                except Exception:
                    pass

    def read(self, lane):
        tr = self.traci.lane
        occ = tr.getLastStepOccupancy(lane) / 100.0
        q = float(tr.getLastStepHaltingNumber(lane))
        v = tr.getLastStepMeanSpeed(lane) / max(tr.getMaxSpeed(lane), 0.1)
        vals = np.array([q, occ, v], dtype=np.float32)
        if lane in self.frozen:
            if lane not in self.last_read:
                self.last_read[lane] = vals
            return self.last_read[lane], True
        self.last_read[lane] = vals
        d = self.distort
        if d and lane in d["lanes"]:
            if d["kind"] == "noise":
                vals = vals * (1 + self.rng.normal(0, d["param"], 3).astype(np.float32))
            elif d["kind"] == "bias":
                vals = vals * (1 + d["param"])
            elif d["kind"] == "dropout" and (int(self.t) // 60) % 2 == 0 and self.rng.random() > d["param"]:
                return vals * 0, False
        return vals, True

    def observe(self, impute_mode=False, message_drop=0.0):
        N = len(self.net.order)
        S = self.slots
        P = max(it.num_phases for it in self.net.intersections.values())
        o = {"slots": np.zeros((N, S, 4), np.float32), "mask": np.zeros((N, S), np.float32),
             "current": np.zeros((N, S), np.float32), "elapsed": np.zeros((N, 1), np.float32),
             "phase_matrix": np.zeros((N, P, S), np.float32), "phase_valid": np.zeros((N, P), np.float32)}
        adm = np.zeros((N, P), bool)
        press = np.zeros((N, P), np.float32)
        occ_all = {}
        for i, tid in enumerate(self.net.order):
            it = self.net.intersections[tid]
            lanes = it.in_lanes + it.out_lanes
            raw = np.stack([self.read(l)[0] for l in lanes])
            avail = np.array([self.read(l)[1] for l in lanes], dtype=bool)
            imp = self.imputers[tid]
            flat = raw.reshape(-1)
            imp.update(flat, self.dt)
            if impute_mode:
                flat, flag = imp.impute(flat, np.repeat(~(imp.failed().reshape(-1, 3).any(1) | ~avail), 3))
                raw = flat.reshape(-1, 3)
            occ = raw[:, 1]
            occ_all[tid] = occ
            nin = len(it.in_lanes)
            cap = it.capacity
            q = raw[:, 0]
            for li, (idx, ia, ob, a, b) in enumerate(it.links):
                s = it.slot_of_link[li]
                o["slots"][i, s] = [q[ia] / cap[ia], occ[ia], raw[ia, 2], 1.0]
                o["mask"][i, s] = 1.0
            st = self.sstate[tid]
            for p, mv in enumerate(it.movements):
                o["phase_valid"][i, p] = 1.0
                for li, (idx, ia, ob, a, b) in enumerate(it.links):
                    if (ia, ob) in mv:
                        o["phase_matrix"][i, p, it.slot_of_link[li]] = 1.0
                press[i, p] = phase_pressure(q, cap, [mv])[0]
            for li, (idx, ia, ob, a, b) in enumerate(it.links):
                if (ia, ob) in it.movements[st.phase]:
                    o["current"][i, it.slot_of_link[li]] = 1.0
            o["elapsed"][i, 0] = st.elapsed / self.cfg["shield"]["g_max_s"]
            m = self.shields[tid].mask(st, np.concatenate([occ[:nin], occ[nin:]]))
            adm[i, :it.num_phases] = m
        return {"obs": o, "adj": self.adj, "admissible": adm, "pressure": press}

    def apply_actions(self, actions):
        for i, tid in enumerate(self.net.order):
            prev = ShieldState(**vars(self.sstate[tid]))
            st = self.shields[tid].advance(self.sstate[tid], int(actions[i]), self.dt)
            self.metrics.violations += violations(prev, st, self.shields[tid]) if self.shields[tid].hard else 0
            self._apply_phase(tid, st.phase if st.transition_left <= 0 else -1, st)

    def _apply_phase(self, tid, phase, st=None):
        it = self.net.intersections[tid]
        n = len(self.traci.trafficlight.getRedYellowGreenState(tid))
        if phase >= 0:
            self.traci.trafficlight.setRedYellowGreenState(tid, it.phase_states[phase])
            return
        cur = it.phase_states[st.phase]
        if st.transition_left > self.shields[tid].all_red:
            self.traci.trafficlight.setRedYellowGreenState(tid, "".join("y" if c in "Gg" else c for c in cur))
        else:
            self.traci.trafficlight.setRedYellowGreenState(tid, "r" * n)

    def step(self, actions, modes=None):
        self.apply_actions(actions)
        rewards = np.zeros(len(self.net.order), np.float32)
        for _ in range(int(self.dt)):
            for ev in self.events:
                self.injector.apply(ev, self.t)
            self._spawn_ev()
            if int(self.t) % 5 == 0:
                self._extra_demand()
            self.traci.simulationStep()
            self.t += 1.0
            for vid in self.traci.simulation.getDepartedIDList():
                self.depart[vid] = self.t
            for vid in self.traci.vehicle.getIDList():
                self.metrics.waiting[vid] = self.traci.vehicle.getAccumulatedWaitingTime(vid)
            for vid in self.traci.simulation.getArrivedIDList():
                self._arrived(vid)
            self.metrics.co2_mg += sum(self.traci.vehicle.getCO2Emission(v) for v in self.traci.vehicle.getIDList())
        self.metrics.inserted += self.traci.simulation.getDepartedNumber()
        label = scripted_label(self.events, self.t)
        for i, tid in enumerate(self.net.order):
            rewards[i] = self._reward(tid, label if modes is None else modes)
        done = self.t >= self.episode_s
        if done:
            self.metrics.pending = collect_pending(self.traci, self.t, self._ff_remaining)
        return rewards, done, {"label": label}

    def _arrived(self, vid):
        t0 = self.depart.pop(vid, None)
        if t0 is None:
            return
        tt = self.t - t0
        if vid in self.ev_ids:
            self.metrics.ev_times.append(tt)
        else:
            self.metrics.travel_times.append(tt)

    def _reward(self, tid, mode):
        it = self.net.intersections[tid]
        nin = len(it.in_lanes)
        q = np.array([self.traci.lane.getLastStepHaltingNumber(l) for l in it.in_lanes], np.float32)
        qn = float((q / it.capacity[:nin]).sum())
        st = self.sstate[tid]
        pr = float(phase_pressure(np.concatenate([q, np.array([self.traci.lane.getLastStepHaltingNumber(l) for l in it.out_lanes], np.float32)]),
                                  it.capacity, [it.movements[st.phase]])[0])
        ev = 0.0
        for l in it.in_lanes:
            for v in self.traci.lane.getLastStepVehicleIDs(l):
                if v in self.ev_ids:
                    ev += self.traci.vehicle.getWaitingTime(v) / 60.0
        occ = [self.traci.lane.getLastStepOccupancy(it.out_lanes[ob]) / 100.0 for ob in it.receiving[st.phase]]
        sp = float(bool(occ) and max(occ) > self.cfg["shield"]["spillback_threshold"])
        return mode_reward(int(mode) if np.isscalar(mode) else int(mode[0]), qn, pr, ev, sp)

    def _ff_remaining(self, vid):
        try:
            route = self.traci.vehicle.getRoute(vid)
            i = self.traci.vehicle.getRouteIndex(vid)
            tot = 0.0
            for e in route[max(i, 0):]:
                tot += self.traci.lane.getLength(e + "_0") / max(self.traci.lane.getMaxSpeed(e + "_0"), 0.1)
            return tot
        except Exception:
            return 0.0

    def stats(self):
        tr = self.traci
        ids = tr.vehicle.getIDList()
        speeds = [tr.vehicle.getSpeed(v) for v in ids] or [0.0]
        total_q = sum(tr.lane.getLastStepHaltingNumber(l) for l in self.detectors())
        return {"mean_speed": float(np.mean(speeds)), "total_queue": float(total_q), "throughput": len(self.metrics.travel_times)}

    def intersection_stats(self, window_s=60.0):
        out = {}
        for tid in self.net.order:
            it = self.net.intersections[tid]
            q = sum(self.traci.lane.getLastStepHaltingNumber(l) for l in it.in_lanes)
            occ = max(self.traci.lane.getLastStepOccupancy(l) for l in it.in_lanes) / 100.0
            inflow = sum(self.traci.lane.getLastStepVehicleNumber(l) for l in it.in_lanes)
            out[tid] = {"queue": float(q), "occ": float(occ), "inflow": float(inflow)}
        return out

    def health(self):
        n = f = 0
        for imp in self.imputers.values():
            fa = imp.failed()
            n += int(fa.sum()); f += fa.size
        return n, (n / f if f else 0.0)

    def flagged(self):
        out = []
        for v in self.ev_ids:
            if v in self.traci.vehicle.getIDList():
                out.append({"id": v, "type": "emergency", "edge": self.traci.vehicle.getRoadID(v),
                            "remaining_route": list(self.traci.vehicle.getRoute(v)[self.traci.vehicle.getRouteIndex(v):])[:8],
                            "active": True})
        return out

    def close(self):
        if self.net is not None:
            self.traci.close()
            self.net = None
