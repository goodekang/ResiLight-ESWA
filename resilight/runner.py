import numpy as np

from .digest import DigestBuilder
from .modes import DEG, EMG, NORM
from .preemption import apply_bonus, arrival_times, preempted_signals


class MaxPressureController:
    def __init__(self, env):
        self.env = env

    def act(self, pack, mode, ctx):
        acts = []
        for i, tid in enumerate(self.env.net.order):
            p = np.where(pack["admissible"][i], pack["pressure"][i], -np.inf)
            acts.append(int(p.argmax()))
        return np.array(acts)


class FixedTimeController:
    def __init__(self, env, plans):
        self.env, self.plans = env, plans

    def act(self, pack, mode, ctx):
        acts = []
        for i, tid in enumerate(self.env.net.order):
            green = self.plans[tid]
            cyc = sum(green)
            x = ctx["t"] % cyc
            acc = 0.0
            for p, g in enumerate(green):
                acc += g
                if x < acc:
                    acts.append(p)
                    break
        return np.array(acts)


def webster_plan(flows, lost_per_phase=5.0, c_min=40.0, c_max=120.0):
    flows = np.maximum(np.asarray(flows, float), 1e-3)
    y = np.minimum(flows / (1800.0), 0.9)
    Y = min(y.sum(), 0.9)
    L = lost_per_phase * len(flows)
    c = float(np.clip((1.5 * L + 5) / (1 - Y), c_min, c_max))
    g = (c - L) * y / y.sum()
    return np.maximum(g + lost_per_phase, 10.0)


class ResiLightController:
    def __init__(self, env, agent, cfg, preemption=True, imputation=True, mode_conditioning=True, message_drop=0.0):
        self.env, self.agent, self.cfg = env, agent, cfg
        self.preemption, self.imputation = preemption, imputation
        self.mode_conditioning, self.message_drop = mode_conditioning, message_drop
        self.edge_tls = {}
        self.explore = False

    def _edge_map(self):
        if self.edge_tls:
            return
        for tid, it in self.env.net.intersections.items():
            for l in it.in_lanes:
                self.edge_tls[self.env.traci.lane.getEdgeID(l)] = tid

    def act(self, pack, mode, ctx):
        self._edge_map()
        m = int(mode) if self.mode_conditioning else NORM
        q = self.agent.q_values(pack["obs"], pack["adj"], m, self.message_drop)
        q = np.where(pack["admissible"], q[:, : pack["admissible"].shape[1]], -np.inf)
        if self.preemption and int(mode) == EMG:
            q = self._preempt(q, pack, ctx.get("scope", "all"))
        acts = q.argmax(1)
        if self.explore:
            eps = self.agent.epsilon()
            for i in range(len(acts)):
                if self.agent.rng.random() < eps:
                    acts[i] = self.agent.rng.choice(np.flatnonzero(pack["admissible"][i]))
        return acts

    def _preempt(self, q, pack, scope):
        env, pcfg = self.env, self.cfg["preemption"]
        idx = {t: i for i, t in enumerate(env.net.order)}
        for ev in env.flagged():
            sig, lens, speeds, ff = [], [], [], []
            for e in ev["remaining_route"]:
                tid = self.edge_tls.get(e)
                lane = e + "_0"
                lens.append(env.traci.lane.getLength(lane))
                speeds.append(env.traci.lane.getLastStepMeanSpeed(lane))
                ff.append(env.traci.lane.getMaxSpeed(lane))
                sig.append(tid)
            arr = arrival_times(lens, speeds, ff)
            pairs = [(s, a) for s, a in zip(sig, arr) if s]
            aff = None if scope == "all" else set(scope)
            sel = preempted_signals([p[0] for p in pairs], [p[1] for p in pairs], pcfg["horizon_s"], aff)
            for tid in sel:
                i = idx[tid]
                it = env.net.intersections[tid]
                nxt = ev["remaining_route"][0]
                serving = [p for p, mv in enumerate(it.movements)
                           if any(env.traci.lane.getEdgeID(it.in_lanes[ia]) == nxt for ia, ob in mv)]
                finite = np.isfinite(q[i])
                q[i] = np.where(finite, apply_bonus(np.where(finite, q[i], 0.0), finite, serving, pcfg["bonus_std_multiplier"]), -np.inf)
        return q


def run_episode(env, controller, seed, events, supervisor=None, oracle=None, fixed_mode=None, collect=None,
                digest_top_k=5, supervisory_cycle_s=60.0, train_agent=None, labels_noise=None, fallback_controller=None):
    pack = env.reset(seed, [] if callable(events) else events)
    if callable(events):
        events = events(env)
        env.events = events
    builder = DigestBuilder(digest_top_k)
    mode, scope, fallback, dt = NORM, "all", False, env.dt
    log = []
    done = False
    while not done:
        t = env.t
        if t % supervisory_cycle_s == 0 and t > 0 or t == 0:
            if oracle is not None:
                mode = oracle.mode_at(t)
            elif supervisor is not None:
                digest = builder.build(env.stats(), env.intersection_stats(), env.health(), env.flagged(), int(mode))
                dec = supervisor.step(digest)
                mode, scope, fallback = dec.mode, dec.scope, dec.fallback
                true = None
                from .disruptions import scripted_label
                true = scripted_label(events, t)
                log.append({"t": t, "true": true, "raw": dec.raw_mode, "mode": dec.mode, "conf": dec.confidence,
                            "valid": dec.valid, "latency": dec.latency, "fallback": dec.fallback, "digest": digest})
            elif fixed_mode is not None:
                mode = fixed_mode
        if train_agent is not None:
            from .disruptions import scripted_label
            mode = scripted_label(events, t)
            if labels_noise is not None:
                from .reward import noisy_label
                mode = noisy_label(mode, labels_noise, env.rng)
        ctl = fallback_controller if (fallback and fallback_controller is not None) else controller
        impute = mode == DEG
        pack = env.observe(impute_mode=impute)
        actions = ctl.act(pack, mode, {"t": t, "scope": scope})
        rewards, done, info = env.step(actions, mode)
        if train_agent is not None:
            nxt = env.observe(impute_mode=info["label"] == DEG)
            train_agent.observe(_transition(pack, nxt, actions, rewards, mode, info["label"], done))
            train_agent.update()
        if collect is not None:
            collect(env, t, mode, info)
        pack = nxt if train_agent is not None else pack
    return env.metrics, log


def _transition(pack, nxt, actions, rewards, mode, next_mode, done):
    return {"obs": pack["obs"], "next_obs": nxt["obs"], "adj": pack["adj"], "action": np.asarray(actions),
            "reward": np.asarray(rewards), "mode": int(mode), "next_mode": int(next_mode), "done": float(done),
            "admissible": pack["admissible"], "next_admissible": nxt["admissible"], "pressure": pack["pressure"]}
