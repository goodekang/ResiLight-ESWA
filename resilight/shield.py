from dataclasses import dataclass, field

import numpy as np


@dataclass
class ShieldState:
    phase: int = 0
    elapsed: float = 0.0
    transition_left: float = 0.0
    target: int = -1


@dataclass
class Shield:
    num_phases: int
    successors: list
    g_min: float = 10.0
    g_max: float = 60.0
    yellow: float = 3.0
    all_red: float = 2.0
    rho_sb: float = 0.9
    spillback: bool = True
    hard: bool = True
    receiving: list = field(default_factory=list)

    def hard_mask(self, st):
        m = np.zeros(self.num_phases, dtype=bool)
        if st.transition_left > 0:
            m[st.target if st.target >= 0 else st.phase] = True
            return m
        if not self.hard:
            m[:] = True
            return m
        if st.elapsed < self.g_min:
            m[st.phase] = True
            return m
        for p in self.successors[st.phase]:
            m[p] = True
        m[st.phase] = st.elapsed < self.g_max
        return m

    def spillback_mask(self, occupancy):
        m = np.ones(self.num_phases, dtype=bool)
        if not self.spillback:
            return m
        for p, lanes in enumerate(self.receiving):
            if lanes and max(occupancy[l] for l in lanes) > self.rho_sb:
                m[p] = False
        return m

    def mask(self, st, occupancy):
        hard = self.hard_mask(st)
        sb = hard & self.spillback_mask(occupancy)
        if sb.any():
            return sb
        if hard.any():
            return hard
        m = np.zeros(self.num_phases, dtype=bool)
        m[st.phase] = True
        return m

    def select(self, q, st, occupancy):
        m = self.mask(st, occupancy)
        return int(np.where(m, q, -np.inf).argmax()), m

    def advance(self, st, action, dt):
        if st.transition_left > 0:
            st.transition_left -= dt
            st.elapsed = 0.0
            if st.transition_left <= 0:
                st.phase, st.target = st.target, -1
            return st
        if action != st.phase:
            st.target = action
            st.transition_left = self.yellow + self.all_red
            st.elapsed = 0.0
        else:
            st.elapsed += dt
        return st

    def max_pressure(self, pressure, st, occupancy):
        m = self.mask(st, occupancy)
        return int(np.where(m, pressure, -np.inf).argmax()), m


def violations(prev, cur, shield):
    n = 0
    if prev.phase == cur.phase and cur.elapsed > shield.g_max:
        n += 1
    if prev.phase != cur.phase and prev.elapsed < shield.g_min and prev.transition_left <= 0:
        n += 1
    if prev.phase != cur.phase and cur.phase not in shield.successors[prev.phase] and prev.transition_left <= 0:
        n += 1
    return n
