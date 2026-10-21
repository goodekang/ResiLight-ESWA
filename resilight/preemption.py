import numpy as np


def arrival_times(route_lengths, mean_speeds, free_flow_speeds):
    t = 0.0
    out = []
    for length, v, vf in zip(route_lengths, mean_speeds, free_flow_speeds):
        v_eff = max(v, 0.1)
        t += max(length / v_eff, length / vf)
        out.append(t)
    return out


def preempted_signals(route_signals, arrivals, horizon_s=90.0, affected=None):
    sel = {}
    for sig, t in zip(route_signals, arrivals):
        if t < horizon_s and (affected is None or sig in affected):
            sel[sig] = min(t, sel.get(sig, t))
    return sel


def apply_bonus(q, admissible, serving, multiplier=10.0):
    q = np.asarray(q, dtype=np.float32).copy()
    adm = np.asarray(admissible, dtype=bool)
    std = float(q[adm].std()) if adm.sum() > 1 else 1.0
    bonus = multiplier * max(std, 1e-3)
    for p in serving:
        q[p] += bonus
    return q
