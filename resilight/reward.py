import numpy as np

from .modes import WEIGHTS


def mode_reward(mode, queue_norm, pressure_norm, ev_delay_norm, spill_ind, table=None):
    w = (WEIGHTS if table is None else table)[int(mode)]
    terms = np.array([queue_norm, abs(pressure_norm), ev_delay_norm, spill_ind], dtype=np.float32)
    return -float(np.dot(w, np.maximum(terms, 0.0)))


def noisy_label(mode, confusion, rng):
    return int(rng.choice(len(confusion), p=confusion[int(mode)]))
