import numpy as np

from .config import load_modes

MODES = ["NORM", "INC", "SRG", "DEG", "EMG"]
MODE_INDEX = {m: i for i, m in enumerate(MODES)}
NORM, INC, SRG, DEG, EMG = range(5)
_CFG = load_modes()
WEIGHTS = np.array(
    [[_CFG["weights"][m][k] for k in ("w_q", "w_p", "w_e", "w_s")] for m in MODES],
    dtype=np.float32,
)
COMPOUND_PRIORITY = [MODE_INDEX[m] for m in _CFG["compound_priority"]]


def mode_weights(mode, table=None):
    t = WEIGHTS if table is None else table
    return t[int(mode)]


def priority_label(active):
    for m in COMPOUND_PRIORITY:
        if m in active:
            return m
    return NORM
