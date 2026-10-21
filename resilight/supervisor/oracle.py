from ..disruptions import active_modes
from ..modes import NORM, priority_label


class OracleSupervisor:
    def __init__(self, events):
        self.events = events

    def mode_at(self, t):
        act = active_modes(self.events, t)
        return priority_label(act) if act else NORM
