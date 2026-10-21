from ..modes import DEG, EMG, INC, NORM, SRG


class RuleDetector:
    def __init__(self, localization=False, deg_threshold=0.15, surge_growth=0.25,
                 inc_occ=0.85, inc_occ_growth=0.2, local_share=0.5):
        self.localization = localization
        self.deg_threshold = deg_threshold
        self.surge_growth = surge_growth
        self.inc_occ = inc_occ
        self.inc_occ_growth = inc_occ_growth
        self.local_share = local_share
        self.prev_queue = None
        self.prev2_queue = None

    def reset(self):
        self.prev_queue = self.prev2_queue = None

    def classify(self, digest):
        if any(v.get("active", True) for v in digest["flagged_vehicles"]):
            return EMG
        if digest["detector_health"]["fraction"] > self.deg_threshold:
            return DEG
        q = digest["network"]["total_queue_veh"]
        growth_2 = None
        if self.prev_queue is not None and self.prev2_queue is not None:
            growth_2 = (q - self.prev2_queue) / max(self.prev2_queue, 1.0)
            consecutive = q > self.prev_queue > self.prev2_queue
        else:
            consecutive = False
        self.prev2_queue, self.prev_queue = self.prev_queue, q
        if self.localization:
            if growth_2 is not None and growth_2 > self.surge_growth and consecutive:
                total = sum(max(r["queue_change_veh"], 0.0) for r in digest["worst_by_queue"]) or 1.0
                top2 = sum(sorted((max(r["queue_change_veh"], 0.0) for r in digest["worst_by_queue"]), reverse=True)[:2])
                return INC if top2 / total > self.local_share else SRG
            w = digest["worst_by_occupancy"][0] if digest["worst_by_occupancy"] else None
            if w and w["occupancy"] > self.inc_occ and w["occupancy_change"] > self.inc_occ_growth:
                return INC
            return NORM
        if growth_2 is not None and growth_2 > self.surge_growth and consecutive:
            return SRG
        w = digest["worst_by_occupancy"][0] if digest["worst_by_occupancy"] else None
        if w and w["occupancy"] > self.inc_occ and w["occupancy_change"] > self.inc_occ_growth:
            return INC
        return NORM
