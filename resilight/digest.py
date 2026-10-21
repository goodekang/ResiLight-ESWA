import json

import numpy as np


def _sign(x, eps=1e-6):
    return 1 if x > eps else (-1 if x < -eps else 0)


class DigestBuilder:
    def __init__(self, top_k=5):
        self.top_k = top_k
        self.prev = None
        self.hist = []

    def build(self, stats, inter, health, flagged, current_mode):
        net = {
            "mean_speed_mps": round(float(stats["mean_speed"]), 2),
            "total_queue_veh": round(float(stats["total_queue"]), 1),
            "throughput_veh_per_window": int(stats["throughput"]),
        }
        self.hist.append((net["mean_speed_mps"], net["total_queue_veh"], net["throughput_veh_per_window"]))
        self.hist = self.hist[-3:]
        if len(self.hist) >= 3:
            a, b, c = self.hist[-3:]
            net["trend_sign"] = {"mean_speed": _sign(c[0] - b[0]) , "total_queue": _sign(c[1] - b[1]),
                                 "throughput": _sign(c[2] - b[2])}
        elif len(self.hist) == 2:
            a, b = self.hist
            net["trend_sign"] = {"mean_speed": _sign(b[0] - a[0]), "total_queue": _sign(b[1] - a[1]),
                                 "throughput": _sign(b[2] - a[2])}
        else:
            net["trend_sign"] = {"mean_speed": 0, "total_queue": 0, "throughput": 0}
        rows = {}
        for sid, d in inter.items():
            prev = (self.prev or {}).get(sid, d)
            rows[sid] = {"id": sid, "queue_veh": round(float(d["queue"]), 1), "occupancy": round(float(d["occ"]), 3),
                         "inflow_veh_per_min": round(float(d["inflow"]), 1),
                         "queue_change_veh": round(float(d["queue"] - prev["queue"]), 1),
                         "occupancy_change": round(float(d["occ"] - prev["occ"]), 3)}
        self.prev = {k: dict(v) for k, v in inter.items()}
        by_q = sorted(rows.values(), key=lambda r: -r["queue_veh"])[: self.top_k]
        by_o = sorted(rows.values(), key=lambda r: -r["occupancy"])[: self.top_k]
        n_fail, frac = health
        return {
            "network": net,
            "worst_by_queue": by_q,
            "worst_by_occupancy": by_o,
            "detector_health": {"failed_detectors": int(n_fail), "fraction": round(float(frac), 3)},
            "flagged_vehicles": flagged,
            "current_mode": current_mode,
        }


def to_json(digest):
    return json.dumps(digest, separators=(",", ":"))


def approx_tokens(digest):
    return int(len(to_json(digest)) / 3.6)


def features(digest, capacity=None, normalize=True, rates_mean=None):
    n = digest["network"]
    q = n["total_queue_veh"]
    sp = n["mean_speed_mps"]
    th = n["throughput_veh_per_window"]
    if normalize and capacity:
        q = q / capacity["network"]
        th = th / max(rates_mean or th, 1.0)
    f = [sp, q, th, *n["trend_sign"].values(),
         digest["detector_health"]["fraction"], len(digest["flagged_vehicles"])]
    for key in ("worst_by_queue", "worst_by_occupancy"):
        rows = digest[key]
        for i in range(5):
            r = rows[i] if i < len(rows) else None
            if r is None:
                f += [0.0, 0.0, 0.0, 0.0]
                continue
            cq = capacity["per_inter"].get(r["id"], 1.0) if (normalize and capacity) else 1.0
            f += [r["queue_veh"] / cq, r["occupancy"], r["inflow_veh_per_min"] / cq, r["queue_change_veh"] / cq]
    return np.asarray(f, dtype=np.float32)


def set_entries(digest, capacity=None):
    rows = digest["worst_by_queue"] + digest["worst_by_occupancy"]
    out = []
    for r in rows:
        cq = capacity["per_inter"].get(r["id"], 1.0) if capacity else 1.0
        out.append([r["queue_veh"] / cq, r["occupancy"], r["inflow_veh_per_min"] / cq,
                    r["queue_change_veh"] / cq, r["occupancy_change"]])
    while len(out) < 10:
        out.append([0.0] * 5)
    return np.asarray(out, dtype=np.float32)
