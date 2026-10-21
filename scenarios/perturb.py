"""Write the disruption schedules used in the experiments as JSON, one file per seed.

    python scenarios/perturb.py --network ingolstadt21 --suite eval
    python scenarios/perturb.py --network ingolstadt21 --suite ood
    python scenarios/perturb.py --network ingolstadt21 --suite compound
    python scenarios/perturb.py --network grid4x4 --suite train --seeds 1 2 3 4 5 6 7 8 9 10

The schedules are read back with load_events(); the run itself (lane closures, detector freezing, ...) is done by
resilight.disruptions.Injector. For the surge classes, --surge-routes additionally writes a copy of the route
file in which the selected OD relations are thinned/duplicated inside the window.
"""
import argparse
import json
import sys
import zlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from resilight.config import load_seeds, load_yaml  # noqa: E402
from resilight.disruptions import Event, evaluation_events, sample_training_events  # noqa: E402


def od_pool(rou_file):
    pool = {}
    for el in ET.parse(rou_file).getroot():
        if el.tag == "trip":
            od = (el.get("from"), el.get("to"))
        elif el.tag == "route":
            e = el.get("edges", "").split()
            od = (e[0], e[-1]) if e else None
        elif el.tag == "vehicle":
            r = el.find("route")
            e = r.get("edges", "").split() if r is not None else []
            od = (e[0], e[-1]) if e else None
        else:
            od = None
        if od and None not in od:
            pool[od] = pool.get(od, 0) + 1
    return [list(k) for k, _ in sorted(pool.items(), key=lambda kv: -kv[1])]


def tls_ids(net_file):
    return sorted(t.get("id") for t in ET.parse(net_file).getroot().iter("tlLogic"))


def dump_events(events):
    return [{"kind": e.kind, "start": e.start, "end": e.end,
             "params": {k: v for k, v in e.params.items() if not k.startswith("_")}} for e in events]


def load_events(path):
    return [Event(d["kind"], d["start"], d["end"], d["params"]) for d in json.load(open(path))["events"]]


def ood_events(name, cfg):
    spec = dict(cfg["ood"][name])
    lo, hi = cfg["evaluation"]["window_s"]
    if "duration_s" in spec:
        hi = lo + spec.pop("duration_s")
    return [Event(spec["type"], lo, hi, spec)]


def suite_events(suite, name, seed, cfg, ids, pool):
    if suite == "eval":
        return evaluation_events([name], cfg["evaluation"])
    if suite == "ood":
        return ood_events(name, cfg)
    if suite == "compound":
        return evaluation_events(cfg["compound"][name], cfg["evaluation"])
    rng = np.random.default_rng(seed)
    return sample_training_events(rng, ids, pool, cfg["training"])


def surge_routes(src, dst, events, seed):
    """Copy src to dst, adding departures on the surged OD relations inside each surge window."""
    rng = np.random.default_rng(seed)
    tree = ET.parse(src)
    root = tree.getroot()
    extra = []
    for ev in events:
        if ev.kind not in ("D2", "demand_surge") or "od" not in ev.params:
            continue
        f = ev.params.get("factor", 1.5)
        od = {tuple(o) for o in ev.params["od"]}
        for el in root:
            if el.tag != "trip" or (el.get("from"), el.get("to")) not in od:
                continue
            t = float(el.get("depart"))
            if ev.start <= t < ev.end and rng.random() < f - 1.0:
                c = ET.Element("trip", dict(el.attrib))
                c.set("id", el.get("id") + "_s")
                c.set("depart", f"{t + rng.uniform(0, 1):.2f}")
                extra.append(c)
    root.extend(extra)
    root[:] = sorted(root, key=lambda e: float(e.get("depart", 0)) if e.tag == "trip" else -1)
    tree.write(dst)
    return len(extra)


def names_for(suite, cfg):
    if suite == "eval":
        return ["D1", "D2", "D3", "D4"]
    if suite == "ood":
        return [k for k in cfg["ood"] if k != "leave_one_out"]
    if suite == "compound":
        return list(cfg["compound"])
    return ["train"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="ingolstadt21")
    ap.add_argument("--suite", choices=["eval", "ood", "compound", "train"], default="eval")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--surge-routes", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_yaml("disruptions.yaml")
    net_cfg = load_yaml("networks.yaml")[a.network]
    base = Path(net_cfg["sumocfg"]).parent
    net_file = next(base.glob("*.net.xml"))
    rou_file = next(base.glob("*.rou.xml"))
    ids, pool = tls_ids(net_file), od_pool(rou_file)
    seeds = a.seeds or load_seeds()["training" if a.suite == "train" else "evaluation"]
    out = Path(a.out or base / "perturb" / a.suite)
    out.mkdir(parents=True, exist_ok=True)

    n = 0
    for name in names_for(a.suite, cfg):
        for seed in seeds:
            rng = np.random.default_rng([seed, zlib.crc32(name.encode()) % 9973])
            events = suite_events(a.suite, name, seed, cfg, ids, pool)
            for ev in events:
                if ev.kind in ("D1", "lane_closure") and "site" not in ev.params:
                    ev.params["site"] = ids[int(rng.integers(len(ids)))] if a.suite == "train" else "central"
                if ev.kind in ("D2", "demand_surge") and "od" not in ev.params:
                    k = ev.params.get("od_relations", 3)
                    ev.params["od"] = pool[:k]
                    ev.params.setdefault("factor", 1.5)
                if ev.kind in ("D3", "detector_freeze"):
                    ev.params["rng_seed"] = int(rng.integers(2 ** 31))
            tag = f"{name}_s{seed}"
            with open(out / f"{tag}.json", "w") as f:
                json.dump({"network": a.network, "suite": a.suite, "name": name, "seed": seed,
                           "events": dump_events(events)}, f, indent=1)
            if a.surge_routes and any(e.kind in ("D2", "demand_surge") for e in events):
                surge_routes(rou_file, out / f"{tag}.rou.xml", events, seed)
            n += 1
    print(f"{n} schedules -> {out}")


if __name__ == "__main__":
    main()
