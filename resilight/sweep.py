import argparse
import copy
import itertools
import json

import numpy as np

from .config import load_modes, load_yaml

VALUES = load_modes()["search_values"]


def perturbed_tables(n=40, lo=0.5, hi=1.5, seed=0):
    from .modes import WEIGHTS
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        f = np.exp(rng.uniform(np.log(lo), np.log(hi), WEIGHTS.shape)).astype(np.float32)
        out.append(np.where(WEIGHTS > 0, WEIGHTS * f, 0.0))
    return out


def coordinate_search(evaluate, table, values=VALUES, rounds=2):
    best = table.copy()
    best_score = evaluate(best)
    for _ in range(rounds):
        for i, j in itertools.product(range(best.shape[0]), range(best.shape[1])):
            for v in values:
                cand = best.copy()
                cand[i, j] = v
                s = evaluate(cand)
                if s < best_score:
                    best, best_score = cand, s
    return best, best_score


def tuning_configs(base, n=24, seed=0):
    rng = np.random.default_rng(seed)
    out = [copy.deepcopy(base)]
    while len(out) < n:
        c = copy.deepcopy(base)
        c["train"]["lr"] = float(base["train"]["lr"] * 10 ** rng.uniform(-0.5, 0.5))
        c["train"]["batch_size"] = int(rng.choice([128, 256, 512]))
        c["train"]["target_update_every"] = int(rng.choice([1000, 2000, 4000]))
        c["control"]["gamma"] = float(rng.choice([0.95, 0.98, 0.99]))
        out.append(c)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["weights", "tuning"], default="weights")
    ap.add_argument("--out", default="results/sweep.json")
    a = ap.parse_args(argv)
    if a.kind == "weights":
        res = [t.tolist() for t in perturbed_tables()]
    else:
        from .config import load_default
        res = tuning_configs(load_default(), load_default()["baseline_tuning"]["configurations"])
    with open(a.out, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
