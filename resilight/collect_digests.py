import argparse
import pickle
from pathlib import Path

import numpy as np

from .config import load_default, load_yaml
from .digest import DigestBuilder
from .disruptions import sample_training_events, scripted_label
from .env import TrafficEnv
from .runner import MaxPressureController, run_episode


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/digests")
    a = ap.parse_args(argv)
    cfg = load_default()
    env = TrafficEnv(cfg, load_yaml("networks.yaml")[a.network], label="digests")
    rng = np.random.default_rng(a.seed)
    ctl = MaxPressureController(env)
    data = []

    def collect(e, t, mode, info):
        if t > 0 and t % cfg["control"]["supervisory_cycle_s"] == 0:
            d = builder.build(e.stats(), e.intersection_stats(), e.health(), e.flagged(), 0)
            data.append((d, scripted_label(e.events, t)))

    for _ in range(a.episodes):
        builder = DigestBuilder(cfg["supervisor"]["digest_top_k"])
        sample = lambda e: sample_training_events(rng, e.net.order, e.heaviest_od(8))
        run_episode(env, ctl, int(rng.integers(1, 10**6)), sample, collect=collect)
    env.close()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{a.network}.pkl", "wb") as f:
        pickle.dump(data, f)
    print(len(data), "labeled digests")


if __name__ == "__main__":
    main()
