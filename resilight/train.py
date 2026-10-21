import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .agent import ResiLightAgent
from .config import load_default, load_seeds, load_yaml
from .disruptions import sample_training_events
from .env import TrafficEnv
from .runner import ResiLightController, run_episode


def variant_flags(name):
    return {
        "full": dict(use_gat=True, use_physics=True),
        "no_physics": dict(use_gat=True, use_physics=False),
        "no_gat": dict(use_gat=False, use_physics=True),
        "no_gat_no_physics": dict(use_gat=False, use_physics=False),
    }[name]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="grid4x4")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--out", default="results/checkpoints")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-mode-conditioning", action="store_true")
    ap.add_argument("--label-noise", default=None)
    ap.add_argument("--lambda-end", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args(argv)

    cfg = load_default()
    if a.lambda_end is not None:
        cfg["physics"]["lambda_end"] = a.lambda_end
    if a.steps:
        cfg["train"]["total_env_steps"] = a.steps
    assert a.seed not in load_seeds()["evaluation"]
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    net_cfg = load_yaml("networks.yaml")[a.network]
    env = TrafficEnv(cfg, net_cfg, label=f"train_{a.seed}")
    agent = ResiLightAgent(cfg, a.device, **variant_flags(a.variant))
    ctl = ResiLightController(env, agent, cfg, mode_conditioning=not a.no_mode_conditioning)
    noise = np.load(a.label_noise) if a.label_noise else None
    out = Path(a.out) / f"{a.network}_{a.variant}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    ep, hist = 0, []
    while agent.steps < cfg["train"]["total_env_steps"]:
        ctl.agent.rng = rng
        ctl.explore = True
        sample = lambda e: sample_training_events(rng, e.net.order, e.heaviest_od(8))
        m, _ = run_episode(env, ctl, int(rng.integers(1, 10**6)), sample, train_agent=agent, labels_noise=noise,
                           digest_top_k=cfg["supervisor"]["digest_top_k"])
        ep += 1
        hist.append({"episode": ep, "steps": agent.steps, "att": m.att(), "epsilon": agent.epsilon(), "lambda": agent.lam()})
        if ep % 10 == 0:
            agent.save(out / "latest.pt")
            (out / "history.json").write_text(json.dumps(hist))
    agent.save(out / "final.pt")
    (out / "history.json").write_text(json.dumps(hist))
    env.close()


if __name__ == "__main__":
    main()
