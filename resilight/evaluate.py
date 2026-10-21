import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .agent import ResiLightAgent
from .config import load_default, load_seeds, load_yaml
from .disruptions import evaluation_events, Event
from .env import TrafficEnv
from .runner import FixedTimeController, MaxPressureController, ResiLightController, run_episode
from .stats import compare_family, degradation, macro_f1, summarize, transfer_gap
from .supervisor import OracleSupervisor, RuleDetector, Supervisor
from .supervisor.backend import make_backend


def build_controller(method, env, cfg, ckpt, device, variant=None):
    if method == "max_pressure":
        return MaxPressureController(env)
    if method == "resilight":
        flags = variant or {}
        agent = ResiLightAgent(cfg, device, use_gat=flags.get("use_gat", True), use_physics=flags.get("use_physics", True))
        agent.load(ckpt)
        return ResiLightController(env, agent, cfg, preemption=flags.get("preemption", True),
                                   mode_conditioning=flags.get("mode_conditioning", True))
    from .baselines import make_baseline
    agent = make_baseline(method, cfg, device)
    agent.load(ckpt)
    return ResiLightController(env, agent, cfg, preemption=False, mode_conditioning=False)


def build_supervisor(kind, cfg, valid_ids, model_path=None, classifier=None):
    s = cfg["supervisor"]
    common = dict(valid_ids=valid_ids, confidence_threshold=s["confidence_threshold"], consecutive=s["consecutive_proposals"],
                  budget_s=s["call_budget_s"], max_failures=s["fallback_after_failures"])
    if kind in ("rule", "rule_localization"):
        return Supervisor(classifier=RuleDetector(localization=kind == "rule_localization"), **common)
    if kind == "llm":
        return Supervisor(backend=make_backend(s, model_path), **common)
    if kind == "llm_no_localization":
        return Supervisor(backend=make_backend(s, model_path), localization_rule=False, **common)
    if kind == "llm_no_rules":
        return Supervisor(backend=make_backend(s, model_path), localization_rule=False, decision_rules=False, **common)
    if kind == "classifier":
        return Supervisor(classifier=classifier, **common)
    raise ValueError(kind)


def evaluate(method, network, seeds, disruption, mode_source, cfg, ckpt, device, supervisor_kind="llm",
             model_path=None, variant=None, out_csv=None):
    net_cfg = load_yaml("networks.yaml")[network]
    env = TrafficEnv(cfg, net_cfg, label=f"eval_{method}")
    rows = []
    for seed in seeds:
        events = evaluation_events(disruption) if disruption else []
        ctl = build_controller(method, env, cfg, ckpt, device, variant)
        sup = oracle = None
        fixed = None
        if method == "resilight" and (variant or {}).get("mode_conditioning", True):
            if mode_source == "oracle":
                oracle = OracleSupervisor(events)
            elif mode_source == "fixed":
                fixed = 0
            else:
                env.reset(seed, [])
                sup = build_supervisor(supervisor_kind, cfg, env.net.order, model_path)
        fb = MaxPressureController(env)
        m, log = run_episode(env, ctl, seed, events, supervisor=sup, oracle=oracle, fixed_mode=fixed,
                             digest_top_k=cfg["supervisor"]["digest_top_k"], fallback_controller=fb)
        row = {"method": method, "network": network, "disruption": "+".join(disruption or ["recurrent"]),
               "mode_source": mode_source, "seed": seed, **m.as_row()}
        if log:
            y = np.array([l["true"] for l in log]); p = np.array([l["mode"] for l in log])
            row["mode_macro_f1"] = macro_f1(y, p)[0]
        rows.append(row)
    env.close()
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="resilight")
    ap.add_argument("--network", default="ingolstadt21")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--disruption", nargs="*", default=None)
    ap.add_argument("--mode-source", choices=["supervisor", "oracle", "fixed"], default="supervisor")
    ap.add_argument("--supervisor", default="llm")
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    cfg = load_default()
    seeds = load_seeds()["evaluation"]
    rows = evaluate(a.method, a.network, seeds, a.disruption, a.mode_source, cfg, a.ckpt, a.device,
                    a.supervisor, a.model_path, None, a.out)
    m, s = summarize([r["att"] for r in rows])
    print(f"{a.method} {a.network} ATT {m:.1f} +- {s:.1f}")


if __name__ == "__main__":
    main()
