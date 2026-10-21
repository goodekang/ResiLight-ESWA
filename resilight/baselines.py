import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .agent import ResiLightAgent, collate
from .config import ROOT, load_yaml
from .models import ResiLightNet
from .runner import FixedTimeController, MaxPressureController, ResiLightController, webster_plan


def _agent(cfg, gat, layers=None, device="cpu"):
    c = copy.deepcopy(cfg)
    if layers:
        c["model"]["gat_layers"] = layers
    return ResiLightAgent(c, device, use_gat=gat, use_physics=False)


class IPPOAgent:
    def __init__(self, cfg, device="cpu", clip=0.2, epochs=4, ent=0.01, gae=0.95):
        m = cfg["model"]
        self.net = ResiLightNet(m["embed_dim"], m["gat_hidden"], 1, m["gat_heads"], m["movement_slots"],
                                m["mode_embed_dim"], use_gat=False).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg["train"]["lr"])
        self.cfg, self.device, self.clip, self.epochs, self.ent, self.gae = cfg, device, clip, epochs, ent, gae
        self.buf, self.steps = [], 0
        self.rng = np.random.default_rng(0)

    def epsilon(self):
        return 0.0

    @torch.no_grad()
    def q_values(self, obs, adj, mode, drop_rate=0.0):
        o = {k: torch.as_tensor(v, device=self.device).unsqueeze(0) for k, v in obs.items()}
        a = torch.as_tensor(adj, device=self.device).unsqueeze(0)
        return self.net(o, a, torch.as_tensor([mode], device=self.device))[0].cpu().numpy()

    def observe(self, item):
        self.buf.append(item)
        self.steps += 1

    def update(self):
        if len(self.buf) < 512:
            return None
        b = collate(self.buf, self.device)
        with torch.no_grad():
            logits = self.net(b["obs"], b["adj"], b["mode"]).masked_fill(~b["admissible"], -1e9)
            logp_old = torch.log_softmax(logits, -1).gather(-1, b["action"].unsqueeze(-1)).squeeze(-1)
            ret = b["reward"].clone()
            g = torch.zeros_like(ret[0])
            for t in reversed(range(len(ret))):
                g = ret[t] + self.cfg["control"]["gamma"] * (1 - b["done"][t]) * g
                ret[t] = g
            adv = (ret - ret.mean()) / (ret.std() + 1e-6)
        for _ in range(self.epochs):
            logits = self.net(b["obs"], b["adj"], b["mode"]).masked_fill(~b["admissible"], -1e9)
            lp = torch.log_softmax(logits, -1)
            logp = lp.gather(-1, b["action"].unsqueeze(-1)).squeeze(-1)
            r = (logp - logp_old).exp()
            loss = -torch.min(r * adv, r.clamp(1 - self.clip, 1 + self.clip) * adv).mean()
            ent = -(lp.exp() * lp.masked_fill(~b["admissible"], 0)).sum(-1).mean()
            self.opt.zero_grad()
            (loss - self.ent * ent).backward()
            self.opt.step()
        self.buf.clear()
        return {"ppo": float(loss)}

    def save(self, path):
        torch.save({"net": self.net.state_dict(), "steps": self.steps}, path)

    def load(self, path):
        s = torch.load(path, map_location=self.device)
        self.net.load_state_dict(s["net"])
        self.steps = s["steps"]


def make_baseline(name, cfg, device="cpu"):
    if name == "idqn":
        return _agent(cfg, gat=False, device=device)
    if name == "mplight":
        return _agent(cfg, gat=False, device=device)
    if name == "colight":
        return _agent(cfg, gat=True, device=device)
    if name == "mgmq":
        return _agent(cfg, gat=True, layers=3, device=device)
    if name == "enb_rl":
        return _agent(cfg, gat=True, layers=3, device=device)
    if name == "ippo":
        return IPPOAgent(cfg, device)
    raise KeyError(name)


def baseline_reward_overrides(name):
    if name == "mplight":
        return {"w": np.array([0.0, 1.0, 0.0, 0.0], np.float32)}
    return {}


def fixed_time_plans(env, flows):
    return {t: webster_plan(flows[t]) for t in env.net.order}


def llm_baseline_prompts(name):
    path = ROOT / load_yaml("baselines.yaml")[name]["prompt"]
    if not Path(path).exists():
        raise FileNotFoundError(f"{path}: place the authors' released prompt templates here")
    return {p.name: p.read_text(encoding="utf-8") for p in Path(path).glob("*")}


class LLMLightController:
    def __init__(self, env, backend, prompts, pause_sim=True):
        self.env, self.backend, self.prompts, self.pause = env, backend, prompts, pause_sim
        self.latency = []

    def describe(self, i, pack):
        o = pack["obs"]
        rows = [f"movement {s}: queue ratio {o['slots'][i, s, 0]:.2f}, occupancy {o['slots'][i, s, 1]:.2f}"
                for s in range(o["mask"].shape[1]) if o["mask"][i, s] > 0]
        return "\n".join(rows)

    def act(self, pack, mode, ctx):
        import time
        acts = []
        for i, tid in enumerate(self.env.net.order):
            adm = np.flatnonzero(pack["admissible"][i])
            msgs = [{"role": "system", "content": self.prompts.get("system.txt", "")},
                    {"role": "user", "content": self.describe(i, pack) + f"\nChoose one phase from {adm.tolist()}."}]
            t0 = time.perf_counter()
            text = self.backend.generate(msgs)
            self.latency.append(time.perf_counter() - t0)
            digits = [int(c) for c in text if c.isdigit() and int(c) in adm]
            acts.append(digits[0] if digits else int(np.where(pack["admissible"][i], pack["pressure"][i], -np.inf).argmax()))
        return np.array(acts)


class ILLMTSCController:
    def __init__(self, env, agent, backend, prompts, cfg):
        self.inner = ResiLightController(env, agent, cfg, preemption=False, mode_conditioning=False)
        self.llm = LLMLightController(env, backend, prompts)

    def act(self, pack, mode, ctx):
        proposal = self.inner.act(pack, 0, ctx)
        review = self.llm.act(pack, mode, ctx)
        keep = np.array([pack["admissible"][i, a] for i, a in enumerate(proposal)])
        return np.where(keep, proposal, review)
