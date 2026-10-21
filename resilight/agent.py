import copy

import numpy as np
import torch
import torch.nn.functional as F

from .models import IndependentMLPNet, ResiLightNet
from .pressure import forward_kl, masked_log_policy, prior_log_probs
from .replay import PrioritizedReplay


def collate(items, device):
    keys = items[0]["obs"].keys()
    def stack(get, k):
        return torch.as_tensor(np.stack([get(i)[k] for i in items]), device=device)
    obs = {k: stack(lambda i: i["obs"], k) for k in keys}
    nxt = {k: stack(lambda i: i["next_obs"], k) for k in keys}
    t = lambda k, dt=torch.float32: torch.as_tensor(np.stack([i[k] for i in items]), dtype=dt, device=device)
    return {
        "obs": obs, "next_obs": nxt,
        "adj": t("adj", torch.bool), "action": t("action", torch.long),
        "reward": t("reward"), "mode": t("mode", torch.long), "next_mode": t("next_mode", torch.long),
        "done": t("done"), "admissible": t("admissible", torch.bool),
        "next_admissible": t("next_admissible", torch.bool), "pressure": t("pressure"),
    }


class ResiLightAgent:
    def __init__(self, cfg, device="cpu", use_gat=True, use_physics=True):
        m, t, p = cfg["model"], cfg["train"], cfg["physics"]
        cls = ResiLightNet if use_gat else IndependentMLPNet
        kw = dict(embed=m["embed_dim"], hidden=m["gat_hidden"], layers=m["gat_layers"], heads=m["gat_heads"],
                  slots=m["movement_slots"], mode_dim=m["mode_embed_dim"])
        self.online = cls(**kw).to(device)
        self.target = copy.deepcopy(self.online)
        self.opt = torch.optim.Adam(self.online.parameters(), lr=t["lr"])
        self.replay = PrioritizedReplay(t["replay_size"], t["per_alpha"])
        self.cfg, self.device, self.use_physics = cfg, device, use_physics
        self.steps = 0
        self.rng = np.random.default_rng(0)

    def epsilon(self):
        t = self.cfg["train"]
        f = min(1.0, self.steps / t["epsilon_decay_steps"])
        return t["epsilon_start"] + f * (t["epsilon_end"] - t["epsilon_start"])

    def lam(self):
        if not self.use_physics:
            return 0.0
        p, total = self.cfg["physics"], self.cfg["train"]["total_env_steps"]
        f = min(1.0, self.steps / (p["lambda_anneal_frac"] * total))
        return p["lambda_start"] + f * (p["lambda_end"] - p["lambda_start"])

    @torch.no_grad()
    def q_values(self, obs, adj, mode, drop_rate=0.0):
        o = {k: torch.as_tensor(v, device=self.device).unsqueeze(0) for k, v in obs.items()}
        a = torch.as_tensor(adj, device=self.device).unsqueeze(0)
        md = torch.as_tensor([mode], device=self.device)
        return self.online(o, a, md, drop_rate)[0].cpu().numpy()

    def observe(self, item):
        self.replay.add(item)
        self.steps += 1

    def update(self):
        t, p = self.cfg["train"], self.cfg["physics"]
        if len(self.replay) < max(t["batch_size"], t["warmup_steps"]):
            return None
        beta = t["per_beta0"] + (1 - t["per_beta0"]) * min(1.0, self.steps / t["total_env_steps"])
        idx, items, w = self.replay.sample(t["batch_size"], beta, self.rng)
        b = collate(items, self.device)
        q = self.online(b["obs"], b["adj"], b["mode"])
        qa = q.gather(-1, b["action"].unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            qn_online = self.online(b["next_obs"], b["adj"], b["next_mode"]).masked_fill(~b["next_admissible"], -1e9)
            a_star = qn_online.argmax(-1, keepdim=True)
            qn = self.target(b["next_obs"], b["adj"], b["next_mode"]).gather(-1, a_star).squeeze(-1)
            y = b["reward"] + self.cfg["control"]["gamma"] * (1 - b["done"]).unsqueeze(-1) * qn
        td = y - qa
        wt = torch.as_tensor(w, device=self.device).view(-1, 1)
        loss_td = (wt * td.pow(2)).mean()
        loss_phys = torch.zeros((), device=self.device)
        lam = self.lam()
        if lam > 0:
            logp = masked_log_policy(q, b["admissible"], p["tau"])
            prior = prior_log_probs(b["pressure"], b["admissible"], p["tau_p"])
            loss_phys = forward_kl(prior, logp, b["admissible"]).mean()
        loss = loss_td + lam * loss_phys
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), t["grad_clip"])
        self.opt.step()
        self.replay.update(idx, td.abs().mean(dim=1).detach().cpu().numpy())
        if self.steps % t["target_update_every"] == 0:
            self.target.load_state_dict(self.online.state_dict())
        return {"td": float(loss_td.detach()), "phys": float(loss_phys.detach()), "lambda": lam}

    def save(self, path):
        torch.save({"online": self.online.state_dict(), "steps": self.steps}, path)

    def load(self, path):
        s = torch.load(path, map_location=self.device)
        self.online.load_state_dict(s["online"])
        self.target.load_state_dict(s["online"])
        self.steps = s["steps"]
