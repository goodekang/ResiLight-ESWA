import torch
import torch.nn as nn
import torch.nn.functional as F

from .modes import MODES

SLOT_FEATURES = 4


class GATLayer(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        assert dim % heads == 0
        self.h, self.dk = heads, dim // heads
        self.W = nn.Linear(dim, dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.dk))
        self.a_dst = nn.Parameter(torch.empty(heads, self.dk))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x, adj, drop_rate=0.0):
        B, N, _ = x.shape
        z = self.W(x).view(B, N, self.h, self.dk)
        s = (z * self.a_src).sum(-1)
        d = (z * self.a_dst).sum(-1)
        e = F.leaky_relu(s.unsqueeze(2) + d.unsqueeze(1), 0.2)
        a = adj
        if drop_rate > 0:
            keep = (torch.rand_like(adj.float()) > drop_rate).bool()
            eye = torch.eye(N, dtype=torch.bool, device=adj.device).unsqueeze(0)
            a = (adj & keep) | eye
        e = e.permute(0, 3, 1, 2).masked_fill(~a.unsqueeze(1), float("-inf"))
        alpha = torch.softmax(e, dim=-1)
        out = torch.einsum("bhij,bjhd->bihd", alpha, z).reshape(B, N, -1)
        return F.elu(out)


class ResiLightNet(nn.Module):
    def __init__(self, embed=128, hidden=128, layers=2, heads=4, slots=12, mode_dim=16,
                 use_gat=True, num_modes=len(MODES)):
        super().__init__()
        self.slots = slots
        self.use_gat = use_gat
        self.slot_id = nn.Embedding(slots, embed)
        self.slot_enc = nn.Sequential(nn.Linear(SLOT_FEATURES + 1, embed), nn.ReLU(), nn.Linear(embed, embed))
        self.node_enc = nn.Sequential(nn.Linear(embed + 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.gat = nn.ModuleList([GATLayer(hidden, heads) for _ in range(layers)]) if use_gat else nn.ModuleList()
        self.mode_emb = nn.Embedding(num_modes, mode_dim)
        self.value = nn.Sequential(nn.Linear(hidden + mode_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.adv = nn.Sequential(nn.Linear(embed + hidden + mode_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def encode(self, obs, adj, drop_rate=0.0):
        slots, mask = obs["slots"], obs["mask"]
        B, N, S, _ = slots.shape
        feat = torch.cat([slots, mask.unsqueeze(-1)], dim=-1)
        ids = torch.arange(S, device=slots.device)
        e = self.slot_enc(feat) + self.slot_id(ids)
        e = e * mask.unsqueeze(-1)
        pooled = e.sum(2) / mask.sum(2, keepdim=True).clamp_min(1.0)
        cur = (obs["current"] * mask).sum(2, keepdim=True) / mask.sum(2, keepdim=True).clamp_min(1.0)
        h = self.node_enc(torch.cat([pooled, cur, obs["elapsed"]], dim=-1))
        for layer in self.gat:
            h = layer(h, adj, drop_rate)
        return h, e

    def forward(self, obs, adj, mode, drop_rate=0.0):
        h, e = self.encode(obs, adj, drop_rate)
        m = self.mode_emb(mode).unsqueeze(1).expand(-1, h.shape[1], -1)
        hm = torch.cat([h, m], dim=-1)
        v = self.value(hm)
        pm = obs["phase_matrix"]
        phase_vec = torch.einsum("bnps,bnsd->bnpd", pm, e)
        P = pm.shape[2]
        a_in = torch.cat([phase_vec, hm.unsqueeze(2).expand(-1, -1, P, -1)], dim=-1)
        a = self.adv(a_in).squeeze(-1)
        valid = obs["phase_valid"]
        a_mean = (a * valid).sum(-1, keepdim=True) / valid.sum(-1, keepdim=True).clamp_min(1.0)
        q = v + a - a_mean
        return q.masked_fill(valid == 0, -1e9)


class IndependentMLPNet(ResiLightNet):
    def __init__(self, **kw):
        kw["use_gat"] = False
        super().__init__(**kw)
