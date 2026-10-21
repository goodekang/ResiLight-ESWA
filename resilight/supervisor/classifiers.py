import numpy as np
import torch
import torch.nn as nn

from ..digest import features, set_entries


class _Wrapper:
    def classify(self, digest):
        raise NotImplementedError


class XGBSupervisor(_Wrapper):
    def __init__(self, capacity=None, normalize=True):
        self.capacity, self.normalize = capacity, normalize
        self.model = None

    def fit(self, digests, labels):
        from xgboost import XGBClassifier
        X = np.stack([features(d, self.capacity, self.normalize) for d in digests])
        self.model = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.9, objective="multi:softprob")
        self.model.fit(X, np.asarray(labels))
        return self

    def classify(self, digest):
        x = features(digest, self.capacity, self.normalize)[None]
        return int(self.model.predict(x)[0])


class MLPSupervisor(_Wrapper):
    def __init__(self, in_dim=48, hidden=128, capacity=None, normalize=True, device="cpu"):
        self.capacity, self.normalize, self.device = capacity, normalize, device
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 5)).to(device)

    def fit(self, digests, labels, epochs=30, lr=1e-3, batch=256):
        X = torch.as_tensor(np.stack([features(d, self.capacity, self.normalize) for d in digests]), device=self.device)
        y = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        for _ in range(epochs):
            perm = torch.randperm(len(X), device=self.device)
            for i in range(0, len(X), batch):
                j = perm[i:i + batch]
                loss = nn.functional.cross_entropy(self.net(X[j]), y[j])
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def classify(self, digest):
        x = torch.as_tensor(features(digest, self.capacity, self.normalize)[None], device=self.device)
        return int(self.net(x).argmax(-1))


class _SAB(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))

    def forward(self, x):
        x = self.n1(x + self.att(x, x, x)[0])
        return self.n2(x + self.ff(x))


class SetTransformer(nn.Module):
    def __init__(self, in_dim=5, global_dim=8, d=64, blocks=2, classes=5):
        super().__init__()
        self.inp = nn.Linear(in_dim, d)
        self.blocks = nn.Sequential(*[_SAB(d) for _ in range(blocks)])
        self.seed = nn.Parameter(torch.randn(1, 1, d))
        self.pool = nn.MultiheadAttention(d, 4, batch_first=True)
        self.out = nn.Sequential(nn.Linear(d + global_dim, d), nn.ReLU(), nn.Linear(d, classes))

    def forward(self, s, g):
        h = self.blocks(self.inp(s))
        p = self.pool(self.seed.expand(len(s), -1, -1), h, h)[0].squeeze(1)
        return self.out(torch.cat([p, g], -1))


class SetTransformerSupervisor(_Wrapper):
    def __init__(self, capacity=None, device="cpu"):
        self.capacity, self.device = capacity, device
        self.net = SetTransformer().to(device)

    def _g(self, d):
        n = d["network"]
        cap = self.capacity["network"] if self.capacity else 1.0
        return [n["mean_speed_mps"], n["total_queue_veh"] / cap, *n["trend_sign"].values(),
                d["detector_health"]["fraction"], float(len(d["flagged_vehicles"])), 0.0]

    def fit(self, digests, labels, epochs=30, lr=1e-3, batch=256):
        S = torch.as_tensor(np.stack([set_entries(d, self.capacity) for d in digests]), device=self.device)
        G = torch.as_tensor(np.asarray([self._g(d) for d in digests], dtype=np.float32), device=self.device)
        y = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        for _ in range(epochs):
            perm = torch.randperm(len(S), device=self.device)
            for i in range(0, len(S), batch):
                j = perm[i:i + batch]
                loss = nn.functional.cross_entropy(self.net(S[j], G[j]), y[j])
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    @torch.no_grad()
    def classify(self, digest):
        s = torch.as_tensor(set_entries(digest, self.capacity)[None], device=self.device)
        g = torch.as_tensor(np.asarray([self._g(digest)], dtype=np.float32), device=self.device)
        return int(self.net(s, g).argmax(-1))


def capacity_stats(net):
    return {"network": float(net["total_storage"]), "per_inter": dict(net["approach_storage"])}
