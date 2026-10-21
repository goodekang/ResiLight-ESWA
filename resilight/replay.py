import numpy as np


class SumTree:
    def __init__(self, capacity):
        self.n = 1
        while self.n < capacity:
            self.n *= 2
        self.t = np.zeros(2 * self.n, dtype=np.float64)

    def set(self, i, v):
        i += self.n
        self.t[i] = v
        i //= 2
        while i >= 1:
            self.t[i] = self.t[2 * i] + self.t[2 * i + 1]
            i //= 2

    def total(self):
        return self.t[1]

    def find(self, s):
        i = 1
        while i < self.n:
            i *= 2
            if s > self.t[i]:
                s -= self.t[i]
                i += 1
        return i - self.n


class PrioritizedReplay:
    def __init__(self, capacity, alpha=0.6, eps=1e-5):
        self.cap, self.alpha, self.eps = capacity, alpha, eps
        self.data = [None] * capacity
        self.tree = SumTree(capacity)
        self.pos = 0
        self.size = 0
        self.max_p = 1.0

    def add(self, item):
        self.data[self.pos] = item
        self.tree.set(self.pos, self.max_p ** self.alpha)
        self.pos = (self.pos + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch, beta, rng):
        total = self.tree.total()
        seg = total / batch
        idx = []
        for k in range(batch):
            s = rng.uniform(seg * k, seg * (k + 1))
            i = min(self.tree.find(s), self.size - 1)
            idx.append(i)
        p = np.array([self.tree.t[i + self.tree.n] for i in idx]) / total
        w = (self.size * p) ** (-beta)
        w = (w / w.max()).astype(np.float32)
        return idx, [self.data[i] for i in idx], w

    def update(self, idx, td):
        for i, d in zip(idx, td):
            p = float(abs(d)) + self.eps
            self.max_p = max(self.max_p, p)
            self.tree.set(i, p ** self.alpha)

    def __len__(self):
        return self.size
