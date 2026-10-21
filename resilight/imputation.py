import numpy as np


class Imputer:
    def __init__(self, n_lanes, alpha=0.1, persistence_s=120.0):
        self.mean = np.zeros(n_lanes, dtype=np.float32)
        self.seen = np.zeros(n_lanes, dtype=bool)
        self.last = np.full(n_lanes, np.nan, dtype=np.float32)
        self.unchanged = np.zeros(n_lanes, dtype=np.float32)
        self.alpha = alpha
        self.persistence = persistence_s

    def update(self, readings, dt):
        r = np.asarray(readings, dtype=np.float32)
        same = np.isclose(r, self.last, equal_nan=False)
        self.unchanged = np.where(same, self.unchanged + dt, 0.0)
        self.last = r.copy()
        self.mean = np.where(self.seen, (1 - self.alpha) * self.mean + self.alpha * r, r)
        self.seen[:] = True

    def failed(self):
        return self.unchanged > self.persistence

    def health(self):
        f = self.failed()
        return int(f.sum()), float(f.mean()) if f.size else 0.0

    def impute(self, readings, available):
        r = np.asarray(readings, dtype=np.float32)
        a = np.asarray(available, dtype=bool)
        filled = np.where(a, r, self.mean)
        return filled, a.astype(np.float32)
