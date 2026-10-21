import numpy as np
from scipy import stats


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def paired_tests(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    t = stats.ttest_rel(a, b).pvalue
    w = stats.wilcoxon(a, b, alternative="two-sided", method="exact" if len(a) <= 25 else "auto").pvalue
    return {"t": float(t), "wilcoxon": float(w), "dz": cohen_dz(d)}


def cohen_dz(d):
    d = np.asarray(d, float)
    s = d.std(ddof=1)
    return float(d.mean() / s) if s > 0 else 0.0


def bootstrap_ci(d, resamples=10000, level=0.95, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(resamples, len(d)))
    means = d[idx].mean(axis=1)
    lo = (1 - level) / 2
    return float(np.quantile(means, lo)), float(np.quantile(means, 1 - lo))


def compare_family(reference, others, resamples=10000):
    rows = {}
    for name, vals in others.items():
        r = paired_tests(vals, reference)
        r["ci"] = bootstrap_ci(np.asarray(vals) - np.asarray(reference), resamples)
        r["mean_diff"] = float(np.mean(vals) - np.mean(reference))
        rows[name] = r
    names = list(rows)
    pt = holm([rows[n]["t"] for n in names])
    pw = holm([rows[n]["wilcoxon"] for n in names])
    for n, a, b in zip(names, pt, pw):
        rows[n]["t_holm"], rows[n]["wilcoxon_holm"] = float(a), float(b)
        rows[n]["significant"] = bool(a < 0.05 and b < 0.05)
    return rows


def summarize(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0


def degradation(disrupted, recurrent):
    return 100.0 * (np.asarray(disrupted, float) / np.asarray(recurrent, float) - 1.0)


def transfer_gap(frozen, native):
    return 100.0 * (np.asarray(frozen, float) / np.asarray(native, float) - 1.0)


def expected_calibration_error(conf, correct, bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(ece)


def macro_f1(y_true, y_pred, k=5):
    f = []
    for c in range(k):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f)), f
