import numpy as np
import torch


def lane_capacity(length_m, spacing_m=7.5):
    return np.maximum(1.0, np.asarray(length_m, dtype=np.float32) / spacing_m)


def phase_pressure(queue, capacity, movements):
    ratio = np.asarray(queue, dtype=np.float32) / np.asarray(capacity, dtype=np.float32)
    out = np.zeros(len(movements), dtype=np.float32)
    for k, pairs in enumerate(movements):
        out[k] = sum(ratio[a] - ratio[b] for a, b in pairs)
    return out


def standardize(pressure, admissible):
    p = np.where(admissible, pressure, 0.0)
    m = np.abs(p).max()
    return pressure / m if m > 1e-8 else pressure


def prior_log_probs(pressure, admissible, tau_p=0.5):
    p = torch.as_tensor(pressure, dtype=torch.float32)
    mask = torch.as_tensor(admissible, dtype=torch.bool)
    scale = torch.where(mask, p.abs(), torch.zeros_like(p)).amax(dim=-1, keepdim=True).clamp_min(1e-8)
    z = (p / scale) / tau_p
    z = z.masked_fill(~mask, float("-inf"))
    z = z - z.amax(dim=-1, keepdim=True)
    return torch.log_softmax(z, dim=-1)


def masked_log_policy(q, admissible, tau=1.0):
    z = (q / tau).masked_fill(~admissible, float("-inf"))
    z = z - z.amax(dim=-1, keepdim=True)
    return torch.log_softmax(z, dim=-1)


def forward_kl(prior_logp, policy_logp, admissible):
    p = prior_logp.exp()
    term = torch.where(admissible, p * (prior_logp - policy_logp), torch.zeros_like(p))
    return term.sum(dim=-1)
