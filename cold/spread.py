"""Within-pair spread (Section 3.2.2) and spread matching (Section 3.3)."""
import math

import numpy as np


def spearman(r):
    return 6.0 / math.pi * math.asin(r / 2.0)


def pair_spread(lam, rho, s_min, s_max, sigma_b, C=12):
    """Closed-form within-pair spread of the lead-to-lead log-SD and mean, Eq. (6)."""
    sd_log_s = math.log(s_max / s_min) / math.sqrt(12.0)
    psi = 1.0 - spearman(lam) - spearman(rho) + spearman(lam * rho)
    k = math.sqrt((C - 1) / C)
    return (math.sqrt(2.0 * psi) * k * sd_log_s,
            math.sqrt(2.0 * (1.0 - lam) * (1.0 - rho)) * k * sigma_b)


def lead_stats(x):
    """x: (N, C, T). Returns lead-to-lead parts of the mean and log-SD, each (N, C)."""
    mu = x.mean(axis=2)
    log_sd = np.log(x.std(axis=2) + 1e-8)
    return mu - mu.mean(axis=1, keepdims=True), log_sd - log_sd.mean(axis=1, keepdims=True)


def target_spread(x):
    """delta_b and delta_s of Eq. (7) for target recordings x of shape (N, C, T)."""
    d_mu, d_logsd = lead_stats(x)
    C = x.shape[1]
    delta = lambda a: math.sqrt(2.0 / C * a.var(axis=0).sum())
    return delta(d_mu), delta(d_logsd)


def bounds(delta_b, delta_s, C=12):
    """Smallest gain ratio s_max / s_min and offset SD that satisfy Eq. (8)."""
    k = math.sqrt(2.0 * (C - 1) / C)
    return math.exp(math.sqrt(12.0) * delta_s / k), delta_b / k
