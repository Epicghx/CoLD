"""Two augmented views of a 12-lead recording.

amp="cold"      per-lead gains and offsets from one Gaussian copula (lam, rho)
amp="standard"  one global gain per view, U(0.9, 1.1)
amp="none"      no amplitude change (the control)
"""
import math
import random

import torch
import torch.nn.functional as F


def crop_shift(x, crop_min=800, crop_max=1000, shift_max=50, noise_std=0.01):
    C, T = x.shape
    L = random.randint(crop_min, min(crop_max, T))
    s0 = random.randint(0, T - L)
    x = x[:, s0:s0 + L]
    if L != T:
        x = F.interpolate(x[None], size=T, mode="linear", align_corners=False)[0]
    if shift_max > 0:
        k = random.randint(-shift_max, shift_max)
        x = torch.roll(x, k, dims=-1)
        if k > 0:
            x[:, :k] = 0
        elif k < 0:
            x[:, k:] = 0
    if noise_std > 0:
        x = x + noise_std * torch.randn_like(x)
    return x


def copula_latents(C, lam, rho):
    """Latents of the two views: corr lam between leads, rho between views."""
    c = math.sqrt(max(0.0, 1.0 - rho * rho))
    a1, a_ = torch.randn(1), torch.randn(1)
    e1, e_ = torch.randn(C), torch.randn(C)
    a2 = rho * a1 + c * a_
    e2 = rho * e1 + c * e_
    xi1 = math.sqrt(lam) * a1 + math.sqrt(1.0 - lam) * e1
    xi2 = math.sqrt(lam) * a2 + math.sqrt(1.0 - lam) * e2
    return xi1, xi2


def normal_cdf(z):
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


class CoLDSampler:
    """Gains LogU(s_min, s_max) and offsets N(0, sigma_b^2) for both views."""

    def __init__(self, s_min=0.5, s_max=2.0, sigma_b=0.5, lam=0.0, rho=0.0):
        assert 0.0 <= lam <= 1.0 and 0.0 <= rho <= 1.0
        self.log_min, self.log_max = math.log(s_min), math.log(s_max)
        self.sigma_b, self.lam, self.rho = sigma_b, lam, rho

    def __call__(self, C):
        xi1, xi2 = copula_latents(C, self.lam, self.rho)
        zeta1, zeta2 = copula_latents(C, self.lam, self.rho)
        span = self.log_max - self.log_min
        s1 = torch.exp(self.log_min + normal_cdf(xi1) * span)
        s2 = torch.exp(self.log_min + normal_cdf(xi2) * span)
        return (s1, self.sigma_b * zeta1), (s2, self.sigma_b * zeta2)


class TwoViews:
    def __init__(self, amp="cold", crop_min=800, crop_max=1000, shift_max=50, noise_std=0.01,
                 s_min=0.5, s_max=2.0, sigma_b=0.5, lam=0.0, rho=0.0, global_gain=(0.9, 1.1)):
        assert amp in ("cold", "standard", "none")
        self.amp = amp
        self.time = dict(crop_min=crop_min, crop_max=crop_max, shift_max=shift_max, noise_std=noise_std)
        self.noise_std = noise_std
        self.sampler = CoLDSampler(s_min, s_max, sigma_b, lam, rho)
        self.global_gain = global_gain

    def _draws(self, C):
        if self.amp == "cold":
            return self.sampler(C)
        if self.amp == "standard":
            return [(torch.full((C,), random.uniform(*self.global_gain)), torch.zeros(C)) for _ in range(2)]
        return [(torch.ones(C), torch.zeros(C)) for _ in range(2)]

    def __call__(self, x):
        views = []
        pre = dict(self.time, noise_std=0.0) if self.amp == "standard" else self.time
        for s, b in self._draws(x.shape[0]):
            u = crop_shift(x.clone(), **pre)
            v = s[:, None] * u + b[:, None]
            if self.noise_std > 0:
                v = v + self.noise_std * torch.randn_like(v)
            views.append(v)
        return views
