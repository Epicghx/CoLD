"""Two-view objectives. Each takes two batches of views and returns the loss;
`self.encoder` is the network kept for fine-tuning."""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(d_in, d_hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.BatchNorm1d(d_hidden), nn.ReLU(inplace=True), nn.Linear(d_hidden, d_out))


def frozen_copy(module):
    m = copy.deepcopy(module)
    for p in m.parameters():
        p.requires_grad = False
    return m


class MoCo(nn.Module):
    """MoCo-v2 with a queue of negatives and a momentum encoder."""

    def __init__(self, encoder, proj_dim=128, proj_hidden=192, queue_size=16384, temperature=0.1,
                 momentum=(0.99, 0.999)):
        super().__init__()
        self.encoder = encoder
        self.proj = mlp(encoder.embed_dim, proj_hidden, proj_dim)
        self.encoder_k, self.proj_k = frozen_copy(encoder), frozen_copy(self.proj)
        self.tau, self.momentum_range = temperature, momentum
        self.register_buffer("m", torch.tensor(momentum[0]))
        self.register_buffer("queue", F.normalize(torch.randn(proj_dim, queue_size), dim=0))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _ema(self):
        m = float(self.m)
        for a, b in ((self.encoder, self.encoder_k), (self.proj, self.proj_k)):
            for pa, pb in zip(a.parameters(), b.parameters()):
                pb.mul_(m).add_(pa.detach(), alpha=1 - m)

    @torch.no_grad()
    def _enqueue(self, k):
        K, B, p = self.queue.shape[1], k.shape[0], int(self.ptr)
        idx = torch.arange(p, p + B, device=k.device) % K
        self.queue[:, idx] = k.T
        self.ptr[0] = (p + B) % K

    def forward(self, v1, v2):
        q = F.normalize(self.proj(self.encoder(v1)), dim=1)
        with torch.no_grad():
            self._ema()
            k = F.normalize(self.proj_k(self.encoder_k(v2)), dim=1)
        logits = torch.cat([(q * k).sum(1, keepdim=True), q @ self.queue.clone()], dim=1) / self.tau
        loss = F.cross_entropy(logits, torch.zeros(len(q), dtype=torch.long, device=q.device))
        self._enqueue(k)
        return loss


class SimCLR(nn.Module):
    def __init__(self, encoder, proj_dim=128, proj_hidden=192, temperature=0.1):
        super().__init__()
        self.encoder, self.proj, self.tau = encoder, mlp(encoder.embed_dim, proj_hidden, proj_dim), temperature

    def forward(self, v1, v2):
        z = torch.cat([F.normalize(self.proj(self.encoder(v)), dim=1) for v in (v1, v2)])
        B = len(v1)
        sim = (z @ z.T / self.tau).masked_fill(torch.eye(2 * B, dtype=torch.bool, device=z.device), float("-inf"))
        target = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
        return F.cross_entropy(sim, target)


class BYOL(nn.Module):
    def __init__(self, encoder, proj_dim=128, proj_hidden=192, pred_hidden=128, momentum=(0.99, 0.999)):
        super().__init__()
        self.encoder = encoder
        self.proj = mlp(encoder.embed_dim, proj_hidden, proj_dim)
        self.pred = mlp(proj_dim, pred_hidden, proj_dim)
        self.encoder_t, self.proj_t = frozen_copy(encoder), frozen_copy(self.proj)
        self.momentum_range = momentum
        self.register_buffer("m", torch.tensor(momentum[0]))

    def forward(self, v1, v2):
        p1, p2 = (F.normalize(self.pred(self.proj(self.encoder(v))), dim=1) for v in (v1, v2))
        with torch.no_grad():
            m = float(self.m)
            for a, b in ((self.encoder, self.encoder_t), (self.proj, self.proj_t)):
                for pa, pb in zip(a.parameters(), b.parameters()):
                    pb.mul_(m).add_(pa.detach(), alpha=1 - m)
            t1, t2 = (F.normalize(self.proj_t(self.encoder_t(v)), dim=1) for v in (v1, v2))
        return 2 - (p1 * t2).sum(1).mean() - (p2 * t1).sum(1).mean()


class BarlowTwins(nn.Module):
    def __init__(self, encoder, proj_dim=512, proj_hidden=512, lambd=5e-3):
        super().__init__()
        self.encoder, self.proj, self.lambd = encoder, mlp(encoder.embed_dim, proj_hidden, proj_dim), lambd

    def forward(self, v1, v2):
        z1, z2 = (self.proj(self.encoder(v)).float() for v in (v1, v2))
        z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
        z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)
        c = z1.T @ z2 / len(z1)
        d = torch.diagonal(c)
        return ((d - 1) ** 2).sum() + self.lambd * ((c ** 2).sum() - (d ** 2).sum())


class VICReg(nn.Module):
    def __init__(self, encoder, proj_dim=768, proj_hidden=768, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super().__init__()
        self.encoder, self.proj = encoder, mlp(encoder.embed_dim, proj_hidden, proj_dim)
        self.coeffs = (sim_coeff, std_coeff, cov_coeff)

    def forward(self, v1, v2):
        z1, z2 = (self.proj(self.encoder(v)).float() for v in (v1, v2))
        N, D = z1.shape
        z1c, z2c = z1 - z1.mean(0), z2 - z2.mean(0)
        std = 0.5 * sum(F.relu(1 - torch.sqrt(z.var(0) + 1e-4)).mean() for z in (z1c, z2c))
        off = ~torch.eye(D, dtype=torch.bool, device=z1.device)
        cov = sum(((z.T @ z) / (N - 1))[off].pow(2).sum() for z in (z1c, z2c)) / D
        a, b, c = self.coeffs
        return a * F.mse_loss(z1, z2) + b * std + c * cov


OBJECTIVES = {"moco": MoCo, "simclr": SimCLR, "byol": BYOL, "barlow": BarlowTwins, "vicreg": VICReg}


def build_objective(name, encoder, **kw):
    return OBJECTIVES[name](encoder, **kw)
