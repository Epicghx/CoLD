"""ViT-Tiny with per-lead patches of 32 samples and attention across leads."""
import math

import torch
import torch.nn as nn


def sincos(length, dim):
    pe = torch.zeros(length, dim)
    pos = torch.arange(length).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, patch_w, embed_dim):
        super().__init__()
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_w, stride=patch_w)

    def forward(self, x):
        B, L, T = x.shape
        return self.proj(x.reshape(B * L, 1, T)).transpose(1, 2)


class ViTEncoder(nn.Module):
    def __init__(self, n_leads=12, in_T=1024, patch_w=32, embed_dim=192, depth=12, heads=3, mlp_ratio=4.0):
        super().__init__()
        self.n_leads, self.Tp, self.embed_dim = n_leads, in_T // patch_w, embed_dim
        self.patch_embed = PatchEmbed(patch_w, embed_dim)
        self.lead_pe = nn.Parameter(torch.randn(1, n_leads, 1, embed_dim) * 0.02)
        self.register_buffer("tpos_pe", sincos(self.Tp, embed_dim).reshape(1, 1, self.Tp, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList([Block(embed_dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        B, L, _ = x.shape
        tok = self.patch_embed(x).reshape(B, L, self.Tp, self.embed_dim)
        tok = (tok + self.lead_pe + self.tpos_pe).reshape(B, L * self.Tp, self.embed_dim)
        tok = torch.cat([self.cls_token.expand(B, -1, -1), tok], dim=1)
        for blk in self.blocks:
            tok = blk(tok)
        return self.norm(tok)[:, 1:].mean(dim=1)
