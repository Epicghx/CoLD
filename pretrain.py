"""Two-view pretraining.

    python pretrain.py --config configs/pretrain_cold_vit.yaml
    python pretrain.py --config configs/pretrain_cold_vit.yaml --synthetic 512 --epochs 1
"""
import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from cold.data import NpyDir, Synthetic, pad_1024
from cold.models import build_encoder
from cold.models.objectives import build_objective
from cold.views import TwoViews


class PairDataset(Dataset):
    def __init__(self, base, views):
        self.base, self.views = base, views

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        v1, v2 = self.views(self.base[i])
        return pad_1024(v1), pad_1024(v2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--synthetic", type=int, default=0, help="use N random recordings instead of the corpus")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.batch_size:
        cfg["optim"]["batch_size"] = args.batch_size
    seed = cfg.get("seed", 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    base = Synthetic(args.synthetic) if args.synthetic else NpyDir(cfg["data"]["root"], cfg["data"].get("file_list"))
    opt = cfg["optim"]
    loader = DataLoader(PairDataset(base, TwoViews(**cfg["views"])), batch_size=opt["batch_size"], shuffle=True,
                        drop_last=True, num_workers=cfg.get("workers", 8), pin_memory=dev == "cuda")

    m = cfg["model"]
    model = build_objective(m["objective"], build_encoder(m["encoder"]), **m.get("objective_kwargs", {})).to(dev)

    lr0 = opt["blr"] * opt["batch_size"] / 256
    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim > 1]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim <= 1]
    optim = torch.optim.AdamW([{"params": decay, "weight_decay": opt["wd"]}, {"params": no_decay, "weight_decay": 0.0}],
                              lr=lr0, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(dev, enabled=dev == "cuda")

    epochs, warmup = args.epochs or opt["epochs"], opt.get("warmup", 10)
    out = Path(args.out) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    for ep in range(epochs):
        # epoch-level schedules: linear warm-up then cosine for the learning rate,
        # cosine from 0.99 to 0.999 for the momentum of MoCo / BYOL targets
        lr = lr0 * (ep + 1) / warmup if ep < warmup else 0.5 * lr0 * (1 + math.cos(math.pi * (ep - warmup) / max(1, epochs - warmup)))
        for g in optim.param_groups:
            g["lr"] = lr
        if hasattr(model, "momentum_range"):
            a, b = model.momentum_range
            model.m.fill_(a + 0.5 * (b - a) * (1 - math.cos(math.pi * ep / max(1, epochs - 1))))

        model.train()
        t0, total = time.time(), 0.0
        for v1, v2 in loader:
            v1, v2 = v1.to(dev, non_blocking=True), v2.to(dev, non_blocking=True)
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                loss = model(v1, v2)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            total += loss.item()
        print(f"epoch {ep + 1}/{epochs}  loss {total / len(loader):.4f}  lr {lr:.2e}  {time.time() - t0:.0f}s", flush=True)
        if (ep + 1) % cfg.get("save_every", 10) == 0 or ep + 1 == epochs:
            torch.save({"encoder": m["encoder"], "state_dict": model.encoder.state_dict(), "epoch": ep + 1, "config": cfg},
                       out / "encoder.pt")


if __name__ == "__main__":
    main()
