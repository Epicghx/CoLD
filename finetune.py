"""Fine-tune a pretrained encoder on PTB-XL, keep the best validation macro-AUROC
checkpoint and evaluate it on the test fold.

    python finetune.py --config configs/finetune_ptbxl.yaml --ckpt runs/cold_vit/encoder.pt
    python finetune.py --config configs/finetune_ptbxl.yaml --ckpt none          # from scratch
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from cold.data import PTBXL, Synthetic, pad_1024
from cold.metrics import multilabel_metrics
from cold.models import build_encoder


class Classifier(nn.Module):
    def __init__(self, encoder, n_classes, dropout=0.3):
        super().__init__()
        self.encoder, self.dropout = encoder, nn.Dropout(dropout)
        self.head = nn.Linear(encoder.embed_dim, n_classes)

    def forward(self, x):
        return self.head(self.dropout(self.encoder(x)))


def param_groups(model, lr, wd, decay):
    """Layer-wise learning-rate decay: the head gets lr, earlier layers lr * decay^depth."""
    enc = model.encoder
    if hasattr(enc, "blocks"):
        stages = [[enc.patch_embed, enc.lead_pe, enc.cls_token]] + [[b] for b in enc.blocks]
        top = [model.head, enc.norm]
    else:
        stages = [[enc.stem], [enc.layer1], [enc.layer2], [enc.layer3], [enc.layer4]]
        top = [model.head]
    n = len(stages)
    groups = [{"params": [p for m in top for p in m.parameters()], "lr": lr}]
    for i, mods in enumerate(stages):
        ps = [m for m in mods if isinstance(m, nn.Parameter)] + [p for m in mods if isinstance(m, nn.Module) for p in m.parameters()]
        groups.append({"params": ps, "lr": lr * decay ** (n - i)})
    for g in groups:
        g["weight_decay"] = wd
    return groups


def time_shift(x, max_shift):
    out = torch.zeros_like(x)
    for i, s in enumerate(torch.randint(-max_shift, max_shift + 1, (len(x),)).tolist()):
        if s >= 0:
            out[i, :, s:] = x[i, :, :x.shape[-1] - s]
        else:
            out[i, :, :s] = x[i, :, -s:]
    return out


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        probs.append(torch.sigmoid(model(pad_1024(x).to(dev)).float()).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="encoder.pt from pretrain.py, or 'none'")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.batch_size:
        cfg["optim"]["batch_size"] = args.batch_size
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if args.synthetic:
        k = 20
        tr, va, te = (Synthetic(args.synthetic, k, s) for s in (1, 2, 3))
        Y_train = tr.y.numpy()
    else:
        d = cfg["data"]
        tr = PTBXL(d["root"], range(1, 9), d["label_set"])
        va = PTBXL(d["root"], [9], d["label_set"], (tr.mu, tr.sd))
        te = PTBXL(d["root"], [10], d["label_set"], (tr.mu, tr.sd))
        frac = d.get("train_frac", 1.0)
        if frac < 1:   # fixed subset shared by all models
            rng = np.random.RandomState(d.get("subset_seed", 1234))
            tr.idx = tr.idx[np.sort(rng.permutation(len(tr.idx))[:round(frac * len(tr.idx))])]
        k, Y_train = len(tr.classes), tr.Y[tr.idx]

    if args.ckpt == "none":
        encoder = build_encoder(cfg["encoder"])
    else:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        encoder = build_encoder(ck["encoder"])
        encoder.load_state_dict(ck["state_dict"])
    model = Classifier(encoder, k, cfg.get("dropout", 0.3)).to(dev)

    o = cfg["optim"]
    epochs = args.epochs or o["epochs"]
    bs, w = o["batch_size"], cfg.get("workers", 4)
    tr_dl = DataLoader(tr, bs, shuffle=True, drop_last=True, num_workers=w)
    va_dl, te_dl = DataLoader(va, bs, num_workers=w), DataLoader(te, bs, num_workers=w)
    groups = param_groups(model, o["blr"], o["wd"], o["layer_decay"])
    optim = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=[g["lr"] * o["max_lr_scale"] for g in groups],
                                                epochs=epochs, steps_per_epoch=len(tr_dl), pct_start=0.1)
    loss_fn = nn.BCEWithLogitsLoss()

    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        for x, y in tr_dl:
            x = pad_1024(time_shift(x, cfg.get("time_shift", 50))).to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                loss = loss_fn(model(x), y.to(dev))
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
        val = multilabel_metrics(*predict(model, va_dl, dev), Y_train)
        print(f"epoch {ep + 1}/{epochs}  val macro-AUROC {val['macro_auroc']:.4f}  macro-AUPRC {val['macro_auprc']:.4f}", flush=True)
        if val["macro_auroc"] > best:
            best, best_state = val["macro_auroc"], {n: t.detach().clone() for n, t in model.state_dict().items()}

    model.load_state_dict(best_state)
    probs, Y = predict(model, te_dl, dev)
    test = multilabel_metrics(probs, Y, Y_train)
    print("test", json.dumps({n: round(v, 4) for n, v in test.items()}))
    out = Path(args.out) / f"{cfg['name']}_{Path(args.ckpt).parent.name or 'scratch'}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "test_metrics.json").write_text(json.dumps(test, indent=2))
    np.save(out / "test_probs.npy", probs)
    torch.save(model.state_dict(), out / "classifier.pt")


if __name__ == "__main__":
    main()
