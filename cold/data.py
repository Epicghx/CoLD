import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def pad_1024(x):
    """Zero-pad the time axis from 1000 to 1024 samples (ViT patches of 32)."""
    return torch.nn.functional.pad(x, (0, 1024 - x.shape[-1]))


class NpyDir(Dataset):
    """Pretraining corpus: one array per .npy file at 100 Hz, (12, T) or (T, 12).
    Recordings are centre-cropped or zero-padded to 1000 samples and standardised per lead."""

    def __init__(self, root, file_list=None):
        self.root = Path(root)
        if file_list:
            names = [n.strip() for n in Path(file_list).read_text().splitlines() if n.strip()]
            self.files = [n if n.endswith(".npy") else n + ".npy" for n in names]
        else:
            self.files = sorted(p.name for p in self.root.glob("*.npy"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x = np.load(self.root / self.files[i]).astype(np.float32)
        if x.shape[0] != 12:
            x = x.T
        T = x.shape[1]
        x = x[:, (T - 1000) // 2:(T - 1000) // 2 + 1000] if T > 1000 else np.pad(x, ((0, 0), (0, 1000 - T)))
        x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
        return torch.from_numpy(x)


class Synthetic(Dataset):
    """Random 12-lead signals for a quick check of the pipeline without data."""

    def __init__(self, n=512, n_classes=0, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, 12, 1000, generator=g)
        self.y = (torch.rand(n, n_classes, generator=g) < 0.2).float() if n_classes else None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i] if self.y is None else (self.x[i], self.y[i])


LABEL_SETS = ("all71", "superclass", "subclass", "form", "rhythm")


def ptbxl_labels(db, scp, label_set):
    codes = db.scp_codes.apply(ast.literal_eval)
    diag = scp[scp.diagnostic == 1]
    if label_set == "superclass":
        mapping, classes = diag.diagnostic_class.to_dict(), ["NORM", "MI", "STTC", "CD", "HYP"]
    elif label_set == "subclass":
        mapping = diag.diagnostic_subclass.to_dict()
        classes = sorted(set(mapping.values()))
    else:
        keep = {"all71": (scp.diagnostic == 1) | (scp.form == 1) | (scp.rhythm == 1),
                "form": scp.form == 1, "rhythm": scp.rhythm == 1}[label_set]
        classes = sorted(scp.index[keep])
        mapping = {c: c for c in classes}
    col = {c: j for j, c in enumerate(classes)}
    Y = np.zeros((len(db), len(classes)), dtype=np.float32)
    for i, d in enumerate(codes):
        for c in d:
            if mapping.get(c) in col:
                Y[i, col[mapping[c]]] = 1
    return Y, classes


class PTBXL(Dataset):
    """PTB-XL at 100 Hz from `prepare_ptbxl.py`. Folds 1-8 train, 9 validation, 10 test.
    Records without a statement of the label set are dropped. Each lead is
    normalised with the mean and SD of the training folds."""

    def __init__(self, root, folds, label_set="all71", stats=None):
        root = Path(root)
        db = pd.read_csv(root / "ptbxl_database.csv")
        scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
        self.X = np.load(root / "X100.npy", mmap_mode="r")
        self.Y, self.classes = ptbxl_labels(db, scp, label_set)
        has_label = self.Y.sum(1) > 0
        self.idx = np.flatnonzero(db.strat_fold.isin(list(folds)).to_numpy() & has_label)
        if stats is None:
            train = np.flatnonzero(db.strat_fold.isin(range(1, 9)).to_numpy() & has_label)
            x = np.asarray(self.X[train], dtype=np.float64)
            stats = (x.mean(axis=(0, 2))[:, None], x.std(axis=(0, 2))[:, None] + 1e-6)
        self.mu, self.sd = (np.asarray(a, dtype=np.float32) for a in stats)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = (np.asarray(self.X[j], dtype=np.float32) - self.mu) / self.sd
        return torch.from_numpy(x), torch.from_numpy(self.Y[j])
