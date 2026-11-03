"""Convert PTB-XL (PhysioNet, v1.0.3) to one array of 100-Hz recordings.

    python prepare_ptbxl.py --src /path/to/ptb-xl --dst data/ptbxl
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="data/ptbxl")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    db = pd.read_csv(src / "ptbxl_database.csv")
    X = np.stack([wfdb.rdsamp(str(src / f))[0].T for f in db.filename_lr]).astype(np.float32)   # (N, 12, 1000)
    np.save(dst / "X100.npy", X)
    for f in ("ptbxl_database.csv", "scp_statements.csv"):
        shutil.copy(src / f, dst / f)
    print(X.shape, "->", dst)


if __name__ == "__main__":
    main()
