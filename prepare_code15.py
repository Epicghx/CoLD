"""python prepare_code15.py --src /path/to/code15 --dst data/code15"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import resample_poly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", default="data/code15")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    exams = pd.read_csv(src / "exams.csv")
    keep = set(exams.sort_values("exam_id").drop_duplicates("patient_id").exam_id)
    n = 0
    for part in sorted(src.glob("exams_part*.hdf5")):
        with h5py.File(part, "r") as f:
            ids, tracings = f["exam_id"][:], f["tracings"]
            for i, eid in enumerate(ids):
                if eid not in keep:
                    continue
                x = resample_poly(tracings[i], 1, 4, axis=0)[12:1012].T
                x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
                np.save(dst / f"{eid}.npy", x.astype(np.float32))
                n += 1
    print(n, "recordings ->", dst)


if __name__ == "__main__":
    main()
