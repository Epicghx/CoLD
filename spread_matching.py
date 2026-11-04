"""python spread_matching.py --ptbxl data/ptbxl"""
import argparse

import numpy as np

from cold.data import PTBXL
from cold.spread import bounds, pair_spread, target_spread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptbxl", required=True)
    ap.add_argument("--s_min", type=float, default=0.5)
    ap.add_argument("--s_max", type=float, default=2.0)
    ap.add_argument("--sigma_b", type=float, default=0.5)
    args = ap.parse_args()

    ds = PTBXL(args.ptbxl, range(1, 9))
    x = (np.asarray(ds.X[ds.idx], dtype=np.float32) - ds.mu) / ds.sd
    delta_b, delta_s = target_spread(x)
    ratio, sigma_b = bounds(delta_b, delta_s)
    spread_s, spread_b = pair_spread(0.0, 0.0, args.s_min, args.s_max, args.sigma_b)
    print(f"target spread       delta_s = {delta_s:.3f}   delta_b = {delta_b:.3f}")
    print(f"bounds              s_max/s_min >= {ratio:.2f}   sigma_b >= {sigma_b:.2f}")
    print(f"CoLD pair spread    log-SD {spread_s:.3f}   mean {spread_b:.3f}  "
          f"(gains [{args.s_min}, {args.s_max}], sigma_b {args.sigma_b})")


if __name__ == "__main__":
    main()
