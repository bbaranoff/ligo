#!/usr/bin/env python3
import argparse
import csv
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np


def to_float(x: str) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return None


def pearson(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 3:
        return None
    xv = np.array(x, dtype=float)
    yv = np.array(y, dtype=float)
    if np.std(xv) == 0 or np.std(yv) == 0:
        return None
    return float(np.corrcoef(xv, yv)[0, 1])


def pick_pairs(rows: List[dict], xkey: str, ykey: str, log: bool) -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    for r in rows:
        x = to_float(r.get(xkey, ""))
        y = to_float(r.get(ykey, ""))
        if x is None or y is None:
            continue
        if log:
            ax, ay = abs(x), abs(y)
            if ax <= 0 or ay <= 0:
                continue
            x, y = math.log10(ax), math.log10(ay)
        xs.append(float(x))
        ys.append(float(y))
    return xs, ys


def bootstrap_ci(x: List[float], y: List[float], n: int, seed: int) -> Optional[Tuple[float, float, float]]:
    if len(x) < 5:
        return None
    rng = random.Random(seed)
    idxs = list(range(len(x)))
    rs = []
    for _ in range(n):
        sample = [rng.choice(idxs) for _ in idxs]
        xb = [x[i] for i in sample]
        yb = [y[i] for i in sample]
        r = pearson(xb, yb)
        if r is not None and not (math.isnan(r) or math.isinf(r)):
            rs.append(r)
    if len(rs) < 10:
        return None
    rs.sort()
    lo = rs[int(0.025 * len(rs))]
    hi = rs[int(0.975 * len(rs))]
    mid = float(np.median(rs))
    return mid, lo, hi


def permutation_pvalue(rows: List[dict], xkey: str, ykey: str, n: int, seed: int, log: bool) -> Optional[Tuple[float, float]]:
    # Permute Y across events (keeps marginal distributions, breaks association)
    xs, ys = pick_pairs(rows, xkey, ykey, log=log)
    if len(xs) < 5:
        return None
    r_obs = pearson(xs, ys)
    if r_obs is None:
        return None

    rng = random.Random(seed)
    ys_perm = ys[:]
    count = 0
    for _ in range(n):
        rng.shuffle(ys_perm)
        r = pearson(xs, ys_perm)
        if r is None:
            continue
        if abs(r) >= abs(r_obs):
            count += 1
    p = (count + 1) / (n + 1)  # add-one smoothing
    return r_obs, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="out.csv")
    ap.add_argument("--targets", nargs="+", default=["E_total_J", "m_sun", "tau_hl_s"],
                    help="Target columns to test against (default: E_total_J m_sun tau_hl_s)")
    ap.add_argument("--moments", nargs="+", default=None,
                    help="Moment columns to test (default: auto-detect rho_* and rhoS_*)")
    ap.add_argument("--log", action="store_true", help="Use log10(abs(.)) for X and Y when possible")
    ap.add_argument("--perm", type=int, default=5000, help="Permutation runs (default 5000)")
    ap.add_argument("--boot", type=int, default=2000, help="Bootstrap runs (default 2000)")
    ap.add_argument("--seed", type=int, default=1337, help="RNG seed")
    args = ap.parse_args()

    with open(args.csv_path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        rows = list(rd)

    if not rows:
        raise SystemExit("Empty CSV")

    # Auto-detect moments
    if args.moments is None:
        keys = rows[0].keys()
        moms = [k for k in keys if k.startswith("rho_") or k.startswith("rhoS_")]
        moms.sort()
    else:
        moms = args.moments

    print(f"Loaded rows: {len(rows)}")
    print(f"Moments: {len(moms)}")
    print(f"Targets: {args.targets}")
    print(f"Mode: {'log10(abs(.))' if args.log else 'linear'}\n")

    # Table header
    print(f"{'moment':<22} {'target':<10} {'n':>4} {'r':>9} {'p_perm':>10} {'boot_med':>10} {'boot_95%':>18}")
    print("-" * 80)

    for m in moms:
        for t in args.targets:
            xs, ys = pick_pairs(rows, m, t, log=args.log)
            n = len(xs)
            r = pearson(xs, ys)
            if r is None or n < 5:
                continue

            rp = permutation_pvalue(rows, m, t, n=args.perm, seed=args.seed, log=args.log)
            ci = bootstrap_ci(xs, ys, n=args.boot, seed=args.seed)

            p = rp[1] if rp else None
            boot_mid, lo, hi = ci if ci else (None, None, None)

            p_str = f"{p:10.4g}" if p is not None else f"{'':>10}"
            boot_str = f"{boot_mid:10.3f}" if boot_mid is not None else f"{'':>10}"
            ci_str = f"[{lo:+.3f},{hi:+.3f}]" if lo is not None else ""

            print(f"{m:<22} {t:<10} {n:4d} {r:+9.3f} {p_str} {boot_str} {ci_str:>18}")


if __name__ == "__main__":
    main()
