#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check tau quantization and test invariants using n = |tau| * fs (dimensionless integer-ish).

Outputs:
- per-event: tau, nu, n=|tau|*fs, frac(n), invariants (n*nu, n/log(nu), n*log(nu))
- histogram of n (top counts)
- summary stats (mean/std/CV) for each invariant
- quick check: how often n is ~integer (within eps)

Usage:
  python3 tau_nu_quant.py results/ --glob "GW*.json" --fs 4096
  python3 tau_nu_quant.py results/ --glob "GW*.json" --fs 4096 --nu-key nu_eff.nu_eff_energy --tau-key tau_hl_s
  python3 tau_nu_quant.py results/GW*.json --fs 4096
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import Counter
from statistics import mean, median, pstdev

# -----------------------------
# Helpers
# -----------------------------
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_path(d: dict, path: str):
    cur = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur

def as_float(x):
    try:
        if x is None or isinstance(x, bool):
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None

def robust_stats(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return None
    mu = mean(xs)
    sd = pstdev(xs) if len(xs) > 1 else 0.0
    cv = (sd / abs(mu)) if mu != 0 else float("inf")
    return {
        "n": len(xs),
        "mean": mu,
        "median": median(xs),
        "std": sd,
        "cv": cv,
        "min": min(xs),
        "max": max(xs),
    }

def collect_files(inputs, glob_pat):
    files = []
    for inp in inputs:
        if os.path.isdir(inp):
            pat = glob_pat or "*.json"
            files += glob.glob(os.path.join(inp, pat))
        else:
            files += glob.glob(inp)
    files = sorted(set(files))
    files = [f for f in files if os.path.isfile(f) and f.lower().endswith(".json")]
    return files

def extract_event_name(path, j):
    for k in ("event", "event_id", "name"):
        v = j.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return os.path.splitext(os.path.basename(path))[0]

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="files/globs/dirs (e.g. results/ or results/GW*.json)")
    ap.add_argument("--glob", default=None, help="glob pattern inside directories (e.g. GW*.json)")
    ap.add_argument("--fs", type=float, default=4096.0, help="sample rate for tau grid (default 4096)")
    ap.add_argument("--tau-key", default="tau_hl_s", help="tau key path (default tau_hl_s)")
    ap.add_argument("--nu-key", default="nu_eff.nu_eff_energy", help="nu key path (default nu_eff.nu_eff_energy)")
    ap.add_argument("--eps-int", type=float, default=1e-6, help="tolerance for 'n is integer' (default 1e-6)")
    ap.add_argument("--take-abs-tau", action="store_true", help="use abs(tau) (recommended)")
    args = ap.parse_args()

    files = collect_files(args.inputs, args.glob)
    if not files:
        print("No JSON files found.")
        return

    rows = []
    n_hist = Counter()
    n_is_int = 0

    for fp in files:
        j = load_json(fp)
        ev = extract_event_name(fp, j)

        tau = as_float(get_path(j, args.tau_key))
        nu = as_float(get_path(j, args.nu_key))

        # fallbacks (if your JSON differs a bit)
        if tau is None:
            for k in ("tau_hl_s", "tau_s", "tau", "tau_sec"):
                tau = as_float(j.get(k))
                if tau is not None:
                    break

        if nu is None:
            # common fallbacks from your pipeline history
            for k in ("nu_eff.nu_eff_energy", "bw_all.mu_f", "bw_nu.mu_f", "nu_eff"):
                nu = as_float(get_path(j, k) if "." in k else j.get(k))
                if nu is not None:
                    break

        if tau is None or nu is None:
            continue

        tau0 = abs(tau) if args.take_abs_tau else tau
        n = tau0 * args.fs  # should be integer-ish if tau is snapped to 1/fs
        n_round = round(n)
        frac = abs(n - n_round)

        if frac <= args.eps_int:
            n_is_int += 1

        # histogram as integer-ish bucket
        n_hist[int(n_round)] += 1

        # invariants on n (dimensionless)
        inv_n_nu = n * nu
        inv_n_log = n / math.log(nu) if nu > 0 and math.log(nu) != 0 else None
        inv_n_logmul = n * math.log(nu) if nu > 0 else None

        rows.append({
            "event": ev,
            "tau": tau,
            "nu": nu,
            "n": n,
            "n_round": n_round,
            "frac": frac,
            "I=n*nu": inv_n_nu,
            "I=n/log(nu)": inv_n_log,
            "I=n*log(nu)": inv_n_logmul,
        })

    print(f"Loaded files: {len(files)}")
    print(f"Valid rows (tau & nu): {len(rows)}")
    if not rows:
        print("No valid rows. Check keys: --tau-key / --nu-key.")
        return

    # Quantization check
    print("\n=== Tau quantization check ===")
    print(f"fs = {args.fs} Hz  => dt = 1/fs = {1.0/args.fs:.9g} s")
    print(f"n = |tau| * fs  (should be ~integer if snapped)")
    print(f"integer-ish within eps={args.eps_int}: {n_is_int}/{len(rows)} = {100.0*n_is_int/len(rows):.1f}%")

    # Histogram
    print("\n=== Histogram of n_round (top 12) ===")
    for val, cnt in n_hist.most_common(12):
        print(f"n={val:4d} : {cnt}")

    # Summaries for invariants
    inv_keys = ["I=n*nu", "I=n/log(nu)", "I=n*log(nu)"]
    print("\n=== Invariant summaries (lower CV = more constant) ===")
    summaries = []
    for k in inv_keys:
        st = robust_stats([r.get(k) for r in rows])
        if st:
            summaries.append((k, st))
    summaries.sort(key=lambda t: t[1]["cv"])

    for k, st in summaries:
        print(
            f"{k:12s}  n={st['n']:2d}  mean={st['mean']:+.6e}  std={st['std']:.6e}  "
            f"CV={st['cv']:.4g}  min={st['min']:+.3e}  max={st['max']:+.3e}"
        )

    best_k, best_st = summaries[0]
    print(f"\nBest by CV: {best_k}  (CV={best_st['cv']:.4g})")

    # Per-event table (sorted by best invariant)
    print(f"\n=== Per-event (sorted by {best_k}) ===")
    rows_sorted = sorted(rows, key=lambda r: (r.get(best_k) is None, r.get(best_k)))
    for r in rows_sorted:
        print(
            f"{r['event']:16s}  tau={r['tau']:+.9g}  nu={r['nu']:.6f}  "
            f"n={r['n']:.6f}  frac={r['frac']:.3g}  {best_k}={r.get(best_k):+.6e}"
        )

if __name__ == "__main__":
    main()
