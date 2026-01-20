#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
from statistics import mean, median, pstdev

def get_path(d, path):
    """Get nested key like 'a.b.c'. Returns None if missing."""
    if not path:
        return None
    cur = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur

def as_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return None
        return float(x)
    except Exception:
        return None

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def robust_stats(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if len(xs) == 0:
        return None
    mu = mean(xs)
    med = median(xs)
    sd = pstdev(xs) if len(xs) > 1 else 0.0
    cv = (sd / abs(mu)) if mu != 0 else float("inf")
    return {"n": len(xs), "mean": mu, "median": med, "std": sd, "cv": cv, "min": min(xs), "max": max(xs)}

def main():
    ap = argparse.ArgumentParser(description="Compute tau-nu invariants from GW*.json")
    ap.add_argument("inputs", nargs="*", help="files or directories")
    ap.add_argument("--glob", default=None, help="glob pattern (applied inside any directory input)")
    ap.add_argument("--tau-key", default="tau_hl_s", help="tau key path (default: tau_hl_s)")
    ap.add_argument("--nu-key", default=None, help="nu key path (e.g. nu_eff.nu_eff_energy)")
    ap.add_argument("--exclude-cls", default=None, help="exclude if refs cls matches (e.g. BNS) – informational only here")
    ap.add_argument("--take-abs-tau", action="store_true", help="use abs(tau) for invariants (often sensible)")
    args = ap.parse_args()

    # Collect files
    files = []
    if args.inputs:
        for inp in args.inputs:
            if os.path.isdir(inp):
                if args.glob:
                    files += glob.glob(os.path.join(inp, args.glob))
                else:
                    files += glob.glob(os.path.join(inp, "*.json"))
            else:
                files.append(inp)
    else:
        # If no inputs provided, use current dir
        if args.glob:
            files = glob.glob(args.glob)
        else:
            files = glob.glob("*.json")

    files = sorted(set(files))
    if not files:
        print("No JSON files found (inputs/glob).")
        return

    # Default nu key fallbacks
    nu_candidates = []
    if args.nu_key:
        nu_candidates.append(args.nu_key)
    nu_candidates += [
        "nu_eff.nu_eff_energy",
        "nu_eff.nu_eff_energy_raw",
        "bw_all.mu_f",
        "bw_nu.mu_f",
        "nu_eff",  # if already a float in some dumps
    ]

    rows = []
    for p in files:
        try:
            j = load_json(p)
        except Exception as e:
            print(f"[skip] {p}: {e}")
            continue

        ev = j.get("event") or os.path.splitext(os.path.basename(p))[0]

        tau = as_float(get_path(j, args.tau_key))
        nu = None
        nu_used = None
        for cand in nu_candidates:
            v = as_float(get_path(j, cand))
            if v is not None:
                nu = v
                nu_used = cand
                break

        if tau is None or nu is None:
            continue

        tau0 = abs(tau) if args.take_abs_tau else tau

        # invariants to test (you can add more)
        inv = {
            "I=tau*nu": tau0 * nu,
            "I=abs(tau)*nu": abs(tau) * nu,
            "I=tau*nu^2": tau0 * (nu ** 2),
            "I=tau*sqrt(nu)": tau0 * math.sqrt(nu) if nu > 0 else None,
            "I=tau*log(nu)": tau0 * math.log(nu) if nu > 0 else None,
            "I=tau/log(nu)": (tau0 / math.log(nu)) if nu > 0 and math.log(nu) != 0 else None,
        }

        rows.append({
            "event": ev,
            "tau": tau,
            "nu": nu,
            "nu_key_used": nu_used,
            **inv
        })

    print(f"Loaded files: {len(files)}")
    print(f"Valid rows (tau & nu): {len(rows)}")
    if not rows:
        print("No valid rows. Check --tau-key / --nu-key and the JSON structure.")
        return

    # Summarize each invariant
    keys = [k for k in rows[0].keys() if k.startswith("I=")]
    summaries = []
    for k in keys:
        xs = [r.get(k) for r in rows]
        st = robust_stats(xs)
        if st:
            summaries.append((k, st))

    # Rank by CV (smaller = more constant)
    summaries.sort(key=lambda t: t[1]["cv"])

    print("\n=== Invariant summary (ranked by CV = std/|mean|) ===")
    for k, st in summaries:
        print(f"{k:14s}  n={st['n']:2d}  mean={st['mean']:+.6e}  std={st['std']:.6e}  CV={st['cv']:.4g}  "
              f"min={st['min']:+.3e}  max={st['max']:+.3e}")

    best_k, best_st = summaries[0]
    print(f"\nBest (most constant by CV): {best_k}  (CV={best_st['cv']:.4g})")

    # Show per-event for the best invariant
    print(f"\n=== Per-event values for best invariant: {best_k} ===")
    for r in rows:
        print(f"{r['event']:16s}  tau={r['tau']:+.6e}  nu={r['nu']:.6f}  {best_k}={r[best_k]:+.6e}  nu_key={r['nu_key_used']}")

if __name__ == "__main__":
    main()
