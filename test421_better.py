#!/usr/bin/env python3
# test421_better.py
#
# Usage:
#   python3 test421_better.py results/ --glob "GW*.json" --refs ligo_refs.json
#   python3 test421_better.py results/*.json --refs ligo_refs.json
#   python3 test421_better.py results/ --glob "GW*.json" --refs ligo_refs.json --exclude-cls BNS
#   python3 test421_better.py results/ --glob "GW*.json" --refs ligo_refs.json --csv out.csv
#   python3 test421_better.py results/ --glob "GW*.json" --refs ligo_refs.json --log
#
# Notes:
# - rho_*  = moments amplitude (∫ F(nu) dE/df dnu)
# - rhoS_* = moments shape-only (∫ F(nu) p(nu) dnu), p = (dE/df) / ∫(dE/df)

import argparse
import csv
import glob
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------
# Helpers
# ---------------------------

def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def trapz(x: np.ndarray, y: np.ndarray) -> float:
    # numpy >= 2.0: trapezoid replaces trapz
    return float(np.trapezoid(y, x))


def almost_constant(arr: np.ndarray, rel_eps: float = 1e-15) -> bool:
    # Scale-aware test: "is std effectively zero compared to typical magnitude?"
    s = float(np.std(arr))
    scale = max(1.0, float(np.mean(np.abs(arr))))
    return s < rel_eps * scale


def pearson(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 3:
        return None
    xv = np.array(x, dtype=float)
    yv = np.array(y, dtype=float)
    if almost_constant(xv) or almost_constant(yv):
        return None
    return float(np.corrcoef(xv, yv)[0, 1])


def log_abs(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    a = abs(v)
    if a <= 0.0:
        return None
    return math.log10(a)


def rel_err(x: Optional[float], ref: Optional[float]) -> Optional[float]:
    if x is None or ref is None:
        return None
    if ref == 0:
        return None
    return abs(x - ref) / abs(ref)


def fmt(x: Optional[float], kind: str = "g") -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return ""
    if kind == "pct":
        return f"{100.0 * x:.3f}"
    if kind == "E":
        return f"{x:.3E}"
    return f"{x:.6g}"


def build_weights() -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    # Candidate F(ν). Keep monotonic/simple first.
    return {
        "F=1": lambda nu: np.ones_like(nu),
        "F=nu": lambda nu: nu,
        "F=nu^2": lambda nu: nu ** 2,
        "F=log(nu)": lambda nu: np.log(np.maximum(nu, 1e-12)),
        "F=sqrt(nu)": lambda nu: np.sqrt(np.maximum(nu, 0.0)),
    }


# ---------------------------
# Data model
# ---------------------------

@dataclass
class EventRow:
    event: str
    path: str

    # from event JSON
    nu_eff_json: Optional[float] = None
    nu_eff_spec: Optional[float] = None
    nu_relerr: Optional[float] = None

    E_internal_json: Optional[float] = None
    E_internal_int: Optional[float] = None
    E_relerr: Optional[float] = None

    E_total_J: Optional[float] = None
    m_sun: Optional[float] = None
    tau_hl_s: Optional[float] = None

    # refs (ligo_refs.json)
    cls: Optional[str] = None
    nu_ref: Optional[float] = None
    tau_ref: Optional[float] = None
    msun_c2_ref: Optional[float] = None
    E_ref_J: Optional[float] = None

    # derived comparisons vs refs
    msun_relerr_ref: Optional[float] = None
    Etotal_relerr_ref: Optional[float] = None

    # moments
    rho: Dict[str, Optional[float]] = field(default_factory=dict)       # amplitude
    rhoS: Dict[str, Optional[float]] = field(default_factory=dict)      # shape


# ---------------------------
# Core computations
# ---------------------------

def compute_row(event_path: str,
                weights: Dict[str, Callable[[np.ndarray], np.ndarray]],
                refs: Optional[dict]) -> EventRow:

    d = load_json(event_path)
    name = d.get("event") or os.path.basename(event_path).replace(".json", "")

    row = EventRow(event=name, path=event_path)

    # Pull JSON values
    row.nu_eff_json = safe_float(d.get("nu_eff", {}).get("nu_eff_energy"))
    row.E_internal_json = safe_float(d.get("E_internal"))
    row.E_total_J = safe_float(d.get("E_total_J"))
    row.m_sun = safe_float(d.get("m_sun"))
    row.tau_hl_s = safe_float(d.get("tau_hl_s"))

    freq = np.array(d.get("freq_Hz", []), dtype=float)
    dEdf = np.array(d.get("dEdf_internal", []), dtype=float)

    # Guards
    if freq.size == 0 or dEdf.size == 0 or freq.size != dEdf.size:
        row.rho = {k: None for k in weights.keys()}
        row.rhoS = {k: None for k in weights.keys()}
    else:
        # Ensure monotonic freq (if not, sort)
        if not np.all(np.diff(freq) > 0):
            idx = np.argsort(freq)
            freq = freq[idx]
            dEdf = dEdf[idx]

        # Internal checks
        row.E_internal_int = trapz(freq, dEdf)
        row.E_relerr = rel_err(row.E_internal_int, row.E_internal_json)

        denom = trapz(freq, dEdf)
        if denom != 0.0:
            row.nu_eff_spec = trapz(freq, freq * dEdf) / denom
            row.nu_relerr = rel_err(row.nu_eff_spec, row.nu_eff_json)

        # Moments
        row.rho = {}
        row.rhoS = {}

        E0 = trapz(freq, dEdf)
        if abs(E0) > 0.0:
            p = dEdf / E0
        else:
            p = dEdf

        for label, F in weights.items():
            w = F(freq)
            row.rho[label] = trapz(freq, w * dEdf)   # amplitude
            row.rhoS[label] = trapz(freq, w * p)     # shape-only

    # Attach refs if provided
    if refs is not None and name in refs:
        r = refs[name]
        row.cls = r.get("cls")
        row.nu_ref = safe_float(r.get("nu_eff"))
        row.tau_ref = safe_float(r.get("tau"))
        row.msun_c2_ref = safe_float(r.get("msun_c2"))
        row.E_ref_J = safe_float(r.get("energy_J"))

        row.msun_relerr_ref = rel_err(row.m_sun, row.msun_c2_ref)
        row.Etotal_relerr_ref = rel_err(row.E_total_J, row.E_ref_J)

    return row


def collect_inputs(paths: List[str], glob_pat: Optional[str]) -> List[str]:
    inputs: List[str] = []

    # NEW: if no paths provided, treat --glob as a file glob
    if not paths and glob_pat:
        inputs.extend(sorted(glob.glob(glob_pat)))
        return inputs

    for p in paths:
        if os.path.isdir(p):
            pat = glob_pat or "*.json"
            inputs.extend(sorted(glob.glob(os.path.join(p, pat))))
        else:
            inputs.extend(sorted(glob.glob(p)))
    return inputs


# ---------------------------
# Output
# ---------------------------

def print_table(rows: List[EventRow], weights: Dict[str, Any], show_shape: bool) -> None:
    wlabels = list(weights.keys())

    base_cols = [
        ("event", 16),
        ("cls", 5),
        ("nu_eff", 8),
        ("nu_ref", 7),
        ("nu_rel%", 7),
        ("Eint_rel%", 8),
        ("E_total_J", 10),
        ("E_ref_J", 9),
        ("E_rel%", 7),
        ("m_sun", 7),
        ("msun_ref", 8),
        ("m_rel%", 7),
        ("tau", 8),
        ("tau_ref", 8),
    ]

    # header
    parts = []
    for h, w in base_cols:
        parts.append(f"{h:>{w}s}")
    for lbl in wlabels:
        parts.append(f"{lbl:>12s}")
        if show_shape:
            parts.append(f"{('S'+lbl[1:]):>12s}")  # F=nu -> S=nu etc.
    print(" | ".join(parts))
    print("-" * (len(" | ".join(parts))))

    for r in rows:
        line = []
        line.append(f"{r.event:>16s}")
        line.append(f"{(r.cls or ''):>5s}")
        line.append(f"{fmt(r.nu_eff_json):>8s}")
        line.append(f"{fmt(r.nu_ref):>7s}")
        line.append(f"{fmt(r.nu_relerr, 'pct'):>7s}")
        line.append(f"{fmt(r.E_relerr, 'pct'):>8s}")
        line.append(f"{fmt(r.E_total_J, 'E'):>10s}")
        line.append(f"{fmt(r.E_ref_J, 'E'):>9s}")
        line.append(f"{fmt(r.Etotal_relerr_ref, 'pct'):>7s}")
        line.append(f"{fmt(r.m_sun):>7s}")
        line.append(f"{fmt(r.msun_c2_ref):>8s}")
        line.append(f"{fmt(r.msun_relerr_ref, 'pct'):>7s}")
        line.append(f"{fmt(r.tau_hl_s):>8s}")
        line.append(f"{fmt(r.tau_ref):>8s}")

        for lbl in wlabels:
            line.append(f"{fmt(r.rho.get(lbl), 'E'):>12s}")
            if show_shape:
                line.append(f"{fmt(r.rhoS.get(lbl), 'g'):>12s}")  # shape is usually O(1..100s)

        print(" | ".join(line))


def write_csv(rows: List[EventRow], weights: Dict[str, Any], out_csv: str, show_shape: bool) -> None:
    wlabels = list(weights.keys())
    fields = [
        "event", "cls",
        "nu_eff_json", "nu_eff_spec", "nu_ref", "nu_relerr",
        "E_internal_json", "E_internal_int", "E_int_relerr",
        "E_total_J", "E_ref_J", "E_total_relerr_ref",
        "m_sun", "msun_c2_ref", "m_relerr_ref",
        "tau_hl_s", "tau_ref",
    ]
    for lbl in wlabels:
        fields.append(f"rho_{lbl}")
        if show_shape:
            fields.append(f"rhoS_{lbl}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            row = {
                "event": r.event,
                "cls": r.cls,
                "nu_eff_json": r.nu_eff_json,
                "nu_eff_spec": r.nu_eff_spec,
                "nu_ref": r.nu_ref,
                "nu_relerr": r.nu_relerr,
                "E_internal_json": r.E_internal_json,
                "E_internal_int": r.E_internal_int,
                "E_int_relerr": r.E_relerr,
                "E_total_J": r.E_total_J,
                "E_ref_J": r.E_ref_J,
                "E_total_relerr_ref": r.Etotal_relerr_ref,
                "m_sun": r.m_sun,
                "msun_c2_ref": r.msun_c2_ref,
                "m_relerr_ref": r.msun_relerr_ref,
                "tau_hl_s": r.tau_hl_s,
                "tau_ref": r.tau_ref,
            }
            for lbl in wlabels:
                row[f"rho_{lbl}"] = r.rho.get(lbl)
                if show_shape:
                    row[f"rhoS_{lbl}"] = r.rhoS.get(lbl)
            wr.writerow(row)

def ratio_stats(rows: List[EventRow]) -> None:
    ks = []
    for r in rows:
        if r.m_sun is None or r.E_total_J is None:
            continue
        if r.E_total_J == 0:
            continue
        ks.append(r.m_sun / r.E_total_J)  # k in (Msun)/(J)

    if len(ks) < 3:
        print("\n=== Mc2 ~ k * EJ : not enough points ===")
        return

    kv = np.array(ks, dtype=float)
    print("\n=== Mc2 ~ k * EJ (pipeline identity check) ===")
    print(f"n         = {len(kv)}")
    print(f"k_mean    = {np.mean(kv):.6E}  (Msun/J)")
    print(f"k_median  = {np.median(kv):.6E}  (Msun/J)")
    print(f"k_std     = {np.std(kv):.6E}")
    print(f"k_min     = {np.min(kv):.6E}")
    print(f"k_max     = {np.max(kv):.6E}")

def corr_mc2_ej(rows: List[EventRow], use_log: bool = False) -> None:
    xs = []
    ys = []
    for r in rows:
        x = r.E_total_J
        y = r.m_sun
        if x is None or y is None:
            continue
        if use_log:
            x = log_abs(x)
            y = log_abs(y)
            if x is None or y is None:
                continue
        xs.append(float(x))
        ys.append(float(y))

    c = pearson(xs, ys)
    tag = "log10(abs(.))" if use_log else "linear"
    if c is None:
        print(f"\n=== corr(Mc2, EJ) [{tag}] : n/a (n={len(xs)}) ===")
        return
    print(f"\n=== corr(Mc2, EJ) [{tag}] : r = {c:+.6f} (n={len(xs)}) ===")

def correlations(rows: List[EventRow],
                 weights: Dict[str, Any],
                 show_shape: bool,
                 use_log: bool) -> None:

    wlabels = list(weights.keys())

    # targets from your JSON + from refs (if present)
    targets = [
        ("E_total_J", lambda r: r.E_total_J),
        ("m_sun",     lambda r: r.m_sun),
        ("tau_hl_s",  lambda r: r.tau_hl_s),
        ("E_ref_J",   lambda r: r.E_ref_J),
        ("msun_c2_ref", lambda r: r.msun_c2_ref),
        ("tau_ref",   lambda r: r.tau_ref),
    ]

    def get_x(r: EventRow, lbl: str, shape: bool) -> Optional[float]:
        v = (r.rhoS.get(lbl) if shape else r.rho.get(lbl))
        if not use_log:
            return v
        return log_abs(v)

    def get_y(v: Optional[float]) -> Optional[float]:
        if not use_log:
            return v
        return log_abs(v)

    print("\n=== Correlations (Pearson) ===")
    if use_log:
        print("Mode: log10(abs(x)) pour X et Y (quand possible)")

    for lbl in wlabels:
        for shape in ([False, True] if show_shape else [False]):
            tag = ("rhoS" if shape else "rho")
            for tname, get_t in targets:
                xs: List[float] = []
                ys: List[float] = []
                for r in rows:
                    x0 = get_x(r, lbl, shape)
                    y0 = get_t(r)
                    y1 = get_y(y0)
                    if x0 is None or y1 is None:
                        continue
                    xs.append(float(x0))
                    ys.append(float(y1))
                corr = pearson(xs, ys)
                if corr is None:
                    continue
                print(f"{tag}:{lbl:>9s} vs {tname:>10s} : r = {corr:+.3f} (n={len(xs)})")


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="JSON files, globs, or a directory")
    ap.add_argument("--glob", default=None, help="Glob if a directory is provided (ex: 'GW*.json')")
    ap.add_argument("--refs", default=None, help="ligo_refs.json")
    ap.add_argument("--exclude-cls", default=None, help="Exclude class from refs (ex: BNS)")
    ap.add_argument("--shape", action="store_true", help="Print shape moments rhoS_* as extra columns")
    ap.add_argument("--log", action="store_true", help="Correlate on log10(abs(.)) when possible")
    ap.add_argument("--csv", default=None, help="Write CSV output")
    args = ap.parse_args()

    inputs = collect_inputs(args.paths, args.glob)
    if not inputs:
        raise SystemExit("No JSON files found (files/globs/dir + --glob).")

    refs = None
    if args.refs:
        refs = load_json(args.refs)

    weights = build_weights()

    rows: List[EventRow] = []
    for p in inputs:
        r = compute_row(p, weights, refs)

        # Filter by cls if requested and refs exist
        if args.exclude_cls and (r.cls == args.exclude_cls):
            continue

        rows.append(r)

    # Sort by event name
    rows.sort(key=lambda r: r.event)

    print_table(rows, weights, show_shape=args.shape)
    correlations(rows, weights, show_shape=args.shape, use_log=args.log)
    corr_mc2_ej(rows, use_log=False)
    corr_mc2_ej(rows, use_log=True)
    ratio_stats(rows)

    if args.csv:
        write_csv(rows, weights, args.csv, show_shape=args.shape)
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
