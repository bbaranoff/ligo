#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def trapz(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


def build_weights() -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
    return {
        "F=1": lambda nu: np.ones_like(nu),
        "F=nu": lambda nu: nu,
        "F=nu^2": lambda nu: nu ** 2,
        "F=log(nu)": lambda nu: np.log(np.maximum(nu, 1e-12)),
        "F=sqrt(nu)": lambda nu: np.sqrt(np.maximum(nu, 0.0)),
    }


def parse_bands(bands: List[str]) -> List[Tuple[float, float]]:
    out = []
    for b in bands:
        a, c = b.split(":")
        out.append((float(a), float(c)))
    return out


def select_band(freq: np.ndarray, dEdf: np.ndarray, fmin: float, fmax: float) -> Tuple[np.ndarray, np.ndarray]:
    m = (freq >= fmin) & (freq <= fmax)
    return freq[m], dEdf[m]


def compute_moments(freq: np.ndarray, dEdf: np.ndarray, weights, shape: bool) -> Dict[str, float]:
    out = {}
    if freq.size < 3:
        return out
    E0 = trapz(freq, dEdf)
    if shape and abs(E0) > 0:
        p = dEdf / E0
    else:
        p = None

    for label, F in weights.items():
        w = F(freq)
        out[f"rho_{label}"] = trapz(freq, w * dEdf)
        if shape:
            out[f"rhoS_{label}"] = trapz(freq, w * p) if p is not None else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Directory or glob (ex: results/ or 'results/GW*.json')")
    ap.add_argument("--glob", default="GW*.json", help="If path is a directory, glob inside it (default GW*.json)")
    ap.add_argument("--mode", choices=["real", "shuffle"], default="real",
                    help="real=use spectrum as is; shuffle=permute dEdf along freq")
    ap.add_argument("--shape", action="store_true", help="Also compute rhoS_* (shape-only)")
    ap.add_argument("--bands", nargs="*", default=None, help="Band cuts like '20:50 50:100 100:200'")
    ap.add_argument("--seed", type=int, default=1337, help="Seed for shuffle mode")
    ap.add_argument("--out", default="out_controls.csv", help="Output CSV")
    args = ap.parse_args()

    if os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, args.glob)))
    else:
        files = sorted(glob.glob(args.path))

    if not files:
        raise SystemExit("No JSON files found")

    weights = build_weights()
    rng = np.random.default_rng(args.seed)

    bands = parse_bands(args.bands) if args.bands else [(None, None)]

    # CSV fields
    base_fields = ["event", "mode", "band"]
    moment_fields = []
    for label in weights.keys():
        moment_fields.append(f"rho_{label}")
        if args.shape:
            moment_fields.append(f"rhoS_{label}")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=base_fields + moment_fields)
        wr.writeheader()

        for fp in files:
            d = load_json(fp)
            event = d.get("event") or os.path.basename(fp).replace(".json", "")
            freq = np.array(d.get("freq_Hz", []), dtype=float)
            dEdf = np.array(d.get("dEdf_internal", []), dtype=float)

            if freq.size == 0 or dEdf.size == 0 or freq.size != dEdf.size:
                continue

            # sort by freq if needed
            if not np.all(np.diff(freq) > 0):
                idx = np.argsort(freq)
                freq = freq[idx]
                dEdf = dEdf[idx]

            # optional shuffle (break structure)
            if args.mode == "shuffle":
                dEdf = dEdf.copy()
                rng.shuffle(dEdf)

            for (fmin, fmax) in bands:
                if fmin is None:
                    fb, eb = freq, dEdf
                    bname = "full"
                else:
                    fb, eb = select_band(freq, dEdf, fmin, fmax)
                    bname = f"{fmin:g}-{fmax:g}"

                moms = compute_moments(fb, eb, weights, shape=args.shape)
                row = {"event": event, "mode": args.mode, "band": bname}
                row.update(moms)
                wr.writerow(row)

    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
