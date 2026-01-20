#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def trapz_integral(freq_hz: List[float], dEdf: List[float]) -> float:
    f = np.asarray(freq_hz, dtype=float)
    y = np.asarray(dEdf, dtype=float)

    if f.ndim != 1 or y.ndim != 1:
        raise ValueError("freq_Hz et dEdf_internal doivent être des listes 1D.")
    if len(f) != len(y):
        raise ValueError(f"Longueurs différentes: len(freq)={len(f)} vs len(dEdf)={len(y)}")
    if len(f) < 2:
        return float("nan")

    # tri si jamais ça arrive non trié
    idx = np.argsort(f)
    f = f[idx]
    y = y[idx]

    return float(np.trapz(y, f))


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def integrate_one(path: str) -> Dict[str, Any]:
    j = load_json(path)

    event = j.get("event") or os.path.splitext(os.path.basename(path))[0]

    freq = j.get("freq_Hz")
    dEdf_internal = j.get("dEdf_internal")

    if freq is None or dEdf_internal is None:
        raise KeyError(f"{path}: JSON ne contient pas freq_Hz ou dEdf_internal")

    E_internal_int = trapz_integral(freq, dEdf_internal)

    SCALE_EJ = safe_float(j.get("SCALE_EJ"))
    E_total_J_int = (E_internal_int * SCALE_EJ) if (SCALE_EJ is not None and np.isfinite(E_internal_int)) else float("nan")

    # Valeurs “référence” si présentes
    E_internal_ref = safe_float(j.get("E_internal"))
    E_total_J_ref  = safe_float(j.get("E_total_J"))
    nu_eff = None
    if isinstance(j.get("nu_eff"), dict):
        nu_eff = safe_float(j["nu_eff"].get("nu_eff_energy"))

    def relerr(a: Optional[float], b: Optional[float]) -> float:
        if a is None or b is None:
            return float("nan")
        if b == 0:
            return float("nan")
        return float((a - b) / b)

    return {
        "event": event,
        "json_path": path,

        "gps": safe_float(j.get("gps")),
        "distance_mpc": safe_float(j.get("distance_mpc")),
        "nu_eff_hz": nu_eff,

        "flow_Hz": safe_float(j.get("flow_Hz")),
        "fhigh_Hz": safe_float(j.get("fhigh_Hz")),
        "n_bins": len(freq),

        "SCALE_EJ": SCALE_EJ,

        "E_internal_int": E_internal_int,
        "E_total_J_int": E_total_J_int,

        "E_internal_ref": E_internal_ref,
        "E_total_J_ref": E_total_J_ref,

        "relerr_E_internal": relerr(E_internal_int, E_internal_ref),
        "relerr_E_total_J": relerr(E_total_J_int, E_total_J_ref),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Intègre le spectre (freq_Hz, dEdf_internal) dans chaque JSON et exporte CSV.")
    ap.add_argument("--glob", required=True, help='Pattern des JSON. Ex: "results/json/*.json"')
    ap.add_argument("--out", default="spectra_integrated.csv", help="CSV de sortie")
    ap.add_argument("--fail-fast", action="store_true", help="Stop au premier JSON cassé (sinon: skip + warning)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"[FATAL] Aucun fichier ne matche: {args.glob}")

    rows: List[Dict[str, Any]] = []
    bad: List[Tuple[str, str]] = []

    for p in paths:
        try:
            rows.append(integrate_one(p))
        except Exception as e:
            if args.fail_fast:
                raise
            bad.append((p, str(e)))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print(f"[OK] Integrated: {len(df)} JSON -> {args.out}")
    if bad:
        print(f"[WARN] {len(bad)} fichiers ignorés:")
        for p, msg in bad[:10]:
            print(f"  - {p}: {msg}")
        if len(bad) > 10:
            print("  ...")

    # mini résumé d’erreur (utile pour sanity check)
    if len(df) > 0:
        with np.errstate(invalid="ignore"):
            e_int = df["relerr_E_internal"].astype(float)
            e_tot = df["relerr_E_total_J"].astype(float)
        print("\n[STATS] relerr_E_internal  (median, p95):",
              np.nanmedian(e_int), np.nanpercentile(e_int, 95))
        print("[STATS] relerr_E_total_J   (median, p95):",
              np.nanmedian(e_tot), np.nanpercentile(e_tot, 95))


if __name__ == "__main__":
    main()
