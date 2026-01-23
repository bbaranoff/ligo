#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import glob
import os
import math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REFS_PATH = os.path.join(HERE, "ligo_refs.json")
RESULTS_DIR = os.path.join(HERE, "results")


def load_refs(path):
    with open(path, "r") as f:
        return json.load(f)


def load_results(results_dir):
    rows = []
    for fn in sorted(glob.glob(os.path.join(results_dir, "GW*.json"))):
        try:
            with open(fn, "r") as f:
                d = json.load(f)
            d["_file"] = os.path.basename(fn)
            rows.append(d)
        except Exception as e:
            print(f"[WARN] impossible de lire {fn}: {e}")
    return rows


def pct_err(x, ref):
    if ref is None or ref == 0 or not np.isfinite(ref):
        return np.nan
    return 100.0 * (x - ref) / ref


def finite(a):
    return np.array([x for x in a if x is not None and np.isfinite(x)])


def print_stats(label, arr):
    arr = finite(arr)
    if len(arr) == 0:
        print(f"{label:<25}: n/a")
        return
    print(
        f"{label:<25}: "
        f"N={len(arr):2d} | "
        f"min={arr.min():.3g} | "
        f"median={np.median(arr):.3g} | "
        f"mean={arr.mean():.3g} | "
        f"max={arr.max():.3g}"
    )


def main():
    refs = load_refs(REFS_PATH)
    results = load_results(RESULTS_DIR)

    print("=" * 100)
    print("📊 STATS LIGO — refs vs results")
    print("=" * 100)
    print(f"Refs chargées        : {len(refs)}")
    print(f"Résultats chargés    : {len(results)}")
    print()

    rows = []

    for r in results:
        ev = r.get("event")
        ref = refs.get(ev, {})
        row = {
            "event": ev,
            "cls": ref.get("cls"),
            "m_calc": r.get("m_sun") or r.get("msun_c2"),
            "m_ref": ref.get("msun_c2"),
            "E_calc": r.get("energy_J"),
            "E_ref": ref.get("energy_J"),
            "tau": r.get("tau_hl_s") or r.get("tau"),
            "nu_eff": (
                r.get("nu_eff", {}).get("nu_eff_energy")
                if isinstance(r.get("nu_eff"), dict)
                else r.get("nu_eff")
            ),
        }
        row["err_m_pct"] = pct_err(row["m_calc"], row["m_ref"])
        row["err_E_pct"] = pct_err(row["E_calc"], row["E_ref"])
        rows.append(row)

    # ------------------------
    # Stats globales
    # ------------------------
    print("\n--- Quantités (résultats bruts) ---")
    print_stats("M☉c² (calc)", [r["m_calc"] for r in rows])
    print_stats("M☉c² (ref)", [r["m_ref"] for r in rows])
    print_stats("Énergie J (calc)", [r["E_calc"] for r in rows])
    print_stats("τ (s)", [r["tau"] for r in rows])
    print_stats("ν_eff (Hz)", [r["nu_eff"] for r in rows])

    # ------------------------
    # Erreurs
    # ------------------------
    err_m = finite([abs(r["err_m_pct"]) for r in rows])
    err_E = finite([abs(r["err_E_pct"]) for r in rows])

    print("\n--- Erreurs relatives ---")
    print_stats("Erreur masse (%)", err_m)
    print_stats("Erreur énergie (%)", err_E)

    # ------------------------
    # Événements problématiques
    # ------------------------
    THR = 50.0
    bad = [r for r in rows if r["err_m_pct"] is not None and abs(r["err_m_pct"]) > THR]

    print("\n--- Événements problématiques (|err masse| > 50%) ---")
    if not bad:
        print("Aucun")
    else:
        for r in sorted(bad, key=lambda x: abs(x["err_m_pct"]), reverse=True):
            print(
                f"{r['event']:20s} | "
                f"cls={r['cls']:>4} | "
                f"m_calc={r['m_calc']:.3g} | "
                f"m_ref={r['m_ref']:.3g} | "
                f"err={r['err_m_pct']:+7.2f}%"
            )

    print("\n--- TOP 25 meilleurs fits (masse) ---")
    good = [r for r in rows if r["err_m_pct"] is not None]
    for r in sorted(good, key=lambda x: abs(x["err_m_pct"]))[:25]:
        print(
            f"{r['event']:20s} | "
            f"err={r['err_m_pct']:+6.2f}%"
        )

    print("\n" + "=" * 100)
    print("Fin stats")
    print("=" * 100)


if __name__ == "__main__":
    main()
