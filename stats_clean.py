#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stats CLEAN : exclut le cluster -1 (outliers) des statistiques
Usage: python stats_clean.py [--clusters clusters.json]
"""

import json
import glob
import os
import math
import numpy as np
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REFS_PATH = os.path.join(HERE, "ligo_refs.json")
RESULTS_DIR = os.path.join(HERE, "results")
CLUSTERS_PATH = os.path.join(HERE, "clusters.json")


def load_refs(path):
    with open(path, "r") as f:
        return json.load(f)


def load_clusters(path):
    """Charge les assignments cluster depuis clusters.json"""
    if not os.path.exists(path):
        print(f"⚠️  Pas de fichier clusters: {path}")
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("event_to_cluster", {})


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


def print_comparison_stats(label, arr_all, arr_clean):
    """Affiche stats avec comparaison ALL vs CLEAN"""
    arr_all = finite(arr_all)
    arr_clean = finite(arr_clean)
    
    if len(arr_all) == 0:
        print(f"{label:<25}: n/a")
        return
    
    print(f"\n{label}:")
    print(f"  ALL   (n={len(arr_all):2d}) : "
          f"min={arr_all.min():.3g} | "
          f"median={np.median(arr_all):.3g} | "
          f"mean={arr_all.mean():.3g} | "
          f"max={arr_all.max():.3g}")
    
    if len(arr_clean) > 0:
        improvement = (arr_all.mean() - arr_clean.mean()) / arr_all.mean() * 100
        print(f"  CLEAN (n={len(arr_clean):2d}) : "
              f"min={arr_clean.min():.3g} | "
              f"median={np.median(arr_clean):.3g} | "
              f"mean={arr_clean.mean():.3g} | "
              f"max={arr_clean.max():.3g}")
        print(f"  → Amélioration: {improvement:+.1f}%")
    else:
        print(f"  CLEAN (n=0)  : aucun événement après filtrage")


def main():
    ap = argparse.ArgumentParser(description="Stats CLEAN (sans cluster -1)")
    ap.add_argument("--clusters", default=CLUSTERS_PATH, help="Fichier clusters.json")
    ap.add_argument("--refs", default=REFS_PATH, help="Fichier refs JSON")
    ap.add_argument("--results-dir", default=RESULTS_DIR, help="Dossier résultats")
    ap.add_argument("--exclude-clusters", nargs="*", type=int, default=[-1],
                    help="Clusters à exclure des stats CLEAN (défaut: -1)")
    ap.add_argument("--threshold", type=float, default=50.0,
                    help="Seuil pour événements problématiques (%)")
    args = ap.parse_args()

    refs = load_refs(args.refs)
    results = load_results(args.results_dir)
    event_to_cluster = load_clusters(args.clusters)

    print("=" * 100)
    print("📊 STATS LIGO — ALL vs CLEAN (sans outliers)")
    print("=" * 100)
    print(f"Refs chargées        : {len(refs)}")
    print(f"Résultats chargés    : {len(results)}")
    print(f"Clusters chargés     : {len(event_to_cluster)} événements")
    print(f"Clusters exclus      : {args.exclude_clusters}")
    print()

    # Stats clustering
    if event_to_cluster:
        from collections import Counter
        cluster_counts = Counter(event_to_cluster.values())
        print("Distribution des clusters:")
        for cid in sorted(cluster_counts.keys()):
            marker = "❌" if cid in args.exclude_clusters else "✅"
            print(f"  Cluster {cid:2} : {cluster_counts[cid]:2} événements {marker}")
        print()

    rows = []
    for r in results:
        ev = r.get("event")
        ref = refs.get(ev, {})
        cid = event_to_cluster.get(ev, None)
        
        row = {
            "event": ev,
            "cluster": cid,
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

    # Filtrage CLEAN
    rows_clean = [r for r in rows if r["cluster"] not in args.exclude_clusters]

    print(f"Événements TOTAL : {len(rows)}")
    print(f"Événements CLEAN : {len(rows_clean)} ({100*len(rows_clean)/len(rows):.1f}%)")
    print(f"Événements exclus: {len(rows) - len(rows_clean)}")

    # ------------------------
    # Erreurs - Comparaison ALL vs CLEAN
    # ------------------------
    print("\n" + "=" * 100)
    print("COMPARAISON : ALL vs CLEAN")
    print("=" * 100)

    err_m_all = finite([abs(r["err_m_pct"]) for r in rows])
    err_m_clean = finite([abs(r["err_m_pct"]) for r in rows_clean])
    
    err_E_all = finite([abs(r["err_E_pct"]) for r in rows])
    err_E_clean = finite([abs(r["err_E_pct"]) for r in rows_clean])

    print_comparison_stats("Erreur masse (%)", err_m_all, err_m_clean)
    print_comparison_stats("Erreur énergie (%)", err_E_all, err_E_clean)

    # ------------------------
    # Distribution des erreurs par cluster
    # ------------------------
    if event_to_cluster:
        print("\n" + "=" * 100)
        print("ERREURS PAR CLUSTER")
        print("=" * 100)
        
        from collections import defaultdict
        cluster_errors = defaultdict(list)
        
        for r in rows:
            if r["err_m_pct"] is not None and np.isfinite(r["err_m_pct"]):
                cid = r["cluster"]
                cluster_errors[cid].append(abs(r["err_m_pct"]))
        
        for cid in sorted(cluster_errors.keys(), key=lambda x: (x is None, x == -1, x)):
            errs = cluster_errors[cid]
            marker = "❌" if cid in args.exclude_clusters else "✅"
            label = f"Cluster {cid if cid is not None else 'None':2}"
            print(f"{label} {marker} : n={len(errs):2} | "
                  f"MAE={np.mean(errs):6.2f}% | "
                  f"median={np.median(errs):6.2f}% | "
                  f"max={np.max(errs):7.2f}%")

    # ------------------------
    # Événements problématiques
    # ------------------------
    THR = args.threshold
    
    print("\n" + "=" * 100)
    print(f"ÉVÉNEMENTS PROBLÉMATIQUES (|err masse| > {THR}%)")
    print("=" * 100)
    
    bad_all = [r for r in rows if r["err_m_pct"] is not None and abs(r["err_m_pct"]) > THR]
    bad_clean = [r for r in rows_clean if r["err_m_pct"] is not None and abs(r["err_m_pct"]) > THR]
    
    print(f"\nALL   : {len(bad_all)} événements")
    print(f"CLEAN : {len(bad_clean)} événements ({100*len(bad_clean)/len(bad_all) if bad_all else 0:.1f}% restants)")
    
    if bad_all:
        print(f"\nListe (ALL, triés par erreur décroissante):")
        for r in sorted(bad_all, key=lambda x: abs(x["err_m_pct"]), reverse=True):
            cid = r["cluster"]
            marker = "❌" if cid in args.exclude_clusters else "✅"
            print(
                f"{r['event']:20s} | "
                f"C={cid if cid is not None else '?':2} {marker} | "
                f"cls={r['cls']:>4} | "
                f"m_calc={r['m_calc']:.3g} | "
                f"m_ref={r['m_ref']:.3g} | "
                f"err={r['err_m_pct']:+7.2f}%"
            )

    # ------------------------
    # Top 25 meilleurs fits - Comparaison
    # ------------------------
    print("\n" + "=" * 100)
    print("TOP 25 MEILLEURS FITS — ALL vs CLEAN")
    print("=" * 100)
    
    good_all = [r for r in rows if r["err_m_pct"] is not None]
    good_clean = [r for r in rows_clean if r["err_m_pct"] is not None]
    
    print("\n--- TOP 25 (ALL) ---")
    for i, r in enumerate(sorted(good_all, key=lambda x: abs(x["err_m_pct"]))[:25], 1):
        cid = r["cluster"]
        marker = "❌" if cid in args.exclude_clusters else "✅"
        print(
            f"{i:2}. {r['event']:20s} | "
            f"err={r['err_m_pct']:+7.2f}% | "
            f"C={cid if cid is not None else '?':2} {marker}"
        )
    
    if good_clean:
        print("\n--- TOP 25 (CLEAN) ---")
        for i, r in enumerate(sorted(good_clean, key=lambda x: abs(x["err_m_pct"]))[:25], 1):
            cid = r["cluster"]
            marker = "✅"
            print(
                f"{i:2}. {r['event']:20s} | "
                f"err={r['err_m_pct']:+7.2f}% | "
                f"C={cid if cid is not None else '?':2} {marker}"
            )

    # ------------------------
    # Quantités physiques - Comparaison
    # ------------------------
    print("\n" + "=" * 100)
    print("QUANTITÉS PHYSIQUES")
    print("=" * 100)
    
    print_comparison_stats(
        "M☉c² (calc)", 
        [r["m_calc"] for r in rows],
        [r["m_calc"] for r in rows_clean]
    )
    
    print_comparison_stats(
        "Énergie J (calc)", 
        [r["E_calc"] for r in rows],
        [r["E_calc"] for r in rows_clean]
    )
    
    print_comparison_stats(
        "τ (s)", 
        [r["tau"] for r in rows],
        [r["tau"] for r in rows_clean]
    )
    
    print_comparison_stats(
        "ν_eff (Hz)", 
        [r["nu_eff"] for r in rows],
        [r["nu_eff"] for r in rows_clean]
    )

    # ------------------------
    # Résumé final
    # ------------------------
    print("\n" + "=" * 100)
    print("RÉSUMÉ")
    print("=" * 100)
    
    if len(err_m_all) > 0 and len(err_m_clean) > 0:
        improvement_mae = (err_m_all.mean() - err_m_clean.mean()) / err_m_all.mean() * 100
        improvement_median = (np.median(err_m_all) - np.median(err_m_clean)) / np.median(err_m_all) * 100
        
        print(f"Erreur masse (MAE)    : {err_m_all.mean():.2f}% → {err_m_clean.mean():.2f}% ({improvement_mae:+.1f}%)")
        print(f"Erreur masse (median) : {np.median(err_m_all):.2f}% → {np.median(err_m_clean):.2f}% ({improvement_median:+.1f}%)")
        print(f"Événements > {THR}%    : {len(bad_all)} → {len(bad_clean)} (-{len(bad_all)-len(bad_clean)})")
        print(f"Événements conservés  : {len(rows_clean)}/{len(rows)} ({100*len(rows_clean)/len(rows):.1f}%)")

    print("\n" + "=" * 100)
    print("Fin stats CLEAN")
    print("=" * 100)


if __name__ == "__main__":
    main()
