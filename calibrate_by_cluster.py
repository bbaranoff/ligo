#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calibration cluster-by-cluster pour LIGO spectral analysis

Pipeline:
1. Charge les features depuis results/GW*.json
2. Fait le clustering (DBSCAN + KMeans)
3. Calibre H_STAR et SCALE_EJ **par cluster** (sauf cluster -1)
4. Ré-analyse tous les événements avec la calibration de leur cluster
5. Génère un rapport comparatif

Usage:
    python calibrate_by_cluster.py --refs ligo_refs.json --event-params event_params.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

# Import des modules locaux (assume que les scripts sont dans le même dossier)
import cluster_latent_kmeans as clk
import ligo_spectral_planck as lsp


def load_cluster_assignments(
    results_glob: str = "results/GW*.json",
    f_split: float = 150.0,
    db_eps: float = 1.4,
    db_min_samples: int = 3,
    k: int = 4,
    seed: int = 42,
) -> Tuple[Dict[str, int], List[clk.Features]]:
    """
    Fait le clustering et retourne un dict {event_name: cluster_id}
    """
    import glob
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, DBSCAN

    paths = sorted(glob.glob(results_glob))
    if not paths:
        raise RuntimeError(f"No files matched: {results_glob}")

    feats: List[clk.Features] = []
    for p in paths:
        ft = clk.compute_features_from_json(p, f_split=f_split)
        if ft is not None:
            feats.append(ft)

    if len(feats) < 2:
        raise RuntimeError(f"Not enough events (n={len(feats)})")

    order = ["logE", "nu_mean", "nu_peak", "nu_invf", "frac_bw", "Q_eff", "peak_rel", "R_LH"]
    X = np.array([f.as_row(order) for f in feats], dtype=float)

    # Impute NaN/inf
    X2 = X.copy()
    for j in range(X2.shape[1]):
        col = X2[:, j]
        m = np.isfinite(col)
        med = float(np.median(col[m])) if np.any(m) else 0.0
        col[~m] = med
        X2[:, j] = col

    # Standardize
    scaler = StandardScaler()
    Z = scaler.fit_transform(X2)

    # DBSCAN pour outliers
    db = DBSCAN(eps=float(db_eps), min_samples=int(db_min_samples))
    labels_db = db.fit_predict(Z)
    inlier_mask = labels_db != -1

    n_in = int(inlier_mask.sum())
    if n_in < max(2, int(k)):
        raise RuntimeError(
            f"Not enough inliers after DBSCAN for k={k} (inliers={n_in})"
        )

    # KMeans sur inliers
    Z_in = Z[inlier_mask]
    km = KMeans(n_clusters=int(k), random_state=int(seed), n_init="auto")
    labels_in = km.fit_predict(Z_in)

    # Labels finaux
    labels = np.full(Z.shape[0], -1, dtype=int)
    labels[inlier_mask] = labels_in

    # Mapping event -> cluster
    event_to_cluster = {ft.event: int(lab) for ft, lab in zip(feats, labels)}

    print(f"[clustering] n_events={len(feats)} inliers={n_in} outliers={len(feats)-n_in} k={k}")

    return event_to_cluster, feats


def calibrate_cluster(
    cluster_id: int,
    events: List[str],
    params: Dict,
    y_obs_dict: Dict[str, float],
    bands: lsp.Bands,
    **analyze_kwargs,
) -> Tuple[float, float]:
    """
    Calibre H_STAR et SCALE_EJ pour un cluster donné
    Retourne (hstar, scale_ej)
    """
    print(f"\n{'='*70}")
    print(f"📊 CALIBRATION CLUSTER {cluster_id} ({len(events)} events)")
    print(f"{'='*70}")

    # Filtre events valides (présents dans refs)
    valid_events = [e for e in events if e in y_obs_dict]
    if len(valid_events) < 2:
        print(f"[WARN] Cluster {cluster_id}: pas assez d'events avec refs ({len(valid_events)})")
        print(f"       -> using default calibration (H_STAR=1.0, SCALE_EJ=1.0)")
        return 1.0, 1.0

    try:
        hstar, scale_ej = lsp.calibrate_least_squares(
            events=valid_events,
            params=params,
            y_obs_dict=y_obs_dict,
            bands=bands,
            **analyze_kwargs,
        )
        print(f"✅ Cluster {cluster_id}: H_STAR={hstar:.6e}, SCALE_EJ={scale_ej:.6e}")
        return hstar, scale_ej
    except Exception as e:
        print(f"[ERROR] Cluster {cluster_id} calibration failed: {e}")
        print(f"        -> using default calibration")
        return 1.0, 1.0


def analyze_with_cluster_calibration(
    event: str,
    cluster_id: int,
    hstar: float,
    scale_ej: float,
    params: Dict,
    bands: lsp.Bands,
    **analyze_kwargs,
) -> Dict:
    """
    Analyse un événement avec la calibration de son cluster
    """
    try:
        result = lsp.analyze_event(
            event=event,
            params=params,
            bands=bands,
            hstar_in=hstar,
            scale_ej_in=scale_ej,
            plot=False,
            return_internals=False,
            **analyze_kwargs,
        )
        return result
    except Exception as e:
        print(f"[ERROR] Failed to analyze {event}: {e}")
        return None


def compute_errors(
    results: Dict[str, Dict],
    refs: Dict[str, Dict],
    ref_key: str = "energy_J",
) -> Dict[str, float]:
    """
    Calcule les erreurs relatives (%) pour chaque événement
    """
    errors = {}
    for event, res in results.items():
        if event not in refs:
            continue
        
        E_calc = res.get("E_internal")
        E_ref = refs[event].get(ref_key)
        
        if E_calc is None or E_ref is None:
            continue
        
        E_calc = float(E_calc)
        E_ref = float(E_ref)
        
        if E_ref > 0:
            error_pct = 100.0 * (E_calc - E_ref) / E_ref
            errors[event] = error_pct
    
    return errors


def write_report(
    out_path: str,
    event_to_cluster: Dict[str, int],
    cluster_calib: Dict[int, Tuple[float, float]],
    results: Dict[str, Dict],
    errors: Dict[str, float],
    refs: Dict[str, Dict],
) -> None:
    """
    Génère un rapport détaillé par cluster
    """
    # Group par cluster
    clusters = defaultdict(list)
    for event, cid in event_to_cluster.items():
        clusters[cid].append(event)
    
    lines = []
    lines.append("="*100)
    lines.append("CALIBRATION PAR CLUSTER - RAPPORT DÉTAILLÉ")
    lines.append("="*100)
    lines.append("")
    
    # Stats globales
    all_errors = [abs(e) for e in errors.values()]
    if all_errors:
        mae_global = np.mean(all_errors)
        med_global = np.median(all_errors)
        lines.append(f"📊 STATISTIQUES GLOBALES")
        lines.append(f"   MAE (all events) : {mae_global:.2f}%")
        lines.append(f"   Médiane         : {med_global:.2f}%")
        lines.append(f"   N événements    : {len(errors)}")
        lines.append("")
    
    # Par cluster
    for cid in sorted(clusters.keys()):
        events = sorted(clusters[cid])
        hstar, scale_ej = cluster_calib.get(cid, (1.0, 1.0))
        
        lines.append(f"\n{'='*100}")
        lines.append(f"CLUSTER {cid} ({len(events)} événements)")
        lines.append(f"{'='*100}")
        lines.append(f"Calibration: H_STAR = {hstar:.6e}, SCALE_EJ = {scale_ej:.6e}, K = {hstar*scale_ej:.6e}")
        lines.append("")
        
        # Stats du cluster
        cluster_errors = [abs(errors[e]) for e in events if e in errors]
        if cluster_errors:
            mae = np.mean(cluster_errors)
            med = np.median(cluster_errors)
            max_err = max(cluster_errors)
            min_err = min(cluster_errors)
            
            lines.append(f"📈 Statistiques:")
            lines.append(f"   MAE     : {mae:.2f}%")
            lines.append(f"   Médiane : {med:.2f}%")
            lines.append(f"   Min/Max : {min_err:.2f}% / {max_err:.2f}%")
            lines.append("")
        
        # Table des événements
        lines.append("Event                | E_calc [J]  | E_ref [J]   | Erreur (%) | M☉c² calc | M☉c² ref")
        lines.append("-" * 100)
        
        for event in events:
            res = results.get(event)
            ref = refs.get(event)
            err = errors.get(event)
            
            if res and ref:
                E_calc = res.get("E_total_J", 0.0)
                E_ref = ref.get("energy_J", 0.0)
                m_calc = res.get("m_sun", 0.0)
                m_ref = ref.get("msun_c2", 0.0)
                err_str = f"{err:+.2f}" if err is not None else "N/A"
                
                lines.append(
                    f"{event:20} | {E_calc:.3e} | {E_ref:.3e} | {err_str:>10} | "
                    f"{m_calc:>9.3f} | {m_ref:>8.3f}"
                )
            else:
                lines.append(f"{event:20} | [no data]")
        
        lines.append("")
    
    # Comparaison clusters
    lines.append(f"\n{'='*100}")
    lines.append("📊 COMPARAISON INTER-CLUSTERS")
    lines.append(f"{'='*100}")
    lines.append("")
    lines.append("Cluster | N events | MAE (%)  | Médiane (%) | H_STAR     | SCALE_EJ   | K=H×S")
    lines.append("-" * 100)
    
    cluster_stats = []
    for cid in sorted(clusters.keys()):
        events = clusters[cid]
        hstar, scale_ej = cluster_calib.get(cid, (1.0, 1.0))
        cluster_errors = [abs(errors[e]) for e in events if e in errors]
        
        if cluster_errors:
            mae = np.mean(cluster_errors)
            med = np.median(cluster_errors)
        else:
            mae = med = float('nan')
        
        cluster_stats.append({
            'cid': cid,
            'n': len(events),
            'mae': mae,
            'med': med,
            'hstar': hstar,
            'scale_ej': scale_ej,
        })
        
        lines.append(
            f"{cid:>7} | {len(events):>8} | {mae:>8.2f} | {med:>11.2f} | "
            f"{hstar:>10.6e} | {scale_ej:>10.6e} | {hstar*scale_ej:>10.6e}"
        )
    
    # Meilleurs/pires fits
    lines.append(f"\n{'='*100}")
    lines.append("🏆 TOP 10 MEILLEURS FITS")
    lines.append(f"{'='*100}")
    
    sorted_errors = sorted(errors.items(), key=lambda x: abs(x[1]))[:10]
    for i, (event, err) in enumerate(sorted_errors, 1):
        cid = event_to_cluster.get(event, -999)
        lines.append(f"{i:2}. {event:20} | Cluster {cid:2} | Erreur: {err:+7.2f}%")
    
    lines.append(f"\n{'='*100}")
    lines.append("💀 TOP 10 PIRES FITS")
    lines.append(f"{'='*100}")
    
    sorted_errors = sorted(errors.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    for i, (event, err) in enumerate(sorted_errors, 1):
        cid = event_to_cluster.get(event, -999)
        lines.append(f"{i:2}. {event:20} | Cluster {cid:2} | Erreur: {err:+7.2f}%")
    
    # Write file
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"\n✅ Rapport écrit: {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Calibration cluster-by-cluster pour LIGO spectral analysis"
    )
    
    # Files
    ap.add_argument("--refs", required=True, help="JSON refs LIGO (ligo_refs.json)")
    ap.add_argument("--event-params", default="event_params.json", help="JSON params events")
    ap.add_argument("--results-glob", default="results/GW*.json", help="Pattern des JSON results")
    
    # Clustering params
    ap.add_argument("--k", type=int, default=4, help="Nombre de clusters KMeans")
    ap.add_argument("--db-eps", type=float, default=1.4, help="DBSCAN eps")
    ap.add_argument("--db-min-samples", type=int, default=3, help="DBSCAN min_samples")
    ap.add_argument("--f-split", type=float, default=150.0, help="Split Hz pour R_LH")
    ap.add_argument("--seed", type=int, default=42)
    
    # Analysis params
    ap.add_argument("--flow", type=float, default=None)
    ap.add_argument("--fhigh", type=float, default=None)
    ap.add_argument("--tau-band", type=float, nargs=2, default=[35.0, 250.0])
    ap.add_argument("--nu-band", type=float, nargs=2, default=[30.0, 350.0])
    ap.add_argument("--no-virgo", action="store_true")
    ap.add_argument("--peak-quantile", type=float, default=99.5)
    
    # Output
    ap.add_argument("--out", default="calibration_by_cluster.txt", help="Rapport de sortie")
    ap.add_argument("--calib-json", default="cluster_calibrations.json", help="JSON calibrations")
    ap.add_argument("--ref-key", default="energy_J", choices=["energy_J", "msun_c2"])
    ap.add_argument("--exclude-cls", nargs="*", default=[], help="Classes à exclure")
    ap.add_argument("--exclude-cluster-minus1", action="store_true", help="Exclure cluster -1 de la calibration")
    
    args = ap.parse_args()
    
    # 1) Load refs
    print("📚 Chargement des références...")
    with open(args.refs, "r") as f:
        refs = json.load(f)
    
    exclude = set(args.exclude_cls)
    y_obs_dict = {}
    for ev, d in refs.items():
        if not isinstance(d, dict):
            continue
        if exclude and d.get("cls", "") in exclude:
            continue
        v = d.get(args.ref_key)
        if v is None or not np.isfinite(float(v)) or float(v) <= 0:
            continue
        y_obs_dict[ev] = float(v)
    
    print(f"   Références chargées: {len(refs)} total, {len(y_obs_dict)} valides")
    
    # 2) Load event params
    print("\n📋 Chargement des paramètres d'événements...")
    params = lsp.load_event_params(args.event_params)
    print(f"   Événements dans params: {len(params)}")
    
    # 3) Clustering
    print("\n🔍 Clustering des événements...")
    event_to_cluster, feats = load_cluster_assignments(
        results_glob=args.results_glob,
        f_split=args.f_split,
        db_eps=args.db_eps,
        db_min_samples=args.db_min_samples,
        k=args.k,
        seed=args.seed,
    )
    
    # Group par cluster
    clusters = defaultdict(list)
    for event, cid in event_to_cluster.items():
        clusters[cid].append(event)
    
    for cid in sorted(clusters.keys()):
        print(f"   Cluster {cid:2}: {len(clusters[cid])} événements")
    
    # 4) Calibrate each cluster
    print("\n🔧 Calibration par cluster...")
    
    bands = lsp.Bands(
        tau_band=(float(args.tau_band[0]), float(args.tau_band[1])),
        nu_band=(float(args.nu_band[0]), float(args.nu_band[1])),
    )
    
    analyze_kwargs = {
        'flow': args.flow,
        'fhigh': args.fhigh,
        'signal_win': None,
        'noise_pad': None,
        'distance_mpc': None,
        'use_virgo': not args.no_virgo,
        'peak_quantile': args.peak_quantile,
    }
    
    cluster_calib = {}
    
    for cid in sorted(clusters.keys()):
        # Skip cluster -1 si demandé
        if cid == -1 and args.exclude_cluster_minus1:
            print(f"\n⏭️  Cluster -1 (outliers): skipped (using default calibration)")
            cluster_calib[cid] = (1.0, 1.0)
            continue
        
        events = clusters[cid]
        hstar, scale_ej = calibrate_cluster(
            cluster_id=cid,
            events=events,
            params=params,
            y_obs_dict=y_obs_dict,
            bands=bands,
            **analyze_kwargs,
        )
        cluster_calib[cid] = (hstar, scale_ej)
    
    # Save calibrations
    calib_data = {
        str(cid): {"H_STAR": float(hstar), "SCALE_EJ": float(scale_ej)}
        for cid, (hstar, scale_ej) in cluster_calib.items()
    }
    with open(args.calib_json, "w") as f:
        json.dump(calib_data, f, indent=2)
    print(f"\n💾 Calibrations sauvegardées: {args.calib_json}")
    
    # 5) Re-analyze all events with cluster-specific calibration
    print("\n🔬 Ré-analyse avec calibration par cluster...")
    results = {}
    
    for event, cid in event_to_cluster.items():
        hstar, scale_ej = cluster_calib[cid]
        print(f"   {event} (cluster {cid})...", end=" ")
        
        res = analyze_with_cluster_calibration(
            event=event,
            cluster_id=cid,
            hstar=hstar,
            scale_ej=scale_ej,
            params=params,
            bands=bands,
            **analyze_kwargs,
        )
        
        if res:
            results[event] = res
            print("✅")
        else:
            print("❌")
    
    # 6) Compute errors
    print("\n📊 Calcul des erreurs...")
    errors = compute_errors(results, refs, ref_key=args.ref_key)
    print(f"   Erreurs calculées pour {len(errors)} événements")
    
    # 7) Write report
    print("\n📝 Génération du rapport...")
    write_report(
        out_path=args.out,
        event_to_cluster=event_to_cluster,
        cluster_calib=cluster_calib,
        results=results,
        errors=errors,
        refs=refs,
    )
    
    # 8) Summary
    print("\n" + "="*70)
    print("✅ CALIBRATION PAR CLUSTER TERMINÉE")
    print("="*70)
    all_errors = [abs(e) for e in errors.values()]
    if all_errors:
        print(f"MAE globale : {np.mean(all_errors):.2f}%")
        print(f"Médiane     : {np.median(all_errors):.2f}%")
    print(f"Rapport     : {args.out}")
    print(f"Calibs JSON : {args.calib_json}")
    print("="*70)


if __name__ == "__main__":
    main()
