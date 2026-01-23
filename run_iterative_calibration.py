#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline complet de calibration par grille exhaustive pour l'analyse spectrale LIGO

Ce script orchestre :
1. Clustering des événements (via cluster_latent_kmeans.py)
2. Calibration par grille exhaustive (H_STAR × SCALE_EJ) par cluster
3. Analyse finale avec calibration optimale
4. Génération de rapports détaillés

Principe de calibration :
- Dans ligo_spectral_planck.py : energy_J = E_internal(H_STAR) × SCALE_EJ
- Pour chaque H_STAR ∈ [0.5, 2.5] par pas de 0.1 :
  * Calculer E_internal(H_STAR) pour tous les événements
  * Trouver le meilleur SCALE_EJ par moindres carrés
  * Calculer RSS = sum((E_ref - SCALE_EJ × E_internal)^2)
- Garder la combinaison (H_STAR, SCALE_EJ) qui minimise RSS

Usage:
    python run_iterative_calibration.py --refs ligo_refs.json --event-params event_params.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

# Import des modules locaux
import cluster_latent_kmeans as clk
import ligo_spectral_planck as lsp


def calibrate_hstar_and_scale_grid(
    events: List[str],
    params: Dict,
    E_ref_dict: Dict[str, float],
    bands: lsp.Bands,
    h_min: float = 0.5,
    h_max: float = 2.5,
    h_step: float = 0.1,
    **analyze_kwargs,
) -> Tuple[float, float]:
    """
    Calibre H_STAR et SCALE_EJ par recherche exhaustive sur grille.
    
    Pour chaque H_STAR dans [h_min, h_max] avec pas h_step :
      1. Calculer E_internal(H_STAR) pour tous les événements
      2. Trouver le meilleur SCALE_EJ par moindres carrés
      3. Calculer l'erreur RSS
    
    Retourner la combinaison (H_STAR, SCALE_EJ) qui minimise RSS.
    """
    print(f"  🔧 Calibration grille H_STAR ∈ [{h_min}, {h_max}] pas={h_step}...")
    
    # Filtrer événements valides
    valid_events = []
    E_ref_list = []
    
    for ev in events:
        if ev not in E_ref_dict:
            continue
        E_ref = float(E_ref_dict[ev])
        if not np.isfinite(E_ref) or E_ref <= 0:
            continue
        valid_events.append(ev)
        E_ref_list.append(E_ref)
    
    if len(valid_events) < 2:
        print(f"    [WARN] Pas assez d'événements valides ({len(valid_events)})")
        return 1.0, 1.0
    
    E_ref = np.array(E_ref_list)
    
    # Grille de valeurs H_STAR à tester
    h_values = np.arange(h_min, h_max + h_step/2, h_step)
    n_tests = len(h_values)
    
    print(f"    Testing {n_tests} valeurs de H_STAR...")
    
    best_h = 1.0
    best_scale = 1.0
    best_rss = float('inf')
    best_mae = float('inf')
    
    for i, h_star in enumerate(h_values):
        # 1) Calculer E_internal(h_star) pour tous les événements
        E_internal_list = []
        for ev in valid_events:
            try:
                result = lsp.analyze_event(
                    event=ev,
                    params=params,
                    bands=bands,
                    hstar_in=h_star,
                    scale_ej_in=1.0,
                    plot=False,
                    return_internals=True,
                    **analyze_kwargs,
                )
                E_int = float(result.get("E_internal", 0.0))
                E_internal_list.append(E_int if np.isfinite(E_int) and E_int > 0 else 0.0)
            except Exception:
                E_internal_list.append(0.0)
        
        E_internal = np.array(E_internal_list)
        
        # 2) Trouver le meilleur SCALE_EJ pour ce H_STAR (solution fermée)
        num = float(np.dot(E_ref, E_internal))
        den = float(np.dot(E_internal, E_internal))
        
        if den <= 0 or not np.isfinite(num):
            continue
        
        scale_ej = num / den
        
        # 3) Calculer l'erreur avec cette combinaison
        E_pred = scale_ej * E_internal
        residuals = E_ref - E_pred
        rss = float(np.sum(residuals ** 2))
        rel_errors = 100 * np.abs(residuals / E_ref)
        mae = float(np.mean(rel_errors))
        
        # 4) Garder si c'est le meilleur
        if rss < best_rss:
            best_rss = rss
            best_h = h_star
            best_scale = scale_ej
            best_mae = mae
        
        # Affichage de chaque test
        print(f"      [{i+1:2}/{n_tests}] H={h_star:.2f}, S={scale_ej:.3e}, MAE={mae:.2f}%, RSS={rss:.3e}")
    
    print(f"    ✅ Optimal: H_STAR={best_h:.2f}, SCALE_EJ={best_scale:.6e}, MAE={best_mae:.2f}%, RSS={best_rss:.3e}")
    
    return best_h, best_scale


def iterative_calibration(
    cluster_id: int,
    events: List[str],
    params: Dict,
    E_ref_dict: Dict[str, float],
    bands: lsp.Bands,
    max_iter: int = 10,
    tol: float = 1e-4,
    h_min: float = 0.5,
    h_max: float = 2.5,
    h_step: float = 0.1,
    **analyze_kwargs,
) -> Tuple[float, float, List[Dict]]:
    """
    Calibration par recherche exhaustive sur grille H_STAR.
    
    Pour chaque H_STAR testé, trouve le meilleur SCALE_EJ,
    puis garde la combinaison qui minimise l'erreur globale.
    """
    print(f"\n{'='*70}")
    print(f"📊 CALIBRATION GRILLE CLUSTER {cluster_id} ({len(events)} events)")
    print(f"{'='*70}")
    
    valid_events = [e for e in events if e in E_ref_dict]
    if len(valid_events) < 2:
        print(f"⚠️  Cluster {cluster_id}: pas assez d'events avec refs ({len(valid_events)})")
        print(f"    -> calibration par défaut (H_STAR=1.0, SCALE_EJ=1.0)")
        return 1.0, 1.0, []
    
    print(f"Événements valides: {len(valid_events)}")
    print(f"Recherche exhaustive: H_STAR ∈ [{h_min}, {h_max}], pas={h_step}")
    
    # Calibration directe par grille
    h_star, scale_ej = calibrate_hstar_and_scale_grid(
        events=valid_events,
        params=params,
        E_ref_dict=E_ref_dict,
        bands=bands,
        h_min=h_min,
        h_max=h_max,
        h_step=h_step,
        **analyze_kwargs,
    )
    
    history = [{
        'iteration': 1,
        'h_star': h_star,
        'scale_ej': scale_ej,
        'K': h_star * scale_ej,
        'delta_h': 0.0,
        'delta_s': 0.0,
    }]
    
    print(f"\n{'='*70}")
    print(f"📊 CALIBRATION FINALE CLUSTER {cluster_id}")
    print(f"{'='*70}")
    print(f"H_STAR   = {h_star:.6e}")
    print(f"SCALE_EJ = {scale_ej:.6e}")
    print(f"K = H×S  = {h_star * scale_ej:.6e}")
    print(f"{'='*70}")
    
    return h_star, scale_ej, history


def compute_errors(
    results: Dict[str, Dict],
    refs: Dict[str, Dict],
    ref_key: str = "energy_J",
) -> Dict[str, float]:
    """Calcule les erreurs relatives (%) pour chaque événement."""
    errors = {}
    for event, res in results.items():
        if event not in refs:
            continue
        
        E_calc = res.get("energy_J")
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
    cluster_history: Dict[int, List[Dict]],
    results: Dict[str, Dict],
    errors: Dict[str, float],
    refs: Dict[str, Dict],
) -> None:
    """Génère un rapport détaillé avec historique de convergence."""
    clusters = defaultdict(list)
    for event, cid in event_to_cluster.items():
        clusters[cid].append(event)
    
    lines = []
    lines.append("="*100)
    lines.append("CALIBRATION ITÉRATIVE PAR CLUSTER - RAPPORT DÉTAILLÉ")
    lines.append("="*100)
    lines.append("")
    
    # Stats globales (tous événements)
    all_errors = [abs(e) for e in errors.values()]
    if all_errors:
        mae_global = np.mean(all_errors)
        med_global = np.median(all_errors)
        lines.append(f"📊 STATISTIQUES GLOBALES (tous événements)")
        lines.append(f"   MAE                  : {mae_global:.2f}%")
        lines.append(f"   Médiane              : {med_global:.2f}%")
        lines.append(f"   N événements         : {len(errors)}")
        lines.append("")
    
    # Stats clean (sans cluster -1 ni clusters à 1 événement)
    clean_errors = []
    for event, err in errors.items():
        cid = event_to_cluster.get(event, -999)
        if cid == -1:
            continue
        if len(clusters[cid]) == 1:
            continue
        clean_errors.append(abs(err))
    
    if clean_errors:
        mae_clean = np.mean(clean_errors)
        med_clean = np.median(clean_errors)
        max_clean = max(clean_errors)
        min_clean = min(clean_errors)
        
        # Compte des événements par catégorie
        n_clean = len(clean_errors)
        n_outliers = sum(1 for e, cid in event_to_cluster.items() if cid == -1 and e in errors)
        n_single = sum(1 for e, cid in event_to_cluster.items() if len(clusters[cid]) == 1 and cid != -1 and e in errors)
        
        lines.append(f"✨ STATISTIQUES CLEAN (sans outliers ni clusters à 1 event)")
        lines.append(f"   MAE                  : {mae_clean:.2f}%")
        lines.append(f"   Médiane              : {med_clean:.2f}%")
        lines.append(f"   Min / Max            : {min_clean:.2f}% / {max_clean:.2f}%")
        lines.append(f"   N événements clean   : {n_clean}")
        lines.append(f"   N outliers (cl=-1)   : {n_outliers}")
        lines.append(f"   N single (cl=1 ev)   : {n_single}")
        lines.append("")
    
    # Par cluster
    for cid in sorted(clusters.keys()):
        events = sorted(clusters[cid])
        hstar, scale_ej = cluster_calib.get(cid, (1.0, 1.0))
        history = cluster_history.get(cid, [])
        
        lines.append(f"\n{'='*100}")
        lines.append(f"CLUSTER {cid} ({len(events)} événements)")
        lines.append(f"{'='*100}")
        lines.append(f"Calibration finale: H_STAR = {hstar:.6e}, SCALE_EJ = {scale_ej:.6e}, K = {hstar*scale_ej:.6e}")
        lines.append("")
        
        # Historique de convergence
        if history:
            lines.append(f"📈 HISTORIQUE DE CONVERGENCE:")
            lines.append(f"Iter | H_STAR      | SCALE_EJ    | K=H×S       | ΔH     | ΔS")
            lines.append("-" * 80)
            for h in history:
                lines.append(
                    f"{h['iteration']:4} | {h['h_star']:.6e} | {h['scale_ej']:.6e} | "
                    f"{h['K']:.6e} | {h['delta_h']:.2e} | {h['delta_s']:.2e}"
                )
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
                E_calc = res.get("energy_J", 0.0)
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
    lines.append("Cluster | N events | Iter | MAE (%)  | Médiane (%) | H_STAR     | SCALE_EJ   | K=H×S")
    lines.append("-" * 100)
    
    for cid in sorted(clusters.keys()):
        events = clusters[cid]
        hstar, scale_ej = cluster_calib.get(cid, (1.0, 1.0))
        history = cluster_history.get(cid, [])
        n_iter = len(history)
        cluster_errors = [abs(errors[e]) for e in events if e in errors]
        
        if cluster_errors:
            mae = np.mean(cluster_errors)
            med = np.median(cluster_errors)
        else:
            mae = med = float('nan')
        
        lines.append(
            f"{cid:>7} | {len(events):>8} | {n_iter:>4} | {mae:>8.2f} | {med:>11.2f} | "
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
    
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"\n✅ Rapport écrit: {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Pipeline complet de calibration itérative LIGO"
    )
    
    # Files
    ap.add_argument("--refs", required=True, help="JSON refs LIGO")
    ap.add_argument("--event-params", default="event_params.json", help="JSON params events")
    ap.add_argument("--results-glob", default="results/GW*.json", help="Pattern des JSON results")
    ap.add_argument("--signal-win", type=float, default=0.6,
                    help="Fenêtre signal (pass-through vers ligo_spectral_planck)")
    ap.add_argument("--noise-pad", type=float, default=800,
                    help="Padding bruit (pass-through vers ligo_spectral_planck)")

    # Clustering params
    ap.add_argument("--k", type=int, default=4, help="Nombre de clusters KMeans")
    ap.add_argument("--db-eps", type=float, default=1.4, help="DBSCAN eps")
    ap.add_argument("--db-min-samples", type=int, default=3, help="DBSCAN min_samples")
    ap.add_argument("--f-split", type=float, default=150.0, help="Split Hz pour R_LH")
    ap.add_argument("--seed", type=int, default=42)
    
    # Iterative calibration
    ap.add_argument("--h-min", type=float, default=0.5, help="H_STAR minimum grille")
    ap.add_argument("--h-max", type=float, default=2.5, help="H_STAR maximum grille")
    ap.add_argument("--h-step", type=float, default=0.1, help="Pas de la grille H_STAR")
    
    # Analysis params
    ap.add_argument("--flow", type=float, default=None)
    ap.add_argument("--fhigh", type=float, default=None)
    ap.add_argument("--tau-band", type=float, nargs=2, default=[35.0, 250.0])
    ap.add_argument("--nu-band", type=float, nargs=2, default=[30.0, 350.0])
    ap.add_argument("--no-virgo", action="store_true")
    ap.add_argument("--peak-quantile", type=float, default=99.5)
    
    # Output
    ap.add_argument("--out", default="calibration_iterative.txt", help="Rapport de sortie")
    ap.add_argument("--calib-json", default="cluster_calibrations_iterative.json", help="JSON calibrations")
    ap.add_argument("--ref-key", default="energy_J", choices=["energy_J", "msun_c2"])
    ap.add_argument("--exclude-cls", nargs="*", default=[], help="Classes à exclure")
    ap.add_argument("--exclude-cluster-minus1", action="store_true", help="Exclure cluster -1")
    
    args = ap.parse_args()
    
    # 1) Load refs
    print("📚 Chargement des références...")
    with open(args.refs, "r") as f:
        refs = json.load(f)
    
    exclude = set(args.exclude_cls)
    E_ref_dict = {}
    for ev, d in refs.items():
        if not isinstance(d, dict):
            continue
        if exclude and d.get("cls", "") in exclude:
            continue
        v = d.get(args.ref_key)
        if v is None or not np.isfinite(float(v)) or float(v) <= 0:
            continue
        E_ref_dict[ev] = float(v)
    
    print(f"   Références chargées: {len(refs)} total, {len(E_ref_dict)} valides")
    
    # 2) Load event params
    print("\n📋 Chargement des paramètres d'événements...")
    params = lsp.load_event_params(args.event_params)
    print(f"   Événements dans params: {len(params)}")
    
    # 3) Clustering
    print("\n🔍 Clustering des événements...")
    event_to_cluster, feats = clk.load_cluster_assignments(
        results_glob=args.results_glob,
        f_split=args.f_split,
        db_eps=args.db_eps,
        db_min_samples=args.db_min_samples,
        k=args.k,
        seed=args.seed,
    )
    
    clusters = defaultdict(list)
    for event, cid in event_to_cluster.items():
        clusters[cid].append(event)
    
    for cid in sorted(clusters.keys()):
        print(f"   Cluster {cid:2}: {len(clusters[cid])} événements")
    
    # 4) Grid calibration par cluster
    print("\n🔧 Calibration par grille exhaustive par cluster...")
    
    bands = lsp.Bands(
        tau_band=(float(args.tau_band[0]), float(args.tau_band[1])),
        nu_band=(float(args.nu_band[0]), float(args.nu_band[1])),
    )
    
    analyze_kwargs = {
        'flow': args.flow,
        'fhigh': args.fhigh,
        'signal_win': args.signal_win,
        'noise_pad': args.noise_pad,
        'distance_mpc': None,
        'use_virgo': not args.no_virgo,
        'peak_quantile': args.peak_quantile,
    }
    
    cluster_calib = {}
    cluster_history = {}
    
    for cid in sorted(clusters.keys()):
        if cid == -1 and args.exclude_cluster_minus1:
            print(f"\n⭐ Cluster -1 (outliers): skipped (using default calibration)")
            cluster_calib[cid] = (1.0, 1.0)
            cluster_history[cid] = []
            continue
        
        events = clusters[cid]
        hstar, scale_ej, history = iterative_calibration(
            cluster_id=cid,
            events=events,
            params=params,
            E_ref_dict=E_ref_dict,
            bands=bands,
            h_min=args.h_min,
            h_max=args.h_max,
            h_step=args.h_step,
            **analyze_kwargs,
        )
        cluster_calib[cid] = (hstar, scale_ej)
        cluster_history[cid] = history
    
    # Save calibrations with history
    calib_data = {
        "event_to_cluster": {
            event: int(cid) for event, cid in event_to_cluster.items()
        },
        "calibrations": {
            str(cid): {
                "H_STAR": float(hstar),
                "SCALE_EJ": float(scale_ej),
                "K": float(hstar * scale_ej),
                "n_iterations": len(cluster_history.get(cid, [])),
                "history": cluster_history.get(cid, []),
            }
            for cid, (hstar, scale_ej) in cluster_calib.items()
        },
        "params": {
            "h_min": args.h_min,
            "h_max": args.h_max,
            "h_step": args.h_step,
            "k_clusters": args.k,
        }
    }
    
    with open(args.calib_json, "w") as f:
        json.dump(calib_data, f, indent=2)
    print(f"\n💾 Calibrations sauvegardées: {args.calib_json}")
    
    # 5) Re-analyze all events with final calibration
    print("\n🔬 Ré-analyse avec calibration finale...")
    results = {}
    
    for event, cid in event_to_cluster.items():
        hstar, scale_ej = cluster_calib[cid]
        print(f"   {event} (cluster {cid}, H={hstar:.3e}, S={scale_ej:.3e})...", end=" ")
        
        try:
            res = lsp.analyze_event(
                event=event,
                params=params,
                bands=bands,
                hstar_in=hstar,
                scale_ej_in=scale_ej,
                plot=False,
                return_internals=False,
                **analyze_kwargs,
            )
            results[event] = res
            print("✅")
        except Exception as e:
            print(f"❌ ({e})")
    
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
        cluster_history=cluster_history,
        results=results,
        errors=errors,
        refs=refs,
    )
    
    # 8) Summary
    print("\n" + "="*70)
    print("✅ CALIBRATION PAR GRILLE EXHAUSTIVE TERMINÉE")
    print("="*70)
    all_errors = [abs(e) for e in errors.values()]
    if all_errors:
        print(f"MAE globale : {np.mean(all_errors):.2f}%")
        print(f"Médiane     : {np.median(all_errors):.2f}%")
    
    # Stats clean
    clean_errors = []
    for event, err in errors.items():
        cid = event_to_cluster.get(event, -999)
        if cid == -1:
            continue
        if len(clusters[cid]) == 1:
            continue
        clean_errors.append(abs(err))
    
    if clean_errors:
        print(f"\n✨ STATS CLEAN (sans outliers ni clusters 1-event):")
        print(f"MAE clean   : {np.mean(clean_errors):.2f}%")
        print(f"Médiane     : {np.median(clean_errors):.2f}%")
        print(f"N clean     : {len(clean_errors)}")
    
    print(f"\nRapport     : {args.out}")
    print(f"Calibs JSON : {args.calib_json}")
    print("="*70)


if __name__ == "__main__":
    main()
