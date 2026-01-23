#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline complet de calibration itérative pour l'analyse spectrale LIGO

Ce script orchestre :
1. Téléchargement des données NPZ (si nécessaire)
2. Clustering des événements
3. Calibration itérative alternée H_STAR ↔ SCALE_EJ par cluster
4. Analyse finale avec calibration optimale
5. Génération de rapports détaillés

Usage:
    python run_iterative_calibration.py --refs ligo_refs.json --event-params event_params.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.optimize import minimize_scalar

# Import des modules locaux
import cluster_latent_kmeans as clk
import ligo_spectral_planck as lsp


def calibrate_hstar_fixed_scale(
    events: List[str],
    params: Dict,
    y_obs_dict: Dict[str, float],
    bands: lsp.Bands,
    scale_ej_fixed: float,
    **analyze_kwargs,
) -> float:
    """
    Calibre H_STAR en gardant SCALE_EJ fixé.
    
    Minimise: sum((y_obs - scale_ej * h * y_model(h))^2)
    où y_model(h) dépend de h via tau_hl_cal = h * tau_hl
    
    Returns:
        float: H_STAR optimal
    """
    print(f"\n  🔧 Calibration H_STAR (SCALE_EJ={scale_ej_fixed:.6e} fixé)...")
    
    # Collecte des données
    data = []
    for ev in events:
        if ev not in y_obs_dict:
            continue
        
        y_obs = float(y_obs_dict[ev])
        if not np.isfinite(y_obs) or y_obs <= 0:
            continue
        
        try:
            # Analyse avec H_STAR=1.0 pour référence
            result = lsp.analyze_event(
                event=ev,
                params=params,
                bands=bands,
                hstar_in=1.0,
                scale_ej_in=scale_ej_fixed,
                plot=False,
                return_internals=True,
                **analyze_kwargs,
            )
            
            y_model = float(result.get("energy_J", 0.0))
            if not np.isfinite(y_model) or y_model <= 0:
                continue
                
            data.append((ev, y_obs, y_model))
            
        except Exception as e:
            print(f"    [WARN] Skip {ev}: {e}")
            continue
    
    if len(data) < 2:
        print(f"    [WARN] Pas assez d'événements valides ({len(data)}), H_STAR=1.0")
        return 1.0
    
    evs = [d[0] for d in data]
    y_obs = np.array([d[1] for d in data])
    y_mod = np.array([d[2] for d in data])
    
    # Optimisation de H_STAR
    # Note: comme tau dépend linéairement de H, y_model(h) n'est pas exactement linéaire en h
    # On fait une optimisation 1D simple
    def loss(h: float) -> float:
        if h <= 0:
            return 1e10
        
        # Ré-analyse avec ce H_STAR
        y_pred = []
        for ev in evs:
            try:
                result = lsp.analyze_event(
                    event=ev,
                    params=params,
                    bands=bands,
                    hstar_in=h,
                    scale_ej_in=scale_ej_fixed,
                    plot=False,
                    return_internals=True,
                    **analyze_kwargs,
                )
                y_pred.append(float(result.get("energy_J", 0.0)))
            except Exception:
                y_pred.append(0.0)
        
        y_pred = np.array(y_pred)
        residuals = y_obs - y_pred
        return float(np.sum(residuals ** 2))
    
    # Approximation initiale: H ~ moyenne des ratios
    h_init = float(np.median(y_obs / y_mod))
    
    # Optimisation
    result = minimize_scalar(
        loss,
        bounds=(0.1 * h_init, 10.0 * h_init),
        method='bounded',
        options={'xatol': 1e-6, 'maxiter': 20}
    )
    
    h_opt = float(result.x)
    
    print(f"    ✅ H_STAR = {h_opt:.6e} (init={h_init:.6e}, loss={result.fun:.3e})")
    
    return h_opt


def calibrate_scale_fixed_hstar(
    events: List[str],
    params: Dict,
    y_obs_dict: Dict[str, float],
    bands: lsp.Bands,
    hstar_fixed: float,
    **analyze_kwargs,
) -> float:
    """
    Calibre SCALE_EJ en gardant H_STAR fixé.
    
    Pour H_STAR fixé, SCALE_EJ est linéaire: y_pred = scale * y_model
    Solution fermée: scale = (y_obs · y_model) / (y_model · y_model)
    
    Returns:
        float: SCALE_EJ optimal
    """
    print(f"\n  🔧 Calibration SCALE_EJ (H_STAR={hstar_fixed:.6e} fixé)...")
    
    data = []
    for ev in events:
        if ev not in y_obs_dict:
            continue
        
        y_obs = float(y_obs_dict[ev])
        if not np.isfinite(y_obs) or y_obs <= 0:
            continue
        
        try:
            result = lsp.analyze_event(
                event=ev,
                params=params,
                bands=bands,
                hstar_in=hstar_fixed,
                scale_ej_in=1.0,  # Calibrer à partir de 1.0
                plot=False,
                return_internals=True,
                **analyze_kwargs,
            )
            
            y_model = float(result.get("energy_J", 0.0))
            if not np.isfinite(y_model) or y_model <= 0:
                continue
                
            data.append((ev, y_obs, y_model))
            
        except Exception as e:
            print(f"    [WARN] Skip {ev}: {e}")
            continue
    
    if len(data) < 2:
        print(f"    [WARN] Pas assez d'événements valides ({len(data)}), SCALE_EJ=1.0")
        return 1.0
    
    y_obs = np.array([d[1] for d in data])
    y_mod = np.array([d[2] for d in data])
    
    # Solution fermée des moindres carrés
    num = float(np.dot(y_obs, y_mod))
    den = float(np.dot(y_mod, y_mod))
    
    if den <= 0 or not np.isfinite(num):
        print(f"    [WARN] Dénominateur invalide, SCALE_EJ=1.0")
        return 1.0
    
    scale_opt = num / den
    
    # Diagnostics
    y_pred = scale_opt * y_mod
    rel_errors = 100 * np.abs((y_pred - y_obs) / y_obs)
    mae = float(np.mean(rel_errors))
    
    print(f"    ✅ SCALE_EJ = {scale_opt:.6e} (MAE={mae:.2f}%)")
    
    return scale_opt


def iterative_calibration(
    cluster_id: int,
    events: List[str],
    params: Dict,
    y_obs_dict: Dict[str, float],
    bands: lsp.Bands,
    max_iter: int = 10,
    tol: float = 1e-4,
    **analyze_kwargs,
) -> Tuple[float, float, List[Dict]]:
    """
    Calibration itérative alternée H_STAR ↔ SCALE_EJ jusqu'à convergence.
    
    Args:
        cluster_id: ID du cluster
        events: Liste d'événements du cluster
        params: Paramètres des événements
        y_obs_dict: Énergies observées de référence
        bands: Bandes de fréquence
        max_iter: Nombre max d'itérations
        tol: Tolérance de convergence (changement relatif)
        **analyze_kwargs: Arguments pour analyze_event
    
    Returns:
        (hstar, scale_ej, history): Valeurs optimales et historique
    """
    print(f"\n{'='*70}")
    print(f"📊 CALIBRATION ITÉRATIVE CLUSTER {cluster_id} ({len(events)} events)")
    print(f"{'='*70}")
    
    # Filtrer événements valides
    valid_events = [e for e in events if e in y_obs_dict]
    if len(valid_events) < 2:
        print(f"⚠️  Cluster {cluster_id}: pas assez d'events avec refs ({len(valid_events)})")
        print(f"    -> calibration par défaut (H_STAR=1.0, SCALE_EJ=1.0)")
        return 1.0, 1.0, []
    
    print(f"Événements valides: {len(valid_events)}")
    print(f"Tolérance: {tol:.2e}, Max itérations: {max_iter}")
    
    # Initialisation
    h_star = 1.0
    scale_ej = 1.0
    history = []
    
    for iteration in range(max_iter):
        print(f"\n--- Itération {iteration + 1}/{max_iter} ---")
        
        # 1) Calibrer H_STAR (SCALE_EJ fixé)
        h_new = calibrate_hstar_fixed_scale(
            events=valid_events,
            params=params,
            y_obs_dict=y_obs_dict,
            bands=bands,
            scale_ej_fixed=scale_ej,
            **analyze_kwargs,
        )
        
        # 2) Calibrer SCALE_EJ (H_STAR fixé)
        scale_new = calibrate_scale_fixed_hstar(
            events=valid_events,
            params=params,
            y_obs_dict=y_obs_dict,
            bands=bands,
            hstar_fixed=h_new,
            **analyze_kwargs,
        )
        
        # 3) Vérifier convergence
        dh = abs(h_new - h_star) / max(abs(h_star), 1e-30)
        ds = abs(scale_new - scale_ej) / max(abs(scale_ej), 1e-30)
        
        history.append({
            'iteration': iteration + 1,
            'h_star': h_new,
            'scale_ej': scale_new,
            'K': h_new * scale_new,
            'delta_h': dh,
            'delta_s': ds,
        })
        
        print(f"\n  📈 H_STAR:   {h_star:.6e} → {h_new:.6e}  (Δ={dh:.2e})")
        print(f"  📈 SCALE_EJ: {scale_ej:.6e} → {scale_new:.6e}  (Δ={ds:.2e})")
        print(f"  📈 K=H×S:    {h_star*scale_ej:.6e} → {h_new*scale_new:.6e}")
        
        converged = (dh < tol) and (ds < tol)
        
        h_star = h_new
        scale_ej = scale_new
        
        if converged:
            print(f"\n✅ CONVERGENCE atteinte à l'itération {iteration + 1}")
            break
    else:
        print(f"\n⚠️  Max itérations atteint sans convergence complète")
    
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
    
    # Stats globales
    all_errors = [abs(e) for e in errors.values()]
    if all_errors:
        mae_global = np.mean(all_errors)
        med_global = np.median(all_errors)
        lines.append(f"📊 STATISTIQUES GLOBALES")
        lines.append(f"   MAE (tous événements) : {mae_global:.2f}%")
        lines.append(f"   Médiane              : {med_global:.2f}%")
        lines.append(f"   N événements         : {len(errors)}")
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
    
    # Write file
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"\n✅ Rapport écrit: {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Pipeline complet de calibration itérative LIGO"
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
    
    # Iterative calibration
    ap.add_argument("--max-iter", type=int, default=10, help="Max itérations calibration")
    ap.add_argument("--tol", type=float, default=1e-4, help="Tolérance convergence")
    
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
    
    # 4) Iterative calibration par cluster
    print("\n🔧 Calibration itérative par cluster...")
    
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
            y_obs_dict=y_obs_dict,
            bands=bands,
            max_iter=args.max_iter,
            tol=args.tol,
            **analyze_kwargs,
        )
        cluster_calib[cid] = (hstar, scale_ej)
        cluster_history[cid] = history
    
    # Save calibrations with history
    calib_data = {
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
            "max_iter": args.max_iter,
            "tol": args.tol,
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
    print("✅ CALIBRATION ITÉRATIVE TERMINÉE")
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
