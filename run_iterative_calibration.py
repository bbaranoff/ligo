#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calibration itérative avec PEAK fixe:
1. SCALE_EJ par moindres carrés → TAU déduit de K=10
2. Optimiser NU
3. Re-calculer SCALE_EJ → nouveau TAU
4. Itérer jusqu'à convergence
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

import cluster_latent_kmeans as clk
from ligo_spectral_gpu_batch import analyze_events_batch_gpu


# GPU version si disponible
try:
    import ligo_spectral_gpu as lsp
    print("[INFO] Utilisation de ligo_spectral_gpu - Mode GPU")
    GPU_MODE = True
except ImportError:
    import ligo_spectral_planck as lsp
    print("[INFO] Utilisation de ligo_spectral_planck - Mode CPU")
    GPU_MODE = False

def filter_analyze_kwargs(kwargs: Dict) -> Dict:
    """
    Filtre les kwargs passés à analyze_event / analyze_events_batch_gpu
    pour éviter les arguments non supportés côté GPU.
    """
    if not kwargs:
        return {}

    out = dict(kwargs)

    # Incompatibilités connues GPU
    out.pop("plot", None)
    out.pop("return_internals", None)
    out.pop("verbose", None)  # déjà passé explicitement

    return out


def compute_scale_ej_optimal(
    events: List[str],
    params: Dict,
    M_ref: np.ndarray,
    bands: lsp.Bands,
    peak_scale: float,
    tau_scale: float,
    window_scale: float,
    **analyze_kwargs,
) -> float:
    """Calcule SCALE_EJ optimal par moindres carrés avec paramètres fixés"""
    
    M_internal_list = []
    results = analyze_events_batch_gpu(
        events=events,
        params=params,
        batch_size=32,        # 8–16 sur RTX 4090
        bands=bands,
        peak_scale=peak_scale,
        tau_scale=tau_scale,
        window_scale=window_scale,
        scale_ej_in=1.0,     # normalisé comme avant
        verbose=False,
        **analyze_kwargs,
    )

    M_internal_list = []
    for r in results:
        if r is None:
            M_internal_list.append(0.0)
            continue

        E_int = r.get("E_internal", 0.0)
        M_int = E_int / (lsp.M_sun * lsp.c * lsp.c)
        M_internal_list.append(M_int if np.isfinite(M_int) and M_int > 0 else 0.0)

    M_internal = np.array(M_internal_list)
    valid_mask = (M_internal > 0) & (M_ref > 0)

    if np.sum(valid_mask) < 2:
        return 1.0

    scale_ej = (
        np.sum(M_ref[valid_mask] * M_internal[valid_mask])
        / np.sum(M_internal[valid_mask] ** 2)
    )

    return float(scale_ej) if np.isfinite(scale_ej) and scale_ej > 0 else 1.0

def calibrate_iterative(
    events: List[str],
    params: Dict,
    M_ref_dict: Dict[str, float],
    bands: lsp.Bands,
    peak_scale: float = 1.0,
    K_target: float = 10.0,
    nu_min: float = 0.5,
    nu_max: float = 1.5,
    nu_step: float = 0.1,
    max_iter: int = 10,
    tol: float = 0.01,  # Plus relax: 1%
    **analyze_kwargs,
) -> Tuple[float, float, float]:
    """
    Calibration itérative:
    - PEAK fixe
    - Alterne: SCALE_EJ (moindres carrés) → TAU (K=10) → optimiser NU
    
    Returns:
        (best_tau, best_nu, best_scale)
    """
    print(f"  🔧 Calibration itérative (PEAK={peak_scale:.2f} fixe, K={K_target}):")
    
    # Filtrer événements
    valid_events, M_ref_list = [], []
    for ev in events:
        if ev not in M_ref_dict:
            continue
        M_ref = float(M_ref_dict[ev])
        if np.isfinite(M_ref) and M_ref > 0:
            valid_events.append(ev)
            M_ref_list.append(M_ref)
    
    if len(valid_events) < 2:
        return 1.0, 1.0, 1.0
    
    M_ref = np.array(M_ref_list)
    nu_values = np.arange(nu_min, nu_max + nu_step/2, nu_step)
    
    # Initialisation
    window_scale = 1.0
    scale_ej = 1.0
    tau_scale = 1.0
    best_mae_global = float('inf')
    best_nu_global = 1.0
    best_tau_global = 1.0
    best_scale_global = 1.0
    no_improvement_count = 0
    
    print(f"     NU ∈ [{nu_min}, {nu_max}] step {nu_step} → {len(nu_values)} valeurs")
    print(f"     Max iterations: {max_iter}, tol: {tol}")
    
    for iteration in range(max_iter):
        print(f"\n  🔄 Itération {iteration+1}/{max_iter}:")
        
        # ÉTAPE 1: SCALE_EJ par moindres carrés (avec NU actuel)
        scale_ej_new = compute_scale_ej_optimal(
            valid_events, params, M_ref, bands,
            peak_scale, tau_scale, window_scale,
            **analyze_kwargs
        )
        
        # ÉTAPE 2: TAU déduit de K = PEAK² × TAU × SCALE
        tau_scale_new = K_target / (peak_scale**2 * scale_ej_new)
        
        print(f"     SCALE_EJ = {scale_ej_new:.6e} → TAU = {tau_scale_new:.6f}")
        
        # ÉTAPE 3: Optimiser NU
        # ÉTAPE 3: Optimiser NU (SUR E_internal, PAS msun_c2)
        best_nu = window_scale
        best_mae = float('inf')

        E_ref = M_ref * (lsp.M_sun * lsp.c * lsp.c)

        from ligo_spectral_gpu_batch import analyze_events_batch_gpu
        akw = filter_analyze_kwargs(analyze_kwargs)

        for nu_test in nu_values:
            results = analyze_events_batch_gpu(
                events=valid_events,
                params=params,
                batch_size=32,
                bands=bands,
                peak_scale=peak_scale,
                tau_scale=tau_scale_new,
                window_scale=nu_test,
                scale_ej_in=scale_ej_new,
                verbose=False,
                **akw,
            )

            E_pred_list = []
            for r in results:
                if r is None:
                    E_pred_list.append(0.0)
                else:
                    E_pred_list.append(r.get("E_internal", 0.0))

            E_pred = np.array(E_pred_list)

            valid_mask = (E_pred > 0) & (E_ref > 0)
            if np.sum(valid_mask) < 2:
                continue

            mae = float(
                np.mean(
                    np.abs((E_ref[valid_mask] - E_pred[valid_mask]) / E_ref[valid_mask])
                ) * 100
            )

            if np.isfinite(mae) and mae < best_mae:
                best_mae = mae
                best_nu = nu_test

        print(f"     NU optimal = {best_nu:.6f} (MAE={best_mae:.2f}%)")
        if not np.isfinite(best_mae):
            print("     ⚠️  MAE non finie → skip optimisation NU")
            best_nu = window_scale
            best_mae = float('inf')

        # Garder meilleur GLOBAL
        if np.isfinite(best_mae) and best_mae < best_mae_global:
            best_mae_global = best_mae
            best_nu_global = best_nu
            best_tau_global = tau_scale_new
            best_scale_global = scale_ej_new
            print(f"     ⭐ Nouveau meilleur global!")
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= 3:
                print(f"     🛑 Arrêt: pas d'amélioration depuis 3 itérations")
                break
        
        # Convergence
        delta_nu = abs(best_nu - window_scale)
        delta_tau = abs(tau_scale_new - tau_scale) / tau_scale if tau_scale > 0 else 0
        delta_scale = abs(scale_ej_new - scale_ej) / scale_ej if scale_ej > 0 else 0
        delta_mae = abs(best_mae - best_mae_global) / best_mae_global if best_mae_global > 0 else 0
        
        window_scale = best_nu
        tau_scale = tau_scale_new
        scale_ej = scale_ej_new
        
        # Convergence plus relax
        if delta_nu < 0.01 and delta_tau < tol and delta_scale < tol:
            print(f"     ✅ Convergence (Δ < {tol})")
            break
    
    # Utiliser meilleur GLOBAL
    K_final = peak_scale**2 * best_tau_global * best_scale_global
    
    print(f"\n    ✅ MEILLEUR GLOBAL:")
    print(f"       PEAK  = {peak_scale:.6f} (fixe)")
    print(f"       TAU   = {best_tau_global:.6f}")
    print(f"       NU    = {best_nu_global:.6f}")
    print(f"       SCALE = {best_scale_global:.6e}")
    print(f"       K     = {K_final:.6f}")
    print(f"       MAE   = {best_mae_global:.2f}%")
    
    return best_tau_global, best_nu_global, best_scale_global


def iterative_calibration(
    cluster_id: int,
    events: List[str],
    params: Dict,
    M_ref_dict: Dict[str, float],
    bands: lsp.Bands,
    peak_scale: float = 1.0,
    K_target: float = 10.0,
    nu_min: float = 0.5,
    nu_max: float = 1.5,
    nu_step: float = 0.1,
    max_iter: int = 10,
    **analyze_kwargs,
) -> Tuple[float, float, float, List[Dict]]:
    """Wrapper pour un cluster"""
    print(f"\n{'='*70}")
    print(f"📊 CALIBRATION CLUSTER {cluster_id} ({len(events)} events)")
    print(f"{'='*70}")
    
    best_tau, best_nu, best_scale = calibrate_iterative(
        events=events,
        params=params,
        M_ref_dict=M_ref_dict,
        bands=bands,
        peak_scale=peak_scale,
        K_target=K_target,
        nu_min=nu_min,
        nu_max=nu_max,
        nu_step=nu_step,
        max_iter=max_iter,
        **analyze_kwargs,
    )
    
    history = [{'tau': best_tau, 'nu': best_nu, 'scale': best_scale}]
    
    K_final = peak_scale**2 * best_tau * best_scale
    
    print(f"\n{'='*70}")
    print(f"📊 CALIBRATION FINALE CLUSTER {cluster_id}")
    print(f"{'='*70}")
    print(f"PEAK_SCALE = {peak_scale:.6e}")
    print(f"TAU_SCALE  = {best_tau:.6e}")
    print(f"NU_SCALE   = {best_nu:.6e}")
    print(f"SCALE_EJ   = {best_scale:.6e}")
    print(f"K = PEAK² × TAU × SCALE = {K_final:.6e}")
    print(f"{'='*70}")
    
    return best_tau, best_nu, best_scale, history


def write_report(calibrations, event_to_cluster, errors, out_path):
    lines = ["="*100, "RAPPORT CALIBRATION ITÉRATIVE", "="*100]
    lines.append(f"\n{'Cluster':>7} | {'N':>4} | {'MAE%':>6} | {'TAU':>8} | {'NU':>8} | {'SCALE':>10} | {'K':>8}")
    lines.append("-"*100)
    
    for cid, cal in sorted(calibrations.items()):
        events = [ev for ev, c in event_to_cluster.items() if c == cid]
        errs = [errors[ev] for ev in events if ev in errors]
        mae = np.mean(np.abs(errs)) if errs else float('nan')
        
        tau = cal.get('tau_scale', 1.0)
        nu = cal.get('window_scale', 1.0)
        scale = cal.get('scale_ej', 1.0)
        peak = cal.get('peak_scale', 1.0)
        K = peak**2 * tau * scale
        
        lines.append(f"{cid:>7} | {len(events):>4} | {mae:>6.2f} | {tau:>8.4f} | {nu:>8.4f} | {scale:>10.4e} | {K:>8.2f}")
    
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"\n✅ Rapport: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    
    # Files
    ap.add_argument("--refs", required=True)
    ap.add_argument("--event-params", default="event_params.json")
    ap.add_argument("--results-glob", default="results/GW*.json")
    ap.add_argument("--signal-win", type=float, default=0.6)
    ap.add_argument("--noise-pad", type=float, default=800)

    # Clustering
    ap.add_argument("--clusters", type=str)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--method", default="hdbscan+kmeans")
    ap.add_argument("--min-cluster-size", type=int, default=5)
    ap.add_argument("--min-samples", type=int, default=3)
    ap.add_argument("--use-logE", action="store_true")
    ap.add_argument("--f-split", type=float, default=150.0)
    ap.add_argument("--seed", type=int, default=42)

    # Calibration
    ap.add_argument("--peak-scale", type=float, default=1.0, help="PEAK fixe")
    ap.add_argument("--k-target", type=float, default=10.0)
    ap.add_argument("--nu-min", type=float, default=0.5)
    ap.add_argument("--nu-max", type=float, default=1.5)
    ap.add_argument("--nu-step", type=float, default=0.1)
    ap.add_argument("--max-iter", type=int, default=10)
    
    # Analysis
    ap.add_argument("--flow", type=float, default=None)
    ap.add_argument("--fhigh", type=float, default=None)
    ap.add_argument("--tau-band", type=float, nargs=2, default=[35.0, 250.0])
    ap.add_argument("--nu-band", type=float, nargs=2, default=[30.0, 350.0])
    ap.add_argument("--no-virgo", action="store_true")
    ap.add_argument("--peak-quantile", type=float, default=99.5)
    
    # Output
    ap.add_argument("--out", default="calibration_iterative.txt")
    ap.add_argument("--calib-json", default="cluster_calibrations_iterative.json")
    ap.add_argument("--exclude-cluster-minus1", action="store_true")
    
    args = ap.parse_args()
    
    # Load
    print("📚 Chargement...")
    with open(args.refs) as f:
        refs = json.load(f)
    
    M_ref_dict = {ev: d["msun_c2"] for ev, d in refs.items() if isinstance(d, dict) and "msun_c2" in d}
    print(f"   {len(M_ref_dict)} références")
    
    with open(args.event_params) as f:
        params = json.load(f)
    
    # Clustering
    if args.clusters and os.path.exists(args.clusters):
        with open(args.clusters) as f:
            event_to_cluster = json.load(f).get('event_to_cluster', {})
    else:
        event_to_cluster, _ = clk.load_cluster_assignments(
            results_glob=args.results_glob, method=args.method, k=args.k,
            min_cluster_size=args.min_cluster_size, min_samples=args.min_samples,
            use_logE=args.use_logE, f_split=args.f_split, seed=args.seed,
        )
    
    clusters = defaultdict(list)
    for ev, cid in event_to_cluster.items():
        clusters[cid].append(ev)
    
    print(f"   {len(clusters)} clusters")
    
    bands = lsp.Bands(tau_band=tuple(args.tau_band), nu_band=tuple(args.nu_band))
    exclude = {-1} if args.exclude_cluster_minus1 else set()
    
    # Calibration
    calibrations, errors = {}, {}
    
    for cid in sorted(clusters.keys()):
        if cid in exclude:
            continue
        
        best_tau, best_nu, best_scale, history = iterative_calibration(
            cluster_id=cid,
            events=clusters[cid],
            params=params,
            M_ref_dict=M_ref_dict,
            bands=bands,
            peak_scale=args.peak_scale,
            K_target=args.k_target,
            nu_min=args.nu_min,
            nu_max=args.nu_max,
            nu_step=args.nu_step,
            max_iter=args.max_iter,
            flow=args.flow, fhigh=args.fhigh,
            signal_win=args.signal_win, noise_pad=args.noise_pad,
            peak_quantile=args.peak_quantile,
        )
        
        calibrations[cid] = {
            'peak_scale': args.peak_scale,
            'tau_scale': best_tau,
            'window_scale': best_nu,
            'scale_ej': best_scale,
            'history': history,
        }
        
        # Erreurs
        for ev in clusters[cid]:
            if ev not in M_ref_dict:
                continue
            try:
                result = lsp.analyze_event(
                    event=ev, params=params, bands=bands,
                    peak_scale=args.peak_scale, tau_scale=best_tau,
                    window_scale=best_nu, scale_ej_in=best_scale,
                    flow=args.flow, fhigh=args.fhigh,
                    signal_win=args.signal_win, noise_pad=args.noise_pad,
                    peak_quantile=args.peak_quantile,
                    plot=False, return_internals=False, verbose=False,
                )
                M_pred = result.get("msun_c2", 0)
                errors[ev] = (M_pred - M_ref_dict[ev]) / M_ref_dict[ev] * 100
            except:
                pass
    
    # Save
    with open(args.calib_json, "w") as f:
        json.dump(calibrations, f, indent=2)
    
    print(f"\n💾 {args.calib_json}")
    
    write_report(calibrations, event_to_cluster, errors, args.out)
    
    print("\n✨ TERMINÉ !")


if __name__ == '__main__':
    main()
