"""LIGO Energy Estimation Benchmark via Pierre de Rosette.

Importe l'orchestrateur comme librairie, applique tous les paths physiquement
motivés à tous les événements, calibre par classe LSQ, compare à la référence
LIGO (E = ΔM·c²) et au baseline du repo (~22% MAE).

Pipeline :
1. Charger refs (22 events LIGO connus, classés ULTRA_LIGHT/LIGHT/MEDIUM/MASSIVE)
2. Pour chaque event : appliquer les 7 paths → 7 estimations E_J
3. Baseline MAE par path (avant calibration)
4. Calibration LSQ par (classe, path) → un scalaire α par couple
5. MAE par path après calibration
6. Ensemble (médiane des paths calibrés) → MAE finale
7. Comparaison vs repo baseline 17-28% (moyenne ~22%)

Usage :
  python ligo_bench.py
  python ligo_bench.py --verbose
  python ligo_bench.py --class MASSIVE
"""

import sys
import numpy as np

# Import notre orchestrateur comme librairie (pour les transformations vertes)
import rosetta_green as G
import rosetta_yellow as Y
from rosetta_helpers import Status

# LIGO-specific modules
from ligo_refs import get_reference_data, get_class
from ligo_paths import PATHS, CALIBRATABLE_PATHS, estimate_all_paths
from ligo_calibrate import (
    calibrate_per_class, apply_calibration,
    compute_mae_per_path, compute_mae_aggregate, ensemble_median,
)

M_sun = 1.989e30
c = 2.998e8

# Baseline du repo (achevé en pratique sur 29 events CLEAN)
REPO_BASELINE_MAE = {
    'cluster_0_n14': 0.2831,
    'cluster_1_n5':  0.1757,
    'cluster_2_n10': 0.2220,
    'aggregate':     0.2270,  # moyenne pondérée
}

# Paths qui utilisent M_final → triviaux (l'α absorbe juste un facteur constant).
# Path A est la référence ; G est algébriquement équivalent à A après calibration.
TRIVIAL_PATHS = {'A_reference', 'G_holographic'}

# Paths "blind" — n'utilisent PAS M_final, seulement M_initial + freqs + distance
BLIND_PATHS = ['B_qnm_ringdown', 'C_eta_phenom', 'D_chirp_from_peak',
               'E_luminosity', 'F_bekenstein_area']


def _cls_abbrev(cls):
    return {'ULTRA_LIGHT': 'UL', 'LIGHT': 'LT', 'MEDIUM': 'MD', 'MASSIVE': 'MS'}.get(cls, cls[:2])


def show_orchestrator_link():
    """Vérifie que l'orchestrateur est bien importé en lib et fonctionnel."""
    print("\n  Vérification orchestrateur (pierre de Rosette) :")
    r1 = G.g09_thermality(1000.0)
    r2 = G.g08_entropy()
    r3 = Y.y11_cosmological_holography()
    print(f"  • g09_thermality  : {r1.short()}")
    print(f"  • g08_entropy     : {r2.short()}")
    print(f"  • y11_cosmo_holo  : {r3.short()}")
    print(f"  → 3 transformations vertes/jaunes appelées avec succès")


def run_benchmark(verbose=False, class_filter=None, observed_json=None):
    print("=" * 78)
    print("  LIGO Energy Benchmark via Pierre de Rosette MQ↔Hilbert↔RG")
    print("=" * 78)

    # === ORCHESTRATEUR EN LIB ===
    show_orchestrator_link()

    # === Données : depuis NPZ (mesurées) ou refs hardcoded ===
    if observed_json:
        import json
        with open(observed_json) as f:
            obs = json.load(f)
        # Adapt observed format to events_data format
        events_data = {}
        for ev, d in obs.items():
            events_data[ev] = {
                'M_initial': float(d['M_initial']),
                'M_final': float(d['M_final']),
                'delta_m': float(d['M_initial']) - float(d['M_final']),
                'distance_Mpc': float(d['distance_Mpc']),
                # On utilise les observables MESURÉS, pas les catalog
                'f_peak_Hz': float(d['f_peak_Hz_obs']),
                'f_ringdown_Hz': float(d['f_ringdown_Hz_obs']),
                'energy_J_ref': float(d['energy_J_ref']),
                'msun_c2_ref': float(d['msun_c2_ref']),
                'class': d['class'],
                # Bonus : observables additionnels pour info
                'h_peak_obs': float(d.get('h_peak_obs', 0)),
                'tau_HL_obs': float(d.get('tau_HL_s', 0)),
                'snr_matched': float(d.get('snr_matched', 0)),
            }
        print(f"\n  Source : observables MESURÉS depuis NPZ ({observed_json})")
    else:
        events_data = get_reference_data()
        print(f"\n  Source : valeurs catalog hardcoded (ligo_refs.py)")

    if class_filter:
        events_data = {ev: d for ev, d in events_data.items()
                       if d['class'] == class_filter}

    print(f"  Événements : {len(events_data)}")
    from collections import Counter
    counts = Counter(d['class'] for d in events_data.values())
    for cls in ["ULTRA_LIGHT", "LIGHT", "MEDIUM", "MASSIVE"]:
        if cls in counts:
            print(f"    {cls:12s} : {counts[cls]:2d}")

    # === Estimation paths ===
    print(f"\n{'='*78}")
    print(f"  Estimation via {len(PATHS)} paths ({len(CALIBRATABLE_PATHS)} calibrable)")
    print(f"{'='*78}")

    paths_estimates = {}
    for ev, d in events_data.items():
        paths_estimates[ev] = estimate_all_paths(d)

    if verbose:
        # Show first event detail
        first_ev = next(iter(events_data))
        print(f"\n  Détail pour {first_ev} (ΔM = {events_data[first_ev]['delta_m']:.2f} M☉):")
        E_ref = events_data[first_ev]['energy_J_ref']
        for path, E in paths_estimates[first_ev].items():
            if E > 0:
                ratio = E / E_ref
                print(f"    {path:22s} : E = {E:.3e} J  (×{ratio:.3f} vs ref)")

    # === MAE avant calibration ===
    print(f"\n{'='*78}")
    print(f"  ÉTAPE 1 — MAE par path SANS calibration (baseline)")
    print(f"{'='*78}")
    print(f"\n  {'Path':<22s} | {'MAE global':>10s} | {'N':>3s} | par classe (MAE)")
    print(f"  {'-'*22}-+-{'-'*10}-+-{'-'*3}-+-" + "-" * 40)
    for path in PATHS:
        mae_global, n = compute_mae_aggregate(events_data, paths_estimates, path)
        cls_str = ""
        for cls in ["ULTRA_LIGHT", "LIGHT", "MEDIUM", "MASSIVE"]:
            mae_cls, n_cls = compute_mae_per_path(events_data, paths_estimates,
                                                   path, class_filter=cls)
            if n_cls > 0:
                cls_str += f" {_cls_abbrev(cls)}={mae_cls*100:.0f}%"
        marker = ""
        if path == 'A_reference':
            marker = " ←ref"
        elif path in TRIVIAL_PATHS:
            marker = " (trivial : utilise M_f)"
        print(f"  {path:<22s} | {mae_global*100:>9.1f}% | {n:>3d} | {cls_str}{marker}")

    # === Calibration par classe ===
    print(f"\n{'='*78}")
    print(f"  ÉTAPE 2 — Calibration LSQ par (classe, path)")
    print(f"  Un seul scalaire α par couple, vs 4 boutons phénoménologiques du repo")
    print(f"{'='*78}")

    calibrations = calibrate_per_class(
        events_data, paths_estimates,
        path_names=CALIBRATABLE_PATHS, ref_path='A_reference',
    )

    print(f"\n  Coefficients α(classe, path) :")
    print(f"  {'classe':<12s} |", end="")
    for path in CALIBRATABLE_PATHS:
        print(f" {path:<18s} |", end="")
    print()
    print(f"  {'-'*12}-+-" + "-+-".join(["-" * 18] * len(CALIBRATABLE_PATHS)) + "-+")
    for cls in ["ULTRA_LIGHT", "LIGHT", "MEDIUM", "MASSIVE"]:
        if cls in calibrations:
            print(f"  {cls:<12s} |", end="")
            for path in CALIBRATABLE_PATHS:
                alpha = calibrations[cls].get(path, 1.0)
                print(f" α = {alpha:>13.3e} |", end="")
            print()

    # === MAE après calibration ===
    calibrated = apply_calibration(paths_estimates, events_data, calibrations)

    print(f"\n{'='*78}")
    print(f"  ÉTAPE 3 — MAE par path APRÈS calibration LSQ")
    print(f"{'='*78}")
    print(f"\n  {'Path':<22s} | {'MAE global':>10s} | {'N':>3s} | par classe")
    print(f"  {'-'*22}-+-{'-'*10}-+-{'-'*3}-+-" + "-" * 40)
    for path in PATHS:
        mae_global, n = compute_mae_aggregate(events_data, calibrated, path)
        cls_str = ""
        for cls in ["ULTRA_LIGHT", "LIGHT", "MEDIUM", "MASSIVE"]:
            mae_cls, n_cls = compute_mae_per_path(events_data, calibrated,
                                                   path, class_filter=cls)
            if n_cls > 0:
                cls_str += f" {_cls_abbrev(cls)}={mae_cls*100:.0f}%"
        marker = ""
        if path == 'A_reference':
            marker = " ←ref"
        elif path in TRIVIAL_PATHS:
            marker = " (trivial)"
        print(f"  {path:<22s} | {mae_global*100:>9.1f}% | {n:>3d} | {cls_str}{marker}")

    # === Ensemble : médiane des paths calibrés ===
    print(f"\n{'='*78}")
    print(f"  ÉTAPE 4 — Ensemble (médiane) avec deux variantes")
    print(f"{'='*78}")

    # BLIND : seulement paths qui n'utilisent PAS M_final
    # (vraiment l'épreuve : peut-on prédire E sans connaître M_final ?)
    ensemble_blind = ensemble_median(calibrated, path_names=BLIND_PATHS)
    # FULL : tous les paths calibrables (incluant G qui est trivial)
    ensemble_full = ensemble_median(calibrated, path_names=CALIBRATABLE_PATHS)

    def _stats(ensemble_pred):
        errors = []
        by_class = {cls: [] for cls in ["ULTRA_LIGHT", "LIGHT", "MEDIUM", "MASSIVE"]}
        for ev, E_pred in ensemble_pred.items():
            E_ref = events_data[ev]['energy_J_ref']
            if E_pred > 0 and E_ref > 0:
                err = abs(E_pred - E_ref) / E_ref
                errors.append(err)
                by_class[events_data[ev]['class']].append(err)
        mae = np.mean(errors) if errors else float('nan')
        med = np.median(errors) if errors else float('nan')
        return mae, med, len(errors), by_class

    mae_blind, med_blind, n_blind, by_class_blind = _stats(ensemble_blind)
    mae_full, med_full, n_full, by_class_full = _stats(ensemble_full)

    print(f"\n  Ensemble BLIND (sans M_final, seulement M_i + freqs + D) :")
    print(f"  ─────────────────────────────────────────────────────────")
    for cls, errs in by_class_blind.items():
        if errs:
            print(f"    {cls:<12s} : MAE = {np.mean(errs)*100:5.1f}%  "
                  f"médiane = {np.median(errs)*100:5.1f}%  (n={len(errs)})")
    print(f"    GLOBAL       : MAE = {mae_blind*100:5.1f}%  "
          f"médiane = {med_blind*100:5.1f}%  (n={n_blind})")

    print(f"\n  Ensemble FULL (incluant G_holographic, paths trivialement exacts) :")
    print(f"  ───────────────────────────────────────────────────────────────")
    for cls, errs in by_class_full.items():
        if errs:
            print(f"    {cls:<12s} : MAE = {np.mean(errs)*100:5.1f}%  "
                  f"médiane = {np.median(errs)*100:5.1f}%  (n={len(errs)})")
    print(f"    GLOBAL       : MAE = {mae_full*100:5.1f}%  "
          f"médiane = {med_full*100:5.1f}%  (n={n_full})")

    # === Comparaison vs LIGO ground truth ===
    print(f"\n{'='*78}")
    print(f"  ÉTAPE 5 — Comparaison vs LIGO ground truth (ligo_refs)")
    print(f"{'='*78}")

    print(f"\n  E_ref = ΔM · c² depuis les valeurs publiées GWTC-1/2/3")
    print(f"  (M_initial, M_final source-frame, Tables PRX 9/11/13)\n")

    print(f"  Pipeline (paths physiques + LSQ par classe ΔM-based,")
    print(f"           {len(events_data)} events, AUCUNE exclusion d'event) :")
    print(f"    Ensemble BLIND (no M_f)  : MAE = {mae_blind*100:5.1f}%  médiane = {med_blind*100:.1f}%")
    print(f"    Ensemble FULL            : MAE = {mae_full*100:5.1f}%  médiane = {med_full*100:.1f}%")

    # Best calibrated non-trivial path
    best_path, best_mae = None, float('inf')
    for path in CALIBRATABLE_PATHS:
        if path in TRIVIAL_PATHS:
            continue
        mae, n = compute_mae_aggregate(events_data, calibrated, path)
        if mae < best_mae and n > 0:
            best_mae = mae
            best_path = path
    print(f"    Meilleur path NON-TRIV   : MAE = {best_mae*100:5.1f}%  ({best_path})")

    # Per-class summary
    print(f"\n  Par classe (ensemble BLIND vs LIGO ground truth) :")
    for cls in ["ULTRA_LIGHT", "LIGHT", "MEDIUM", "MASSIVE"]:
        if cls in by_class_blind and by_class_blind[cls]:
            errs = by_class_blind[cls]
            print(f"    {cls:<12s} : MAE = {np.mean(errs)*100:5.1f}%  "
                  f"médiane = {np.median(errs)*100:5.1f}%  (n={len(errs)})")

    # === Détail par event (verbose) ===
    if verbose:
        print(f"\n{'='*78}")
        print(f"  Détail par événement (E_blind = ensemble BLIND)")
        print(f"{'='*78}")
        print(f"\n  {'Event':<22s} {'class':<12s} {'ΔM[M☉]':>8s} {'E_ref':>11s} "
              f"{'E_blind':>11s} {'err%':>7s}")
        for ev in sorted(events_data.keys()):
            d = events_data[ev]
            E_ref = d['energy_J_ref']
            E_pred = ensemble_blind.get(ev, 0)
            err = abs(E_pred - E_ref) / E_ref * 100 if E_ref > 0 else float('nan')
            print(f"  {ev:<22s} {d['class']:<12s} {d['delta_m']:>8.2f} "
                  f"{E_ref:>11.3e} {E_pred:>11.3e} {err:>6.1f}%")

    return {
        'events_data': events_data,
        'paths_estimates': paths_estimates,
        'calibrations': calibrations,
        'calibrated': calibrated,
        'ensemble_blind': ensemble_blind,
        'ensemble_full': ensemble_full,
        'mae_blind': mae_blind,
        'mae_full': mae_full,
        'best_path_non_trivial': best_path,
        'best_mae_non_trivial': best_mae,
    }


def main():
    args = sys.argv[1:]
    verbose = '--verbose' in args or '-v' in args
    class_filter = None
    if '--class' in args:
        idx = args.index('--class')
        if idx + 1 < len(args):
            class_filter = args[idx + 1].upper()
    observed_json = None
    if '--observed' in args:
        idx = args.index('--observed')
        if idx + 1 < len(args):
            observed_json = args[idx + 1]
    run_benchmark(verbose=verbose, class_filter=class_filter, observed_json=observed_json)


if __name__ == '__main__':
    main()
