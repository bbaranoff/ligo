"""Régression statistique sur Ej prédits vs Ej références.

Pour chaque path et pour l'ensemble :
- Régression linéaire E_pred = slope × E_ref + intercept
- slope ≈ 1, intercept ≈ 0 → prédicteur fidèle
- R² → fraction de variance expliquée
- p-value → significativité de la corrélation
- MAE relative

Par classe + verdict global.

Usage : python3 ligo_ej_stats.py [--observed observed.json]
"""

import json
import sys
import numpy as np

try:
    from scipy import stats as scistat
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("scipy requis"); sys.exit(1)

from ligo_paths import PATHS, CALIBRATABLE_PATHS, estimate_all_paths
from ligo_calibrate import (
    calibrate_per_class, apply_calibration, ensemble_median,
)

M_sun = 1.989e30
c = 2.998e8

BLIND_PATHS = ['B_qnm_ringdown', 'C_eta_phenom', 'D_chirp_from_peak',
               'E_luminosity', 'F_bekenstein_area']


def verdict_r2(r2):
    if r2 >= 0.9:  return "✓✓ excellent"
    if r2 >= 0.7:  return "✓  bon       "
    if r2 >= 0.4:  return "~  modéré    "
    if r2 >= 0.1:  return "✗  faible    "
    return                "✗✗ aléatoire"


def verdict_slope(slope, intercept_ratio):
    """slope ≈ 1 et intercept ≈ 0 → idéal."""
    s_dev = abs(slope - 1.0)
    i_dev = abs(intercept_ratio)  # intercept relativisé à mean(E_ref)
    if s_dev < 0.1 and i_dev < 0.1:  return "✓ fidèle"
    if s_dev < 0.3 and i_dev < 0.3:  return "~ biaisé léger"
    return                                "✗ biaisé fort"


def regression_block(name, E_ref, E_pred, indent="    "):
    """Calcule et affiche les stats de régression."""
    if len(E_ref) < 3:
        print(f"{indent}{name:<14s} : insuffisant (n={len(E_ref)})")
        return None
    slope, intercept, r, p, se = scistat.linregress(E_ref, E_pred)
    mae = float(np.mean(np.abs(E_pred - E_ref) / E_ref))
    intercept_ratio = intercept / np.mean(E_ref)
    r2 = r**2
    v_r2 = verdict_r2(r2)
    v_slope = verdict_slope(slope, intercept_ratio)
    print(f"{indent}{name:<14s} n={len(E_ref):>3d}  slope={slope:>6.3f}  "
          f"R²={r2:>5.3f}  p={p:>8.2e}  MAE={mae*100:>5.1f}%   "
          f"{v_r2}  {v_slope}")
    return {'slope': slope, 'r2': r2, 'p': p, 'mae': mae, 'n': len(E_ref)}


def main():
    path = 'observed.json'
    if '--observed' in sys.argv:
        idx = sys.argv.index('--observed')
        if idx + 1 < len(sys.argv):
            path = sys.argv[idx + 1]

    with open(path) as f:
        obs = json.load(f)

    # Build events_data (idem ligo_bench.py)
    events_data = {}
    for ev, d in obs.items():
        events_data[ev] = {
            'M_initial': float(d['M_initial']),
            'M_final': float(d['M_final']),
            'delta_m': float(d['M_initial']) - float(d['M_final']),
            'distance_Mpc': float(d['distance_Mpc']),
            'f_peak_Hz': float(d['f_peak_Hz_obs']),
            'f_ringdown_Hz': float(d['f_ringdown_Hz_obs']),
            'energy_J_ref': float(d['energy_J_ref']),
            'msun_c2_ref': float(d['msun_c2_ref']),
            'class': d['class'],
        }

    # Estimate paths + calibrate (same pipeline as bench)
    paths_estimates = {ev: estimate_all_paths(d) for ev, d in events_data.items()}
    calibrations = calibrate_per_class(
        events_data, paths_estimates,
        path_names=CALIBRATABLE_PATHS, ref_path='A_reference',
    )
    calibrated = apply_calibration(paths_estimates, events_data, calibrations)
    ensemble_blind = ensemble_median(calibrated, path_names=BLIND_PATHS)

    print(f"\n{'='*78}")
    print(f"  RÉGRESSION E_pred vs E_ref — {len(events_data)} events")
    print(f"{'='*78}\n")

    # ===== Par path =====
    print(f"  RÉGRESSION GLOBALE PAR PATH (calibré LSQ par classe)")
    print(f"  {'-'*72}")
    for path_name in PATHS:
        E_ref_list, E_pred_list = [], []
        for ev, paths_dict in calibrated.items():
            E_ref = events_data[ev]['energy_J_ref']
            E_pred = paths_dict.get(path_name, 0.0)
            if E_pred > 0 and E_ref > 0:
                E_ref_list.append(E_ref)
                E_pred_list.append(E_pred)
        regression_block(path_name, np.array(E_ref_list), np.array(E_pred_list))

    # ===== Ensemble BLIND =====
    print(f"\n  ENSEMBLE BLIND (médiane des paths sauf A et G)")
    print(f"  {'-'*72}")
    E_ref_arr, E_pred_arr, classes = [], [], []
    for ev, E_pred in ensemble_blind.items():
        E_ref = events_data[ev]['energy_J_ref']
        if E_pred > 0 and E_ref > 0:
            E_ref_arr.append(E_ref)
            E_pred_arr.append(E_pred)
            classes.append(events_data[ev]['class'])
    E_ref_arr = np.array(E_ref_arr)
    E_pred_arr = np.array(E_pred_arr)
    classes = np.array(classes)

    res_global = regression_block("GLOBAL", E_ref_arr, E_pred_arr)

    # Per class
    print(f"\n  ENSEMBLE BLIND PAR CLASSE")
    print(f"  {'-'*72}")
    for cls in ['ULTRA_LIGHT', 'LIGHT', 'MEDIUM', 'MASSIVE']:
        mask = classes == cls
        if mask.sum() < 3:
            continue
        regression_block(cls, E_ref_arr[mask], E_pred_arr[mask])

    # ===== Homogénéité des erreurs entre classes (ANOVA) =====
    print(f"\n  HOMOGÉNÉITÉ DES ERREURS ENTRE CLASSES")
    print(f"  {'-'*72}")
    errors_per_class = {}
    for ev, E_pred in ensemble_blind.items():
        E_ref = events_data[ev]['energy_J_ref']
        if E_pred > 0 and E_ref > 0:
            err = abs(E_pred - E_ref) / E_ref
            cls = events_data[ev]['class']
            errors_per_class.setdefault(cls, []).append(err)

    class_errors = [np.array(v) for v in errors_per_class.values() if len(v) >= 2]
    class_names = [k for k, v in errors_per_class.items() if len(v) >= 2]
    if len(class_errors) >= 2:
        f_stat, anova_p = scistat.f_oneway(*class_errors)
        k_stat, kw_p = scistat.kruskal(*class_errors)
        print(f"    classes testées  : {class_names}")
        print(f"    ANOVA F          : {f_stat:>6.3f}, p = {anova_p:.3e}")
        print(f"    Kruskal-Wallis H : {k_stat:>6.3f}, p = {kw_p:.3e}")
        if anova_p > 0.05 and kw_p > 0.05:
            print(f"    verdict          : ✓ erreurs HOMOGÈNES — calibration équilibrée")
        else:
            print(f"    verdict          : ⚠ erreurs INHOMOGÈNES — certaines classes mal fittées")

    # ===== Verdict global =====
    print(f"\n  {'='*72}")
    print(f"  VERDICT RÉGRESSION")
    print(f"  {'='*72}")
    if res_global:
        r2 = res_global['r2']
        slope = res_global['slope']
        mae = res_global['mae']
        print(f"  Global  R² = {r2:.3f}  slope = {slope:.3f}  MAE = {mae*100:.1f}%")
        if r2 > 0.9 and abs(slope - 1) < 0.15:
            print(f"  → Le pipeline EST un prédicteur fidèle de E_J LIGO")
        elif r2 > 0.7:
            print(f"  → Bon prédicteur, mais biais résiduel ; calibration améliorable")
        elif r2 > 0.4:
            print(f"  → Prédicteur partiel : capture la tendance, pas la précision")
        else:
            print(f"  → Faible corrélation : le calibration LSQ absorbe la variance")


if __name__ == '__main__':
    main()
