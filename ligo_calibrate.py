"""Calibration physique par classe — UN SEUL scalaire par (classe, path).

Vire les 4 boutons phénoménologiques TAU/NU/SCALE/PEAK du repo.

Modèle :
  E_predicted(event) = α(class, path) × E_path(event)

α optimisé par LSQ par classe :
  α* = argmin_α Σ_{events in class} (E_ref - α·E_path)²
     = (Σ E_ref · E_path) / (Σ E_path²)

Chaque (class, path) a son propre α. Path A n'est pas calibré (référence exacte).
"""

import numpy as np
from collections import defaultdict


def lsq_scalar(E_ref_arr, E_path_arr):
    """LSQ : trouve α minimisant ||E_ref - α·E_path||².

    Solution analytique : α* = (E_ref · E_path) / (E_path · E_path)
    """
    E_ref = np.asarray(E_ref_arr, float)
    E_path = np.asarray(E_path_arr, float)
    mask = np.isfinite(E_ref) & np.isfinite(E_path) & (E_path > 0) & (E_ref > 0)
    if mask.sum() < 1:
        return 1.0
    num = float(np.sum(E_ref[mask] * E_path[mask]))
    den = float(np.sum(E_path[mask] ** 2))
    if den <= 0:
        return 1.0
    return num / den


def calibrate_per_class(events_data: dict, paths_estimates: dict,
                        path_names=None, ref_path='A_reference'):
    """Calibre chaque (class, path) via LSQ.

    Args:
        events_data: dict[event_name] -> dict avec 'class', 'energy_J_ref'
        paths_estimates: dict[event_name] -> dict[path_name] -> E_J
        path_names: liste des paths à calibrer (sauf ref_path)
        ref_path: nom du path qui sert de ground truth (default 'A_reference')

    Returns:
        calibrations: dict[class_name][path_name] -> α scalar
    """
    # Group events by class
    events_by_class = defaultdict(list)
    for ev, info in events_data.items():
        events_by_class[info['class']].append(ev)

    if path_names is None:
        # Auto-discover paths from first event
        first_ev = next(iter(paths_estimates))
        path_names = [p for p in paths_estimates[first_ev] if p != ref_path]

    calibrations = {}
    for cls, ev_list in events_by_class.items():
        calibrations[cls] = {}
        for path in path_names:
            E_ref_arr = np.array([
                events_data[ev]['energy_J_ref'] for ev in ev_list
            ])
            E_path_arr = np.array([
                paths_estimates[ev].get(path, 0.0) for ev in ev_list
            ])
            alpha = lsq_scalar(E_ref_arr, E_path_arr)
            calibrations[cls][path] = alpha
    return calibrations


def apply_calibration(paths_estimates: dict, events_data: dict, calibrations: dict):
    """Applique la calibration : E_calibrated = α(class, path) × E_path."""
    calibrated = {}
    for ev, paths_dict in paths_estimates.items():
        cls = events_data[ev]['class']
        calibrated[ev] = {}
        for path, E in paths_dict.items():
            if path in calibrations.get(cls, {}):
                alpha = calibrations[cls][path]
                calibrated[ev][path] = alpha * E
            else:
                calibrated[ev][path] = E  # path A (référence)
    return calibrated


def compute_mae_per_path(events_data: dict, predictions: dict, path_name: str,
                         class_filter=None):
    """MAE relative pour un path donné.

    MAE = mean(|E_pred - E_ref| / E_ref)

    class_filter : si fourni, ne garde que les events de cette classe.
    """
    errors = []
    for ev, paths_dict in predictions.items():
        if class_filter and events_data[ev]['class'] != class_filter:
            continue
        E_pred = paths_dict.get(path_name, 0.0)
        E_ref = events_data[ev]['energy_J_ref']
        if E_pred > 0 and E_ref > 0:
            err = abs(E_pred - E_ref) / E_ref
            errors.append(err)
    if not errors:
        return float('nan'), 0
    return float(np.mean(errors)), len(errors)


def compute_mae_aggregate(events_data: dict, predictions: dict, path_name: str):
    """MAE globale sur tous les événements pour un path."""
    return compute_mae_per_path(events_data, predictions, path_name, class_filter=None)


def ensemble_median(predictions: dict, path_names=None, exclude_ref=True):
    """Ensemble : pour chaque event, prend la médiane des paths.

    Réduit la variance individuelle des paths.
    """
    ensemble = {}
    for ev, paths_dict in predictions.items():
        if path_names:
            vals = [paths_dict[p] for p in path_names if p in paths_dict]
        else:
            vals = [v for k, v in paths_dict.items()
                    if not (exclude_ref and k == 'A_reference')]
        vals = [v for v in vals if v > 0]
        if vals:
            ensemble[ev] = float(np.median(vals))
        else:
            ensemble[ev] = 0.0
    return ensemble
