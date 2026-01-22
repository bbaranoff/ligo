#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calibration cluster-by-cluster pour LIGO spectral analysis

Pipeline:
1) Charge les features depuis results/GW*.json
2) Clustering (DBSCAN -> outliers ; KMeans -> clusters inliers)
3) Calibre (H_STAR, SCALE_EJ) par cluster via LSQ (clé dégénérée K = H_STAR*SCALE_EJ)
4) Ré-analyse tous les événements avec la calibration de leur cluster
5) Génère un rapport comparatif

Usage:
  python calibrate_by_cluster.py --refs ligo_refs.json --event-params event_params.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np

# modules locaux
import cluster_latent_kmeans as clk
import ligo_spectral_planck as lsp


# ----------------------------
# Clustering
# ----------------------------

def load_cluster_assignments(
    results_glob: str = "results/GW*.json",
    f_split: float = 150.0,
    db_eps: float = 1.4,
    db_min_samples: int = 3,
    k: int = 4,
    seed: int = 42,
) -> Tuple[Dict[str, int], List[clk.Features]]:
    """
    Fait le clustering et retourne:
      - event_to_cluster: {event_name: cluster_id}
      - feats: liste de Features (dans le même ordre que le clustering)
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

    # impute NaN/inf par médiane colonne
    X2 = X.copy()
    for j in range(X2.shape[1]):
        col = X2[:, j]
        m = np.isfinite(col)
        med = float(np.median(col[m])) if np.any(m) else 0.0
        col[~m] = med
        X2[:, j] = col

    scaler = StandardScaler()
    Z = scaler.fit_transform(X2)

    db = DBSCAN(eps=float(db_eps), min_samples=int(db_min_samples))
    labels_db = db.fit_predict(Z)
    inlier_mask = labels_db != -1

    n_in = int(inlier_mask.sum())
    if n_in < max(2, int(k)):
        raise RuntimeError(f"Not enough inliers after DBSCAN for k={k} (inliers={n_in})")

    Z_in = Z[inlier_mask]
    km = KMeans(n_clusters=int(k), random_state=int(seed), n_init="auto")
    labels_in = km.fit_predict(Z_in)

    labels = np.full(Z.shape[0], -1, dtype=int)
    labels[inlier_mask] = labels_in

    event_to_cluster = {ft.event: int(lab) for ft, lab in zip(feats, labels)}
    print(f"[clustering] n_events={len(feats)} inliers={n_in} outliers={len(feats)-n_in} k={k}")
    return event_to_cluster, feats


# ----------------------------
# Helpers: appels LSP propres
# ----------------------------

def _analyze_event_safe(
    *,
    event: str,
    params: Dict,
    bands: lsp.Bands,
    flow: Optional[float],
    fhigh: Optional[float],
    signal_win: Optional[float],
    noise_pad: Optional[float],
    distance_mpc: Optional[float],
    use_virgo: bool,
    peak_norm: bool,
    peak_quantile: float,
    hstar_in: float,
    scale_ej_in: float,
) -> Dict:
    """
    Appelle lsp.analyze_event avec une signature 100% explicite.
    => zéro risque de "multiple values for keyword argument".
    """
    return lsp.analyze_event(
        event=event,
        params=params,
        bands=bands,
        flow=flow,
        fhigh=fhigh,
        signal_win=signal_win,
        noise_pad=noise_pad,
        distance_mpc=distance_mpc,
        use_virgo=use_virgo,
        peak_norm=peak_norm,
        peak_quantile=peak_quantile,
        hstar_in=hstar_in,
        scale_ej_in=scale_ej_in,
        plot=False,
        return_internals=False,
    )


def _calibrate_lsq_safe(
    *,
    events: List[str],
    params: Dict,
    y_obs_dict: Dict[str, float],
    bands: lsp.Bands,
    flow: Optional[float],
    fhigh: Optional[float],
    signal_win: Optional[float],
    noise_pad: Optional[float],
    distance_mpc: Optional[float],
    use_virgo: bool,
    peak_norm: bool,
    peak_quantile: float,
) -> Tuple[float, float]:
    """
    Appelle lsp.calibrate_least_squares avec params explicites.
    """
    return lsp.calibrate_least_squares(
        events=events,
        params=params,
        y_obs_dict=y_obs_dict,
        bands=bands,
        flow=flow,
        fhigh=fhigh,
        signal_win=signal_win,
        noise_pad=noise_pad,
        distance_mpc=distance_mpc,
        use_virgo=use_virgo,
        peak_norm=peak_norm,
        peak_quantile=peak_quantile,
    )


# ----------------------------
# Calibration par cluster
# ----------------------------

def calibrate_cluster(
    *,
    cluster_id: int,
    events: List[str],
    params: Dict,
    y_obs_dict: Dict[str, float],
    bands: lsp.Bands,
    flow: Optional[float],
    fhigh: Optional[float],
    signal_win: Optional[float],
    noise_pad: Optional[float],
    distance_mpc: Optional[float],
    use_virgo: bool,
    peak_norm: bool,
    peak_quantile: float,
) -> Tuple[float, float]:
    """
    Calibre H_STAR et SCALE_EJ pour un cluster donné.
    Retourne (hstar, scale_ej).
    """
    print(f"\n{'='*70}")
    print(f"📊 CALIBRATION CLUSTER {cluster_id} ({len(events)} events)")
    print(f"{'='*70}")

    valid_events = [e for e in events if e in y_obs_dict]
    if len(valid_events) < 2:
        print(f"[WARN] Cluster {cluster_id}: pas assez d'events avec refs ({len(valid_events)})")
        print("       -> using default calibration (H_STAR=1.0, SCALE_EJ=1.0)")
        return 1.0, 1.0

    try:
        hstar, scale_ej = _calibrate_lsq_safe(
            events=valid_events,
            params=params,
            y_obs_dict=y_obs_dict,
            bands=bands,
            flow=flow,
            fhigh=fhigh,
            signal_win=signal_win,
            noise_pad=noise_pad,
            distance_mpc=distance_mpc,
            use_virgo=use_virgo,
            peak_norm=peak_norm,
            peak_quantile=peak_quantile,
        )
        print(f"✅ Cluster {cluster_id}: H_STAR={hstar:.6e}, SCALE_EJ={scale_ej:.6e}")
        return float(hstar), float(scale_ej)
    except Exception as e:
        print(f"[ERROR] Cluster {cluster_id} calibration failed: {e}")
        print("        -> using default calibration")
        return 1.0, 1.0


def analyze_with_cluster_calibration(
    *,
    event: str,
    cluster_id: int,
    hstar: float,
    scale_ej: float,
    params: Dict,
    bands: lsp.Bands,
    flow: Optional[float],
    fhigh: Optional[float],
    signal_win: Optional[float],
    noise_pad: Optional[float],
    distance_mpc: Optional[float],
    use_virgo: bool,
    peak_norm: bool,
    peak_quantile: float,
) -> Optional[Dict]:
    """
    Analyse un événement avec la calibration de son cluster.
    """
    try:
        return _analyze_event_safe(
            event=event,
            params=params,
            bands=bands,
            flow=flow,
            fhigh=fhigh,
            signal_win=signal_win,
            noise_pad=noise_pad,
            distance_mpc=distance_mpc,
            use_virgo=use_virgo,
            peak_norm=peak_norm,
            peak_quantile=peak_quantile,
            hstar_in=hstar,
            scale_ej_in=scale_ej,
        )
    except Exception as e:
        gps = None
        try:
            gps = params.get(event, {}).get("gps")
        except Exception:
            pass
        if gps is not None:
            print(f"[WARN] skip {event}: analyze_event failed (gps={gps}): {e}")
        else:
            print(f"[WARN] skip {event}: analyze_event failed: {e}")
        return None


# ----------------------------
# Erreurs / reporting
# ----------------------------

def compute_errors(
    results: Dict[str, Dict],
    refs: Dict[str, Dict],
    ref_key: str = "energy_J",
) -> Dict[str, float]:
    """
    Erreur relative (%) signée: 100*(calc-ref)/ref
    """
    errors: Dict[str, float] = {}
    for event, res in results.items():
        ref = refs.get(event)
        if not isinstance(ref, dict):
            continue

        E_calc = res.get("E_total_J")
        E_ref = ref.get(ref_key)
        if E_calc is None or E_ref is None:
            continue

        E_calc = float(E_calc)
        E_ref = float(E_ref)
        if not np.isfinite(E_calc) or not np.isfinite(E_ref) or E_ref <= 0:
            continue

        errors[event] = 100.0 * (E_calc - E_ref) / E_ref

    return errors


def write_report(
    out_path: str,
    event_to_cluster: Dict[str, int],
    cluster_calib: Dict[int, Tuple[float, float]],
    results: Dict[str, Dict],
    errors: Dict[str, float],
    refs: Dict[str, Dict],
    ref_key: str,
) -> None:
    clusters = defaultdict(list)
    for event, cid in event_to_cluster.items():
        clusters[cid].append(event)

    lines: List[str] = []
    lines.append("=" * 100)
    lines.append("CALIBRATION PAR CLUSTER - RAPPORT DÉTAILLÉ")
    lines.append("=" * 100)
    lines.append("")

    abs_errs = [abs(v) for v in errors.values()]
    if abs_errs:
        lines.append("📊 STATISTIQUES GLOBALES")
        lines.append(f"   MAE (abs) : {np.mean(abs_errs):.2f}%")
        lines.append(f"   Médiane   : {np.median(abs_errs):.2f}%")
        lines.append(f"   N         : {len(abs_errs)}")
        lines.append("")

    for cid in sorted(clusters.keys()):
        evs = sorted(clusters[cid])
        hstar, scale_ej = cluster_calib.get(cid, (1.0, 1.0))

        lines.append("")
        lines.append("=" * 100)
        lines.append(f"CLUSTER {cid} ({len(evs)} événements)")
        lines.append("=" * 100)
        lines.append(f"Calibration: H_STAR={hstar:.6e} SCALE_EJ={scale_ej:.6e} K={hstar*scale_ej:.6e}")
        lines.append("")

        ce = [abs(errors[e]) for e in evs if e in errors]
        if ce:
            lines.append("📈 Statistiques:")
            lines.append(f"   MAE     : {np.mean(ce):.2f}%")
            lines.append(f"   Médiane : {np.median(ce):.2f}%")
            lines.append(f"   Min/Max : {min(ce):.2f}% / {max(ce):.2f}%")
            lines.append("")

        lines.append(f"Event                | E_calc [{ref_key}] | E_ref [{ref_key}]  | err(%)  | m_sun(calc) | msun(ref)")
        lines.append("-" * 100)

        for e in evs:
            res = results.get(e)
            ref = refs.get(e, {})
            if not res or not isinstance(ref, dict):
                lines.append(f"{e:20} | [no data]")
                continue

            E_calc = float(res.get("E_total_J", float("nan")))
            E_ref = float(ref.get(ref_key, float("nan")))
            m_calc = float(res.get("m_sun", float("nan")))
            m_ref = float(ref.get("msun_c2", float("nan")))
            err = errors.get(e)
            err_s = f"{err:+7.2f}" if err is not None else "  N/A  "
            lines.append(f"{e:20} | {E_calc:>12.3e} | {E_ref:>12.3e} | {err_s:>7} | {m_calc:>10.3f} | {m_ref:>8.3f}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n✅ Rapport écrit: {out_path}")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Calibration cluster-by-cluster pour LIGO spectral analysis")

    ap.add_argument("--refs", required=True, help="JSON refs LIGO (ligo_refs.json)")
    ap.add_argument("--event-params", default="event_params.json", help="JSON params events")
    ap.add_argument("--results-glob", default="results/GW*.json", help="Pattern des JSON results")

    # IMPORTANT: peak_norm est un bool de comportement pipeline (comme ton run_all)
    ap.add_argument("--peak-norm", action="store_true",
                    help="Normalise l'amplitude sur le pic (même comportement que run_all.sh)")
    ap.add_argument("--peak-quantile", type=float, default=99.5)

    # clustering
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--db-eps", type=float, default=1.4)
    ap.add_argument("--db-min-samples", type=int, default=3)
    ap.add_argument("--f-split", type=float, default=150.0)
    ap.add_argument("--seed", type=int, default=42)

    # analysis knobs
    ap.add_argument("--flow", type=float, default=None)
    ap.add_argument("--fhigh", type=float, default=None)
    ap.add_argument("--tau-band", type=float, nargs=2, default=[35.0, 250.0])
    ap.add_argument("--nu-band", type=float, nargs=2, default=[30.0, 350.0])
    ap.add_argument("--no-virgo", action="store_true")
    ap.add_argument("--signal-win", type=float, default=None)
    ap.add_argument("--noise-pad", type=float, default=None)
    ap.add_argument("--distance-mpc", type=float, default=None)

    # output
    ap.add_argument("--out", default="calibration_by_cluster.txt")
    ap.add_argument("--calib-json", default="cluster_calibrations.json")
    ap.add_argument("--ref-key", default="energy_J", choices=["energy_J", "msun_c2"])
    ap.add_argument("--exclude-cls", nargs="*", default=[])
    ap.add_argument("--exclude-cluster-minus1", action="store_true",
                    help="Exclure cluster -1 de la calibration (utilise défaut 1,1)")

    args = ap.parse_args()

    # refs
    print("📚 Chargement des références...")
    with open(args.refs, "r") as f:
        refs = json.load(f)

    exclude = set(args.exclude_cls)
    y_obs_dict: Dict[str, float] = {}
    for ev, d in refs.items():
        if not isinstance(d, dict):
            continue
        if exclude and d.get("cls", "") in exclude:
            continue
        v = d.get(args.ref_key)
        if v is None:
            continue
        fv = float(v)
        if not np.isfinite(fv) or fv <= 0:
            continue
        y_obs_dict[ev] = fv

    print(f"   Références chargées: {len(refs)} total, {len(y_obs_dict)} valides")

    # params
    print("\n📋 Chargement des paramètres d'événements...")
    params = lsp.load_event_params(args.event_params)
    print(f"   Événements dans params: {len(params)}")

    # clustering
    print("\n🔍 Clustering des événements...")
    event_to_cluster, _feats = load_cluster_assignments(
        results_glob=args.results_glob,
        f_split=args.f_split,
        db_eps=args.db_eps,
        db_min_samples=args.db_min_samples,
        k=args.k,
        seed=args.seed,
    )

    clusters = defaultdict(list)
    for ev, cid in event_to_cluster.items():
        clusters[cid].append(ev)
    for cid in sorted(clusters.keys()):
        print(f"   Cluster {cid:2}: {len(clusters[cid])} événements")

    # bands / analysis options
    bands = lsp.Bands(
        tau_band=(float(args.tau_band[0]), float(args.tau_band[1])),
        nu_band=(float(args.nu_band[0]), float(args.nu_band[1])),
    )

    use_virgo = not bool(args.no_virgo)
    peak_norm = bool(args.peak_norm)
    peak_quantile = float(args.peak_quantile)

    flow = args.flow
    fhigh = args.fhigh
    signal_win = args.signal_win
    noise_pad = args.noise_pad
    distance_mpc = args.distance_mpc

    # calibrate clusters
    print("\n🔧 Calibration par cluster...")
    cluster_calib: Dict[int, Tuple[float, float]] = {}

    for cid in sorted(clusters.keys()):
        if cid == -1 and args.exclude_cluster_minus1:
            print("\n⏭️  Cluster -1 (outliers): skipped (using default calibration)")
            cluster_calib[cid] = (1.0, 1.0)
            continue

        hstar, scale_ej = calibrate_cluster(
            cluster_id=cid,
            events=clusters[cid],
            params=params,
            y_obs_dict=y_obs_dict,
            bands=bands,
            flow=flow,
            fhigh=fhigh,
            signal_win=signal_win,
            noise_pad=noise_pad,
            distance_mpc=distance_mpc,
            use_virgo=use_virgo,
            peak_norm=peak_norm,
            peak_quantile=peak_quantile,
        )
        cluster_calib[cid] = (hstar, scale_ej)

    # save calib json
    calib_data = {str(cid): {"H_STAR": float(h), "SCALE_EJ": float(s)} for cid, (h, s) in cluster_calib.items()}
    with open(args.calib_json, "w") as f:
        json.dump(calib_data, f, indent=2)
    print(f"\n💾 Calibrations sauvegardées: {args.calib_json}")

    # re-analyze
    print("\n🔬 Ré-analyse avec calibration par cluster...")
    results: Dict[str, Dict] = {}
    for ev, cid in event_to_cluster.items():
        hstar, scale_ej = cluster_calib.get(cid, (1.0, 1.0))
        print(f"   {ev} (cluster {cid})...", end=" ")

        res = analyze_with_cluster_calibration(
            event=ev,
            cluster_id=cid,
            hstar=hstar,
            scale_ej=scale_ej,
            params=params,
            bands=bands,
            flow=flow,
            fhigh=fhigh,
            signal_win=signal_win,
            noise_pad=noise_pad,
            distance_mpc=distance_mpc,
            use_virgo=use_virgo,
            peak_norm=peak_norm,
            peak_quantile=peak_quantile,
        )
        if res is not None:
            results[ev] = res
            print("✅")
        else:
            print("❌")

    # errors
    print("\n📊 Calcul des erreurs...")
    errors = compute_errors(results, refs, ref_key=args.ref_key)
    print(f"   Erreurs calculées pour {len(errors)} événements")

    # report
    print("\n📝 Génération du rapport...")
    write_report(
        out_path=args.out,
        event_to_cluster=event_to_cluster,
        cluster_calib=cluster_calib,
        results=results,
        errors=errors,
        refs=refs,
        ref_key=args.ref_key,
    )

    # summary
    print("\n" + "=" * 70)
    print("✅ CALIBRATION PAR CLUSTER TERMINÉE")
    print("=" * 70)
    abs_errs = [abs(v) for v in errors.values()]
    if abs_errs:
        print(f"MAE globale : {np.mean(abs_errs):.2f}%")
        print(f"Médiane     : {np.median(abs_errs):.2f}%")
    print(f"Rapport     : {args.out}")
    print(f"Calibs JSON : {args.calib_json}")
    print("=" * 70)


if __name__ == "__main__":
    main()
