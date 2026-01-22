#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calibration cluster-by-cluster pour LIGO spectral analysis

Pipeline:
1) Charge refs + event_params
2) Calcule features en appelant analyze_event(return_internals=True)
3) Clustering: DBSCAN -> outliers (-1) + KMeans sur inliers (k clusters)
4) Calibration LSQ par cluster: fit K = H_STAR*SCALE_EJ (dégénéré)
5) Ré-analyse et produit un tableau final avec erreurs (%)

Important:
- On passe peak_norm / peak_quantile UNE SEULE FOIS (jamais via **kwargs).
- Si --k 0 : pas de KMeans -> tous les inliers reçoivent cluster=0.
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

# sklearn (DBSCAN/KMeans)
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

import ligo_spectral_planck  # doit être dans le même dossier / PYTHONPATH


# -------------------------
# Utils
# -------------------------

def jload(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def jdump(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def isfinite_pos(x: float) -> bool:
    return np.isfinite(x) and x > 0.0

def fmt_sci(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.3e}"

def signed_pct(err: float) -> str:
    # err already in percent
    sign = "+" if err >= 0 else "-"
    return f"{sign}{abs(err):.2f}"

def rel_err_pct(pred: float, ref: float) -> float:
    # signed relative error: (pred-ref)/ref *100
    if not isfinite_pos(ref):
        return float("nan")
    return 100.0 * (pred - ref) / ref


# -------------------------
# Feature extraction
# -------------------------

FEATURE_KEYS = ["logE", "nu_mean", "nu_peak", "nu_invf", "frac_bw", "Q_eff", "peak_rel", "R_LH"]

def extract_features_from_internals(intern: Dict[str, Any]) -> Dict[str, float]:
    """
    intern = retour de analyze_event(..., return_internals=True)
    Doit contenir les features calculées côté ligo_spectral_planck.
    """
    feats = {}
    for k in FEATURE_KEYS:
        v = intern.get(k, None)
        feats[k] = float(v) if v is not None and np.isfinite(v) else float("nan")
    return feats


# -------------------------
# Clustering
# -------------------------

@dataclass
class ClusterResult:
    labels: Dict[str, int]          # event -> cluster_id  (-1 for outlier)
    inliers: List[str]
    outliers: List[str]
    kmeans_k: int

def cluster_events(
    events: List[str],
    feats_by_event: Dict[str, Dict[str, float]],
    db_eps: float,
    db_min_samples: int,
    k: int,
    seed: int
) -> ClusterResult:
    """
    DBSCAN sur features standardisées.
    - Outliers => label -1
    - Inliers => si k>0 : KMeans(k) sur inliers, labels 0..k-1
                si k==0 : tous les inliers label 0
    """
    X = []
    kept = []
    for ev in events:
        f = feats_by_event.get(ev, {})
        row = [f.get(key, float("nan")) for key in FEATURE_KEYS]
        if not all(np.isfinite(row)):
            continue
        X.append(row)
        kept.append(ev)

    if len(kept) < max(db_min_samples, 2):
        raise RuntimeError(f"Pas assez d'events avec features valides pour clusteriser (n={len(kept)}).")

    X = np.asarray(X, float)
    Xs = StandardScaler().fit_transform(X)

    db = DBSCAN(eps=db_eps, min_samples=db_min_samples)
    db_labels = db.fit_predict(Xs)  # -1 outliers, else cluster id (DBSCAN internal)

    inliers = [kept[i] for i, lab in enumerate(db_labels) if lab != -1]
    outliers = [kept[i] for i, lab in enumerate(db_labels) if lab == -1]

    labels: Dict[str, int] = {}
    for ev in outliers:
        labels[ev] = -1

    # If no inliers, we keep everything as -1
    if len(inliers) == 0:
        return ClusterResult(labels=labels, inliers=[], outliers=outliers, kmeans_k=0)

    if k <= 0:
        for ev in inliers:
            labels[ev] = 0
        return ClusterResult(labels=labels, inliers=inliers, outliers=outliers, kmeans_k=0)

    # KMeans on inliers
    Xin = np.asarray([Xs[kept.index(ev)] for ev in inliers], float)
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    km_lab = km.fit_predict(Xin)

    for ev, lab in zip(inliers, km_lab):
        labels[ev] = int(lab)

    return ClusterResult(labels=labels, inliers=inliers, outliers=outliers, kmeans_k=k)


# -------------------------
# Calibration LSQ
# -------------------------

def calibrate_lsq_K(
    events: List[str],
    y_obs_dict: Dict[str, float],
    params: Dict[str, Any],
    bands: Dict[str, Tuple[float, float]],
    analyze_common: Dict[str, Any],
    peak_norm: bool,
    peak_quantile: float,
    use_virgo: bool
) -> Tuple[float, float, Dict[str, float]]:
    """
    Fit K = H_STAR*SCALE_EJ, en minimisant sum((y_obs - K*y_model)^2)
    où y_model = Energie interne calculée avec hstar_in=1 scale_ej_in=1.

    Retour:
      H_STAR (choisi 1), SCALE_EJ (=K_hat), stats dict
    """
    y_models = []
    y_obs = []
    used = []

    for ev in events:
        yref = y_obs_dict.get(ev, None)
        if yref is None or (not isfinite_pos(float(yref))):
            continue

        try:
            intern = ligo_spectral_planck.analyze_event(
                event=ev,
                params=params,
                bands=bands,
                use_virgo=use_virgo,
                peak_norm=peak_norm,
                peak_quantile=peak_quantile,
                hstar_in=1.0,
                scale_ej_in=1.0,
                plot=False,
                return_internals=True,
                **analyze_common,
            )
        except Exception as e:
            gps = params.get(ev, {}).get("gps_time", "NA")
            print(f"[WARN] skip {ev}: analyze_event failed (gps={gps}): {e}")
            continue

        # l'énergie interne doit être dans intern["energy_J"] (ou similaire)
        em = intern.get("energy_J", None)
        if em is None or (not isfinite_pos(float(em))):
            continue

        y_models.append(float(em))
        y_obs.append(float(yref))
        used.append(ev)

    if len(used) < 2:
        raise RuntimeError(f"[FATAL] Pas assez d'events valides pour LSQ (n={len(used)})")

    y_models = np.asarray(y_models, float)
    y_obs = np.asarray(y_obs, float)

    # Fit K in least squares: y_obs ~ K * y_model
    # K_hat = (y_model^T y_obs) / (y_model^T y_model)
    denom = float(np.dot(y_models, y_models))
    if denom <= 0:
        raise RuntimeError("[FATAL] denom <= 0 dans LSQ")

    K_hat = float(np.dot(y_models, y_obs)) / denom

    # Errors (relative)
    y_pred = K_hat * y_models
    rel = np.abs((y_pred - y_obs) / y_obs)
    stats = {
        "events_used": len(used),
        "K_hat": K_hat,
        "rel_MAE": float(np.mean(rel) * 100.0),
        "rel_MED": float(np.median(rel) * 100.0),
    }

    # degenerate: choose H_STAR=1 ; SCALE_EJ=K
    return 1.0, K_hat, stats


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Calibration cluster-by-cluster pour LIGO spectral analysis")
    ap.add_argument("--refs", required=True, help="JSON refs LIGO (ligo_refs.json)")
    ap.add_argument("--event-params", default="event_params.json", help="JSON params events")
    ap.add_argument("--peak-norm", action="store_true", help="Normalise l'amplitude sur le pic")
    ap.add_argument("--peak-quantile", type=float, default=0.999)

    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--db-eps", type=float, default=1.4)
    ap.add_argument("--db-min-samples", type=int, default=3)
    ap.add_argument("--f-split", type=float, default=150.0)  # si utilisé en interne pour features

    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--flow", type=float, default=20.0)
    ap.add_argument("--fhigh", type=float, default=600.0)
    ap.add_argument("--tau-band", type=float, nargs=2, default=[30.0, 400.0])
    ap.add_argument("--nu-band", type=float, nargs=2, default=[20.0, 600.0])
    ap.add_argument("--no-virgo", action="store_true")

    ap.add_argument("--signal-win", type=float, default=0.8)
    ap.add_argument("--noise-pad", type=float, default=900.0)
    ap.add_argument("--distance-mpc", type=float, default=410.0)

    ap.add_argument("--out", default="cluster_calibrations.json")
    ap.add_argument("--ref-key", choices=["energy_J", "msun_c2"], default="energy_J")
    ap.add_argument("--exclude-cls", nargs="*", default=[])
    ap.add_argument("--exclude-cluster-minus1", action="store_true",
                    help="Exclure cluster -1 de la calibration (utilise défaut 1,1)")
    args = ap.parse_args()

    refs = jload(args.refs)
    params = jload(args.event_params)

    # y_obs_dict from refs
    y_obs_dict: Dict[str, float] = {}
    for r in refs:
        ev = r.get("event", None) or r.get("name", None)
        if not ev:
            continue
        cls = r.get("cls", None)
        if cls in set(args.exclude_cls):
            continue
        y = r.get(args.ref_key, None)
        if y is None:
            continue
        try:
            y = float(y)
        except Exception:
            continue
        if not isfinite_pos(y):
            continue
        y_obs_dict[ev] = y

    # events available
    events = [ev for ev in y_obs_dict.keys() if ev in params]
    events = sorted(events)

    print("📚 Chargement des références...")
    print(f"   Références chargées: {len(refs)} total, {len(y_obs_dict)} valides")
    print("\n📋 Chargement des paramètres d'événements...")
    print(f"   Événements dans params: {len(params)}")
    print("\n🔍 Clustering des événements...")

    # Common analyze args (NE PAS inclure peak_norm/peak_quantile ici)
    analyze_common = dict(
        flow=float(args.flow),
        fhigh=float(args.fhigh),
        signal_win=float(args.signal_win),
        noise_pad=float(args.noise_pad),
        distance_mpc=float(args.distance_mpc),
    )

    # bands (if your analyze_event uses it)
    bands = dict(
        tau_band=(float(args.tau_band[0]), float(args.tau_band[1])),
        nu_band=(float(args.nu_band[0]), float(args.nu_band[1])),
        f_split=float(args.f_split),
    )

    use_virgo = not bool(args.no_virgo)

    # compute features
    feats_by_event: Dict[str, Dict[str, float]] = {}
    ok_events = []
    for ev in events:
        try:
            intern = ligo_spectral_planck.analyze_event(
                event=ev,
                params=params,
                bands=bands,
                use_virgo=use_virgo,
                peak_norm=bool(args.peak_norm),
                peak_quantile=float(args.peak_quantile),
                hstar_in=1.0,
                scale_ej_in=1.0,
                plot=False,
                return_internals=True,
                **analyze_common,
            )
            feats_by_event[ev] = extract_features_from_internals(intern)
            ok_events.append(ev)
        except Exception as e:
            gps = params.get(ev, {}).get("gps_time", "NA")
            print(f"[WARN] skip {ev}: cannot extract features (gps={gps}): {e}")

    if len(ok_events) < 5:
        raise SystemExit(f"[FATAL] Trop peu d'événements feature-ok: {len(ok_events)}")

    cres = cluster_events(
        events=ok_events,
        feats_by_event=feats_by_event,
        db_eps=float(args.db_eps),
        db_min_samples=int(args.db_min_samples),
        k=int(args.k),
        seed=int(args.seed),
    )

    # Count clusters
    cluster_ids = sorted(set(cres.labels.values()))
    counts = {cid: 0 for cid in cluster_ids}
    for ev, cid in cres.labels.items():
        counts[cid] += 1

    print(f"[clustering] n_events={len(ok_events)} inliers={len(cres.inliers)} outliers={len(cres.outliers)} k={args.k}")
    for cid in sorted(counts.keys()):
        print(f"   Cluster {cid:>2}: {counts[cid]} événements")

    # Calibrate per cluster
    print("\n🔧 Calibration par cluster...")
    cluster_calibs: Dict[str, Any] = {}

    for cid in sorted(counts.keys()):
        evs = [ev for ev, lab in cres.labels.items() if lab == cid]
        if cid == -1 and args.exclude_cluster_minus1:
            cluster_calibs[str(cid)] = {
                "H_STAR": 1.0,
                "SCALE_EJ": 1.0,
                "events_used": 0,
                "note": "excluded cluster -1 => default calibration",
            }
            continue

        try:
            Hs, Sc, stats = calibrate_lsq_K(
                events=evs,
                y_obs_dict=y_obs_dict,
                params=params,
                bands=bands,
                analyze_common=analyze_common,
                peak_norm=bool(args.peak_norm),
                peak_quantile=float(args.peak_quantile),
                use_virgo=use_virgo,
            )
            cluster_calibs[str(cid)] = {
                "H_STAR": Hs,
                "SCALE_EJ": Sc,
                **stats,
            }
            print(f"✅ Cluster {cid}: H_STAR={Hs:.6e}, SCALE_EJ={Sc:.6e} | rel_MAE={stats['rel_MAE']:.2f}% rel_MED={stats['rel_MED']:.2f}%")
        except Exception as e:
            print(f"[ERROR] Cluster {cid} calibration failed: {e}")
            cluster_calibs[str(cid)] = {
                "H_STAR": 1.0,
                "SCALE_EJ": 1.0,
                "events_used": 0,
                "note": f"failed => default calibration: {e}",
            }

    jdump(args.out, cluster_calibs)
    print(f"\n💾 Calibrations sauvegardées: {args.out}")

    # Re-run events with cluster calibration and print summary
    print("\n=== SYNTHÈSE PROGRESSIVE (E affinée par cluster) ===")
    header = "Event | cluster | ν_eff | ν_ref | τ[s] | τ_ref | M⊙c² | M⊙c²_r | Energie[J] | E_aff[J] | E_ref[J] | err_raw[%] | err_aff[%] | Notes"
    print(header)
    print("-" * len(header))

    rows = []
    for ev in ok_events:
        cid = cres.labels.get(ev, -1)
        calib = cluster_calibs.get(str(cid), {"H_STAR": 1.0, "SCALE_EJ": 1.0})
        Hs = float(calib.get("H_STAR", 1.0))
        Sc = float(calib.get("SCALE_EJ", 1.0))

        try:
            intern_raw = ligo_spectral_planck.analyze_event(
                event=ev,
                params=params,
                bands=bands,
                use_virgo=use_virgo,
                peak_norm=bool(args.peak_norm),
                peak_quantile=float(args.peak_quantile),
                hstar_in=1.0,
                scale_ej_in=1.0,
                plot=False,
                return_internals=True,
                **analyze_common,
            )
            intern_aff = ligo_spectral_planck.analyze_event(
                event=ev,
                params=params,
                bands=bands,
                use_virgo=use_virgo,
                peak_norm=bool(args.peak_norm),
                peak_quantile=float(args.peak_quantile),
                hstar_in=Hs,
                scale_ej_in=Sc,
                plot=False,
                return_internals=True,
                **analyze_common,
            )
        except Exception as e:
            gps = params.get(ev, {}).get("gps_time", "NA")
            print(f"[WARN] skip {ev}: analyze_event failed (gps={gps}): {e}")
            continue

        # fields
        nu_eff = float(intern_raw.get("nu_eff", float("nan")))
        tau_s = float(intern_raw.get("tau_s", float("nan")))
        e_raw = float(intern_raw.get("energy_J", float("nan")))
        e_aff = float(intern_aff.get("energy_J", float("nan")))

        # refs
        # Here we keep nu_ref/tau_ref/notes from refs if present
        rref = next((r for r in refs if (r.get("event") == ev or r.get("name") == ev)), {})
        nu_ref = float(rref.get("nu_ref", float("nan"))) if rref.get("nu_ref") is not None else float("nan")
        tau_ref = float(rref.get("tau_ref", float("nan"))) if rref.get("tau_ref") is not None else float("nan")
        e_ref = float(rref.get("energy_J", float("nan"))) if args.ref_key == "energy_J" else float("nan")
        note = str(rref.get("notes", rref.get("note", rref.get("cls", ""))))

        err_raw = rel_err_pct(e_raw, e_ref)
        err_aff = rel_err_pct(e_aff, e_ref)

        rows.append((cid, ev, nu_eff, nu_ref, tau_s, tau_ref, e_raw, e_aff, e_ref, err_raw, err_aff, note))

    # sort by cluster then event
    rows.sort(key=lambda x: (x[0], x[1]))

    for (cid, ev, nu_eff, nu_ref, tau_s, tau_ref, e_raw, e_aff, e_ref, err_raw, err_aff, note) in rows:
        print(
            f"{ev} | {cid:>7} | "
            f"{nu_eff:6.1f} | {nu_ref:6.1f} | "
            f"{tau_s:+.5f} | {tau_ref:+.5f} | "
            f"{fmt_sci(e_raw)} | {fmt_sci(e_aff)} | {fmt_sci(e_ref)} | "
            f"{signed_pct(err_raw):>9} | {signed_pct(err_aff):>9} | {note}"
        )


if __name__ == "__main__":
    main()
