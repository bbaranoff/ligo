#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline clustering "from raw LIGO results json" (results/GW*.json)

But:
  1) Écrémer les outliers en amont avec DBSCAN (label = -1)
  2) Clustering fin du noyau restant avec KMeans (labels 0..k-1)

Inputs:
  - freq_Hz
  - dEdf_internal
  - (optionnel) E_total_J si présent (pas utilisé ici)

Features:
  logE, nu_mean, nu_peak, nu_invf, frac_bw, Q_eff, peak_rel, R_LH

Sortie:
  - clusters report (inclut CLUSTER -1 = outliers DBSCAN)
  - optionnel: CSV features+label
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA


# -----------------------
# Helpers numériques
# -----------------------

def trapz(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapz(y, x))


def safe_log10(x: float, eps: float = 1e-300) -> float:
    return float(np.log10(max(float(x), eps)))


def as_float_array(a) -> np.ndarray:
    return np.asarray(a, dtype=float)


def spectral_cdf_quantiles(
    f: np.ndarray,
    w: np.ndarray,
    qs=(0.10, 0.50, 0.90),
) -> List[float]:
    """
    Quantiles de la distribution définie par w(f) >= 0 (normalisée via intégrale).
    Retourne f(q) pour q in qs.
    """
    f = as_float_array(f)
    w = np.maximum(as_float_array(w), 0.0)

    tot = trapz(w, f)
    if not np.isfinite(tot) or tot <= 0.0 or f.size < 4:
        return [float("nan") for _ in qs]

    # CDF via intégrale cumulée trapz (version discrète stable)
    df = np.diff(f)
    mid = 0.5 * (w[:-1] + w[1:])
    cum = np.concatenate([[0.0], np.cumsum(mid * df)])
    cdf = cum / cum[-1]

    out = []
    for q in qs:
        out.append(float(np.interp(float(q), cdf, f)))
    return out


# -----------------------
# Features
# -----------------------

@dataclass
class Features:
    event: str
    logE: float
    nu_mean: float
    nu_peak: float
    nu_invf: float
    frac_bw: float
    Q_eff: float
    peak_rel: float
    R_LH: float

    def as_row(self, order: List[str]) -> List[float]:
        d = self.__dict__
        return [float(d[k]) for k in order]


def compute_features_from_json(path: str, f_split: float) -> Optional[Features]:
    with open(path, "r") as f:
        d = json.load(f)

    event = d.get("event") or os.path.splitext(os.path.basename(path))[0]

    f_hz = d.get("freq_Hz")
    dedf = d.get("dEdf_internal")
    if f_hz is None or dedf is None:
        return None

    f_hz = as_float_array(f_hz)
    dedf = np.maximum(as_float_array(dedf), 0.0)

    if f_hz.size < 8 or dedf.size != f_hz.size:
        return None

    # Energie interne (unités internes)
    E_int = trapz(dedf, f_hz)
    logE = safe_log10(E_int)

    # nu_mean (barycentre énergétique)
    nu_mean = 0.0
    if np.isfinite(E_int) and E_int > 0:
        nu_mean = float(trapz(f_hz * dedf, f_hz) / E_int)

    # nu_peak (fréquence au max)
    idx = int(np.argmax(dedf)) if dedf.size else 0
    nu_peak = float(f_hz[idx]) if f_hz.size else 0.0

    # nu_invf = (∫dEdf) / (∫dEdf/f)
    denom_invf = trapz(dedf / np.maximum(f_hz, 1e-12), f_hz)
    nu_invf = float(E_int / denom_invf) if np.isfinite(denom_invf) and denom_invf > 0 else 0.0

    # Bande passante: f10, f90 via CDF(dEdf)
    f10, _f50, f90 = spectral_cdf_quantiles(f_hz, dedf, qs=(0.10, 0.50, 0.90))
    dnu_80 = float(f90 - f10) if np.isfinite(f90) and np.isfinite(f10) else float("nan")

    # frac_bw = dnu_80 / nu_mean
    frac_bw = float(dnu_80 / max(nu_mean, 1e-12)) if np.isfinite(dnu_80) else float("nan")

    # Q_eff ~ nu_mean / dnu_80
    Q_eff = float(max(nu_mean, 1e-12) / max(dnu_80, 1e-12)) if np.isfinite(dnu_80) else float("nan")

    # peak_rel = peak / mean(dEdf)
    mean_spec = float(np.mean(dedf)) if dedf.size else 0.0
    peak_rel = float(np.max(dedf) / max(mean_spec, 1e-30)) if mean_spec > 0 else 0.0

    # R_LH = log10(E_low / E_high) autour d'un split
    mL = f_hz < float(f_split)
    mH = f_hz >= float(f_split)
    E_low = trapz(dedf[mL], f_hz[mL]) if np.any(mL) else 0.0
    E_high = trapz(dedf[mH], f_hz[mH]) if np.any(mH) else 0.0
    R_LH = float(np.log10((E_low + 1e-30) / (E_high + 1e-30)))

    return Features(
        event=event,
        logE=logE,
        nu_mean=nu_mean,
        nu_peak=nu_peak,
        nu_invf=nu_invf,
        frac_bw=frac_bw,
        Q_eff=Q_eff,
        peak_rel=peak_rel,
        R_LH=R_LH,
    )


# -----------------------
# Report
# -----------------------

def fmt(x: float, spec: str) -> str:
    if x is None or not np.isfinite(x):
        return "NA"
    return format(float(x), spec)


def write_clusters_report(
    out_path: str,
    feats: List[Features],
    labels: np.ndarray,
    order: List[str],
    f_split: float,
    header_extra: str = "",
) -> None:
    clusters: Dict[int, List[Features]] = {}
    for ft, lab in zip(feats, labels):
        clusters.setdefault(int(lab), []).append(ft)

    lines = []
    head = f"features: {', '.join(order)} | f_split={float(f_split):.1f}"
    if header_extra:
        head += " | " + header_extra
    lines.append(head)
    lines.append("")

    for cid in sorted(clusters.keys()):
        group = clusters[cid]

        mat = np.array([g.as_row(order) for g in group], dtype=float)
        means = np.nanmean(mat, axis=0)

        means_s = " | ".join(f"{k}={fmt(v, '.3g')}" for k, v in zip(order, means))
        lines.append(f"=== CLUSTER {cid} ({len(group)}) ===")
        lines.append("means: " + means_s)
        lines.append("Event | logE | nu_mean | frac_bw | Q_eff | R_LH")
        lines.append("-|-|-|-|-|-")

        group_sorted = sorted(group, key=lambda g: (g.nu_mean, g.logE))
        for g in group_sorted:
            lines.append(
                f"{g.event} | "
                f"{fmt(g.logE, '.3f')} | "
                f"{fmt(g.nu_mean, '.1f')} | "
                f"{fmt(g.frac_bw, '.3f')} | "
                f"{fmt(g.Q_eff, '.3f')} | "
                f"{fmt(g.R_LH, '.3f')}"
            )
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# -----------------------
# Main
# -----------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/GW*.json", help="pattern des JSON results")
    ap.add_argument("--k", type=int, default=4, help="nombre de clusters KMeans (sur inliers)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--f-split", type=float, default=150.0, help="split Hz pour R_LH")

    # DBSCAN (filtre outliers)
    ap.add_argument("--db-eps", type=float, default=1.4, help="DBSCAN eps (en espace standardisé)")
    ap.add_argument("--db-min-samples", type=int, default=3, help="DBSCAN min_samples")

    # PCA optionnelle pour KMeans (pas pour DBSCAN)
    ap.add_argument("--pca", type=int, default=0, help="PCA dim pour KMeans (0 = pas de PCA)")

    ap.add_argument("--out", default="clusters_dbscan_kmeans.txt")
    ap.add_argument("--csv", default=None, help="optionnel: export features+label en CSV")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"[FATAL] no files matched: {args.glob}")

    feats: List[Features] = []
    for p in paths:
        ft = compute_features_from_json(p, f_split=args.f_split)
        if ft is not None:
            feats.append(ft)

    if len(feats) < 2:
        raise SystemExit(f"[FATAL] not enough events (n={len(feats)})")

    order = ["logE", "nu_mean", "nu_peak", "nu_invf", "frac_bw", "Q_eff", "peak_rel", "R_LH"]

    X = np.array([f.as_row(order) for f in feats], dtype=float)

    # Remplace NaN/inf par médiane colonne (robuste)
    X2 = X.copy()
    for j in range(X2.shape[1]):
        col = X2[:, j]
        m = np.isfinite(col)
        med = float(np.median(col[m])) if np.any(m) else 0.0
        col[~m] = med
        X2[:, j] = col

    # Standardisation globale
    scaler = StandardScaler()
    Z = scaler.fit_transform(X2)

    # (1) DBSCAN pour écrémer les outliers
    db = DBSCAN(eps=float(args.db_eps), min_samples=int(args.db_min_samples))
    labels_db = db.fit_predict(Z)
    inlier_mask = labels_db != -1

    n_in = int(inlier_mask.sum())
    n_out = int((~inlier_mask).sum())
    if n_in < max(2, int(args.k)):
        raise SystemExit(
            f"[FATAL] not enough inliers after DBSCAN for k={args.k} "
            f"(inliers={n_in}, outliers={n_out}). "
            f"Try increasing --db-eps or decreasing --db-min-samples."
        )

    # (2) KMeans sur les inliers uniquement (éventuellement en PCA)
    Z_in = Z[inlier_mask]
    if args.pca and int(args.pca) > 0:
        pca = PCA(n_components=int(args.pca), random_state=int(args.seed))
        Zk_in = pca.fit_transform(Z_in)
    else:
        Zk_in = Z_in

    km = KMeans(
        n_clusters=int(args.k),
        random_state=int(args.seed),
        n_init="auto",
    )
    labels_in = km.fit_predict(Zk_in)

    # Labels finaux: -1 pour outliers, 0..k-1 pour inliers
    labels = np.full(Z.shape[0], -1, dtype=int)
    labels[inlier_mask] = labels_in

    header_extra = (
        f"DBSCAN(eps={args.db_eps:.3g},min_samples={int(args.db_min_samples)}) "
        f"-> inliers={n_in} outliers={n_out} ; "
        f"KMeans(k={int(args.k)})"
    )
    write_clusters_report(
        out_path=args.out,
        feats=feats,
        labels=labels,
        order=order,
        f_split=args.f_split,
        header_extra=header_extra,
    )
    print(f"[OK] wrote {args.out} (n={len(feats)} inliers={n_in} outliers={n_out} k={args.k} pca={args.pca})")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["event", "cluster"] + order)
            for ft, lab, row in zip(feats, labels, X):
                w.writerow([ft.event, int(lab)] + [float(x) for x in row])
        print(f"[OK] wrote {args.csv}")


if __name__ == "__main__":
    main()
