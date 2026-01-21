#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clustering "from raw LIGO results json" (results/GW*.json)
- No LIGO refs
- No masses
- Uses only freq_Hz + dEdf_internal (and optionally E_total_J if present)
Features:
  logE, nu_mean, nu_peak, nu_invf, frac_bw, Q_eff, peak_rel, R_LH
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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

def spectral_cdf_quantiles(f: np.ndarray, w: np.ndarray, qs=(0.10, 0.50, 0.90)) -> List[float]:
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

    # Energie interne (dans tes unités internes)
    E_int = trapz(dedf, f_hz)
    logE = safe_log10(E_int)

    # nu_mean (barycentre énergétique)
    den = E_int
    nu_mean = 0.0
    if np.isfinite(den) and den > 0:
        nu_mean = float(trapz(f_hz * dedf, f_hz) / den)

    # nu_peak (fréquence au max)
    idx = int(np.argmax(dedf)) if dedf.size else 0
    nu_peak = float(f_hz[idx]) if f_hz.size else 0.0

    # nu_invf = (∫dEdf) / (∫dEdf/f)
    # => correspond au "barycentre 1/f" (harmonique-ish), utile pour chirp/low freq
    denom_invf = trapz(dedf / np.maximum(f_hz, 1e-12), f_hz)
    nu_invf = float(E_int / denom_invf) if np.isfinite(denom_invf) and denom_invf > 0 else 0.0

    # Bande passante: f10, f90 via CDF(dEdf)
    f10, f50, f90 = spectral_cdf_quantiles(f_hz, dedf, qs=(0.10, 0.50, 0.90))
    dnu_80 = float(f90 - f10) if np.isfinite(f90) and np.isfinite(f10) else float("nan")

    # frac_bw = dnu_80 / nu_mean (dimensionless)
    frac_bw = float(dnu_80 / max(nu_mean, 1e-12)) if np.isfinite(dnu_80) else float("nan")

    # Q_eff ~ nu_mean / dnu_80 (l’inverse de frac_bw)
    Q_eff = float(max(nu_mean, 1e-12) / max(dnu_80, 1e-12)) if np.isfinite(dnu_80) else float("nan")

    # peak_rel = peak / mean(dEdf) sur la bande (évite l’échelle brute)
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
) -> None:
    # regroupe par cluster
    clusters: Dict[int, List[Features]] = {}
    for ft, lab in zip(feats, labels):
        clusters.setdefault(int(lab), []).append(ft)

    lines = []
    lines.append(f"features: {', '.join(order)} | f_split={float(f_split):.1f}")
    lines.append("")

    for cid in sorted(clusters.keys()):
        group = clusters[cid]

        # moyennes
        mat = np.array([g.as_row(order) for g in group], dtype=float)
        means = np.nanmean(mat, axis=0)

        means_s = " | ".join(
            f"{k}={fmt(v, '.3g')}" if k not in ("E_int",) else f"{k}={fmt(v, '.3g')}"
            for k, v in zip(order, means)
        )
        lines.append(f"=== CLUSTER {cid} ({len(group)}) ===")
        lines.append("means: " + means_s)
        lines.append("Event | logE | nu_mean | frac_bw | Q_eff | R_LH")
        lines.append("-|-|-|-|-|-")

        # tri: par nu_mean puis logE (comme un “profil”)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/GW*.json", help="pattern des JSON results")
    ap.add_argument("--k", type=int, default=4, help="nombre de clusters KMeans")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--f-split", type=float, default=150.0, help="split Hz pour R_LH")
    ap.add_argument("--pca", type=int, default=2, help="PCA dim (0 = pas de PCA)")
    ap.add_argument("--out", default="clusters_kmeans.txt")
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

    if len(feats) < max(2, args.k):
        raise SystemExit(f"[FATAL] not enough events for k={args.k} (n={len(feats)})")

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

    scaler = StandardScaler()
    Z = scaler.fit_transform(X2)

    if args.pca and args.pca > 0:
        pca = PCA(n_components=int(args.pca), random_state=int(args.seed))
        Zc = pca.fit_transform(Z)
    else:
        Zc = Z

    km = KMeans(n_clusters=int(args.k), random_state=int(args.seed), n_init="auto")
    labels = km.fit_predict(Zc)

    write_clusters_report(
        out_path=args.out,
        feats=feats,
        labels=labels,
        order=order,
        f_split=args.f_split,
    )
    print(f"[OK] wrote {args.out} (n={len(feats)} k={args.k} pca={args.pca})")

    if args.csv:
        # CSV simple: event, label, puis features
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["event", "cluster"] + order)
            for ft, lab, row in zip(feats, labels, X):
                w.writerow([ft.event, int(lab)] + [float(x) for x in row])
        print(f"[OK] wrote {args.csv}")


if __name__ == "__main__":
    main()
