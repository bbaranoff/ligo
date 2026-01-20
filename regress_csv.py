#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
from typing import List, Tuple

import numpy as np
import pandas as pd


def _norm_cdf(x: float) -> float:
    # CDF normale standard via erf (pas besoin de scipy)
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def ols_fit(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    X: (n, p) avec intercept déjà inclus si désiré
    y: (n,)
    returns: beta (p,), se_beta (p,), y_hat (n,), r2
    """
    n, p = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    resid = y - y_hat

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    dof = n - p
    if dof <= 0:
        raise SystemExit(f"[FATAL] dof<=0 (n={n}, p={p}). Trop de features ou pas assez de points.")

    sigma2 = ss_res / dof
    XtX_inv = np.linalg.inv(X.T @ X)
    var_beta = sigma2 * np.diag(XtX_inv)
    se_beta = np.sqrt(np.maximum(var_beta, 0.0))

    return beta, se_beta, y_hat, r2


def compute_vif(X_no_intercept: np.ndarray, names: List[str]) -> pd.DataFrame:
    """
    VIF_i = 1 / (1 - R²_i) où R²_i est obtenu en régressant Xi sur les autres Xj.
    X_no_intercept: (n, k) sans intercept
    """
    n, k = X_no_intercept.shape
    vifs = []
    for i in range(k):
        y = X_no_intercept[:, i]
        X_others = np.delete(X_no_intercept, i, axis=1)

        # intercept + others
        X = np.column_stack([np.ones(n), X_others])
        beta, _, y_hat, r2 = ols_fit(X, y)
        # r2 de la regression de Xi sur les autres
        if r2 is None or np.isnan(r2):
            vif = float("nan")
        else:
            denom = 1.0 - r2
            vif = float("inf") if denom <= 0 else 1.0 / denom
        vifs.append(vif)

    return pd.DataFrame({"feature": names, "VIF": vifs}).sort_values("VIF", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Régression OLS sur un CSV (linear ou log) + VIF + résidus par event.")
    ap.add_argument("--csv", required=True, help="Chemin CSV (ex: spectra_integrated.csv)")
    ap.add_argument("--y", required=True, help="Colonne cible (ex: E_total_J_int)")
    ap.add_argument("--x", nargs="+", required=True, help="Colonnes explicatives (ex: nu_eff_hz bw_all_sigmaf_hz)")
    ap.add_argument("--mode", choices=["linear", "log"], default="linear", help="linear: y ~ X ; log: log(y) ~ log(X)")
    ap.add_argument("--dropna", action="store_true", help="Dropna strict sur y/x (sinon fait déjà le minimum)")
    ap.add_argument("--out-residuals", default="residuals.csv", help="CSV de sortie des résidus par event")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    cols = ["event", args.y] + args.x
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print("[FATAL] Colonnes manquantes:", ", ".join(missing))
        print("\n[INFO] Colonnes disponibles:")
        for col in df.columns.tolist():
            print("  -", col)

        print("\n[HINT] Suggestions (fuzzy match):")
        for m in missing:
            close = difflib.get_close_matches(m, df.columns.tolist(), n=5, cutoff=0.4)
            print(f"  {m} -> {close}")

        raise SystemExit(2)
    d = df[cols].copy()
    d = d.dropna(subset=[args.y] + args.x)  # minimum vital

    # Numeric cast
    d[args.y] = pd.to_numeric(d[args.y], errors="coerce")
    for c in args.x:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=[args.y] + args.x)

    if len(d) < 3:
        raise SystemExit(f"[FATAL] Pas assez de lignes après filtrage: {len(d)}")

    y = d[args.y].to_numpy(dtype=float)
    X_raw = d[args.x].to_numpy(dtype=float)

    if args.mode == "log":
        # log exige >0
        mask = (y > 0) & np.all(X_raw > 0, axis=1)
        d = d.loc[mask].copy()
        y = np.log(d[args.y].to_numpy(dtype=float))
        X_raw = np.log(d[args.x].to_numpy(dtype=float))

    n = len(d)
    X = np.column_stack([np.ones(n), X_raw])  # intercept

    beta, se_beta, y_hat, r2 = ols_fit(X, y)

    # t-stats et p-values (approx normale si n grand, sinon ça reste une approximation correcte en pratique)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = beta / se_beta
    p_vals = np.array([2.0 * (1.0 - _norm_cdf(abs(float(t)))) if np.isfinite(t) else float("nan") for t in t_stats])

    # Sortie résultats
    names = ["intercept"] + args.x
    out = pd.DataFrame({
        "term": names,
        "beta": beta,
        "se": se_beta,
        "t": t_stats,
        "p_approx_norm": p_vals,
    })

    print("\n===== OLS =====")
    print(f"CSV   : {args.csv}")
    print(f"y     : {args.y}")
    print(f"x     : {args.x}")
    print(f"mode  : {args.mode}")
    print(f"n     : {n}")
    print(f"R²    : {r2:.6f}\n")
    print(out.to_string(index=False))

    # VIF (sur X sans intercept)
    vif_df = compute_vif(X_raw, args.x)
    print("\n===== VIF (colinéarité, >10 = gros souci) =====")
    print(vif_df.to_string(index=False))

    # Résidus (en échelle log si mode=log)
    resid = y - y_hat
    resid_df = pd.DataFrame({
        "event": d["event"].to_numpy(),
        "y_used": y,
        "y_hat": y_hat,
        "resid": resid,
    }).sort_values("resid", ascending=False)

    resid_df.to_csv(args.out_residuals, index=False)
    print(f"\n[OK] Résidus exportés -> {args.out_residuals}")

    # petit top outliers
    print("\n===== Top 5 résidus (positifs) =====")
    print(resid_df.head(5).to_string(index=False))
    print("\n===== Top 5 résidus (négatifs) =====")
    print(resid_df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
