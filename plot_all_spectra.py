#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import cumulative_trapezoid, trapezoid

os.makedirs("plots", exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (11, 6),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
})

def _as_float_array(x, name):
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name}: attendu 1D, got shape={a.shape}")
    return a

def load_event_json(path):
    with open(path, "r") as f:
        d = json.load(f)

    event = d.get("event", os.path.basename(path).replace(".json", ""))

    f_hz = _as_float_array(d.get("freq_Hz", []), "freq_Hz")
    dEdf = _as_float_array(d.get("dEdf_internal", []), "dEdf_internal")

    if f_hz.size == 0 or dEdf.size == 0 or f_hz.size != dEdf.size:
        raise ValueError(f"{event}: freq_Hz/dEdf_internal manquants ou tailles incohérentes")

    # tri par fréquence + nettoyage
    order = np.argsort(f_hz)
    f_hz = f_hz[order]
    dEdf = dEdf[order]

    # ferme la porte aux NaN/Inf/negatifs (fermeture “E>=0”)
    f_hz = np.nan_to_num(f_hz, nan=0.0, posinf=0.0, neginf=0.0)
    dEdf = np.nan_to_num(dEdf, nan=0.0, posinf=0.0, neginf=0.0)
    dEdf = np.maximum(dEdf, 0.0)

    # enlève les points non physiques
    m = (f_hz > 0.0)
    f_hz = f_hz[m]
    dEdf = dEdf[m]

    out = {
        "event": event,
        "f_hz": f_hz,
        "dEdf": dEdf,
        "distance_mpc": d.get("distance_mpc", None),
        "E_total_J": d.get("E_total_J", None),
        "nu_eff": (d.get("nu_eff") or {}).get("nu_eff_energy", None),
    }
    return out

def safe_total_energy(f_hz, dEdf):
    # intégrale physique : E = ∫ dE/df df
    E = float(trapezoid(dEdf, f_hz))
    if not np.isfinite(E) or E <= 0.0:
        return 0.0
    return E

def plot_combined_spectra(events, out_png, smooth_sigma=2.0):
    plt.figure()
    n_ok = 0

    for ev in events:
        f = ev["f_hz"]
        y = ev["dEdf"]

        if f.size < 4:
            continue

        # lissage léger pour lisibilité (sans changer l’énergie totale “trop”)
        y_s = gaussian_filter1d(y, sigma=smooth_sigma) if smooth_sigma and smooth_sigma > 0 else y
        y_s = np.maximum(y_s, 1e-300)

        plt.loglog(f, y_s, lw=1.1, alpha=0.85, label=ev["event"])
        n_ok += 1

        # marque nu_eff si présent
        nu_eff = ev.get("nu_eff", None)
        if isinstance(nu_eff, (int, float)) and np.isfinite(nu_eff) and nu_eff > 0:
            plt.axvline(nu_eff, lw=0.6, alpha=0.15)

    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("dE/df (unités internes ou J/Hz si calibré)")
    plt.title(f"Spectres (N={n_ok})")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_combined_normalized(events, out_png, smooth_sigma=2.0):
    plt.figure()
    n_ok = 0

    for ev in events:
        f = ev["f_hz"]
        y = ev["dEdf"]
        E = safe_total_energy(f, y)
        if E <= 0.0:
            continue

        yN = y / E
        yN = gaussian_filter1d(yN, sigma=smooth_sigma) if smooth_sigma and smooth_sigma > 0 else yN
        yN = np.maximum(yN, 1e-300)

        plt.loglog(f, yN, lw=1.1, alpha=0.85, label=ev["event"])
        n_ok += 1

    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("dE/df normalisé (∫=1)")
    plt.title(f"Spectres normalisés (N={n_ok})")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_combined_cumulative(events, out_png):
    plt.figure()
    n_ok = 0

    for ev in events:
        f = ev["f_hz"]
        y = ev["dEdf"]
        E = safe_total_energy(f, y)
        if E <= 0.0:
            continue

        # cumul correct : E(f) = ∫_{f0..f} dE/df df
        Ecum = cumulative_trapezoid(y, f, initial=0.0)

        plt.semilogx(f, Ecum, lw=1.1, alpha=0.85, label=ev["event"])
        n_ok += 1

    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Énergie cumulée (mêmes unités que dE/df)")
    plt.title(f"Énergie cumulée (N={n_ok})")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_peaks(events, out_png):
    # scatter des pics (freq_peak, dEdf_peak)
    xs, ys, labels = [], [], []
    for ev in events:
        f = ev["f_hz"]
        y = ev["dEdf"]
        if f.size < 4:
            continue
        i = int(np.argmax(y))
        xs.append(float(f[i]))
        ys.append(float(y[i]))
        labels.append(ev["event"])

    plt.figure()
    plt.scatter(xs, ys, s=30)
    for x, y, lab in zip(xs, ys, labels):
        plt.annotate(lab, (x, y), fontsize=8, alpha=0.8)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Fréquence du pic (Hz)")
    plt.ylabel("Valeur max dE/df")
    plt.title("Pics spectraux (tous événements)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def plot_energy_vs_distance(events, out_png):
    xs, ys, labels = [], [], []
    for ev in events:
        dist = ev.get("distance_mpc", None)
        if dist is None:
            continue
        f = ev["f_hz"]
        y = ev["dEdf"]
        E = safe_total_energy(f, y)
        if E <= 0.0:
            continue
        xs.append(float(dist))
        ys.append(float(E))
        labels.append(ev["event"])

    plt.figure()
    plt.scatter(xs, ys, s=30)
    for x, y, lab in zip(xs, ys, labels):
        plt.annotate(lab, (x, y), fontsize=8, alpha=0.8)

    plt.yscale("log")
    plt.xlabel("Distance (Mpc)")
    plt.ylabel("Énergie totale (∫ dE/df df)")
    plt.title("Énergie totale vs distance (diagnostic)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def main(json_dir="results"):
    paths = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    if not paths:
        raise SystemExit(f"Aucun JSON trouvé dans {json_dir}/")

    events = []
    bad = 0
    for p in paths:
        try:
            events.append(load_event_json(p))
        except Exception as e:
            bad += 1
            print(f"[SKIP] {p}: {e}")

    print(f"[INFO] loaded={len(events)} skipped={bad}")

    plot_combined_spectra(events, "plots/spectres_combines.png", smooth_sigma=2.0)
    plot_combined_normalized(events, "plots/spectres_normalises.png", smooth_sigma=2.0)
    plot_combined_cumulative(events, "plots/energie_cumulee.png")
    plot_peaks(events, "plots/pics_spectraux.png")
    plot_energy_vs_distance(events, "plots/energie_vs_distance.png")

    print("[OK] plots écrits dans plots/")

if __name__ == "__main__":
    main()

