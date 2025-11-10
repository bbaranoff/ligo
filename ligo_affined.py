#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STPL v1.5 — Pipeline spectral cohérent H1–L1
--------------------------------------------
• Extraction cohérente H1/L1 autour d'un événement GWOSC
• dE/df pondéré par la cohérence γ²(f), intégration sur bande utile
• Mode physique par défaut (constantes c, G incluses) → E en joules
• Option --auto-calib : aligne l'énergie intégrée sur des références publiées (événements connus)
• Choix robuste du signe du délai (±τ_guess) par max(E) sur la bande

Exemple (physique direct, plots) :
  python ligo_affined.py --event GW150914 --distance-mpc 410 \
    --flow 35 --fhigh 250 --signal-win 0.8 --noise-pad 900 \
    --smooth-sigma 1 --slides 200 --plot

Exemple (physique + calibration auto) :
  python ligo_affined.py --event GW150914 --distance-mpc 410 \
    --flow 35 --fhigh 250 --signal-win 0.8 --noise-pad 900 \
    --smooth-sigma 1 --slides 200 --auto-calib --plot
"""

import argparse, os, numpy as np, matplotlib.pyplot as plt
from numpy import trapezoid
from gwosc import datasets
from gwpy.timeseries import TimeSeries
from scipy.signal import butter, sosfiltfilt

try:
    from scipy.signal.windows import tukey
except Exception:
    from scipy.signal import get_window
    def tukey(M, alpha=0.1): return get_window(('tukey', float(alpha)), M)
try:
    from scipy.ndimage import gaussian_filter1d
except Exception:
    gaussian_filter1d = None

# Constantes (mode physique)
c = 299_792_458.0
G = 6.67430e-11
pi = np.pi
Mpc = 3.085677581491367e22

# Références d'énergie (ordre de grandeur, publications LIGO)
REF_E_J = {
    # ~3 M_sun c^2
    "GW150914": 5.0e47,
    # ~1 M_sun c^2 (événement plus léger)
    "GW151226": 1.0e47,
    # ~2 M_sun c^2
    "GW170104": 3.5e47,
    # ~3 M_sun c^2
    "GW170814": 5.0e47,
}

# ---------------- utils ----------------
def fetch(det, t0, t1):
    os.makedirs("data", exist_ok=True)
    return TimeSeries.fetch_open_data(det, t0, t1, cache=True)

def bandpass(x, fs, f1=20.0, f2=512.0, order=4):
    sos = butter(order, [f1, f2], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)

def estimate_delay(h1, h2, fs, search_ms=10.0):
    """Délai H1→L1 (s) via corrélation croisée rapide, fenêtre de recherche ±search_ms."""
    N = min(h1.size, h2.size)
    w = tukey(N, 0.2)
    X = np.fft.rfft(h1[:N]*w); Y = np.fft.rfft(h2[:N]*w)
    r = np.fft.irfft(X*np.conj(Y), n=N)
    r = np.concatenate([r[N//2:], r[:N//2]])
    lags = (np.arange(-N//2, N//2))/fs
    mask = (lags >= -search_ms/1000.0) & (lags <= search_ms/1000.0)
    return lags[mask][np.argmax(r[mask])]

# ---------------- noyau dE/df ----------------
def energy_with_tau(tsH, tsL, tau, flow, fhigh, r_m, smooth_sigma=0.0):
    """
    Calcule (E, f, dE/df, γ²) pour un délai donné τ.
    - FFT fenêtrée (Tukey), déphasage e^{-i2π f τ} sur L1
    - Re{C} ≥ 0 tronqué à ReC_max (bornage physique)
    - Pondération par γ²(f)
    - dE/df_phys = (c^3/(8πG)) * r^2 * (2π f)^2 * ReC * γ²
      (équivalent au flux gw en domaine fréquentiel, à normalisation près)
    """
    xH = np.asarray(tsH.value, float)
    xL = np.asarray(tsL.value, float)
    fs = float(tsH.sample_rate.value)
    N = min(xH.size, xL.size)
    dt = 1.0/fs
    T = N*dt
    w = tukey(N, 0.2)
    U = (w**2).sum()

    H1 = np.fft.rfft(xH[:N]*w)
    H2 = np.fft.rfft(xL[:N]*w)
    f = np.fft.rfftfreq(N, d=dt)

    # Décalage temporel de L1
    phase = np.exp(-1j*2*pi*f*tau)
    H2a = H2 * phase

    # Spectres puissance & cohérence (normalisation Welch-like)
    S1 = (2.0/(fs*U))*np.abs(H1)**2
    S2 = (2.0/(fs*U))*np.abs(H2a)**2
    C  = (2.0/(fs*U))*(H1*np.conj(H2a))

    # Tronquage réaliste + cohérence
    ReC_max = np.sqrt(S1*S2 + 1e-300)
    ReC = np.clip(np.real(C), 0.0, ReC_max)
    gamma2 = np.clip((np.abs(C)**2)/(S1*S2 + 1e-300), 0.0, 1.0)

    # dE/df physique (avec constantes) + pondération γ²
    K_phys = (c**3)/(8.0*pi*G) * (r_m**2)
    dEdf = K_phys * (2*pi*f)**2 * ReC * gamma2

    # Bande utile + lissage optionnel
    band = (f >= max(flow,1e-9)) & (f <= fhigh)
    if smooth_sigma and gaussian_filter1d is not None:
        dEdf = gaussian_filter1d(dEdf, sigma=float(smooth_sigma))
        gamma2 = gaussian_filter1d(gamma2, sigma=float(smooth_sigma))

    E = float(trapezoid(dEdf[band], f[band]))
    return E, f, dEdf, gamma2

# ---------------- analyse principale ----------------
def analyze(tsH, tsL, gps, distance_mpc, flow, fhigh, noise_pad,
            signal_win, smooth_sigma=0.0, slides=0,
            auto_calib=False, event_name=""):

    r_m = distance_mpc * Mpc
    fs = float(tsH.sample_rate.value)

    # Fenêtre courte autour de l’événement
    half = signal_win/2.0
    wH = tsH.crop(gps-half, gps+half)
    wL = tsL.crop(gps-half, gps+half)
    hH = bandpass(np.asarray(wH.value,float), fs, flow, fhigh)
    hL = bandpass(np.asarray(wL.value,float), fs, flow, fhigh)

    # Délai initial et choix du signe par max(E) sur la bande
    tau_guess = estimate_delay(hH, hL, fs, search_ms=10.0)
    # Reconvertit hH/hL en TimeSeries pour energy_with_tau
    tsH_short = TimeSeries(hH, sample_rate=fs)
    tsL_short = TimeSeries(hL, sample_rate=fs)

    E_pos, f, dEdf_pos, gamma2_pos = energy_with_tau(
        tsH_short, tsL_short, +abs(tau_guess), flow, fhigh, r_m, smooth_sigma)
    E_neg, _, dEdf_neg, gamma2_neg = energy_with_tau(
        tsH_short, tsL_short, -abs(tau_guess), flow, fhigh, r_m, smooth_sigma)

    if E_pos >= E_neg:
        tau = +abs(tau_guess)
        dEdf = dEdf_pos; gamma2 = gamma2_pos; E = E_pos
    else:
        tau = -abs(tau_guess)
        dEdf = dEdf_neg; gamma2 = gamma2_neg; E = E_neg

    band = (f >= max(flow,1e-9)) & (f <= fhigh)
    f_use = f[band]
    dEdf_use = dEdf[band]
    E_cum = np.cumsum(dEdf_use) * (f_use[1]-f_use[0])
    # Moments énergétiques
    denom = max(trapezoid(dEdf_use, f_use), 1e-300)
    nu_eff = trapezoid(f_use*dEdf_use, f_use)/denom
    h_eff = E/max(nu_eff,1e-300)

    # Calibration auto (événements connus)
    calib_applied = False
    calib_factor = 1.0
    if auto_calib and event_name in REF_E_J and E > 0:
        target = REF_E_J[event_name]
        calib_factor = target / E
        E *= calib_factor
        dEdf_use = dEdf_use * calib_factor
        E_cum = E_cum * calib_factor
        h_eff = h_eff * calib_factor
        calib_applied = True

    # Time-slides (p-value empirique)
    p_value = None
    if slides and slides > 0:
        E_null = []
        for s in range(1, int(slides)+1):
            for sign in (+1, -1):
                E_s, _, dEdf_s, _ = energy_with_tau(
                    tsH_short, tsL_short, tau + sign*float(s), flow, fhigh, r_m, smooth_sigma)
                # applique la même calibration si utilisée
                E_null.append(E_s * calib_factor)
        E_null = np.array(E_null, float)
        if E_null.size > 0:
            p_value = float(np.mean(E_null >= E))

    # Affichage
    print("\n=== RÉSULTATS COHÉRENTS H1–L1 (STPL v1.5) ===")
    print(f"délai H1→L1 : {tau*1e3:.3f} ms")
    print(f"E = {E:.3e} J")
    print(f"ν_eff = {nu_eff:.1f} Hz ; h_eff = {h_eff:.3e} J·s")
    if calib_applied:
        print(f"[auto-calib] {event_name} → facteur ≈ {calib_factor:.3e}")
    if p_value is not None:
        print(f"p-value (±1..±{slides}s) ≈ {p_value:.3e}")
    print("=================================================\n")

    return f_use, dEdf_use, E_cum, gamma2[band], tau, E, nu_eff, h_eff, p_value, calib_applied, calib_factor

# ---------------- main + plots ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--distance-mpc", type=float, required=True)
    ap.add_argument("--tpad", type=float, default=128.0)
    ap.add_argument("--flow", type=float, default=20.0)
    ap.add_argument("--fhigh", type=float, default=512.0)
    ap.add_argument("--signal-win", type=float, default=0.8)
    ap.add_argument("--noise-pad", type=float, default=900.0)
    ap.add_argument("--smooth-sigma", type=float, default=0.0)
    ap.add_argument("--slides", type=int, default=0)
    ap.add_argument("--auto-calib", action="store_true",
                    help="Aligne E sur des références publiées pour GW150914/151226/170104/170814")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    gps = datasets.event_gps(args.event)
    t0, t1 = gps - args.tpad, gps + args.tpad
    print(f"\nChargement des données LIGO : {args.event}")
    H1 = fetch("H1", t0, t1)
    L1 = fetch("L1", t0, t1)

    f_use, dEdf_use, E_cum, gamma2_use, tau, E, nu_eff, h_eff, pval, used_cal, k = analyze(
        H1, L1, gps, args.distance_mpc, args.flow, args.fhigh, args.noise_pad,
        args.signal_win, smooth_sigma=args.smooth_sigma, slides=args.slides,
        auto_calib=args.auto_calib, event_name=args.event
    )

    if args.plot:
        plt.figure(figsize=(9,4))
        plt.loglog(f_use, dEdf_use)
        plt.xlabel("Fréquence (Hz)"); plt.ylabel("dE/df (J/Hz)")
        title = "Spectre d'énergie cohérente"
        if used_cal:
            title += " [auto-calib]"
        plt.title(title); plt.tight_layout()

        plt.figure(figsize=(9,4))
        plt.semilogx(f_use, E_cum)
        plt.xlabel("Fréquence (Hz)"); plt.ylabel("E(<f) (J)")
        plt.title("Énergie cumulée"); plt.tight_layout()

        plt.figure(figsize=(9,4))
        plt.semilogx(f_use, np.clip(gamma2_use,0,1))
        plt.xlabel("Fréquence (Hz)"); plt.ylabel(r"$\gamma^2(f)$"); plt.ylim(0,1.05)
        plt.title("Cohérence spectrale H1–L1"); plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
