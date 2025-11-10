#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse cohérente spectrale H1–L1 (sans G, sans c)
--------------------------------------------------
Pipeline purement spectral :
- Corrélation H1/L1 autour de l’événement GWOSC
- Estimation dE/df relative et cohérence spectrale
- Normalisation optionnelle (--norm)
- p-value empirique via time-slides (--slides)

Usage :
python ligo_affined.py --event GW150914 --distance-mpc 410 \
  --flow 35 --fhigh 250 --signal-win 0.8 --noise-pad 900 \
  --smooth-sigma 1 --norm band --slides 100 --plot
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

# Constantes neutres
Mpc = 3.085677581491367e22
pi = np.pi


# ============================================================
def fetch(det, t0, t1):
    os.makedirs("data", exist_ok=True)
    return TimeSeries.fetch_open_data(det, t0, t1, cache=True)


def bandpass(x, fs, f1=20.0, f2=512.0, order=4):
    sos = butter(order, [f1, f2], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def psd_welch(ts, seglen=4.0, overlap=2.0, fmin=10.0, fmax=1024.0):
    """Spectre de puissance médian par fenêtre glissante."""
    from numpy.fft import rfft, rfftfreq
    x = np.asarray(ts.value, float)
    fs = float(ts.sample_rate.value)
    Nseg = int(seglen * fs)
    Nhop = int((seglen - overlap) * fs)
    win = tukey(Nseg, 0.2)
    U = (win**2).sum()
    specs = []
    for i in range(0, x.size - Nseg + 1, Nhop):
        seg = bandpass(x[i:i + Nseg], fs, fmin, fmax)
        Xk = rfft(seg * win)
        specs.append((2.0 / (fs * U)) * np.abs(Xk)**2)
    S = np.median(np.stack(specs), axis=0)
    f = rfftfreq(Nseg, d=1.0 / fs)
    return f, S


def estimate_delay(h1, h2, fs, search_ms=10.0):
    """Délai H1→L1 (s) via corrélation croisée rapide."""
    N = min(h1.size, h2.size)
    w = tukey(N, 0.2)
    X = np.fft.rfft(h1[:N] * w)
    Y = np.fft.rfft(h2[:N] * w)
    r = np.fft.irfft(X * np.conj(Y), n=N)
    r = np.concatenate([r[N//2:], r[:N//2]])
    lags = (np.arange(-N//2, N//2)) / fs
    mask = (lags >= -search_ms/1000.0) & (lags <= search_ms/1000.0)
    return lags[mask][np.argmax(r[mask])]


# ============================================================
def compute_dEdf(f, ReC, r):
    """dE/df relatif (sans G ni c)."""
    K = (r**2) / (8.0 * pi)
    return K * (f**2) * ReC


def apply_normalization(f_use, dEdf_use, norm, r):
    """Normalisation facultative pour comparaisons."""
    if norm == "none":
        return dEdf_use
    if norm == "r2":
        return dEdf_use / max(r**2, 1e-30)
    if norm == "unity":
        s = np.max(dEdf_use) if dEdf_use.size else 1.0
        return dEdf_use / max(s, 1e-30)
    if norm == "band":
        S = trapezoid(dEdf_use, f_use) if f_use.size > 1 else 1.0
        return dEdf_use / max(S, 1e-30)
    return dEdf_use


# ============================================================
def analyze(tsH, tsL, gps, distance_mpc, flow, fhigh, noise_pad,
            signal_win, smooth_sigma=None, norm="none",
            slides=0, plot=False):

    r = distance_mpc * Mpc
    fs = float(tsH.sample_rate.value)

    # PSD hors signal
    noiseH = tsH.crop(gps - noise_pad - 40, gps - 10)
    noiseL = tsL.crop(gps - noise_pad - 40, gps - 10)
    fH, S1 = psd_welch(noiseH, fmin=flow, fmax=fhigh)
    fL, S2 = psd_welch(noiseL, fmin=flow, fmax=fhigh)
    if fH.size != fL.size:
        S2 = np.interp(fH, fL, S2)
    f = fH

    # Fenêtre courte autour de l’événement
    half = signal_win / 2.0
    hH = bandpass(np.asarray(tsH.crop(gps - half, gps + half).value, float), fs, flow, fhigh)
    hL = bandpass(np.asarray(tsL.crop(gps - half, gps + half).value, float), fs, flow, fhigh)

    # Délai cohérent
    tau_guess = estimate_delay(hH, hL, fs)
    N = min(len(hH), len(hL))
    dt = 1 / fs
    T = N * dt
    w = tukey(N, 0.2)
    H1 = np.fft.rfft(hH * w)
    H2 = np.fft.rfft(hL * w)
    f_short = np.fft.rfftfreq(N, d=dt)

    def energy_with_tau(tau, floor_rel=1e-7):
        phase = np.exp(-1j * 2 * np.pi * f_short * tau)
        H2a = H2 * phase
        C = (dt * H1) * np.conj(dt * H2a) * (2.0 / T)
        S1_tot = (2.0 / T) * np.abs(dt * H1)**2
        S2_tot = (2.0 / T) * np.abs(dt * H2a)**2
        # Tronquage du réel positif
        ReC_max = np.sqrt(S1_tot * S2_tot + 1e-300)
        ReC_min = floor_rel * ReC_max
        ReC = np.clip(np.real(C), ReC_min, ReC_max)
        dEdf = compute_dEdf(f_short, ReC, r)
        band = (f_short >= max(flow, 1e-9)) & (f_short <= fhigh)
        return float(trapezoid(dEdf[band], f_short[band])), dEdf, S1_tot, S2_tot, ReC

    # Choix du signe τ
    E_pos, dEdf_pos, S1_pos, S2_pos, ReC_pos = energy_with_tau(+tau_guess)
    E_neg, dEdf_neg, S1_neg, S2_neg, ReC_neg = energy_with_tau(-tau_guess)
    if E_neg > E_pos:
        tau, dEdf, S1_tot, S2_tot, ReC = -tau_guess, dEdf_neg, S1_neg, S2_neg, ReC_neg
    else:
        tau, dEdf, S1_tot, S2_tot, ReC = +tau_guess, dEdf_pos, S1_pos, S2_pos, ReC_pos

    # Bande utile
    band = (f_short >= max(flow, 1e-9)) & (f_short <= fhigh)
    f_use = f_short[band]
    dEdf_use = np.nan_to_num(dEdf[band])
    if smooth_sigma and gaussian_filter1d is not None:
        dEdf_use = gaussian_filter1d(dEdf_use, sigma=float(smooth_sigma))

    # Normalisation
    dEdf_use = apply_normalization(f_use, dEdf_use, norm, r)

    # Intégrations relatives
    E = trapezoid(dEdf_use, f_use)
    nu_eff = trapezoid(f_use * dEdf_use, f_use) / max(trapezoid(dEdf_use, f_use), 1e-300)
    h_eff = E / max(nu_eff, 1e-300)
    E_cum = np.cumsum(dEdf_use) * (f_use[1] - f_use[0])
    gamma2 = np.abs(ReC)**2 / (S1_tot * S2_tot + 1e-300)
    gamma2 = np.clip(gamma2[band], 0, 1)
    gamma2 = gaussian_filter1d(gamma2, sigma=2)
    # p-value (time-slides ±1..±N s)
    p_value = None
    if slides and slides > 0:
        E_null = []
        for s in range(1, int(slides)+1):
            for sign in (+1, -1):
                phase = np.exp(-1j * 2 * np.pi * f_short * (tau + sign * float(s)))
                H2a_s = H2 * phase
                C_s = (dt * H1) * np.conj(dt * H2a_s) * (2.0 / T)
                S1_tot_s = (2.0 / T) * np.abs(dt * H1)**2
                S2_tot_s = (2.0 / T) * np.abs(dt * H2a_s)**2
                ReC_max_s = np.sqrt(S1_tot_s * S2_tot_s + 1e-300)
                ReC_s = np.clip(np.real(C_s), 1e-7*ReC_max_s, ReC_max_s)
                dEdf_s = compute_dEdf(f_short, ReC_s, r)
                band_s = (f_short >= max(flow, 1e-9)) & (f_short <= fhigh)
                f_s = f_short[band_s]
                dEdf_s_use = np.nan_to_num(dEdf_s[band_s])
                if smooth_sigma and gaussian_filter1d is not None:
                    dEdf_s_use = gaussian_filter1d(dEdf_s_use, sigma=float(smooth_sigma))
                dEdf_s_use = apply_normalization(f_s, dEdf_s_use, norm, r)
                E_s = trapezoid(dEdf_s_use, f_s)
                E_null.append(E_s)
        E_null = np.array(E_null, float)
        if E_null.size > 0:
            p_value = float(np.mean(E_null >= E))

    # Résumé
    print("\n=== RÉSULTATS COHÉRENTS H1–L1 (spectral pur) ===")
    print(f"délai H1→L1 : {tau * 1e3:.3f} ms")
    print(f"E_rel = {E:.3e} (échelle relative)")
    print(f"ν_eff = {nu_eff:.1f} Hz ; h_eff_rel = {h_eff:.3e}")
    if p_value is not None:
        print(f"p-value (±1..±{slides}s) ≈ {p_value:.3e}")
    print("=================================================\n")

    # Plots
    if plot:
        plt.figure(figsize=(9, 4))
        plt.loglog(f_use, dEdf_use)
        plt.xlabel("Fréquence (Hz)")
        plt.ylabel("dE/df (relatif)")
        plt.title("Spectre d'énergie cohérente")
        plt.tight_layout()

        plt.figure(figsize=(9, 4))
        plt.semilogx(f_use, E_cum)
        plt.xlabel("Fréquence (Hz)")
        plt.ylabel("E(<f) relatif")
        plt.title("Énergie cumulée")
        plt.tight_layout()

        plt.figure(figsize=(9, 4))
        plt.semilogx(f_use, gamma2)
        plt.xlabel("Fréquence (Hz)")
        plt.ylabel("γ²(f)")
        plt.ylim(0, 1.05)
        plt.title("Cohérence spectrale H1–L1")
        plt.tight_layout()
        plt.show()


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--distance-mpc", type=float, required=True)
    ap.add_argument("--tpad", type=float, default=128.0)
    ap.add_argument("--flow", type=float, default=20.0)
    ap.add_argument("--fhigh", type=float, default=512.0)
    ap.add_argument("--signal-win", type=float, default=0.3)
    ap.add_argument("--noise-pad", type=float, default=50.0)
    ap.add_argument("--smooth-sigma", type=float, default=None)
    ap.add_argument("--norm", choices=["none","r2","unity","band"], default="none")
    ap.add_argument("--slides", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    gps = datasets.event_gps(args.event)
    H1 = fetch("H1", gps - args.tpad, gps + args.tpad)
    L1 = fetch("L1", gps - args.tpad, gps + args.tpad)
    analyze(H1, L1, gps, args.distance_mpc, args.flow, args.fhigh,
            args.noise_pad, args.signal_win,
            smooth_sigma=args.smooth_sigma, norm=args.norm,
            slides=args.slides, plot=args.plot)


if __name__ == "__main__":
    main()
