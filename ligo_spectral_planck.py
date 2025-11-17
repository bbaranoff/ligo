#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyse spectrale cohérente H1–L1 (version PUR)
------------------------------------------------
Aucune calibration externe :
 - pas de REF_E_J
 - pas de REF_TAU
 - pas de calib_factor
 - pas de ratio
 - aucune normalisation vers les valeurs officielles LIGO

Pipeline :
 1. Chargement H1/L1
 2. Crop autour GPS
 3. Filtrage bande (Butterworth)
 4. Détection τ via cross-corr upsamplée
 5. Énergie cohérente E(τ)
 6. ν_eff & h_eff
 7. p-value (slides)
 8. sortie JSON + plots
"""

import argparse, os, numpy as np, matplotlib.pyplot as plt, json
from scipy.integrate import trapezoid
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

# Constantes
c = 299_792_458.0
G = 6.67430e-11
pi = np.pi
Mpc = 3.085677581491367e22
ANTENNA_AVG = 5.0


# -------------------------------------------------------------
# Utils
# -------------------------------------------------------------

def fetch(det, t0, t1):
    os.makedirs("data", exist_ok=True)
    return TimeSeries.fetch_open_data(det, t0, t1, cache=True)

def bandpass(x, fs, f1=20, f2=512, order=4):
    sos = butter(order, [f1, f2], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)

# ---- Cross-corr upsample + interpolation ----

def _parabolic_interpolation(y, idx):
    if idx <= 0 or idx >= len(y)-1:
        return float(idx), float(y[idx])
    ym1, y0, yp1 = y[idx-1], y[idx], y[idx+1]
    denom = (ym1 - 2*y0 + yp1)
    if denom == 0:
        return float(idx), float(y[idx])
    delta = 0.5*(ym1 - yp1)/denom
    peak_x = idx + delta
    peak_y = y[idx] - 0.25*(ym1 - yp1)*delta
    return peak_x, peak_y

def estimate_delay_upsampled(h1, h2, fs, search_ms=15.0, upsample=16):
    N = min(len(h1), len(h2))
    if N < 1024:
        return 0.0

    w = np.blackman(N)
    x = h1[:N] * w
    y = h2[:N] * w

    Nfft = int(2**np.ceil(np.log2(N * upsample * 2)))
    X = np.fft.rfft(x, n=Nfft)
    Y = np.fft.rfft(y, n=Nfft)

    eps = 1e-12
    Xw = X / (np.abs(X) + eps)
    Yw = Y / (np.abs(Y) + eps)

    R = np.fft.irfft(Xw * np.conj(Yw), n=Nfft)
    R = np.roll(R, Nfft//2)
    lags = np.arange(-Nfft//2, Nfft//2) / (fs * upsample)

    mask = np.abs(lags) <= (search_ms / 1000.0)
    if not np.any(mask):
        return 0.0

    seg = R[mask]
    rel_idx = int(np.argmax(seg))
    global_start = np.nonzero(mask)[0][0]
    idx_global = global_start + rel_idx

    peak_x, _ = _parabolic_interpolation(R, idx_global)
    lag_index = (peak_x - (Nfft//2)) / upsample
    tau = lag_index / fs
    return float(tau)


# -------------------------------------------------------------
# ÉNERGIE COHÉRENTE
# -------------------------------------------------------------

def compute_dEdf(f, ReC, r_m):
    K = (c**3)/(8*pi*G) * (r_m**2) * ANTENNA_AVG
    return K * (2*pi*f)**2 * ReC

def energy_with_tau(tsH, tsL, tau, flow, fhigh, r_m, smooth_sigma=0.0):
    xH = np.asarray(tsH.value, float)
    xL = np.asarray(tsL.value, float)
    fs = float(tsH.sample_rate.value)
    N = min(len(xH), len(xL))
    dt = 1.0/fs

    w = tukey(N, 0.2)
    U = (w**2).sum()

    H1 = np.fft.rfft(xH[:N]*w)
    H2 = np.fft.rfft(xL[:N]*w)
    f = np.fft.rfftfreq(N, d=dt)

    phase = np.exp(-1j*2*pi*f*tau)
    H2a = H2 * phase

    S1 = (2/(fs*U))*np.abs(H1)**2
    S2 = (2/(fs*U))*np.abs(H2a)**2
    C  = (2/(fs*U))*(H1*np.conj(H2a))

    ReC_max = np.sqrt(S1*S2 + 1e-300)
    ReC = np.clip(np.real(C), 0.0, ReC_max)
    gamma2 = np.clip((np.abs(C)**2)/(S1*S2 + 1e-300), 0.0, 1.0)

    dEdf = compute_dEdf(f, ReC * gamma2, r_m)

    band = (f >= flow) & (f <= fhigh)

    if smooth_sigma and gaussian_filter1d is not None:
        dEdf = gaussian_filter1d(dEdf, sigma=float(smooth_sigma))
        gamma2 = gaussian_filter1d(gamma2, sigma=float(smooth_sigma))

    E = float(trapezoid(dEdf[band], f[band])) if np.any(band) else 0.0
    return E, f, dEdf, gamma2


# -------------------------------------------------------------
# Analyse complète
# -------------------------------------------------------------

def analyze(H1, L1, gps, distance_mpc, flow, fhigh, signal_win, smooth_sigma, slides, event_name, json_output, plot):

    r_m = distance_mpc * Mpc
    fs = float(H1.sample_rate.value)
    half = signal_win / 2.0

    # Crop
    try:
        wH = H1.crop(gps - half, gps + half)
        wL = L1.crop(gps - half, gps + half)
    except:
        wH, wL = H1, L1

    # Filtrage bande
    hH = bandpass(np.asarray(wH.value, float), fs, flow, fhigh)
    hL = bandpass(np.asarray(wL.value, float), fs, flow, fhigh)

    tsH = TimeSeries(hH, sample_rate=fs)
    tsL = TimeSeries(hL, sample_rate=fs)

    # τ via cross-corr
    tau_guess = estimate_delay_upsampled(hH, hL, fs, search_ms=12.0, upsample=16)

    # Choix du signe : énergie max
    E_pos, f, dEdf_pos, g2_pos = energy_with_tau(tsH, tsL, +abs(tau_guess), flow, fhigh, r_m, smooth_sigma)
    E_neg, _, dEdf_neg, g2_neg = energy_with_tau(tsH, tsL, -abs(tau_guess), flow, fhigh, r_m, smooth_sigma)

    if E_pos >= E_neg:
        tau = +abs(tau_guess)
        dEdf = dEdf_pos
        gamma2 = g2_pos
        E = E_pos
    else:
        tau = -abs(tau_guess)
        dEdf = dEdf_neg
        gamma2 = g2_neg
        E = E_neg

    # Bande utile
    band = (f >= flow) & (f <= fhigh)
    f_use = f[band]
    dEdf_use = dEdf[band]

    df = f_use[1] - f_use[0] if len(f_use) > 1 else 1.0
    E_cum = np.cumsum(dEdf_use) * df
    denom = max(trapezoid(dEdf_use, f_use), 1e-300)

    nu_eff = trapezoid(f_use * dEdf_use, f_use) / denom
    h_eff = E / max(nu_eff, 1e-300)

    # p-value (facultatif)
    p_value = None
    if slides > 0:
        E_null = []
        for s in range(1, slides+1):
            for sign in (+1, -1):
                E_s, _, _, _ = energy_with_tau(tsH, tsL,
                                               tau + sign*s,
                                               flow, fhigh, r_m,
                                               smooth_sigma)
                E_null.append(E_s)
        if len(E_null) > 0:
            E_null = np.array(E_null, float)
            p_value = float(np.mean(E_null >= E))

    # ---------------------- PRINT ----------------------
    print("\n=== ANALYSE COHÉRENTE PUR H1–L1 ===")
    print(f"Événement : {event_name}")
    print(f"Distance  : {distance_mpc} Mpc")
    print(f"délai H1→L1 : {tau*1000:+.3f} ms")
    print(f"E total = {E:.3e} J")
    print(f"ν_eff = {nu_eff:.1f} Hz | h_eff = {h_eff:.3e} J·s")
    if p_value is not None:
        print(f"p-value = {p_value:.3e}")
    print("====================================================\n")

    # JSON
    if json_output:
        os.makedirs("results", exist_ok=True)
        out = {
            "event": event_name,
            "tau_s": float(tau),
            "freq_Hz": f_use.tolist(),
            "dEdf_J_Hz": dEdf_use.tolist(),
            "E_total_J": float(E),
            "nu_eff_Hz": float(nu_eff),
            "h_eff_Js": float(h_eff)
        }
        with open(f"results/{event_name}.json", "w") as fj:
            json.dump(out, fj, indent=2)

    # Plots
    if plot:
        plt.figure(figsize=(9,4.3))
        plt.loglog(f_use, dEdf_use, lw=1.4)
        plt.xlabel("Fréquence (Hz)"); plt.ylabel("dE/df (J/Hz)")
        plt.title(f"Spectrum — {event_name}")
        plt.grid(True, which="both", ls=":", alpha=0.4)
        plt.tight_layout()

        plt.figure(figsize=(9,4.3))
        plt.semilogx(f_use, E_cum)
        plt.xlabel("Fréquence (Hz)"); plt.ylabel("E(<f) (J)")
        plt.title("Énergie cumulée")
        plt.grid(True, which="both", ls=":", alpha=0.4)
        plt.tight_layout()
        plt.show()

    return E, tau, nu_eff, h_eff


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--distance-mpc", required=True, type=float)
    ap.add_argument("--tpad", type=float, default=1024.0)
    ap.add_argument("--flow", type=float, default=None)
    ap.add_argument("--fhigh", type=float, default=None)
    ap.add_argument("--signal-win", type=float, default=None)
    ap.add_argument("--smooth-sigma", type=float, default=1.0)
    ap.add_argument("--slides", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--json", action="store_true")

    args = ap.parse_args()

    cfg = {
        "flow": 35,
        "fhigh": 300,
        "win": 1.0
    }

    flow = args.flow or cfg["flow"]
    fhigh = args.fhigh or cfg["fhigh"]
    signal_win = args.signal_win or cfg["win"]

    gps = datasets.event_gps(args.event)
    print(f"Chargement : {args.event} @ GPS {gps}")

    H1 = fetch("H1", gps - args.tpad, gps + args.tpad)
    L1 = fetch("L1", gps - args.tpad, gps + args.tpad)

    analyze(H1, L1, gps,
            args.distance_mpc,
            flow, fhigh,
            signal_win,
            args.smooth_sigma,
            args.slides,
            args.event,
            args.json,
            args.plot)


if __name__ == "__main__":
    main()

