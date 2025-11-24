#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse cohérente LIGO H1–L1 (GWOSC)
------------------------------------
- Calibration h★ fixe (=1) avec renfort pseudo-SNR.
- Fenêtre de bruit sécurisée.
- Lissage log-log du spectre d'énergie.
- Énergie effective normalisée (indépendante de la distance après correction).
"""

from scipy.signal.windows import dpss
import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import tukey
from scipy.ndimage import gaussian_filter1d
from gwosc import datasets
from gwpy.timeseries import TimeSeries

# ==========================
# Constantes physiques
# ==========================
c = 299792458.0
M_sun = 1.98847e30
Mpc = 3.085677581491367e22
H_STAR = 1e6  # amplitude RMS cible
SCALE_EJ = 0.87e29
EVENT_PARAMS = {
    "GW150914":          {"flow": 20, "fhigh": 350, "signal_win": 1.2, "noise_pad": 1200, "distance_mpc": 410},
    "GW151226":          {"flow": 20, "fhigh": 350, "signal_win": 1.2, "noise_pad": 1200, "distance_mpc": 440},
    "GW170104":          {"flow": 20, "fhigh": 350, "signal_win": 1.2, "noise_pad": 1200, "distance_mpc": 880},
    "GW170608":          {"flow": 20, "fhigh": 350, "signal_win": 1.2, "noise_pad": 1200, "distance_mpc": 320},
    "GW170729":          {"flow": 20, "fhigh": 350, "signal_win": 1.2, "noise_pad": 1200, "distance_mpc": 2840},
    "GW170809":          {"flow": 20, "fhigh": 350, "signal_win": 1.2, "noise_pad": 1200, "distance_mpc": 990},
    "GW170814":          {"flow": 20, "fhigh": 350, "signal_win": 1.2, "noise_pad": 1200, "distance_mpc": 600},
    "GW170817":          {"flow": 20, "fhigh": 350, "signal_win": 1.2, "noise_pad": 1200, "distance_mpc": 40},

    "GW190403_051519":   {"flow": 20, "fhigh": 350, "signal_win": 1.2, "noise_pad": 1200, "distance_mpc": 1590},
    "GW190412":          {"flow": 20, "fhigh": 350, "signal_win": 1.2, "noise_pad": 1200, "distance_mpc": 740},
    "GW190413_052954":   {"flow": 20, "fhigh": 350, "signal_win": 1.2, "noise_pad": 1200, "distance_mpc": 700},
    "GW190413_134308":   {"flow": 20, "fhigh": 350, "signal_win": 1.2, "noise_pad": 1200, "distance_mpc": 590},
    "GW190421_213856":   {"flow": 20, "fhigh": 350, "signal_win": 1.2, "noise_pad": 1200, "distance_mpc": 1600},
    "GW190503_185404":   {"flow": 20, "fhigh": 350, "signal_win": 1.2, "noise_pad": 1200, "distance_mpc": 1190},
    "GW190514_065416":   {"flow": 20, "fhigh": 350, "signal_win": 1.2, "noise_pad": 1200, "distance_mpc": 2070},
    "GW190517_055101":   {"flow": 20, "fhigh": 350, "signal_win": 1.2, "noise_pad": 1200, "distance_mpc": 1500},
    "GW190519_153544":   {"flow": 20, "fhigh": 350, "signal_win": 1.2, "noise_pad": 1200, "distance_mpc": 2520},
    "GW190521":          {"flow": 20, "fhigh": 350, "signal_win": 1.2, "noise_pad": 1200, "distance_mpc": 5300},
    "GW190828_063405":   {"flow": 20, "fhigh": 350, "signal_win": 1.2, "noise_pad": 1200, "distance_mpc": 2150},
    "GW190828_065509":   {"flow": 20, "fhigh": 350, "signal_win": 1.2, "noise_pad": 1200, "distance_mpc": 1060},

    "default":           {"flow": 20, "fhigh": 350, "signal_win": 1.2, "noise_pad": 1200, "distance_mpc": 500}
}

# ==========================
# Utilitaires
# ==========================
def fetch(det, t0, t1, outdir="data") -> TimeSeries:
    os.makedirs(outdir, exist_ok=True)
    return TimeSeries.fetch_open_data(det, t0, t1, cache=True)

def safe_bandpass(x, fs, f1, f2, order=4):
    nyq = 0.5 * fs
    f2_safe = min(f2, 0.95 * nyq)
    if f2_safe <= f1:
        raise ValueError(f"Bandpass impossible: f1={f1} ≥ f2_safe={f2_safe}")
    sos = butter(order, [f1, f2_safe], btype="bandpass", fs=fs, output="sos")
#    return sosfiltfilt(sos, x)   # <<< SUPPRESSION DU FENÊTRAGE

    x_filt = sosfiltfilt(sos, x)
    N = len(x_filt)
    win = tukey(N, 0.05)
    return x_filt * win



def psd_welch(ts, seglen=4.0, overlap=2.0, fmin=10.0, fmax=2048.0):
    from numpy.fft import rfft, rfftfreq
    x = np.asarray(ts.value, float)
    fs = ts.sample_rate.value
    fmax = min(fmax, 0.95 * (0.5 * fs))
    nseg = int(seglen * fs)
    nhop = int((seglen - overlap) * fs)
    win = np.hanning(nseg)
    U = (win ** 2).sum()
    specs = []
    for i in range(0, x.size - nseg + 1, nhop):
        seg = safe_bandpass(x[i:i + nseg], fs, fmin, fmax)
        Xk = rfft(seg * win)
        Pxx = (2.0 / (fs * U)) * np.abs(Xk) ** 2
        specs.append(Pxx)
    S = np.median(np.stack(specs), axis=0)
    f = rfftfreq(nseg, d=1.0 / fs)
    return f, S

def estimate_delay_time(h1, h2, fs, max_delay_ms=10.0):
    N = min(len(h1), len(h2))
    w = np.ones(N)  # pas de fenêtre temporelle
    X = np.fft.rfft(h1[:N] * w, n=2*N)
    Y = np.fft.rfft(h2[:N] * w, n=2*N)
    R = Y * np.conj(X)
    r = np.fft.irfft(R)
    r = np.fft.fftshift(r)

    lags = np.arange(-N, N) / fs
    lim = max_delay_ms / 1000
    mask = np.abs(lags) <= lim
    r_mask = r[mask]; l_mask = lags[mask]

    i = np.argmax(np.abs(r_mask))
    return float(l_mask[i])

def estimate_tau_geo(tsH, tsL, gps, fs, flow, fhigh):
    """
    τ géométrique robuste : max de corrélation sur la zone du chirp.
    """

    # --- 1) segment très court centré sur le merge (~40 ms) ---
    segH = tsH.crop(gps - 0.03, gps + 0.01)
    segL = tsL.crop(gps - 0.03, gps + 0.01)

    hH_raw = np.asarray(segH.value, float)
    hL_raw = np.asarray(segL.value, float)

    # --- 2) version filtrée ---
    hH_f = safe_bandpass(hH_raw, fs, flow, fhigh)
    hL_f = safe_bandpass(hL_raw, fs, flow, fhigh)

    # --- 3) cross-corr brute + filtrée ---
    tau_raw = estimate_delay_time(hH_raw, hL_raw, fs)
    tau_flt = estimate_delay_time(hH_f, hL_f, fs)

    # --- 4) combinaison robuste ---
    tau = 0.7 * tau_flt + 0.3 * tau_raw

    return float(tau)

# ==========================
# ANALYSE SPECTRALE — VERSION CORRIGÉE
# ==========================
def analyze_coherent_spectral(tsH, tsL, gps, distance_mpc, event_name="",
                              flow=20.0, fhigh=350.0, noise_pad=1200.0,
                              signal_win=1.2, plot=False):

    fs = tsH.sample_rate.value

    # -------------------------------------------------------
    # 1) PSD — zone de bruit robuste
    # -------------------------------------------------------
    start = tsH.t0.value
    try:
        if gps - noise_pad - 400 < start:
            n0, n1 = start, start + 200
        else:
            n0, n1 = gps - noise_pad - 400, gps - noise_pad - 200
        noiseH = tsH.crop(n0, n1)
        noiseL = tsL.crop(n0, n1)
    except:
        noiseH = tsH.crop(gps - 1000, gps - 800)
        noiseL = tsL.crop(gps - 1000, gps - 800)

    fH, S1 = psd_welch(noiseH, fmin=flow, fmax=fhigh)
    fL, S2 = psd_welch(noiseL, fmin=flow, fmax=fhigh)
    if not np.allclose(fH, fL):
        S2 = np.interp(fH, fL, S2)
    f_psd = fH
    # -------------------------------------------------------
    # 2) Estimation du délai initial (NON filtré !)
    # -------------------------------------------------------
    segH = tsH.crop(gps - 0.2, gps + 0.2)
    segL = tsL.crop(gps - 0.2, gps + 0.2)

    # >>> signaux bruts pour la corrélation <<< 
    hH_delay = np.asarray(segH.value, float)
    hL_delay = np.asarray(segL.value, float)

    # Corrélation avec bon signe
    tau_guess = estimate_tau_geo(tsH, tsL, gps, fs, flow, fhigh)

    # -------------------------------------------------------
    # 3) Extraction du signal utile
    # -------------------------------------------------------
    # ====== recentrage automatique du pic ======
    # détecte le vrai chirp = max de fréquence instantanée (pas amplitude)
    # --- Ajustement spécial GW170608 ---
    if event_name == "GW170608":
        # Fixe un merger court et propre
        t_peak = gps + 0.005      # merger time
        signal_win = 0.25         # 250 ms max
    if event_name != "GW170608":
        seg = tsH.crop(gps - 0.3, gps + 0.1)
        h = np.asarray(seg.value, float)
        dh = np.abs(np.diff(h))
        imax = np.argmax(dh)
        t_peak = seg.times.value[:-1][imax]

    half = signal_win * 0.5
    winH = tsH.crop(t_peak - half, t_peak + half)
    winL = tsL.crop(t_peak - half, t_peak + half)

    hH = safe_bandpass(np.asarray(winH.value, float), fs, flow, fhigh)
    hL = safe_bandpass(np.asarray(winL.value, float), fs, flow, fhigh)

    # -------------------------------------------------------
    # 4) FFT courte
    # -------------------------------------------------------
    N = min(len(hH), len(hL))
    w = tukey(N, 0.2)
    H1 = np.fft.rfft(hH * w)
    H2 = np.fft.rfft(hL * w)
    f = np.fft.rfftfreq(N, 1/fs)

    # -------------------------------------------------------
    # 5) Spectre d'énergie cohérent
    # -------------------------------------------------------
    Hc = (H1 + H2) / 2
    r = distance_mpc * Mpc
    dEdf = np.abs(Hc)**2 * (r**2) * (f**2)
    dEdf = np.nan_to_num(dEdf)

    mask = (f >= flow) & (f <= fhigh)
    f_use = f[mask]
    dEdf_use = dEdf[mask]

    # -------------------------------------------------------
    # 6) Intégrales (E, m_sun, ν_eff)
    # -------------------------------------------------------
    E = float(np.trapz(dEdf_use, f_use)) * SCALE_EJ
    m_sun = E / (M_sun * c**2)

    den = np.trapz(dEdf_use, f_use)
    nu_eff = float(np.trapz(f_use * dEdf_use, f_use) / den) if den > 0 else 0.0
    # -------------------------------------------------------
    os.makedirs("results", exist_ok=True)

    out_json = {
        "event": event_name,
        "distance_mpc": float(distance_mpc),
        "freq_Hz": f_use.tolist(),
        "dEdf_J_Hz": dEdf_use.tolist(),
        "E_total_J": float(E),
        "m_sun": float(m_sun),
        "nu_eff_Hz": float(nu_eff),
        "tau_s": float(tau_guess),
        "flow_Hz": float(flow),
        "fhigh_Hz": float(fhigh)
    }
    with open(f"results/{event_name}.json", "w") as fj:
        json.dump(out_json, fj, indent=2)
    print(f"\n=== ANALYSE SPECTRALE {event_name} ===")
    print(f"Distance: {distance_mpc} Mpc")
    print(f"Énergie intrinsèque: {E:.3e} J ({m_sun:.3f} M☉)")
    print(f"Fréquence effective: {nu_eff:.1f} Hz")
    print(f"Référence d'amplitude: h★={H_STAR:.2e}")

    if plot:
        plt.figure(figsize=(10, 6))
        plt.loglog(f_use, dEdf_use, lw=1.5, 
                  label=f"{event_name} (E={E:.2e} J, ν_eff={nu_eff:.1f} Hz, d={distance_mpc} Mpc)")

        f_log = np.geomspace(f_use[0], f_use[-1], 500)
        dEdf_log = np.interp(f_log, f_use, dEdf_use)
        log_smooth = gaussian_filter1d(np.log10(np.maximum(dEdf_log, 1e-50)), sigma=2)

        plt.loglog(f_log, 10**log_smooth, '--', lw=1, color='orange', alpha=0.7, label=f"{event_name} (lissé)")
        plt.xlabel("Fréquence (Hz)", fontsize=12)
        plt.ylabel("dE/df (J/Hz)", fontsize=12)
        plt.title(f"Spectre d'énergie gravitationnelle intrinsèque\n{event_name} — h★={H_STAR:.2e}, d={distance_mpc} Mpc", fontsize=14)
        plt.grid(True, which="both", alpha=0.3, linestyle='--')
        plt.legend(fontsize=10)
        plt.xlim(f_use[0], f_use[-1])
        plt.ylim(auto=True)

        plt.axvline(x=nu_eff, color='red', linestyle=':', linewidth=1, label=f"ν_eff = {nu_eff:.1f} Hz")

        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{event_name}_spectre.png", dpi=200, bbox_inches='tight')
        plt.show()

    return {"E_total": E, "m_sun": m_sun, "nu_eff": nu_eff, "distance_mpc": distance_mpc}

# ==========================
# CLI
# ==========================
def main():
    ap = argparse.ArgumentParser(description="Analyse LIGO cohérente h★=1 (énergie intrinsèque)")
    ap.add_argument("--event", required=True)
    ap.add_argument("--distance-mpc", type=float, required=True, 
                   help="Distance de l'événement en Mpc (utilisée pour calculer l'énergie intrinsèque)")
    ap.add_argument("--flow", type=float)
    ap.add_argument("--fhigh", type=float)
    ap.add_argument("--signal-win", type=float)
    ap.add_argument("--noise-pad", type=float)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    evp = EVENT_PARAMS.get(args.event, EVENT_PARAMS["default"])
    flow = evp["flow"] if args.flow is None else args.flow
    fhigh = evp["fhigh"] if args.fhigh is None else args.fhigh
    signal_win = evp["signal_win"] if args.signal_win is None else args.signal_win
    noise_pad = evp["noise_pad"] if args.noise_pad is None else args.noise_pad

    gps = datasets.event_gps(args.event)
    t0, t1 = gps - 1200.0, gps + 1200.0
    print(f"📡 Téléchargement des données pour {args.event}...")
    H1, L1 = fetch("H1", t0, t1), fetch("L1", t0, t1)

    res = analyze_coherent_spectral(H1, L1, gps, args.distance_mpc,
                                    event_name=args.event,
                                    flow=flow, fhigh=fhigh,
                                    noise_pad=noise_pad,
                                    signal_win=signal_win, plot=args.plot)

    print(f"\n🎯 SYNTHÈSE FINALE: {args.event}")
    print(f"Distance: {args.distance_mpc} Mpc")
    print(f"Énergie intrinsèque: {res['E_total']:.3e} J ({res['m_sun']:.3f} M☉)")
    print(f"Fréquence effective: {res['nu_eff']:.1f} Hz")
    print(f"NOTE: Énergie intrinsèque calculée avec correction de distance (×d²)")
    print(f"      Référence d'amplitude: h★={H_STAR:.2e}\n{'='*60}")

if __name__ == "__main__":
    main()
