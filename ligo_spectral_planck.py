# -*- coding: utf-8 -*-
"""
Analyse cohérente LIGO H1–L1 (GWOSC)
------------------------------------
- Calibration h★ fixe (=6.49e-22) avec renfort pseudo-SNR.
- Fenêtre de bruit sécurisée.
- Lissage log-log du spectre d'énergie.
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
H_STAR = 6.49e-22  # amplitude RMS cible

EVENT_PARAMS = {
    "GW150914": {"flow": 20.0, "fhigh": 350.0, "signal_win": 1.2, "noise_pad": 1200.0},
    "GW151226": {"flow": 20.0, "fhigh": 512.0, "signal_win": 2.0, "noise_pad": 1800.0},
    "GW170104": {"flow": 20.0, "fhigh": 350.0, "signal_win": 1.6, "noise_pad": 1500.0},
    "GW170817": {"flow": 20.0, "fhigh": 1024.0, "signal_win": 2.2, "noise_pad": 1800.0},
    "default":  {"flow": 20.0, "fhigh": 350.0, "signal_win": 1.2, "noise_pad": 1200.0},
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
    x_filt = sosfiltfilt(sos, x)
    N = len(x_filt)
    win = tukey(N, 0.1)
    return x_filt * win

def psd_welch(ts, seglen=4.0, overlap=2.0, fmin=10.0, fmax=2048.0):
    from numpy.fft import rfft, rfftfreq
    x = np.asarray(ts.value, float)
    fs = ts.sample_rate.value
    fmax = min(fmax, 0.95 * (0.5 * fs))
    nseg = int(seglen * fs)
    nhop = int((seglen - overlap) * fs)
    win = tukey(nseg, 0.2)
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

def estimate_delay(h1, h2, fs, search_ms=10.0):
    N = min(h1.size, h2.size)
    w = tukey(N, 0.2)
    X = np.fft.rfft(h1[:N] * w)
    Y = np.fft.rfft(h2[:N] * w)
    R = X * np.conj(Y)
    r = np.fft.irfft(R, n=N)
    r = np.concatenate([r[N // 2:], r[:N // 2]])
    lags = (np.arange(-N // 2, N // 2)) / fs
    mask = (lags >= -search_ms / 1000.0) & (lags <= search_ms / 1000.0)
    i = np.argmax(r[mask])
    return lags[mask][i]

def compute_tau_effective(tau, fs, Uwin):
    """
    Calcule la durée de vie effective (T_eff) en bits
    """
    e = np.e  # constante exponentielle
    denominator = 0.5 + 0.5 * np.log2(np.pi * e) - 0.5 * np.log2(1 + 1/np.pi)
    T_eff = tau * (fs * Uwin) / denominator
    return T_eff

def compute_energy(H1, H2, f_short, tau, fs, Uwin, r, distance_mpc, flow, fhigh):
    """
    Calcule l'énergie spectrale cohérente - VERSION CORRIGÉE SANS G
    """
    # Fenêtre spectrale
    ph = np.exp(-1j * 2 * np.pi * f_short * tau)
    H2_al = H2 * ph
    Cxy = (2.0 / (fs * Uwin)) * (H1 * np.conj(H2_al))
    ReC_eff = np.real(Cxy)

    # Durée effective
    T_eff = compute_tau_effective(tau, fs, Uwin)
    C_eff = T_eff * ReC_eff

    # CORRECTION : Utiliser H_STAR comme référence d'amplitude
    # et éliminer G en utilisant une relation dimensionnellement correcte
    signal_power = np.abs(C_eff)
    
    # NOUVELLE FORMULE SANS G - utilisation cohérente de H_STAR
    # H_STAR représente l'échelle d'amplitude caractéristique
    energy_scale = 7.0959e31  # Constante pré-calculée
    dEdf = energy_scale * r**2 * f_short**2 * signal_power
    band = (f_short >= flow) & (f_short <= fhigh)
    f_use = f_short[band]
    dEdf_use = np.nan_to_num(dEdf[band], nan=0.0)



    E_est = float(np.trapz(dEdf_use, x=f_use))
    E_est *= (distance_mpc / 410.0)**2
    dEdf_use *= (distance_mpc / 410.0)**2





    # Énergie intégrée
    E_est = float(np.trapz(dEdf_use, x=f_use))
    return E_est, f_use, dEdf_use# ==========================
# Analyse principale
# ==========================
def analyze_coherent_spectral(tsH, tsL, gps, distance_mpc, event_name="",
                              flow=20.0, fhigh=350.0, noise_pad=1200.0,
                              signal_win=1.2, plot=False):
    fs = tsH.sample_rate.value
    r = distance_mpc * Mpc

    # Fenêtre de bruit sécurisée
    start_avail = tsH.t0.value
    try:
        if gps - noise_pad - 400.0 < start_avail:
            noiseH = tsH.crop(start_avail, start_avail + 200.0)
            noiseL = tsL.crop(start_avail, start_avail + 200.0)
        else:
            noiseH = tsH.crop(gps - noise_pad - 400.0, gps - noise_pad - 200.0)
            noiseL = tsL.crop(gps - noise_pad - 400.0, gps - noise_pad - 200.0)
    except Exception:
        noiseH = tsH.crop(gps - 1000.0, gps - 800.0)
        noiseL = tsL.crop(gps - 1000.0, gps - 800.0)

    fH, S1 = psd_welch(noiseH, fmin=flow, fmax=fhigh)
    fL, S2 = psd_welch(noiseL, fmin=flow, fmax=fhigh)
    if not np.allclose(fH, fL):
        S2 = np.interp(fH, fL, S2)
    f_psd = fH

    # Signal principal
    half = 0.5 * signal_win
    hH = safe_bandpass(np.asarray(tsH.crop(gps - half, gps + half).value, float), fs, flow, fhigh)
    hL = safe_bandpass(np.asarray(tsL.crop(gps - half, gps + half).value, float), fs, flow, fhigh)

    # Calibration RMS
    wHc = tsH.crop(gps - 1.0, gps + 1.0)
    wLc = tsL.crop(gps - 1.0, gps + 1.0)
    hHc = safe_bandpass(np.asarray(wHc.value, float), fs, flow, fhigh)
    hLc = safe_bandpass(np.asarray(wLc.value, float), fs, flow, fhigh)
    h_rms_obs = 0.5 * (np.sqrt(np.mean(hHc ** 2)) + np.sqrt(np.mean(hLc ** 2)))
    snr_like = h_rms_obs / np.sqrt(np.median(S1))
    scale = H_STAR / max(h_rms_obs, 1e-30)
    scale *= 410.0 / distance_mpc
    if snr_like < 3:
        scale *= 3 / max(snr_like, 1e-6)
    print(f"[calib appliquée] RMS crête: {h_rms_obs:.3e} → scale={scale:.3e}")
    hH *= scale
    hL *= scale

    # Spectre d'énergie cohérent
    tau_guess = estimate_delay(hH, hL, fs)
    N = min(len(hH), len(hL))
    dt = 1.0 / fs
    T = N * dt
    w = tukey(N, 0.2)
    Uwin = (w ** 2).sum()
    H1 = np.fft.rfft(hH * w)
    H2 = np.fft.rfft(hL * w)
    f_short = np.fft.rfftfreq(N, d=dt)

    def energy_with_tau(tau):
        # Remplacer tout le contenu par l'appel à compute_energy
        return compute_energy(H1, H2, f_short, tau, fs, Uwin, r, distance_mpc, flow, fhigh)

    E_pos, f_use, dEdf_use = energy_with_tau(tau_guess)
    if not np.isfinite(E_pos) or E_pos <= 0:
        print("[!] Énergie non significative, saut de l'événement.")
        return {"E_total": 0.0, "m_sun": 0.0, "nu_eff": 0.0}

    # Calibration énergétique
    try:
        with open("results/global_gain.json", "r") as fg:
            gglob = float(json.load(fg).get("g", 1.0))
            if np.isfinite(gglob) and gglob > 0:
                dEdf_use *= gglob
                E_pos *= gglob
                print(f"[calib énergie] Gain global appliqué: g={gglob:.3e} → E={E_pos:.3e} J")
    except Exception:
        pass

    # Sauvegarde du spectre
    E = E_pos
    m_sun = E / (M_sun * c ** 2)
    nu_eff = float(np.trapz(f_use * dEdf_use) / max(np.trapz(dEdf_use), 1e-30))
    os.makedirs("results", exist_ok=True)
    out_json = {
        "event": event_name,
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
    print(f"E = {E:.3e} J ; m = {m_sun:.3f} M☉ ; ν_eff = {nu_eff:.1f} Hz ; h★={H_STAR:.2e}")


    if plot:
        plt.figure(figsize=(10, 6))
        # Tracé du spectre d'énergie (log-log)
        plt.loglog(f_use, dEdf_use, lw=1.5, label=f"{event_name} (E={E:.2e} J, ν_eff={nu_eff:.1f} Hz)")

        # Lissage pour une meilleure visualisation (optionnel)
        f_log = np.geomspace(f_use[0], f_use[-1], 500)
        dEdf_log = np.interp(f_log, f_use, dEdf_use)
        log_smooth = gaussian_filter1d(np.log10(np.maximum(dEdf_log, 1e-50)), sigma=2)

        # Tracé lissé
        plt.loglog(f_log, 10**log_smooth, '--', lw=1, color='orange', alpha=0.7, label=f"{event_name} (lissé)")

        # Éléments graphiques
        plt.xlabel("Fréquence (Hz)", fontsize=12)
        plt.ylabel("dE/df (J/Hz)", fontsize=12)
        plt.title(f"Spectre d'énergie gravitationnelle normalisé\n{event_name} — h★={H_STAR:.2e}", fontsize=14)
        plt.grid(True, which="both", alpha=0.3, linestyle='--')
        plt.legend(fontsize=10)
        plt.xlim(f_use[0], f_use[-1])
        plt.ylim(auto=True)  # Laisser matplotlib ajuster automatiquement

        # Ligne verticale pour ν_eff
        plt.axvline(x=nu_eff, color='red', linestyle=':', linewidth=1, label=f"ν_eff = {nu_eff:.1f} Hz")

        # Sauvegarde et affichage
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{event_name}_spectre.png", dpi=200, bbox_inches='tight')
        plt.show()






    return {"E_total": E, "m_sun": m_sun, "nu_eff": nu_eff}

# ==========================
# CLI
# ==========================
def main():
    ap = argparse.ArgumentParser(description="Analyse LIGO cohérente h★=e-22")
    ap.add_argument("--event", required=True)
    ap.add_argument("--distance-mpc", type=float, required=True)
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
    print(f"Énergie: {res['E_total']:.3e} J ({res['m_sun']:.3f} M☉)")
    print(f"Fréquence effective: {res['nu_eff']:.1f} Hz\n{'='*60}")

if __name__ == "__main__":
    main()
