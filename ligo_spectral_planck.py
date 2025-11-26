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
from numba import njit
from scipy.signal.windows import dpss
import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import tukey
from scipy.ndimage import gaussian_filter1d
from gwosc import datasets
from gwpy.timeseries import TimeSeries
from scipy.integrate import trapezoid
# ==========================
# Constantes physiques
# ==========================
c = 299792458.0
M_sun = 1.98847e30
Mpc = 3.085677581491367e22
SCALE_EJ = 8e-12
H_STAR = 1
# ==========================
# Utilitaires
# ==========================
def load_event_params():
    path = os.path.join(os.path.dirname(__file__), "event_params.json")
    print("[DEBUG] Chargement JSON depuis :", path)
    with open(path, "r") as f:
        return json.load(f)

EVENT_PARAMS = load_event_params()

def fetch(det, t0, t1, outdir="data") -> TimeSeries:
    os.makedirs(outdir, exist_ok=True)
    return TimeSeries.fetch_open_data(det, t0, t1, cache=False)
def nu_eff_energy(f, dEdf):
    """
    Fréquence effective officielle :
    ν_eff = ∫ f (dE/df) df / ∫ (dE/df) df
    """
    num = trapezoid(f * dEdf, f)
    den = trapezoid(dEdf, f)
    return num / den if den > 0 else 0.0


def nu_mean_simple(f):
    """
    Moyenne arithmétique des fréquences.
    """
    return float(np.mean(f))


def nu_rms(f):
    """
    Moyenne quadratique (root-mean-square).
    """
    return float(np.sqrt(np.mean(f * f)))


def nu_eff_heuristic(f):
    """
    Ton intuition :
        ν_eff ≈ sqrt(2 × moyenne_simple)
    """
    nu_m = np.mean(f)
    return float(np.sqrt(2.0 * nu_m))


def compute_all_nu(f, dEdf):
    """
    Retourne un dict contenant toutes les fréquences 'effectives' utiles.
    """
    return {
        "nu_eff_energy":     nu_eff_energy(f, dEdf),
        "nu_mean_simple":    nu_mean_simple(f),
        "nu_rms":            nu_rms(f),
        "nu_eff_heuristic":  nu_eff_heuristic(f),
    }

def fetch(det, t0, t1, outdir="data") -> TimeSeries:
    os.makedirs(outdir, exist_ok=True)
    return TimeSeries.fetch_open_data(det, t0, t1, cache=True)

def safe_bandpass(x, fs, f1, f2, order=4):
    nyq = 0.5 * fs
    f2_safe = min(f2, 0.95 * nyq)
    if f2_safe <= f1:
        f2_safe = f1 + 10.0

    sos = butter(order, [f1, f2_safe], btype="bandpass", fs=fs, output="sos")
    xf = sosfiltfilt(sos, x)

    # fenêtre douce = seulement 5%
    N = len(xf)
    win = tukey(N, 0.05)

    return xf * win

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

def estimate_delay_time(h1, h2, fs, max_delay_ms=15.0):
    N = min(len(h1), len(h2))
    x = h1[:N] - np.mean(h1[:N])
    y = h2[:N] - np.mean(h2[:N])

    X = np.fft.rfft(x, n=2*N)
    Y = np.fft.rfft(y, n=2*N)
    R = Y * np.conj(X)

    r = np.fft.irfft(R)
    r = np.fft.fftshift(r)

    lags = np.arange(-N, N) / fs
    lim = max_delay_ms * 1e-3
    m = np.abs(lags) <= lim

    i = np.argmax(np.abs(r[m]))
    return float(lags[m][i])

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
    tau = estimate_delay_time(hH_f, hL_f, fs)
    return float(tau)

@njit(cache=True, fastmath=True)
def coherent_energy_numba(H1, H2, S1, S2, phi, f, r):
    n = H1.shape[0]
    dEdf = np.empty(n, dtype=np.float64)
    eps = 1e-22

    for i in range(n):

        # PSD sécurisées
        s1 = S1[i] if S1[i] > eps else eps
        s2 = S2[i] if S2[i] > eps else eps

        w1 = 1.0 / s1
        w2 = 1.0 / s2

        # phase propre
        ph = phi[i] if not np.isnan(phi[i]) else 0.0
        c = np.cos(ph)
        s = np.sin(ph)
        eiphi = c + 1j*s

        # MOYENNE COHÉRENTE CORRECTE
        num = H1[i] * w1 + eiphi * H2[i] * w2
        den = w1 + w2 + eps

        Hc = num / den

        # énergie GR
        dEdf[i] = (r*r) * (f[i]*f[i]) * (Hc.real*Hc.real + Hc.imag*Hc.imag)

    return dEdf

# ==========================
# ANALYSE SPECTRALE — VERSION CORRIGÉE
# ==========================
def analyze_coherent_spectral(tsH, tsL, gps, distance_mpc, V1=None,
                              event_name="", flow=20.0, fhigh=350.0,
                              noise_pad=50.0, signal_win=1.2, plot=False):

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
    g = fH
    # -------------------------------------------------------
    # 2) Estimation du délai initial (NON filtré !)
    # -------------------------------------------------------
    segH = tsH.crop(gps - 0.2, gps + 0.2)
    segL = tsL.crop(gps - 0.2, gps + 0.2)

    # >>> signaux bruts pour la corrélation <<< 
    hH_delay = np.asarray(segH.value, float)
    hL_delay = np.asarray(segL.value, float)

    hH = safe_bandpass(hH_delay, fs, flow, fhigh)
    hL = safe_bandpass(hL_delay, fs, flow, fhigh)
    # Corrélation avec bon signe
    tau_guess = estimate_tau_geo(tsH, tsL, gps, fs, flow, fhigh)

    # -------------------------------------------------------
    # 3) Extraction du signal utile
    # -------------------------------------------------------
    # ====== recentrage automatique du pic ======
    # détecte le vrai chirp = max de fréquence instantanée (pas amplitude)
    # --- Ajustement spécial GW170608 ---

    peakH1 = np.max(np.abs(hH))
    peakL1 = np.max(np.abs(hL))
    peak = max(peakH1, peakL1)

    if peak > 0:
        hH = hH / peakH1 * H_STAR
        hL = hL / peakL1 * H_STAR

    # -------------------------------------------------------
    # 4) FFT courte
    # -------------------------------------------------------
    N = min(len(hH), len(hL))
    w = tukey(N, 0.2)
    H1 = np.fft.rfft(hH * w)
    H2 = np.fft.rfft(hL * w)
    f  = np.fft.rfftfreq(N, 1/fs)

    # -------------------------------------------------------
    # 5) Spectre d'énergie cohérent (version correcte)
    # -------------------------------------------------------
    # Phase cohérente entre H1 et H2 (retard tau_guess)
    phi = 2 * np.pi * f * tau_guess
    # Combinaison cohérente pondérée par PSD
    eps = 1e-30
    r = distance_mpc * Mpc
    # Version compilée Numba (15× plus rapide)
    dEdf = coherent_energy_numba(H1, H2, S1, S2, phi, f, r)

    nu_eff = compute_all_nu(f, dEdf)
    E = float(trapezoid(dEdf, f)) * SCALE_EJ
    m_sun = E / (M_sun * c**2)
    # -------------------------------------------------------
    os.makedirs("results", exist_ok=True)


    out_json = {
        "event": event_name,
        "distance_mpc": float(distance_mpc),
        "freq_Hz": g.tolist(),
        "dEdf_J_Hz": dEdf.tolist(),
        "E_total_J": float(E),
        "m_sun": float(m_sun),
        "nu_eff": compute_all_nu(f, dEdf),
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
        plt.loglog(g, dEdf, lw=1.5, 
                  label=f"{event_name} (E={E:.2e} J, ν_eff={nu_eff:.1f} Hz, d={distance_mpc} Mpc)")

        f_log = np.geomspace(g[0], g[-1], 500)
        dEdf_log = np.interp(f_log, g, dEdf)
        log_smooth = gaussian_filter1d(np.log10(np.maximum(dEdf_log, 1e-50)), sigma=2)

        plt.loglog(f_log, 10**log_smooth, '--', lw=1, color='orange', alpha=0.7, label=f"{event_name} (lissé)")
        plt.xlabel("Fréquence (Hz)", fontsize=12)
        plt.ylabel("dE/df (J/Hz)", fontsize=12)
        plt.title(f"Spectre d'énergie gravitationnelle intrinsèque\n{event_name} — h★={H_STAR:.2e}, d={distance_mpc} Mpc", fontsize=14)
        plt.grid(True, which="both", alpha=0.3, linestyle='--')
        plt.legend(fontsize=10)
        plt.xlim(g[0], g[-1])
        plt.ylim(auto=True)

        plt.axvline(x=nu_eff, color='red', linestyle=':', linewidth=1, label=f"ν_eff = {nu_eff:.1f} Hz")

        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{event_name}_spectre.png", dpi=200, bbox_inches='tight')
        plt.show()

    return {"E_total": E, "m_sun": m_sun, "nu_eff": nu_eff, "distance_mpc": distance_mpc}

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
from numba import njit
from scipy.signal.windows import dpss
import argparse, os, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import tukey
from scipy.ndimage import gaussian_filter1d
from gwosc import datasets
from gwpy.timeseries import TimeSeries
from scipy.integrate import trapezoid
# ==========================
# Constantes physiques
# ==========================
c = 299792458.0
M_sun = 1.98847e30
Mpc = 3.085677581491367e22
SCALE_EJ = 8e-12
H_STAR = 1
# ==========================
# Utilitaires
# ==========================
def load_event_params():
    path = os.path.join(os.path.dirname(__file__), "event_params.json")
    print("[DEBUG] Chargement JSON depuis :", path)
    with open(path, "r") as f:
        return json.load(f)

EVENT_PARAMS = load_event_params()

def fetch(det, t0, t1, outdir="data") -> TimeSeries:
    os.makedirs(outdir, exist_ok=True)
    return TimeSeries.fetch_open_data(det, t0, t1, cache=True)

def safe_bandpass(x, fs, f1, f2, order=4):
    nyq = 0.5 * fs
    f2_safe = min(f2, 0.95 * nyq)
    if f2_safe <= f1:
        f2_safe = f1 + 10.0

    sos = butter(order, [f1, f2_safe], btype="bandpass", fs=fs, output="sos")
    xf = sosfiltfilt(sos, x)

    # fenêtre douce = seulement 5%
    N = len(xf)
    win = tukey(N, 0.05)

    return xf * win

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

def estimate_delay_time(h1, h2, fs, max_delay_ms=15.0):
    N = min(len(h1), len(h2))
    x = h1[:N] - np.mean(h1[:N])
    y = h2[:N] - np.mean(h2[:N])

    X = np.fft.rfft(x, n=2*N)
    Y = np.fft.rfft(y, n=2*N)
    R = Y * np.conj(X)

    r = np.fft.irfft(R)
    r = np.fft.fftshift(r)

    lags = np.arange(-N, N) / fs
    lim = max_delay_ms * 1e-3
    m = np.abs(lags) <= lim

    i = np.argmax(np.abs(r[m]))
    return float(lags[m][i])

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
    tau = estimate_delay_time(hH_f, hL_f, fs)
    return float(tau)

@njit(cache=True, fastmath=True)
def coherent_energy_numba(H1, H2, S1, S2, phi, f, r):
    n = H1.shape[0]
    dEdf = np.empty(n, dtype=np.float64)
    eps = 1e-22

    for i in range(n):

        # PSD sécurisées
        s1 = S1[i] if S1[i] > eps else eps
        s2 = S2[i] if S2[i] > eps else eps

        w1 = 1.0 / s1
        w2 = 1.0 / s2

        # phase propre
        ph = phi[i] if not np.isnan(phi[i]) else 0.0
        c = np.cos(ph)
        s = np.sin(ph)
        eiphi = c + 1j*s

        # MOYENNE COHÉRENTE CORRECTE
        num = H1[i] * w1 + eiphi * H2[i] * w2
        den = w1 + w2 + eps

        Hc = num / den

        # énergie GR
        dEdf[i] = (r*r) * (f[i]*f[i]) * (Hc.real*Hc.real + Hc.imag*Hc.imag)

    return dEdf

@njit(cache=True, fastmath=True)
def coherent_energy_numba_3(H1, H2, H3, S1, S2, S3, phi12, phi13, f, r):
    n = H1.shape[0]
    dEdf = np.empty(n, dtype=np.float64)
    eps = 1e-22

    for i in range(n):

        s1 = S1[i] if S1[i] > eps else eps
        s2 = S2[i] if S2[i] > eps else eps
        s3 = S3[i] if S3[i] > eps else eps

        w1 = 1.0 / s1
        w2 = 1.0 / s2
        w3 = 1.0 / s3

        ph12 = phi12[i] if not np.isnan(phi12[i]) else 0.0
        ph13 = phi13[i] if not np.isnan(phi13[i]) else 0.0

        e12 = np.cos(ph12) + 1j*np.sin(ph12)
        e13 = np.cos(ph13) + 1j*np.sin(ph13)

        num = H1[i] * w1 + e12 * H2[i] * w2 + e13 * H3[i] * w3
        den = w1 + w2 + w3 + eps

        Hc = num / den

        dEdf[i] = (r*r) * (f[i]**2) * (Hc.real*Hc.real + Hc.imag*Hc.imag)

    return dEdf

# ==========================
# ANALYSE SPECTRALE — VERSION CORRIGÉE
# ==========================
def analyze_coherent_spectral(tsH, tsL, gps, distance_mpc, V1=None,
                              event_name="", flow=20.0, fhigh=350.0,
                              noise_pad=50.0, signal_win=1.2, plot=False):

    fs = tsH.sample_rate.value

    use_virgo = (V1 is not None)
    hV = None
    H3 = None
    S3 = None

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
    if use_virgo and H3 is not None:
        try:
            fV, S3 = psd_welch(V1, fmin=flow, fmax=fhigh)
            if not np.allclose(f_psd, fV):
                S3 = np.interp(f_psd, fV, S3)
        except Exception:
            use_virgo = False
            S3 = None
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
    """
    if event_name == "GW170608":
        # Fixe un merger court et propre
        t_peak = gps + 0.01      # merger time
        signal_win = 0.25         # 250 ms max
    if event_name != "GW170608":
    """
    seg = tsH.crop(gps - 0.3, gps + 0.1)
    h = np.asarray(seg.value, float)
    dh = np.abs(np.diff(h))
    imax = np.argmax(dh)
    t_peak = seg.times.value[:-1][imax]

    half = signal_win * 0.5
    winH = tsH.crop(t_peak - half, t_peak + half)
    winL = tsL.crop(t_peak - half, t_peak + half)
    if use_virgo:
            try:
                winV = V1.crop(t_peak - half, t_peak + half)
                hV = safe_bandpass(np.asarray(winV.value, float), fs, flow, fhigh)

                # normalisation comme H1/L1
                if peak > 0:
                    hV = hV / peak * H_STAR

            except Exception:
                use_virgo = False
                hV = None
        
    hH = safe_bandpass(np.asarray(winH.value, float), fs, flow, fhigh)
    hL = safe_bandpass(np.asarray(winL.value, float), fs, flow, fhigh)
    N = min(len(hH), len(hL))
    w = tukey(N, 0.2)
    peakH1 = np.max(np.abs(hH))
    peakL1 = np.max(np.abs(hL))
    peak = max(peakH1, peakL1)

    if peak > 0:
        hH = hH / peak * H_STAR
        hL = hL / peak * H_STAR

    # -------------------------------------------------------
    # 4) FFT courte
    # -------------------------------------------------------
    H1 = np.fft.rfft(hH * w)
    H2 = np.fft.rfft(hL * w)
    if use_virgo and hV is not None:
        H3 = np.fft.rfft(hV * w)
    else:
        H3 = None
    if use_virgo:
        if H3 is None or np.all(H3 == 0):
            print("[WARN] Virgo désactivé : signal non exploitable")
            use_virgo = False
    f  = np.fft.rfftfreq(N, 1/fs)

    # -------------------------------------------------------
    # 5) Spectre d'énergie cohérent (version correcte)
    # -------------------------------------------------------

    # Phase cohérente entre H1 et H2 (retard tau_guess)
    phi = 2 * np.pi * f * tau_guess
    if use_virgo and hV is not None:
        phi12 = phi
        phi13 = phi  # on utilise même tau_guess pour V1, sinon incohérent
    else:
        phi12 = phi
        phi13 = None
    r = distance_mpc * Mpc
    # Combinaison cohérente pondérée par PSD
    eps = 1e-30
    # Version compilée Numba (15× plus rapide)
    if use_virgo and H3 is not None and S3 is not None:
        phi12 = phi
        phi13 = phi
        dEdf = coherent_energy_numba_3(H1, H2, H3, S1, S2, S3, phi12, phi13, f, r)
    else:
        dEdf = coherent_energy_numba(H1, H2, S1, S2, phi, f, r)
    dEdf = np.nan_to_num(dEdf)

    # Bande utile
    mask = (f >= flow) & (f <= fhigh)
    f_use = f[mask]
    dEdf_use = dEdf[mask]

    # -------------------------------------------------------
    # 6) Intégrales (E, m_sun, ν_eff)
    # -------------------------------------------------------

    den = trapezoid(dEdf_use, f_use)
    nu_eff = float(trapezoid(f_use * dEdf_use, f_use) / den) if den > 0 else 0.0
    E = float(trapezoid(dEdf_use, f_use)) * SCALE_EJ
    m_sun = E / (M_sun * c**2)
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

    # ----------------------------
    # Arguments CLI
    # ----------------------------
    ap = argparse.ArgumentParser(description="Analyse spectrale cohérente LIGO/Virgo (h★ → énergie intrinsèque)")

    ap.add_argument("--event", type=str, help="Nom d'événement")
    ap.add_argument("--list-events", action="store_true")

    ap.add_argument("--flow", type=float)
    ap.add_argument("--fhigh", type=float)
    ap.add_argument("--signal-win", type=float)
    ap.add_argument("--noise-pad", type=float)
    ap.add_argument("--distance-mpc", type=float)
    ap.add_argument("--plot", action="store_true")

    args = ap.parse_args()

    # ----------------------------
    # Liste des événements
    # ----------------------------
    if args.list_events:
        print("=== Événements disponibles (JSON) ===")
        for k in sorted(EVENT_PARAMS.keys()):
            print(" -", k)
        return

    if not args.event:
        raise ValueError("Veuillez fournir --event <GWxxxxxx>")

    ev = args.event

    # ----------------------------
    # Paramètres événement
    # ----------------------------
    if ev not in EVENT_PARAMS:
        raise ValueError(f"Événement '{ev}' absent dans event_params.json")

    evp = EVENT_PARAMS[ev]

    flow        = args.flow        if args.flow        is not None else evp["flow"]
    fhigh       = args.fhigh       if args.fhigh       is not None else evp["fhigh"]
    signal_win  = args.signal_win  if args.signal_win  is not None else evp["signal_win"]
    noise_pad   = args.noise_pad   if args.noise_pad   is not None else evp["noise_pad"]

    # Distance JSON + override CLI
    distance_mpc = args.distance_mpc if args.distance_mpc is not None else evp["distance_mpc"]

    gps = datasets.event_gps(ev)

    t0_noise = gps - noise_pad
    t1_noise = gps + noise_pad
    # Fenêtres absolues
    # bruits avant/après = noise_pad
    # signal = signal_win
    H1, L1 = fetch("H1", t0_noise, t1_noise), fetch("L1", t0_noise, t1_noise)

    # Tentative Virgo
    try:
        V1 = fetch("V1", t0_noise, t1_noise)
        print(" → Virgo OK")
    except Exception:
        V1 = None
        print(" → Virgo non disponible")
    print(f"📡 Téléchargement des données pour {args.event}...")

    res = analyze_coherent_spectral(H1, L1, gps, distance_mpc, V1,
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
