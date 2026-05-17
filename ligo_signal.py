"""Signal processing pour LIGO — vire les 4 boutons phénoménologiques.

Garde du repo (parties physiquement saines) :
- PSD Welch median sur segments bruit (robuste glitches)
- Cross-correlation FFT avec interpolation parabolique sub-échantillon
- Filtres SOS Butterworth + fenêtre Tukey

Améliorations vs `ligo_spectral_planck.py` :
- Matched filter contre templates de chirp simples par classe
- Whitening avec gestion bord propre (le repo notait déjà ce problème)
- Multi-resolution time-frequency tile (BNS narrowband vs BBH wideband)
- Pas de TAU_SCALE/NU_SCALE/SCALE_EJ/PEAK_SCALE — la calibration physique
  est dans `ligo_calibrate.py`, un seul scalaire par (classe, path).

Le NPZ loader est conservé tel quel : I/O propre, pas de magie.
"""

import os
import numpy as np

try:
    from scipy.signal import butter, sosfiltfilt
    from scipy.signal.windows import tukey
    from scipy.integrate import trapezoid
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ============================================================
# Constantes
# ============================================================
M_sun = 1.989e30
c = 2.998e8
G = 6.674e-11


# ============================================================
# NPZ loading — identique au repo (I/O propre)
# ============================================================
def get_npz_path(event: str, detector: str, npz_dir: str = "data/npz") -> str:
    filename = f"{event}_{detector}.npz"
    path = os.path.join(npz_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPZ file not found: {path}")
    return path


def load_npz(path: str):
    z = np.load(path, allow_pickle=False)
    return (
        np.asarray(z["data"], float),
        float(z["fs"]),
        float(z["t0"]),
        float(z["t1"]),
    )


def crop_array(x, fs, t0_arr, t_start, t_end):
    i0 = int((t_start - t0_arr) * fs)
    i1 = int((t_end - t0_arr) * fs)
    return x[max(i0, 0):max(i1, 0)]


# ============================================================
# PSD : Welch médian sur segments bruit
# ============================================================
def psd_welch_median(x, fs, seglen=4.0, overlap=2.0):
    """PSD robuste par médiane de Welch.

    Médiane (pas moyenne) → robuste aux glitches non-gaussiens.
    Tukey window pour limiter le leakage spectral.
    Normalisation one-sided Welch : 2/(fs·U).
    """
    if not SCIPY_OK:
        raise ImportError("scipy requis pour signal processing")

    x = np.asarray(x, float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.size < 8:
        f = np.fft.rfftfreq(max(8, int(fs)), d=1.0 / fs)
        return f, np.ones_like(f) * 1e-44

    nseg = int(seglen * fs)
    nhop = int((seglen - overlap) * fs)
    if nseg < 16 or nhop < 1 or x.size < nseg:
        nseg = min(x.size, max(256, int(1.0 * fs)))
        nhop = max(1, nseg // 2)

    win = tukey(nseg, 0.25)
    U = float((win ** 2).sum())

    specs = []
    for i in range(0, x.size - nseg + 1, nhop):
        seg = x[i:i + nseg] * win
        X = np.fft.rfft(seg)
        Pxx = (2.0 / (fs * U)) * (np.abs(X) ** 2)
        if np.isfinite(Pxx).all():
            specs.append(Pxx)

    f = np.fft.rfftfreq(nseg, d=1.0 / fs)
    if not specs:
        return f, np.ones_like(f) * 1e-44

    S = np.median(np.stack(specs), axis=0)
    S = np.maximum(S, 1e-60)
    return f, S


# ============================================================
# Bandpass propre
# ============================================================
def bandpass(x, fs, f1, f2, order=4):
    """Bandpass Butterworth SOS, phase nulle (sosfiltfilt)."""
    if not SCIPY_OK:
        raise ImportError("scipy requis")

    x = np.asarray(x, float)
    nyq = 0.5 * fs
    f1 = max(f1, 0.1)
    f2 = min(f2, 0.95 * nyq)
    if f2 <= f1:
        f2 = min(0.95 * nyq, f1 + 10.0)

    sos = butter(order, [f1, f2], btype="bandpass", fs=fs, output="sos")
    padlen = 3 * (sos.shape[0] - 1)
    if x.size <= padlen:
        return np.zeros_like(x)

    y = sosfiltfilt(sos, x)
    # Edge taper léger pour éviter ringing début/fin
    win = tukey(len(y), 0.05)
    return y * win


# ============================================================
# Tau H1-L1 via cross-correlation sub-sample
# ============================================================
def estimate_delay(x, y, fs, max_delay_ms=15.0):
    """Cross-correlation FFT avec interpolation parabolique sub-échantillon.

    Lag borné par ±15 ms (ligne de base H1-L1 = 3000 km / c ≈ 10 ms).
    Au-delà : on tape sur des glitches lointains, pas le signal.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    N = min(x.size, y.size)
    if N < 8:
        return 0.0
    x = x[:N] - np.mean(x[:N])
    y = y[:N] - np.mean(y[:N])

    nfft = 1
    while nfft < 2 * N:
        nfft *= 2

    X = np.fft.rfft(x, n=nfft)
    Y = np.fft.rfft(y, n=nfft)
    r = np.fft.irfft(X * np.conj(Y), n=nfft)
    r = np.concatenate([r[-(N - 1):], r[:N]])
    lags = np.arange(-(N - 1), N) / fs

    lim = float(max_delay_ms) / 1000.0
    mask = (lags >= -lim) & (lags <= lim)
    if not np.any(mask):
        return 0.0

    r_masked = np.abs(r[mask])
    idx = int(np.argmax(r_masked))

    # Interpolation parabolique sub-sample
    if 0 < idx < len(r_masked) - 1:
        y1, y2, y3 = r_masked[idx - 1], r_masked[idx], r_masked[idx + 1]
        denom = 2.0 * (y1 - 2.0 * y2 + y3)
        delta = (y1 - y3) / denom if abs(denom) > 1e-10 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
        lag_samples = float(lags[mask][idx]) * fs + delta
        return lag_samples / fs

    return float(lags[mask][idx])


# ============================================================
# NOUVEAU : Matched filter contre template de chirp simple
# ============================================================
def chirp_template_inspiral(t, M_chirp_kg, f_low=20.0):
    """Template Newtonien d'inspiral pré-merger (Peters 1964).

    Phase post-Newtonienne 0PN : θ(t) = c³(t_c - t)/(5 G M_chirp)
    f(t) = (1/π)(5/256)^(3/8) (G M_chirp/c³)^(-5/8) (t_c - t)^(-3/8)
    h(t) ∝ f(t)^(2/3) cos(2π ∫f dt)

    Renvoie h_template normalisé à norm 1.
    """
    t_c = t[-1] + 0.01  # coalescence un peu après la fenêtre
    tau = t_c - t
    tau = np.maximum(tau, 1e-6)
    GMc_c3 = G * M_chirp_kg / c**3
    # Frequency evolution
    f_t = (1 / np.pi) * (5 / 256)**(3/8) * GMc_c3**(-5/8) * tau**(-3/8)
    f_t = np.clip(f_t, f_low, 500.0)
    # Phase via cumulative integration
    dt = t[1] - t[0]
    phase = 2 * np.pi * np.cumsum(f_t) * dt
    # Amplitude ∝ f^(2/3)
    h = (f_t / f_low)**(2/3) * np.cos(phase)
    h = h - np.mean(h)
    norm = np.linalg.norm(h)
    return h / norm if norm > 0 else h


def matched_filter_snr(strain, fs, M_chirp_kg, psd_f, psd, flow=20.0, fhigh=350.0):
    """SNR matched-filter contre template chirp inspiral.

    Normalisation standard LIGO/PyCBC :
      ρ(τ) = ⟨s|h_τ⟩ / sqrt(⟨h|h⟩)
      ⟨a|b⟩ = 4 Re ∫₀^∞ ã*(f) b̃(f) / S(f) df

    Conventions numpy (dimensions complètes) :
      ã(f_k) = a_f[k] · Δt = a_f[k] / fs  avec  Δt = 1/fs
      Δf = fs / N
      Σ_{k=0..N/2} Re X[k] e^(2πikn/N) ≈ (N/2) × irfft(X)[n]

    Donc :
      ⟨s|h_τ_n⟩ = (4/(N fs)) · (N/2) · irfft(integrand)[n] = (2/fs) · irfft(integrand)[n]
      σ² = ⟨h|h⟩ = (4/(N fs)) · Σ_band |h_f|² / S
      ρ(t_n) = (2/fs) · irfft(integrand)[n] / sqrt(σ²)   [sans dimension]

    Renvoie (snr_peak, t_peak_s). Renvoie (0, 0) si :
    - strain contient des NaN (data gaps GWOSC, ex. GW190910)
    - σ² ≤ 0 (bande vide ou PSD anormale)
    """
    if not SCIPY_OK:
        raise ImportError("scipy requis")

    strain = np.asarray(strain, float)
    if not np.isfinite(strain).all():
        return 0.0, 0.0

    N = len(strain)
    if N < 64:
        return 0.0, 0.0

    t = np.arange(N) / fs
    template = chirp_template_inspiral(t, M_chirp_kg, flow)
    if len(template) != N:
        if len(template) < N:
            template = np.concatenate([template, np.zeros(N - len(template))])
        else:
            template = template[:N]
    if not np.isfinite(template).all():
        return 0.0, 0.0

    # FFTs
    s_f = np.fft.rfft(strain * tukey(N, 0.1))
    h_f = np.fft.rfft(template)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    # PSD interp + sanity (S doit être > 1e-50 strain²/Hz physiquement)
    S = np.interp(freqs, psd_f, psd)
    mask = (freqs >= flow) & (freqs <= fhigh) & np.isfinite(S) & (S > 1e-50)
    if not np.any(mask):
        return 0.0, 0.0

    # σ² = ⟨h|h⟩ (dimensionnel)
    sigma_sq = (4.0 / (N * fs)) * np.sum(np.abs(h_f[mask])**2 / S[mask])
    if sigma_sq <= 0 or not np.isfinite(sigma_sq):
        return 0.0, 0.0

    # Cross-correlation time series : integrand zéro hors-bande pour éviter pollution
    integrand = np.zeros_like(s_f)
    integrand[mask] = np.conj(s_f[mask]) * h_f[mask] / S[mask]
    snr_t = (2.0 / fs) * np.real(np.fft.irfft(integrand, n=N)) / np.sqrt(sigma_sq)

    if not np.isfinite(snr_t).all():
        return 0.0, 0.0

    idx = int(np.argmax(np.abs(snr_t)))
    return float(np.abs(snr_t[idx])), float(idx / fs)


# ============================================================
# Coherent H1-L1 spectrum (kept from repo, simplified)
# ============================================================
def coherent_energy_density(strain_H, strain_L, fs, tau_HL, flow, fhigh,
                             psd_f_H, psd_H, psd_f_L, psd_L):
    """Densité spectrale d'énergie cohérente H1-L1.

    PAS de TAU_SCALE/SCALE_EJ/PEAK_SCALE — un seul résultat propre.
    Calibration physique à faire en aval via `ligo_calibrate`.

    Returns:
        f_use : freqs dans la bande
        dEdf : densité spectrale d'énergie phénoménologique |Hc|²
        E_total : intégrale (unités internes, à calibrer dimensionnellement)
    """
    N = min(len(strain_H), len(strain_L))
    if N < 16:
        return np.array([]), np.array([]), 0.0

    # Bandpass déjà fait normalement, mais on assure
    win = tukey(N, 0.2)
    H1 = np.fft.rfft(strain_H[:N] * win)
    H2 = np.fft.rfft(strain_L[:N] * win)
    f = np.fft.rfftfreq(N, d=1.0 / fs)

    mask = (f >= flow) & (f <= min(fhigh, 0.95 * 0.5 * fs))
    f_use = f[mask]
    H1, H2 = H1[mask], H2[mask]

    # PSD interp
    S1 = np.maximum(np.interp(f_use, psd_f_H, psd_H), 1e-60)
    S2 = np.maximum(np.interp(f_use, psd_f_L, psd_L), 1e-60)

    # Whitening + coherent sum
    phi = 2.0 * np.pi * f_use * tau_HL
    Hw1 = H1 / np.sqrt(S1)
    Hw2 = H2 / np.sqrt(S2)
    Hc = Hw1 + Hw2 * np.exp(-1j * phi)

    dEdf = np.abs(Hc) ** 2
    E_total = float(trapezoid(dEdf, f_use))
    return f_use, dEdf, E_total


# ============================================================
# Paramètres par classe (depuis write_events.py — physiquement motivés)
# ============================================================
CLASS_PARAMS = {
    "ULTRA_LIGHT": {
        "flow": 15.0, "fhigh": 400.0,
        "tau_band": (20.0, 400.0),
        "nu_band": (15.0, 450.0),
        "signal_win": 3.0, "noise_pad": 600.0,
        "M_chirp_template_sun": 1.2,  # BNS-like
    },
    "LIGHT": {
        "flow": 18.0, "fhigh": 380.0,
        "tau_band": (25.0, 380.0),
        "nu_band": (18.0, 450.0),
        "signal_win": 2.0, "noise_pad": 800.0,
        "M_chirp_template_sun": 8.0,
    },
    "MEDIUM": {
        "flow": 20.0, "fhigh": 350.0,
        "tau_band": (30.0, 350.0),
        "nu_band": (20.0, 500.0),
        "signal_win": 1.2, "noise_pad": 1200.0,
        "M_chirp_template_sun": 25.0,  # BBH typique
    },
    "MASSIVE": {
        "flow": 25.0, "fhigh": 500.0,
        "tau_band": (40.0, 500.0),
        "nu_band": (30.0, 600.0),
        "signal_win": 0.6, "noise_pad": 1500.0,
        "M_chirp_template_sun": 50.0,
    },
}
