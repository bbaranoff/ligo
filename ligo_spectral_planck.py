#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Analyse spectrale coherente GWOSC (H1-L1, V1 optionnel)

Lecture des données
-------------------
Le script lit les fichiers NPZ depuis le répertoire data/npz/:
- {event}_H1.npz : données du détecteur Hanford
- {event}_L1.npz : données du détecteur Livingston
- {event}_V1.npz : données du détecteur Virgo (optionnel)

Chaque fichier NPZ contient:
- data: array de strain
- fs: fréquence d'échantillonnage
- t0, t1: temps GPS de début et fin

Paramètres de calibration
--------------------------
Trois paramètres contrôlent la calibration des résultats :
- TAU_SCALE (--tau-scale)    : facteur multiplicatif pour tau (défaut: 1.0)
- SCALE_EJ (--scale-ej)       : facteur d'échelle d'énergie (défaut: 4.37e37)
- PEAK_SCALE (--peak-scale)   : facteur multiplicatif pour amplitude des signaux (défaut: 1.0)

Par défaut (sans arguments), ces valeurs donnent les résultats de référence.
Pour utiliser des valeurs calibrées issues d'une analyse par clusters, 
passez ces paramètres explicitement :

  python ligo_spectral_planck.py --event GW150914 \\
    --tau-scale 0.9 --scale-ej 1.9041e37 --peak-scale 1.0

Sorties
-------
- dEdf_internal: spectre interne (avant SCALE_EJ)
- E_total_J = SCALE_EJ * integral(dE/df)
- m_sun = E_total_J / (M_sun*c^2)
- tau_hl_s: delay H1-L1 estime par correlation sur une bande dediee (tau-band)
- nu_eff_energy: barycentre energetique sur une bande dediee (nu-band)

Calibration par moindres carrés
--------------------------------
- Minimise sum((y_obs - (H_STAR * SCALE_EJ * y_model))^2) sur tous les events
- Deux modes:
  1. --calibrate-lsq : calcule H_STAR et SCALE_EJ optimaux
  2. --use-calibrated : utilise les valeurs pré-calculées

Dependances: numpy, scipy, gwpy, gwosc, numba (optionnel)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import tukey
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import trapezoid
from scipy.optimize import minimize

from gwosc import datasets

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*args, **kwargs):
        def deco(fn):
            return fn
        return deco

# ------------------
# Constantes
# ------------------
c = 299792458.0
M_sun = 1.98847e30
Mpc = 3.085677581491367e22

DEFAULT_TAU_SCALE = 1.0
DEFAULT_SCALE_EJ = 1.0
DEFAULT_PEAK_SCALE = 1.0
SCALE_EJ_NORM = 5.3e37
# ------------------
# NPZ local loading
# ------------------
def get_npz_path(event: str, detector: str, npz_dir: str = "data/npz") -> str:
    """Retourne le chemin vers le fichier NPZ local."""
    filename = f"{event}_{detector}.npz"
    path = os.path.join(npz_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPZ file not found: {path}")
    return path


def load_npz(path: str) -> Tuple[np.ndarray, float, float, float]:
    """Charge un fichier NPZ.
    
    Returns:
        data: array de strain
        fs: fréquence d'échantillonnage
        t0: temps GPS de début
        t1: temps GPS de fin
    """
    z = np.load(path)
    return (
        np.asarray(z["data"], float),
        float(z["fs"]),
        float(z["t0"]),
        float(z["t1"]),
    )


def crop_array(x: np.ndarray, fs: float, t0_arr: float, t_start: float, t_end: float) -> np.ndarray:
    """Extrait un segment temporel d'un array."""
    i0 = int((t_start - t0_arr) * fs)
    i1 = int((t_end - t0_arr) * fs)
    return x[max(i0, 0):max(i1, 0)]


# ------------------
# IO params
# ------------------
def load_event_params(path: str) -> Dict[str, Any]:
    """Charge les paramètres des événements depuis un JSON."""
    with open(path, "r") as f:
        return json.load(f)


def _load_default_event_params() -> Dict[str, Any]:
    """Best-effort load of event_params.json next to this script."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        p = os.path.join(here, "event_params.json")
        if os.path.exists(p):
            return load_event_params(p)
    except Exception:
        pass
    return {}


EVENT_PARAMS: Dict[str, Any] = _load_default_event_params()


def get_event_gps(event: str, params: Dict[str, Any]) -> float:
    """Récupère le temps GPS d'un événement."""
    evp = params.get(event, {})
    if isinstance(evp, dict):
        if "gps" in evp:
            return float(evp["gps"])
        if "GPS" in evp:
            return float(evp["GPS"])
    return float(datasets.event_gps(event))


# ------------------
# Utils
# ------------------
def robust_peak(x: np.ndarray, q: float = 99.5) -> float:
    """Calcule un peak robuste via percentile."""
    x = np.asarray(x, float)
    if x.size == 0:
        return 0.0
    return float(np.percentile(np.abs(x), q))


def safe_bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order: int = 4) -> np.ndarray:
    """Filtre passe-bande avec fenêtre Tukey."""
    x = np.asarray(x, float)
    nyq = 0.5 * fs
    f2_safe = min(float(f2), 0.95 * nyq)
    f1_safe = max(float(f1), 0.1)
    if f2_safe <= f1_safe:
        f2_safe = min(0.95 * nyq, f1_safe + 10.0)
    sos = butter(order, [f1_safe, f2_safe], btype="bandpass", fs=fs, output="sos")
    padlen = 3 * (sos.shape[0] - 1)
    if x.size <= padlen:
        return np.zeros_like(x)

    y = sosfiltfilt(sos, x)
    win = tukey(len(y), 0.05)
    return y * win


def psd_welch_median(
    x: np.ndarray,
    fs: float,
    seglen: float = 4.0,
    overlap: float = 2.0,
    fmin: float = 10.0,
    fmax: float = 1024.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """PSD robuste par médiane de Welch (fenêtre Tukey), tolérante aux NaN."""
    x = np.asarray(x, float)
    if x.size < 8:
        f = np.fft.rfftfreq(max(8, int(fs)), d=1.0 / fs)
        return f, np.ones_like(f) * 1e-44

    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    nseg = int(seglen * fs)
    nhop = int((seglen - overlap) * fs)
    if nseg < 16 or nhop < 1 or x.size < nseg:
        nseg = min(x.size, max(256, int(1.0 * fs)))
        nhop = max(1, nseg // 2)

    win = tukey(nseg, 0.25)
    U = float((win ** 2).sum())

    specs = []
    for i in range(0, x.size - nseg + 1, nhop):
        seg = x[i:i + nseg]
        if seg.size < 8:
            continue

        if not np.isfinite(seg).all():
            seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)

        seg = seg - float(np.mean(seg))
        seg = safe_bandpass(seg, fs, fmin, fmax)
        if not np.isfinite(seg).all():
            continue

        X = np.fft.rfft(seg * win)
        Pxx = (2.0 / (fs * U)) * (np.abs(X) ** 2)

        if np.isfinite(Pxx).all():
            specs.append(Pxx)

    f = np.fft.rfftfreq(nseg, d=1.0 / fs)

    if not specs:
        return f, np.ones_like(f) * 1e-44

    S = np.median(np.stack(specs, axis=0), axis=0)

    S = np.asarray(S, float)
    bad = (~np.isfinite(S)) | (S <= 0.0)
    if bad.all():
        S = np.ones_like(S) * 1e-44
    else:
        fill = float(np.nanmedian(S[~bad]))
        if not np.isfinite(fill) or fill <= 0:
            fill = 1e-44
        S[bad] = fill

    S = np.maximum(S, 1e-60)
    return f, S


def sanitize_psd(f: np.ndarray, S: np.ndarray, name: str = "PSD") -> Tuple[np.ndarray, np.ndarray]:
    """Nettoie une PSD : garantit finie, >0, et interp stable."""
    f = np.asarray(f, float)
    S = np.asarray(S, float)

    if f.size != S.size or f.size < 8:
        raise RuntimeError(f"[FATAL] {name}: PSD vide/invalide")

    bad = (~np.isfinite(S)) | (S <= 0.0)
    if bad.all():
        return f, np.ones_like(S) * 1e-44

    fill = float(np.nanmedian(S[~bad]))
    if not np.isfinite(fill) or fill <= 0:
        fill = 1e-44

    S2 = S.copy()
    S2[bad] = fill
    S2 = np.maximum(S2, 1e-60)
    return f, S2


def spectral_bandwidth_features(freq_hz: np.ndarray, spec: np.ndarray) -> dict:
    """
    Calcule les caractéristiques de bande passante spectrale.
    freq_hz : array 1D croissant
    spec    : array 1D >=0 (ex: |H(f)|^2, ou densité d'énergie spectrale)
    Retourne f10, f50, f90, dnu_80, mu_f, sigma_f.
    """
    f = np.asarray(freq_hz, dtype=float)
    s = np.asarray(spec, dtype=float)

    if f.ndim != 1 or s.ndim != 1 or f.size != s.size or f.size < 4:
        return {
            "f10": np.nan, "f50": np.nan, "f90": np.nan,
            "dnu_80": np.nan, "mu_f": np.nan, "sigma_f": np.nan
        }

    s = np.maximum(s, 0.0)

    total = float(trapezoid(s, f))
    if not np.isfinite(total) or total <= 0:
        return {
            "f10": np.nan, "f50": np.nan, "f90": np.nan,
            "dnu_80": np.nan, "mu_f": np.nan, "sigma_f": np.nan
        }

    df = np.diff(f)
    mid = 0.5 * (s[:-1] + s[1:])
    cum = np.concatenate(([0.0], np.cumsum(mid * df)))
    cdf = cum / cum[-1]

    f10 = np.interp(0.10, cdf, f)
    f50 = np.interp(0.50, cdf, f)
    f90 = np.interp(0.90, cdf, f)
    dnu_80 = f90 - f10

    mu_f = float(trapezoid(f * s, f) / total)
    var = float(trapezoid(((f - mu_f) ** 2) * s, f) / total)
    sigma_f = np.sqrt(max(var, 0.0))

    return {
        "f10": float(f10),
        "f50": float(f50),
        "f90": float(f90),
        "dnu_80": float(dnu_80),
        "mu_f": float(mu_f),
        "sigma_f": float(sigma_f),
    }


def estimate_delay_time(x: np.ndarray, y: np.ndarray, fs: float, max_delay_ms: float = 15.0) -> float:
    """Cross-correlation FFT avec interpolation sub-échantillon, renvoie le lag (s) max abs corr."""
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
    R = X * np.conj(Y)
    r = np.fft.irfft(R, n=nfft)

    r = np.concatenate([r[-(N-1):], r[:N]])
    lags = np.arange(-(N-1), N) / fs

    lim = float(max_delay_ms) / 1000.0
    mask = (lags >= -lim) & (lags <= lim)
    if not np.any(mask):
        return 0.0

    r_masked = np.abs(r[mask])
    idx = np.argmax(r_masked)
    
    # Interpolation parabolique sub-échantillon
    if 0 < idx < len(r_masked) - 1:
        y1 = r_masked[idx - 1]
        y2 = r_masked[idx]
        y3 = r_masked[idx + 1]
        
        denom = 2.0 * (y1 - 2.0 * y2 + y3)
        if abs(denom) > 1e-10:
            delta = (y1 - y3) / denom
            delta = np.clip(delta, -0.5, 0.5)
        else:
            delta = 0.0
        
        lag_samples = lags[mask][idx] * fs + delta
        lag = lag_samples / fs
    else:
        lag = lags[mask][idx]
    
    return float(lag)


def nu_eff_energy(f: np.ndarray, dEdf: np.ndarray) -> float:
    """nu_eff = int f dEdf / int dEdf, garde-fous den>0."""
    f = np.asarray(f, float)
    w = np.maximum(np.asarray(dEdf, float), 0.0)
    den = float(trapezoid(w, f))
    if (not np.isfinite(den)) or den <= 0.0:
        return 0.0
    num = float(trapezoid(f * w, f))
    if not np.isfinite(num):
        return 0.0
    nu = num / den
    return float(nu) if np.isfinite(nu) else 0.0


@njit(cache=True, fastmath=True)
def coherent_energy_numba(H1, H2, S1, S2, phi, f, r):
    """Calcul optimisé de l'énergie cohérente (H1+H2 seulement)."""
    n = H1.shape[0]
    out = np.empty(n, dtype=np.float64)
    eps = 1e-30
    for i in range(n):
        s1 = S1[i] if S1[i] > eps else eps
        s2 = S2[i] if S2[i] > eps else eps
        w1 = 1.0 / s1
        w2 = 1.0 / s2
        ph = phi[i]
        c = math.cos(ph)
        s = math.sin(ph)
        re = c
        im = s

        a = H1[i] * w1
        b = H2[i] * w2
        br = b.real * re - b.imag * im
        bi = b.real * im + b.imag * re

        numr = a.real + br
        numi = a.imag + bi
        den = w1 + w2 + eps

        Hcr = numr / den
        Hci = numi / den

        ff = f[i]
        out[i] = (r * r) * (ff * ff) * (Hcr * Hcr + Hci * Hci)

    return out


@dataclass
class Bands:
    tau_band: Tuple[float, float]
    nu_band: Tuple[float, float]


def analyze_event(
    event: str,
    params: Dict[str, Any],
    flow: Optional[float] = None,
    fhigh: Optional[float] = None,
    signal_win: Optional[float] = None,
    noise_pad: Optional[float] = None,
    distance_mpc: Optional[float] = None,
    bands: Optional[Bands] = None,
    use_virgo: bool = True,
    npz_dir: str = "data/npz",
    peak_norm: bool = False,
    peak_quantile: float = 99.5,
    tau_scale: float = DEFAULT_TAU_SCALE,
    scale_ej_in: float = DEFAULT_SCALE_EJ,
    peak_scale: float = DEFAULT_PEAK_SCALE,
    plot: bool = False,
    return_internals: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Analyse spectrale d'un événement gravitationnel.
    
    Args:
        event: Nom de l'événement (ex: GW150914)
        params: Dictionnaire des paramètres d'événements
        flow, fhigh: Bande de fréquence d'analyse [Hz]
        signal_win: Fenêtre temporelle du signal [s]
        noise_pad: Padding avant le signal pour estimer le bruit [s]
        distance_mpc: Distance de l'événement [Mpc]
        bands: Bandes pour tau et nu
        use_virgo: Utiliser Virgo si disponible
        npz_dir: Répertoire contenant les fichiers NPZ
        peak_norm: Normaliser par le peak
        peak_quantile: Quantile pour le peak
        tau_scale: Facteur multiplicatif pour tau
        scale_ej_in: Facteur d'échelle d'énergie
        peak_scale: Facteur multiplicatif pour amplitude des signaux
        plot: Générer des plots
        return_internals: Retourner les données internes (pour calibration)
        verbose: Afficher les messages de debug
    """
    # -----------------------------
    # 0) Resolve per-event params with full fallback chain
    # -----------------------------
    # Get event-specific params, fallback to "default" section
    evp = params.get(event, {}) if isinstance(params.get(event), dict) else {}
    defaults = params.get("default", {}) if isinstance(params.get("default"), dict) else {}

    # GPS is mandatory
    gps = None
    for k in ("gps", "GPS", "gps_time"):
        if k in evp and evp[k] is not None:
            gps = float(evp[k])
            break
    if gps is None:
        # best effort fallback to GWOSC catalog
        try:
            gps = float(datasets.event_gps(event))
        except Exception:
            raise RuntimeError(f"[FATAL] GPS time missing for {event}")

    # Distance with comprehensive fallback
    if distance_mpc is None or (isinstance(distance_mpc, (int, float)) and float(distance_mpc) <= 0.0):
        for k in ("distance_mpc", "distance_Mpc", "luminosity_distance", "luminosity_distance_mpc"):
            if k in evp and evp[k] is not None and float(evp[k]) > 0:
                distance_mpc = float(evp[k])
                break
    if distance_mpc is None or float(distance_mpc) <= 0.0:
        raise RuntimeError(f"[FATAL] distance_mpc missing/invalid for {event}")

    # Analysis parameters: explicit arg → event params → default params → hardcoded
    if flow is None:
        flow = evp.get("flow") or defaults.get("flow") or 20.0
    if fhigh is None:
        fhigh = evp.get("fhigh") or defaults.get("fhigh") or 350.0
    if signal_win is None:
        signal_win = evp.get("signal_win") or defaults.get("signal_win") or 1.2
    if noise_pad is None:
        noise_pad = evp.get("noise_pad") or defaults.get("noise_pad") or 1200.0

    flow = float(flow)
    fhigh = float(fhigh)
    signal_win = float(signal_win)
    noise_pad = float(noise_pad)
    
    # Load tau_band and nu_band from event params if not provided
    if bands is None:
        tau_band_param = evp.get("tau_band") or defaults.get("tau_band") or [35.0, 250.0]
        nu_band_param = evp.get("nu_band") or defaults.get("nu_band") or [30.0, 350.0]
        bands = Bands(
            tau_band=tuple(tau_band_param) if isinstance(tau_band_param, list) else tau_band_param,
            nu_band=tuple(nu_band_param) if isinstance(nu_band_param, list) else nu_band_param
        )
    
    if verbose:
        print(f"\n⚙️  Paramètres adaptatifs pour {event}:")
        print(f"   flow={flow:.1f} Hz, fhigh={fhigh:.1f} Hz")
        print(f"   signal_win={signal_win:.2f} s, noise_pad={noise_pad:.1f} s")
        print(f"   tau_band={bands.tau_band}, nu_band={bands.nu_band}")
        print(f"   distance={distance_mpc:.0f} Mpc")

    # -----------------------------
    # 1) Load NPZ files
    # -----------------------------
    if verbose:
        print(f"\n📡 Chargement des données NPZ pour {event}...")

    try:
        pH = get_npz_path(event, "H1", npz_dir)
        pL = get_npz_path(event, "L1", npz_dir)
    except FileNotFoundError as e:
        raise RuntimeError(f"[FATAL] Fichiers NPZ manquants pour {event}: {e}")

    hH, fsH, t0H, t1H = load_npz(pH)
    hL, fsL, t0L, t1L = load_npz(pL)

    if fsH != fsL:
        raise RuntimeError("Sample rate mismatch H1/L1")

    fs = fsH
    used = ["H1", "L1"]

    # ---------- Virgo (V1) ----------
    hV = fsV = t0V = t1V = None

    if use_virgo:
        try:
            pV = get_npz_path(event, "V1", npz_dir)
            hV, fsV, t0V, t1V = load_npz(pV)

            if fsV != fs:
                raise RuntimeError("Sample rate mismatch V1")

            used.append("V1")
            if verbose:
                print(f"[INFO] Virgo (V1) chargé pour {event}")
        except (FileNotFoundError, Exception) as e:
            if verbose:
                print(f"[INFO] Virgo (V1) indisponible pour {event}: {e}")
            hV = None

    # -----------------------------
    # 2) Noise segment for PSD
    # -----------------------------
    n0 = gps - noise_pad - 400.0
    n1 = gps - noise_pad - 200.0
    # clamp to NPZ bounds
    if n0 < t0H:
        n0 = t0H
        n1 = min(n0 + 200.0, t1H)

    noiseH = crop_array(hH, fs, t0H, n0, n1)
    noiseL = crop_array(hL, fs, t0L, n0, n1)

    # PSD H1/L1 (obligatoires)
    f_psd_H, S1 = psd_welch_median(noiseH, fs, fmin=flow, fmax=fhigh)
    f_psd_L, S2 = psd_welch_median(noiseL, fs, fmin=flow, fmax=fhigh)

    # PSD V1 (optionnelle)
    if hV is not None:
        noiseV = crop_array(hV, fs, t0V, n0, n1)
        f_psd_V, S3 = psd_welch_median(noiseV, fs, fmin=flow, fmax=fhigh)
    else:
        f_psd_V = S3 = None

    f_psd_H, S1 = sanitize_psd(f_psd_H, S1, name="PSD_H1")
    f_psd_L, S2 = sanitize_psd(f_psd_L, S2, name="PSD_L1")

    # -----------------------------
    # 3) Tau (H1-L1) estimation
    # -----------------------------
    tb0, tb1 = bands.tau_band
    seg_tau_H = crop_array(hH, fs, t0H, gps - 0.2, gps + 0.2)
    seg_tau_L = crop_array(hL, fs, t0L, gps - 0.2, gps + 0.2)

    x_tau = safe_bandpass(seg_tau_H, fs, *bands.tau_band)
    y_tau = safe_bandpass(seg_tau_L, fs, *bands.tau_band)

    tau_hl = -abs(estimate_delay_time(x_tau, y_tau, fs))
    # Apply tau_scale to tau
    tau_hl_s = float(tau_scale) * tau_hl

    # -----------------------------
    # 4) Signal segment
    # -----------------------------
    s0 = gps - (signal_win / 2.0)
    s1 = gps + (signal_win / 2.0)
    hH_raw = crop_array(hH, fs, t0H, s0, s1)
    hL_raw = crop_array(hL, fs, t0L, s0, s1)

    if hV is not None:
        hV_raw = crop_array(hV, fs, t0V, s0, s1)
    else:
        hV_raw = None

    # Apply peak_scale to raw segments
    hH = hH_raw * float(peak_scale)
    hL = hL_raw * float(peak_scale)

    # Bandpass for analysis band
    hH_f = safe_bandpass(hH, fs, flow, fhigh)
    hL_f = safe_bandpass(hL, fs, flow, fhigh)

    peak_ref = max(robust_peak(hH_f, peak_quantile), robust_peak(hL_f, peak_quantile))
    if peak_norm and peak_ref > 0:
        hH_f = hH_f / peak_ref
        hL_f = hL_f / peak_ref

    N = min(hH_f.size, hL_f.size)
    if N < 16:
        raise RuntimeError(f"Segment signal trop court (N={N})")

    win = tukey(N, 0.2)
    H1 = np.fft.rfft(hH_f[:N] * win).astype(np.complex128)
    H2 = np.fft.rfft(hL_f[:N] * win).astype(np.complex128)
    f = np.fft.rfftfreq(N, d=1.0 / fs).astype(np.float64)

    mask = (f >= flow) & (f <= min(fhigh, 0.95 * (0.5 * fs)))
    if not np.any(mask):
        raise RuntimeError(f"[FATAL] Bande vide après masque: flow={flow} fhigh={fhigh} fs={fs} event={event}")

    f_use = f[mask]
    H1 = H1[mask]
    H2 = H2[mask]

    S1i = np.interp(f_use, f_psd_H, S1).astype(np.float64)
    S2i = np.interp(f_use, f_psd_L, S2).astype(np.float64)
    
    # PAS d'ajustement du PSD pour H_PEAK
    # Le PSD est calculé sur le bruit (sans H_PEAK)
    # Donc quand on blanchit (H_PEAK × signal) / sqrt(PSD_bruit),
    # l'effet de H_PEAK est préservé dans l'énergie finale

    S1i = np.nan_to_num(S1i, nan=1e-44, posinf=1e-44, neginf=1e-44)
    S2i = np.nan_to_num(S2i, nan=1e-44, posinf=1e-44, neginf=1e-44)
    S1i = np.maximum(S1i, 1e-60)
    S2i = np.maximum(S2i, 1e-60)

    phi = (2.0 * np.pi * f_use * tau_hl_s).astype(np.float64)
    r = float(distance_mpc) * Mpc

    # whitened coherent sum (H1 + H2 seulement, V1 ignoré pour simplifier)
    Hw1 = H1 / np.sqrt(S1i)
    Hw2 = H2 / np.sqrt(S2i)

    bad1 = ~np.isfinite(Hw1)
    bad2 = ~np.isfinite(Hw2)
    if np.any(bad1) and not np.any(bad2):
        Hw1 = np.zeros_like(Hw1)
    elif np.any(bad2) and not np.any(bad1):
        Hw2 = np.zeros_like(Hw2)

    Hc = Hw1 + Hw2 * np.exp(-1j * phi)

    # Note: L'ajout de Virgo (V1) nécessiterait un délai H-V et est omis pour simplifier
    # Si vous voulez l'ajouter, calculez tau_HV et ajoutez H3 de manière similaire

    # taper edges in frequency
    bw = max(1e-9, float(f_use[-1] - f_use[0]))
    edge = min(20.0, 0.10 * bw)
    edge = max(edge, 2.0)

    w = np.ones_like(f_use, dtype=np.float64)
    lo = f_use < (f_use[0] + edge)
    hi = f_use > (f_use[-1] - edge)
    w[lo] = 0.5 - 0.5 * np.cos(np.pi * (f_use[lo] - f_use[0]) / edge)
    w[hi] = 0.5 - 0.5 * np.cos(np.pi * (f_use[-1] - f_use[hi]) / edge)
    psi2 = w * w

    dEdf = (np.abs(Hc) ** 2) * psi2
    dEdf = np.nan_to_num(dEdf, nan=0.0, posinf=0.0, neginf=0.0)

    # -------------------------------------------------
    # Énergie spectrale
    # -------------------------------------------------
    # E_internal = énergie brute (sans SCALE_EJ, pour calibration)
    E_internal = float(trapezoid(dEdf, f_use))
    E_internal *= SCALE_EJ_NORM
    # energy_J = énergie calibrée (avec SCALE_EJ)
    energy_J = E_internal * scale_ej_in

    # -------------------------------------------------
    # Bandwidths & nu_eff
    # -------------------------------------------------
    if energy_J <= 0.0 or not np.any(dEdf > 0):
        bw_all = bw_nu = {
            "f10": np.nan, "f50": np.nan, "f90": np.nan,
            "dnu_80": np.nan, "mu_f": np.nan, "sigma_f": np.nan,
        }
    else:
        bw_all = spectral_bandwidth_features(f_use, dEdf)

    nb0, nb1 = bands.nu_band
    mask_nu = (f_use >= nb0) & (f_use <= nb1)

    bw_nu = (
        spectral_bandwidth_features(f_use[mask_nu], dEdf[mask_nu])
        if np.any(mask_nu)
        else {
            "f10": np.nan, "f50": np.nan, "f90": np.nan,
            "dnu_80": np.nan, "mu_f": np.nan, "sigma_f": np.nan,
        }
    )

    nu_eff = nu_eff_energy(f_use[mask_nu], dEdf[mask_nu]) if np.any(mask_nu) else 0.0
    nu_eff_raw = nu_eff_energy(f_use, dEdf)


    # -------------------------------------------------
    # Mass equivalent
    # -------------------------------------------------
    E_total = float(energy_J)
    m_sun_val = float(E_total / (M_sun * c * c))

    # -------------------------------------------------
    # LSQ / calibration mode (internals only)
    # -------------------------------------------------
    if return_internals:
        # Segments bruts pour peak/power (sans H_PEAK car déjà dans FFT)
        hL_cal = safe_bandpass(hL_raw, fs, tb0, tb1)
        peak_raw = robust_peak(hL_cal, peak_quantile)
        pwr_raw = float(np.mean(hL_cal * hL_cal))

        return {
            "event": event,
            "E_internal": float(E_internal),   # ✅ BRUT
            "energy_J": float(energy_J),       # ✅ CALIBRÉ
            "peak_raw": float(peak_raw),
            "pwr_raw": float(pwr_raw),
        }

    # -------------------------------------------------
    # FULL OUTPUT
    # -------------------------------------------------
    out = {
        "event": event,
        "gps": float(gps),
        "distance_mpc": float(distance_mpc),
        "used": used,
        "flow_Hz": float(flow),
        "fhigh_Hz": float(fhigh),
        "tau_band_Hz": [float(tb0), float(tb1)],
        "nu_band_Hz": [float(nb0), float(nb1)],
        "tau_hl_s": float(tau_hl_s),
        "TAU_SCALE": float(tau_scale),
        "SCALE_EJ": float(scale_ej_in),
        "PEAK_SCALE": float(peak_scale),
        "E_internal": float(E_internal),
        "energy_J": float(energy_J),
        "msun_c2": float(m_sun_val),
        "nu_eff": {
            "nu_eff_energy": float(nu_eff),
            "nu_eff_energy_raw": float(nu_eff_raw),
        },
        "bw_all": bw_all,
        "bw_nu": bw_nu,
        "freq_Hz": f_use.tolist(),
        "dEdf_internal": dEdf.tolist(),
    }

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", f"{event}.json"), "w") as f:
        json.dump(out, f, indent=2)

    if verbose:
        print(f"=== ANALYSE SPECTRALE {event} ===")
        print(f"E_total = {E_total:.3e} J ({m_sun_val:.6f} M_sun)")
        print(f"nu_eff  = {nu_eff:.2f} Hz")
        print(f"Tau (H1-L1): {tau_hl:.6e} s")

    if plot:
        import matplotlib.pyplot as plt
        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.loglog(f_use, np.maximum(dEdf, 1e-80), lw=1.4, label=f"{event}")

        f_log = np.geomspace(max(f_use[0], 1e-6), f_use[-1], 500)
        d_log = np.interp(f_log, f_use, np.maximum(dEdf, 1e-80))
        log_smooth = gaussian_filter1d(np.log10(d_log), sigma=2)
        plt.loglog(f_log, 10 ** log_smooth, "--", lw=1.2, alpha=0.8, label="smoothed")

        if nu_eff > 0:
            plt.axvline(x=nu_eff, linestyle=":", lw=1.0, label=f"nu_eff={nu_eff:.1f} Hz")

        plt.xlabel("Frequence (Hz)")
        plt.ylabel("dE/df (interne)")
        plt.title(f"Spectre coherent H1-L1 - {event}")
        plt.grid(True, which="both", alpha=0.25, linestyle="--")
        plt.legend()
        out_plot = os.path.join("plots", f"{event}_spectre.png")
        plt.savefig(out_plot, dpi=200, bbox_inches="tight")
        plt.close()
        print("[plot]", out_plot)

    return out


def calibrate_least_squares(
    events,
    params,
    y_obs_dict,
    bands,
    event_to_cluster=None,
    exclude_clusters=None,
    **analyze_kwargs
):
    """
    Calibre TAU_SCALE et SCALE_EJ par moindres carrés.
    
    Minimise: sum((y_obs[i] - TAU_SCALE * SCALE_EJ * y_model[i])^2)
    
    où y_model[i] est l'énergie interne pour l'event i avec TAU_SCALE=1, SCALE_EJ=1
    et y_obs[i] est l'énergie observée (en M_sun) depuis une référence externe.
    """
    
    print("\n🔧 Calibration par moindres carrés...")
    print(f"Événements: {len(events)}")
    
    # Étape 1: Collecter les y_model (E_internal avec TAU_SCALE=1, SCALE_EJ=1)
    data = []

    for ev in events:

        # --- exclusion par cluster ---
        if exclude_clusters and event_to_cluster:
            cls = event_to_cluster.get(ev)
            if cls in exclude_clusters:
                continue

        # --- valeur observée ---
        yobs = y_obs_dict.get(ev)
        if yobs is None:
            continue

        yobs = float(yobs)
        if not np.isfinite(yobs) or yobs <= 0.0:
            continue

        try:
            intern = analyze_event(
                event=ev,
                params=params,
                bands=bands,
                tau_scale=1.0,
                scale_ej_in=1.0,
                peak_scale=1.0,
                return_internals=True,
                plot=False,
                verbose=False,
                **analyze_kwargs,
            )
        except Exception as e:
            print(f"[WARN] skip {ev}: {e}")
            continue

        y_model = float(intern.get("E_internal", 0.0))
        if not np.isfinite(y_model) or y_model <= 0.0:
            continue

        peak_raw = float(intern.get("peak_raw", 0.0))
        data.append((ev, yobs, y_model, peak_raw))

    if len(data) < 2:
        raise RuntimeError(f"[FATAL] Pas assez d'events valides pour LSQ (n={len(data)})")

    # Vecteurs
    evs = [t[0] for t in data]
    y_obs = np.array([t[1] for t in data], dtype=float)   # "vérité" (ex: M_sun ou J selon ton dict)
    y_mod = np.array([t[2] for t in data], dtype=float)   # E_internal (TAU_SCALE=1, SCALE_EJ=1)
    p_raw = np.array([t[3] for t in data], dtype=float)   # peak_raw (utile si tu veux lever la dégénérescence)

    # --------------------------------------------
    # IMPORTANT (math):
    # Si ton modèle est: y_pred = (TAU_SCALE * SCALE_EJ) * y_model
    # alors seul le produit K = TAU_SCALE*SCALE_EJ est identifiable par LSQ sur y_obs.
    #
    # -> Option A (par défaut): on fit K en closed-form, puis on fixe TAU_SCALE=1 et SCALE_EJ=K.
    # -> Option B (si tu passes peak_target dans analyze_kwargs): on ajoute une contrainte d'amplitude
    #    sur peak_raw pour estimer TAU_SCALE et SCALE_EJ séparément.
    # --------------------------------------------

    peak_target = analyze_kwargs.get("peak_target", None)  # optionnel

    if peak_target is None:
        # === Option A : fit K seulement (closed-form) ===
        num = float(np.dot(y_obs, y_mod))
        den = float(np.dot(y_mod, y_mod))
        if den <= 0.0 or (not np.isfinite(num)) or (not np.isfinite(den)):
            raise RuntimeError("[FATAL] LSQ degenerate (dot products invalid)")

        K = num / den
        tau_scale = 1.0
        scale_ej = float(K)

        # petits diagnostics
        y_pred = scale_ej * y_mod
        rel = (y_pred - y_obs) / np.maximum(np.abs(y_obs), 1e-30)
        mae = float(np.mean(np.abs(rel)))
        med = float(np.median(np.abs(rel)))

        print("\n=== LSQ (degenerate) : fit K = TAU_SCALE*SCALE_EJ ===")
        print(f"events_used = {len(evs)}")
        print(f"K_hat       = {K:.6e}")
        print(f"-> choose TAU_SCALE=1 ; SCALE_EJ=K")
        print(f"rel_MAE     = {100*mae:.2f}%")
        print(f"rel_MED     = {100*med:.2f}%")

        return float(tau_scale), float(scale_ej)

    # === Option B : fit (TAU_SCALE, SCALE_EJ) avec contrainte peak ===
    # On force: TAU_SCALE * peak_raw ≈ peak_target  (linéaire en TAU_SCALE)
    # et:      (TAU_SCALE * SCALE_EJ) * y_mod ≈ y_obs
    #
    # Loss = ||y_obs - (T*S)*y_mod||^2 / ||y_obs||^2  +  λ * ||peak_target - T*peak_raw||^2 / ||peak_target||^2
    # λ règle le poids de la contrainte amplitude.
    lam = float(analyze_kwargs.get("peak_lambda", 1.0))

    # normalisations
    y_norm = float(np.dot(y_obs, y_obs)) + 1e-30
    p_norm = float((peak_target ** 2) * len(p_raw)) + 1e-30

    def loss_logparams(x: np.ndarray) -> float:
        # x = [logT, logS] pour garantir positivité
        logT, logS = float(x[0]), float(x[1])
        T = math.exp(logT)
        S = math.exp(logS)

        y_pred = (T * S) * y_mod
        e_y = y_obs - y_pred
        L_y = float(np.dot(e_y, e_y)) / y_norm

        # contrainte amplitude (ignore p_raw<=0)
        m = (p_raw > 0) & np.isfinite(p_raw)
        if np.any(m):
            p_pred = T * p_raw[m]
            e_p = (peak_target - p_pred)
            L_p = float(np.dot(e_p, e_p)) / p_norm
        else:
            L_p = 0.0

        return L_y + lam * L_p

    # init raisonnable: T depuis peak, S depuis K
    # T0 = peak_target / median(peak_raw) ; S0 = K/T0
    p_med = float(np.median(p_raw[p_raw > 0])) if np.any(p_raw > 0) else 1.0
    T0 = float(peak_target) / max(p_med, 1e-30)

    num = float(np.dot(y_obs, y_mod))
    den = float(np.dot(y_mod, y_mod)) + 1e-30
    K0 = num / den

    S0 = float(K0) / max(T0, 1e-30)

    x0 = np.array([math.log(max(T0, 1e-30)), math.log(max(S0, 1e-30))], dtype=float)

    res = minimize(loss_logparams, x0=x0, method="Nelder-Mead", options={"maxiter": 5000})

    logT, logS = float(res.x[0]), float(res.x[1])
    tau_scale = math.exp(logT)
    scale_ej = math.exp(logS)

    # diagnostics
    y_pred = (tau_scale * scale_ej) * y_mod
    rel = (y_pred - y_obs) / np.maximum(np.abs(y_obs), 1e-30)
    mae = float(np.mean(np.abs(rel)))
    med = float(np.median(np.abs(rel)))

    print("\n=== LSQ (non-degenerate) : fit TAU_SCALE & SCALE_EJ (avec peak_target) ===")
    print(f"events_used = {len(evs)}")
    print(f"peak_target = {peak_target:.6e}  lambda={lam}")
    print(f"TAU_SCALE   = {tau_scale:.6e}")
    print(f"SCALE_EJ    = {scale_ej:.6e}")
    print(f"K=T*S       = {(tau_scale*scale_ej):.6e}")
    print(f"rel_MAE     = {100*mae:.2f}%")
    print(f"rel_MED     = {100*med:.2f}%")
    print(f"opt_status  = {res.success}  it={res.nit}  f={res.fun:.3e}")

    return float(tau_scale), float(scale_ej)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyse spectrale coherente GWOSC (H1-L1, V1 optionnel)")

    ap.add_argument("--event", required=False, help="Nom d'evenement (ex: GW150914)")
    ap.add_argument("--event-params", default="event_params.json", help="JSON params events")
    ap.add_argument("--list-events", action="store_true", help="Liste les events du JSON")

    ap.add_argument("--no-virgo", action="store_true", help="N'essaie pas V1")

    ap.add_argument("--flow", type=float, default=None)
    ap.add_argument("--fhigh", type=float, default=None)
    ap.add_argument("--signal-win", type=float, default=None)
    ap.add_argument("--noise-pad", type=float, default=None)
    ap.add_argument("--distance-mpc", type=float, default=None)

    ap.add_argument("--tau-band", type=float, nargs=2, default=[35.0, 250.0], metavar=("F1", "F2"))
    ap.add_argument("--nu-band", type=float, nargs=2, default=[30.0, 350.0], metavar=("F1", "F2"))

    ap.add_argument("--tau-scale", type=float, default=DEFAULT_TAU_SCALE, 
                    help=f"Facteur multiplicatif pour tau (défaut: {DEFAULT_TAU_SCALE})")
    ap.add_argument("--scale-ej", type=float, default=DEFAULT_SCALE_EJ,
                    help=f"Facteur d'échelle d'énergie (défaut: {DEFAULT_SCALE_EJ})")
    ap.add_argument("--peak-scale", type=float, default=DEFAULT_PEAK_SCALE,
                    help=f"Facteur multiplicatif pour amplitude des signaux (défaut: {DEFAULT_PEAK_SCALE})")
    ap.add_argument("--peak-norm", action="store_true", help="Normalise par peak_ref (debug)")
    ap.add_argument("--peak-quantile", type=float, default=99.5)
    ap.add_argument("--refs", type=str, default=None, help="JSON refs LIGO (ligo_refs.json)")
    ap.add_argument("--ref-key", default="energy_J", choices=["energy_J", "msun_c2"])
    ap.add_argument("--exclude-cls", nargs="*", default=[])
    ap.add_argument("--calibrate-lsq", action="store_true")
    ap.add_argument("--cal-out", type=str, default="calibrated.json", help="Fichier sortie calibration LSQ")
    ap.add_argument("--use-calibrated", type=str, default=None, help="Charge H_STAR/SCALE_EJ depuis un JSON")
    ap.add_argument("--peak-target", type=float, default=None, help="Cible peak pour lever degenerescence (optionnel)")
    ap.add_argument("--peak-lambda", type=float, default=1.0, help="Poids contrainte peak (optionnel)")

    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true", help="Afficher les messages de debug")
    ap.add_argument(
        "--clusters-json",
        type=str,
        default=None,
        help="JSON mapping event -> cluster (ex: clusters.json)"
    )

    ap.add_argument(
        "--exclude-clusters",
        type=int,
        nargs="*",
        default=[],
        help="Liste des IDs de clusters à exclure (ex: -1 3)"
    )

    args = ap.parse_args()
    event_to_cluster = {}
    # --- exclusion par cluster (globale) ---
    if args.clusters_json:
        with open(args.clusters_json, "r") as f:
            cj = json.load(f)
        event_to_cluster = cj.get("event_to_cluster", {})

    params = load_event_params(args.event_params)
    def _load_json(path: str) -> dict:
        with open(path, "r") as f:
            return json.load(f)

    # 1) use-calibrated
    if args.use_calibrated:
        cal = _load_json(args.use_calibrated)
        args.tau_scale = float(cal.get("TAU_SCALE", cal.get("H_STAR", DEFAULT_TAU_SCALE)))
        args.scale_ej = float(cal.get("SCALE_EJ", DEFAULT_SCALE_EJ))
        args.peak_scale = float(cal.get("PEAK_SCALE", cal.get("H_PEAK", DEFAULT_PEAK_SCALE)))

    # 2) calibrate-lsq
    refs = {}
    if args.refs:
        with open(args.refs, "r") as f:
            refs = json.load(f)

    # clé de vérité : énergie (recommandé) ou msun_c2
    ref_key = getattr(args, "ref_key", "energy_J")

    exclude = set(getattr(args, "exclude_cls", []) or [])

    y_obs_dict = {}
    for ev, d in refs.items():
        if not isinstance(d, dict):
            continue
        if exclude and d.get("cls", "") in exclude:
            continue
        v = d.get(ref_key, None)
        if v is None:
            continue
        try:
            v = float(v)
        except Exception:
            continue
        if not np.isfinite(v) or v <= 0:
            continue
        y_obs_dict[ev] = v

    # si pas d'events donnés -> prendre ceux des refs
    # si pas d'events donnés -> prendre ceux des refs
    events = list(getattr(args, "events", []) or [])
    if not events:
        events = sorted(y_obs_dict.keys())

    # --- exclusion par cluster (globale, AVANT LSQ) ---
    if args.exclude_clusters:
        events = [
            ev for ev in events
            if event_to_cluster.get(ev) not in args.exclude_clusters
        ]

    print(
        f"[cal] refs_loaded={len(refs)} "
        f"y_obs={len(y_obs_dict)} "
        f"events_after_cluster_exclusion={len(events)} "
        f"ref_key={ref_key}"
    )
    bands = Bands(
        tau_band=(float(args.tau_band[0]), float(args.tau_band[1])),
        nu_band=(float(args.nu_band[0]), float(args.nu_band[1])),
    )

    do_cal = bool(args.calibrate_lsq)
    do_use_cal = bool(args.use_calibrated)

    if do_cal:
        # calibration ONLY
        if not args.refs:
            raise RuntimeError("[FATAL] --calibrate-lsq nécessite --refs")
        tau_scale, scale = calibrate_least_squares(
            events=events,
            params=params,
            y_obs_dict=y_obs_dict,
            bands=bands,
            flow=args.flow,
            fhigh=args.fhigh,
            signal_win=args.signal_win,
            noise_pad=args.noise_pad,
            distance_mpc=None,
            use_virgo=not args.no_virgo,
            peak_quantile=args.peak_quantile,
        )
    elif do_use_cal:
        tau_scale, scale = args.tau_scale, args.scale_ej
    else:
        # NO calibration: analyze events with defaults or provided constants
        tau_scale, scale = args.tau_scale, args.scale_ej
    
    if do_cal or do_use_cal:
        print(f"[cal] TAU_SCALE={tau_scale:.6e} SCALE_EJ={scale:.6e}")
        with open(args.cal_out, "w") as f:
            json.dump({
                "TAU_SCALE": float(tau_scale), 
                "SCALE_EJ": float(scale), 
                "PEAK_SCALE": float(args.peak_scale)
            }, f, indent=2)
        print(f"[cal] wrote {args.cal_out}")
        return

    if args.list_events:
        for k in sorted(params.keys()):
            print(k)
        return

    if not args.event:
        raise SystemExit("Erreur: fournir --event ou --list-events")

    ev = args.event
    if ev not in params:
        raise SystemExit(f"Erreur: event '{ev}' absent dans {args.event_params}")

    evp = params.get(ev, {}) if isinstance(params.get(ev), dict) else {}

    flow = float(args.flow) if args.flow is not None else float(evp.get("flow", 20.0))
    fhigh = float(args.fhigh) if args.fhigh is not None else float(evp.get("fhigh", 512.0))
    signal_win = float(args.signal_win) if args.signal_win is not None else float(evp.get("signal_win", 0.2))
    noise_pad = float(args.noise_pad) if args.noise_pad is not None else float(evp.get("noise_pad", 50.0))

    distance_mpc = float(args.distance_mpc) if args.distance_mpc is not None else float(evp.get("distance_mpc", 0.0))
    if distance_mpc <= 0:
        raise SystemExit("Erreur: distance_mpc doit etre >0 (JSON ou --distance-mpc)")

    bands = Bands(
        tau_band=(float(args.tau_band[0]), float(args.tau_band[1])),
        nu_band=(float(args.nu_band[0]), float(args.nu_band[1])),
    )
    # --- garde-fou : interdiction d'analyser un event exclu ---
    if args.event and args.exclude_clusters:
        cls = event_to_cluster.get(args.event, None)
        if cls in args.exclude_clusters:
            print(f"[SKIP] event '{args.event}' appartient au cluster exclu {cls}")
            return

    analyze_event(
        event=ev,
        params=params,
        flow=flow,
        fhigh=fhigh,
        signal_win=signal_win,
        noise_pad=noise_pad,
        distance_mpc=distance_mpc,
        bands=bands,
        use_virgo=not args.no_virgo,
        peak_norm=bool(args.peak_norm),
        peak_quantile=float(args.peak_quantile),
        tau_scale=float(args.tau_scale),
        scale_ej_in=float(args.scale_ej),
        peak_scale=float(args.peak_scale),
        plot=bool(args.plot),
        return_internals=False,
        verbose=bool(args.verbose),
    )


if __name__ == "__main__":
    main()
