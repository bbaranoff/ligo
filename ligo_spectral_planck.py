#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Analyse spectrale coherente GWOSC (H1-L1, V1 optionnel)

Sorties
- dEdf_internal: spectre interne (avant SCALE_EJ)
- E_total_J = SCALE_EJ * integral(dE/df)
- m_sun = E_total_J / (M_sun*c^2)
- tau_hl_s: delay H1-L1 estime par correlation sur une bande dediee (tau-band)
- nu_eff_energy: barycentre energetique sur une bande dediee (nu-band), avec garde-fous

Calibration double (non degeneree)
- H_STAR via amplitude: calibrer sur un event A en imposant un peak_ref (quantile)
- SCALE_EJ via energie: calibrer sur un event B en imposant une energie cible (en M_sun c^2)

Important: H_STAR et SCALE_EJ ne doivent pas etre calibres sur le meme event (sinon degenerescence).

Dependances: numpy, scipy, gwpy, gwosc, numba (optionnel)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import tukey
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import trapezoid

from gwosc import datasets
from gwpy.timeseries import TimeSeries

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

DEFAULT_HSTAR = 1.0
DEFAULT_SCALE_EJ = 1.0

# ------------------
# IO params
# ------------------

def load_event_params(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


# Module-level default params for compatibility with helper scripts (e.g. run_all.sh)
def _load_default_event_params() -> Dict[str, Any]:
    """Best-effort load of event_params.json next to this script.

    Some helper scripts import this module and expect a preloaded
    EVENT_PARAMS dictionary. The main() still supports --event-params;
    this is only a convenience fallback.
    """
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
    evp = params.get(event, {})
    if isinstance(evp, dict):
        if "gps" in evp:
            return float(evp["gps"])
        if "GPS" in evp:
            return float(evp["GPS"])
    # fallback GWOSC
    return float(datasets.event_gps(event))


def fetch(det: str, t0: float, t1: float, cache: bool = True) -> TimeSeries:
    # gwpy: open data
    return TimeSeries.fetch_open_data(det, t0, t1, cache=cache)


# ------------------
# Utils
# ------------------

def robust_peak(x: np.ndarray, q: float = 99.5) -> float:
    x = np.asarray(x, float)
    if x.size == 0:
        return 0.0
    return float(np.percentile(np.abs(x), q))


def safe_bandpass(x: np.ndarray, fs: float, f1: float, f2: float, order: int = 4) -> np.ndarray:
    x = np.asarray(x, float)
    nyq = 0.5 * fs
    f2_safe = min(float(f2), 0.95 * nyq)
    f1_safe = max(float(f1), 0.1)
    if f2_safe <= f1_safe:
        f2_safe = min(0.95 * nyq, f1_safe + 10.0)
    sos = butter(order, [f1_safe, f2_safe], btype="bandpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, x)
    # fenetre douce
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

    # GWOSC peut contenir NaN/Inf : si on laisse passer, PSD -> NaN -> E=0.
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

        if not np.isfinite(seg).all():
            seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)

        seg = seg - float(np.mean(seg))

        seg = safe_bandpass(seg, fs, fmin, fmax)
        if not np.isfinite(seg).all():
            # filtre qui diverge / segment foireux -> on jette
            continue

        X = np.fft.rfft(seg * win)
        Pxx = (2.0 / (fs * U)) * (np.abs(X) ** 2)

        if np.isfinite(Pxx).all():
            specs.append(Pxx)

    f = np.fft.rfftfreq(nseg, d=1.0 / fs)

    if not specs:
        return f, np.ones_like(f) * 1e-44  # fallback bruit blanc

    S = np.median(np.stack(specs, axis=0), axis=0)

    # sécurité numérique
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

    # sécurité : pas de négatif
    s = np.maximum(s, 0.0)

    # intégrale totale
    total = float(trapezoid(s, f))
    if not np.isfinite(total) or total <= 0:
        return {
            "f10": np.nan, "f50": np.nan, "f90": np.nan,
            "dnu_80": np.nan, "mu_f": np.nan, "sigma_f": np.nan
        }

    # cumul normalisé
    # (trapz cumulatif via intégration par trapèzes)
    df = np.diff(f)
    mid = 0.5 * (s[:-1] + s[1:])
    cum = np.concatenate(([0.0], np.cumsum(mid * df)))
    cdf = cum / cum[-1]

    # percentiles
    f10 = np.interp(0.10, cdf, f)
    f50 = np.interp(0.50, cdf, f)
    f90 = np.interp(0.90, cdf, f)
    dnu_80 = f90 - f10

    # moments
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
    """Cross-correlation FFT, renvoie le lag (s) max abs corr. """
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

    # rearrange lags to [-N..N]
    r = np.concatenate([r[-(N-1):], r[:N]])
    lags = np.arange(-(N-1), N) / fs

    lim = float(max_delay_ms) / 1000.0
    mask = (lags >= -lim) & (lags <= lim)
    if not np.any(mask):
        return 0.0

    idx = np.argmax(np.abs(r[mask]))
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
        # e^{i phi}
        re = c
        im = s

        # num = H1*w1 + e^{i phi} H2*w2
        a = H1[i] * w1
        b = H2[i] * w2
        # e^{i phi} * b
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
    flow: float,
    fhigh: float,
    signal_win: float,
    noise_pad: float,
    distance_mpc: float,
    bands: Bands,
    use_virgo: bool,
    peak_norm: bool,
    peak_quantile: float,
    hstar_in: float,
    scale_ej_in: float,
    calibrate_hstar_event: Optional[str] = None,
    hpeak_target: Optional[float] = None,
    calibrate_scale_event: Optional[str] = None,
    msun_target: Optional[float] = None,
    plot: bool = False,
) -> Dict[str, Any]:

    gps = get_event_gps(event, params)

    # fetch window: bruit + marge
    t0 = gps - max(1200.0, noise_pad + 450.0)
    t1 = gps + max(200.0, noise_pad + 50.0)

    print(f"\n\U0001F4E1 Telechargement des donnees H1/L1{'/V1' if use_virgo else ''} pour {event}...")
    tsH = fetch("H1", t0, t1)
    tsL = fetch("L1", t0, t1)

    tsV = None
    used = ["H1", "L1"]
    if use_virgo:
        try:
            tsV = fetch("V1", t0, t1)
            used.append("V1")
        except Exception as e:
            print(f"[INFO] Virgo (V1) indisponible pour {event}: {e}")
            tsV = None

    fs = float(tsH.sample_rate.value)

    # ---- 1) PSD sur une fenetre de bruit "sure" ----
    n0 = gps - noise_pad - 400.0
    n1 = gps - noise_pad - 200.0
    if n0 < float(tsH.t0.value):
        n0 = float(tsH.t0.value)
        n1 = n0 + 200.0

    # ---- 1) PSD sur une fenetre de bruit "sure" ----
    noiseH = np.asarray(tsH.crop(n0, n1).value, float)
    noiseL = np.asarray(tsL.crop(n0, n1).value, float)

    # GWOSC peut contenir NaN/Inf -> on neutralise avant PSD
    noiseH = np.nan_to_num(noiseH, nan=0.0, posinf=0.0, neginf=0.0)
    noiseL = np.nan_to_num(noiseL, nan=0.0, posinf=0.0, neginf=0.0)

    # PSD par detecteur (Welch median robuste)
    f_psd_H, S1 = psd_welch_median(noiseH, fs, fmin=flow, fmax=fhigh)
    f_psd_L, S2 = psd_welch_median(noiseL, fs, fmin=flow, fmax=fhigh)

    # Sanitize: PSD finie et strictement > 0 pour whitening
    f_psd_H, S1 = sanitize_psd(f_psd_H, S1, name="PSD_H1")
    f_psd_L, S2 = sanitize_psd(f_psd_L, S2, name="PSD_L1")

    # ---- 2) tau (sur tau-band) ----
    tb0, tb1 = bands.tau_band
    seg_tau_H = np.asarray(tsH.crop(gps - 0.2, gps + 0.2).value, float)
    seg_tau_L = np.asarray(tsL.crop(gps - 0.2, gps + 0.2).value, float)

    x_tau = safe_bandpass(seg_tau_H, fs, tb0, tb1)
    y_tau = safe_bandpass(seg_tau_L, fs, tb0, tb1)
    tau_hl = estimate_delay_time(x_tau, y_tau, fs)

    # convention: signe negatif comme dans tes runs (L arrive apres H => tau negatif)
    # Ici: estimate_delay_time renvoie lag tel que x(t) ~ y(t+lag). On garde ton choix:
    tau_hl = -abs(float(tau_hl))

    # ---- 3) segment signal ----
    s0 = gps - (signal_win / 2.0)
    s1 = gps + (signal_win / 2.0)
    hH_raw = np.asarray(tsH.crop(s0, s1).value, float)
    hL_raw = np.asarray(tsL.crop(s0, s1).value, float)

    # ---- 4) calibration H_STAR event A ----
    hstar = float(hstar_in)
    if calibrate_hstar_event and hpeak_target is not None and event == calibrate_hstar_event:
        # peak ref sur L1 (plus stable); bande tau-band
        hL_cal = safe_bandpass(hL_raw, fs, tb0, tb1)
        meas = robust_peak(hL_cal, peak_quantile)

        if meas > 0 and np.isfinite(meas):
            hstar = float(hpeak_target) / float(meas)
        else:
            hstar = float(hstar_in)

        # >>> LIGNE CONTRACTUELLE POUR run_all.sh <<<
        print(f"H_STAR = {hstar}")

        print(
            f"[CAL] H_STAR sur {event}: "
            f"peak_meas={meas:.3e} -> target={hpeak_target:.3e} => H_STAR={hstar:.6g}"
        )


    # applique H_STAR (gain) avant tout
    print("[DEBUG]:", hstar, flush=True)  # ← Force l'écriture immédiate
    hH = hH_raw * hstar
    hL = hL_raw * hstar

    # ---- 5) bande energie (flow,fhigh) ----
    hH_f = safe_bandpass(hH, fs, flow, fhigh)
    hL_f = safe_bandpass(hL, fs, flow, fhigh)

    peak_ref = max(robust_peak(hH_f, peak_quantile), robust_peak(hL_f, peak_quantile))

    # normalisation par pic OPTIONNELLE (sinon conserve amplitude physique)
    if peak_norm and peak_ref > 0:
        hH_f = hH_f / peak_ref
        hL_f = hL_f / peak_ref

    # ---- 6) FFT courte ----
    N = min(hH_f.size, hL_f.size)
    if N < 16:
        raise RuntimeError(f"Segment signal trop court (N={N})")

    win = tukey(N, 0.2)
    H1 = np.fft.rfft(hH_f[:N] * win).astype(np.complex128)
    H2 = np.fft.rfft(hL_f[:N] * win).astype(np.complex128)
    f = np.fft.rfftfreq(N, d=1.0 / fs).astype(np.float64)

    mask = (f >= flow) & (f <= min(fhigh, 0.95 * (0.5 * fs)))
    if not np.any(mask):
        raise RuntimeError(
            f"[FATAL] Bande vide apres masque: flow={flow} fhigh={fhigh} fs={fs} event={event}"
        )
    f_use = f[mask]
    H1 = H1[mask]
    H2 = H2[mask]

    S1i = np.interp(f_use, f_psd_H, S1).astype(np.float64)
    S2i = np.interp(f_use, f_psd_L, S2).astype(np.float64)

    # garde-fou final
    S1i = np.nan_to_num(S1i, nan=1e-44, posinf=1e-44, neginf=1e-44)
    S2i = np.nan_to_num(S2i, nan=1e-44, posinf=1e-44, neginf=1e-44)
    S1i = np.maximum(S1i, 1e-60)
    S2i = np.maximum(S2i, 1e-60)

    phi = (2.0 * np.pi * f_use * tau_hl).astype(np.float64)
    r = float(distance_mpc) * Mpc

    # ---- 6b) Energie L2 pondérée (whitened, cohérente) ----
    eps = 1e-60
    S1i = np.maximum(S1i, eps)
    S2i = np.maximum(S2i, eps)

    Hw1 = H1 / np.sqrt(S1i)
    Hw2 = H2 / np.sqrt(S2i)

    bad1 = ~np.isfinite(Hw1)
    bad2 = ~np.isfinite(Hw2)

    if np.any(bad1) and not np.any(bad2):
        Hw1 = np.zeros_like(Hw1)     # L1-only
    elif np.any(bad2) and not np.any(bad1):
        Hw2 = np.zeros_like(Hw2)     # H1-only

    # somme cohérente (phasage via tau_hl)
    Hc = Hw1 + Hw2 * np.exp(-1j * phi)

    # ψ(f) : taper doux pour éviter les bords (10% de la bande)
    bw = max(1e-9, float(f_use[-1] - f_use[0]))
    edge = min(20.0, 0.10 * bw)   # max 20 Hz de taper
    edge = max(edge, 2.0)         # min 2 Hz pour éviter edge ~ 0

    w = np.ones_like(f_use, dtype=np.float64)

    lo = f_use < (f_use[0] + edge)
    hi = f_use > (f_use[-1] - edge)

    w[lo] = 0.5 - 0.5 * np.cos(np.pi * (f_use[lo] - f_use[0]) / edge)
    w[hi] = 0.5 - 0.5 * np.cos(np.pi * (f_use[-1] - f_use[hi]) / edge)

    psi2 = w * w

    dEdf = (np.abs(Hc) ** 2) * psi2
    dEdf = np.nan_to_num(dEdf, nan=0.0, posinf=0.0, neginf=0.0)
    E_internal = float(trapezoid(dEdf, f_use))
    if not np.isfinite(E_internal) or E_internal <= 0.0:
        # debug minimal mais ultra utile
        print(
            f"[WARN] E_internal<=0 pour {event}: "
            f"minS1={S1i.min():.3e} minS2={S2i.min():.3e} "
            f"sum|H1|={np.sum(np.abs(H1)):.3e} sum|H2|={np.sum(np.abs(H2)):.3e} "
            f"sum dEdf={np.sum(dEdf):.3e}",
            flush=True
        )


        # ---- 6b) largeur spectrale (feature manquante) ----
    if E_internal <= 0.0 or not np.any(dEdf > 0):
        bw_all = bw_nu = {"dnu_80": 0.0, "sigma_f": 0.0, "f10": 0.0, "f90": 0.0}
    else:
        bw_all = spectral_bandwidth_features(f_use, dEdf)
    # nu-band mask (défini UNE fois, réutilisé ensuite)
    nb0, nb1 = bands.nu_band
    mask_nu = (f_use >= nb0) & (f_use <= nb1)

    bw_nu = spectral_bandwidth_features(f_use[mask_nu], dEdf[mask_nu]) if np.any(mask_nu) else {
        "f10": np.nan, "f50": np.nan, "f90": np.nan,
        "dnu_80": np.nan, "mu_f": np.nan, "sigma_f": np.nan
    }

    # ---- 7) calibration SCALE_EJ event B ----
    scale_ej = float(scale_ej_in)

    if calibrate_scale_event and msun_target is not None and event == calibrate_scale_event:

        # énergie cible physique
        E_target = float(msun_target) * (M_sun * c * c)

        if E_internal > 0.0 and np.isfinite(E_internal):
            scale_ej = E_target / E_internal
        else:
            raise RuntimeError(
                f"[FATAL] SCALE_EJ: E_internal invalide ({E_internal}) pour {event}"
            )

        # >>> LIGNE CONTRACTUELLE POUR run_all.sh <<<
        print(f"SCALE_EJ = {scale_ej}")

        print(
            f"[CAL] SCALE_EJ sur {event}: "
            f"E_internal={E_internal:.3e} -> "
            f"target={E_target:.3e} => "
            f"SCALE_EJ={scale_ej:.6g}"
        )

    E_total = float(E_internal * scale_ej)
    m_sun_val = float(E_total / (M_sun * c * c))

    # ---- 8) nu_eff sur nu-band ----
    nu_eff = nu_eff_energy(f_use[mask_nu], dEdf[mask_nu]) if np.any(mask_nu) else 0.0
    nu_eff_raw = nu_eff_energy(f_use, dEdf)

    out = {
        "bw_all": bw_all,
        "bw_nu": bw_nu,
        "event": event,
        "gps": float(gps),
        "distance_mpc": float(distance_mpc),
        "used": used,
        "flow_Hz": float(flow),
        "fhigh_Hz": float(fhigh),
        "tau_band_Hz": [float(tb0), float(tb1)],
        "nu_band_Hz": [float(nb0), float(nb1)],
        "tau_hl_s": float(tau_hl),
        "peak_quantile": float(peak_quantile),
        "peak_norm": bool(peak_norm),
        "peak_ref": float(peak_ref),
        "H_STAR": float(hstar),
        "SCALE_EJ": float(scale_ej),
        "E_internal": float(E_internal),
        "E_total_J": float(E_total),
        "m_sun": float(m_sun_val),
        "nu_eff": {
            "nu_eff_energy": float(nu_eff),
            "nu_eff_energy_raw": float(nu_eff_raw),
        },
        "freq_Hz": f_use.tolist(),
        "dEdf_internal": dEdf.tolist(),
    }

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", f"{event}.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"=== ANALYSE SPECTRALE {event} ===")
    print(f"GPS: {gps}")
    print(f"Distance: {distance_mpc} Mpc")
    print(f"Tau (H1-L1): {tau_hl:.6e} s")
    print(f"Energie intrins. (J): {E_total:.3e}  ({m_sun_val:.6f} M_sun)")
    print(f"nu_eff (energy): {nu_eff:.2f} Hz (raw: {nu_eff_raw:.2f} Hz)")
    print(f"[DEBUG] flow={flow} fhigh={fhigh}")
    print(f"[BW] all: dnu_80={bw_all['dnu_80']:.2f} Hz  sigma_f={bw_all['sigma_f']:.2f} Hz  "
          f"f10={bw_all['f10']:.1f} f90={bw_all['f90']:.1f}")
    print(f"[BW] nu : dnu_80={bw_nu['dnu_80']:.2f} Hz  sigma_f={bw_nu['sigma_f']:.2f} Hz  "
          f"f10={bw_nu['f10']:.1f} f90={bw_nu['f90']:.1f}")
    if plot:
        import matplotlib.pyplot as plt
        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.loglog(f_use, np.maximum(dEdf, 1e-80), lw=1.4, label=f"{event}")

        # lissage log-log pour visuel
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

    ap.add_argument("--hstar", type=float, default=DEFAULT_HSTAR)
    ap.add_argument("--scale-ej", type=float, default=DEFAULT_SCALE_EJ)

    ap.add_argument("--peak-norm", action="store_true", help="Normalise par peak_ref (debug)")
    ap.add_argument("--peak-quantile", type=float, default=99.5)

    ap.add_argument("--calibrate-hstar", type=str, default=None, help="Event de calibration H_STAR")
    ap.add_argument("--hpeak-target", type=float, default=None, help="Cible peak_ref (strain) pour H_STAR")

    ap.add_argument("--calibrate-scale", type=str, default=None, help="Event de calibration SCALE_EJ")
    ap.add_argument("--msun-target", type=float, default=None, help="Cible energie en M_sun c^2")

    ap.add_argument("--plot", action="store_true")

    args = ap.parse_args()

    params = load_event_params(args.event_params)

    if args.list_events:
        for k in sorted(params.keys()):
            print(k)
        return

    if not args.event:
        raise SystemExit("Erreur: fournir --event ou --list-events")

    ev = args.event
    if ev not in params:
        raise SystemExit(f"Erreur: event '{ev}' absent dans {args.event_params}")

    # securite degenerate
    if args.calibrate_hstar and args.calibrate_scale and args.calibrate_hstar == args.calibrate_scale:
        raise SystemExit("[FATAL] H_STAR et SCALE_EJ ne peuvent pas etre calibres sur le meme event")

    evp = params.get(ev, {}) if isinstance(params.get(ev), dict) else {}

    flow = float(args.flow) if args.flow is not None else float(evp.get("flow", 20.0))
    fhigh = float(args.fhigh) if args.fhigh is not None else float(evp.get("fhigh", 512.0))
    signal_win = float(args.signal_win) if args.signal_win is not None else float(evp.get("signal_win", 0.2))
    noise_pad = float(args.noise_pad) if args.noise_pad is not None else float(evp.get("noise_pad", 50.0))

    distance_mpc = float(args.distance_mpc) if args.distance_mpc is not None else float(evp.get("distance_mpc", 0.0))
    if distance_mpc <= 0:
        raise SystemExit("Erreur: distance_mpc doit etre >0 (JSON ou --distance-mpc)")

    bands = Bands(tau_band=(float(args.tau_band[0]), float(args.tau_band[1])),
                  nu_band=(float(args.nu_band[0]), float(args.nu_band[1])))

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
        hstar_in=float(args.hstar),
        scale_ej_in=float(args.scale_ej),
        calibrate_hstar_event=args.calibrate_hstar,
        hpeak_target=args.hpeak_target,
        calibrate_scale_event=args.calibrate_scale,
        msun_target=args.msun_target,
        plot=bool(args.plot),
    )


if __name__ == "__main__":
    main()
