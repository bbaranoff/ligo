#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ligo_spectral_gpu.py - Version GPU STANDALONE
==============================================

Version GPU-accélérée qui N'UTILISE PAS ligo_cuda_backend
pour éviter les conflits d'imports.

Speedup attendu: 3-5× sur RTX 4090

Usage:
    from ligo_spectral_gpu import analyze_event_gpu
    
    result = analyze_event_gpu(
        event="GW150914",
        params=params,
        npz_dir="data/npz"
    )
"""

import os
import numpy as np  # Import numpy DIRECT
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from gwosc import datasets

try:
    import cupy as cp
    from cupyx.scipy.signal import butter as cp_butter, sosfiltfilt as cp_sosfiltfilt
    from cupyx.scipy.signal.windows import tukey as cp_tukey
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

# Constantes physiques
c = 299792458.0
M_sun = 1.98847e30
Mpc = 3.085677581491367e22
# Constante de Planck (J·s)
h = 6.62607015e-34

# Action classique typique (J·s)
S_classique = 3.193015e4

# Rapport classique / quantique
SCALE_EJ_NORM = S_classique / h


# ============================================================================
# NPZ Loading (copié depuis ligo_spectral_planck)
# ============================================================================

def get_npz_path(event: str, detector: str, npz_dir: str = "data/npz") -> str:
    """Retourne le chemin vers le fichier NPZ local."""
    filename = f"{event}_{detector}.npz"
    path = os.path.join(npz_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPZ file not found: {path}")
    return path


def load_npz(path: str) -> Tuple[np.ndarray, float, float, float]:
    """Charge un fichier NPZ - VERSION NUMPY PURE"""
    z = np.load(path, allow_pickle=False)
    return (
        np.array(z["data"], dtype=float),  # Utilise np.array au lieu de np.asarray
        float(z["fs"]),
        float(z["t0"]),
        float(z["t1"]),
    )


# ============================================================================
# GPU Helper Functions
# ============================================================================

def bandpass_gpu(data_gpu, fs, flow, fhigh, order=4):
    """Bandpass filter sur GPU - OPTIMISÉ sans sync"""
    # Convertir vers GPU si nécessaire
    if not isinstance(data_gpu, cp.ndarray):
        data_gpu = cp.asarray(data_gpu)
    
    nyq = 0.5 * fs
    low = flow / nyq
    high = min(fhigh / nyq, 0.999)
    
    # Limiter order pour éviter instabilités
    if low < 0.01:
        low = 0.01
    if high > 0.99:
        high = 0.99
    
    try:
        sos = cp_butter(order, [low, high], btype='band', output='sos')
        # sosfiltfilt peut être lent, utiliser une seule passe
        # return cp_sosfiltfilt(sos, data_gpu)
        
        # Alternative: FFT bandpass (plus rapide)
        n = len(data_gpu)
        nfft = n
        
        # FFT
        data_fft = cp.fft.rfft(data_gpu, n=nfft)
        freqs = cp.fft.rfftfreq(nfft, d=1.0/fs)
        
        # Masque bande passante
        mask = (freqs >= flow) & (freqs <= fhigh)
        data_fft[~mask] = 0
        
        # IFFT
        result = cp.fft.irfft(data_fft, n=nfft)
        
        return result[:n]
        
    except Exception as e:
        print(f"   ⚠️  Bandpass GPU failed: {e}, using passthrough")
        return data_gpu


def estimate_delay_gpu(x_gpu, y_gpu, fs):
    """Corrélation croisée sur GPU"""
    n = len(x_gpu) + len(y_gpu) - 1
    nfft = 2 ** int(np.ceil(np.log2(n)))
    
    X = cp.fft.rfft(x_gpu, n=nfft)
    Y = cp.fft.rfft(y_gpu, n=nfft)
    
    R = X * cp.conj(Y)
    r = cp.fft.irfft(R, n=nfft)
    
    lag_idx = int(cp.argmax(cp.abs(r)))
    
    if lag_idx > nfft // 2:
        lag_idx -= nfft
    
    delay = lag_idx / fs
    return float(delay)


def welch_psd_gpu(data_gpu, fs, nperseg=2048, noverlap=None):
    """Welch PSD sur GPU - OPTIMISÉ avec vectorisation"""
    if not isinstance(data_gpu, cp.ndarray):
        data_gpu = cp.asarray(data_gpu)
    
    if noverlap is None:
        noverlap = nperseg // 2
    
    n = len(data_gpu)
    step = nperseg - noverlap
    n_segs = max(1, (n - nperseg) // step + 1)
    
    # Limiter nombre de segments pour vitesse
    if n_segs > 100:
        # Utiliser segments plus espacés
        step = (n - nperseg) // 100
        n_segs = min(100, (n - nperseg) // step + 1)
    
    window = cp_tukey(nperseg, 0.25)
    
    nfreqs = nperseg // 2 + 1
    psd_sum = cp.zeros(nfreqs, dtype=cp.float64)
    
    # Batch processing avec stride tricks (plus rapide que boucle)
    for i in range(n_segs):
        start = i * step
        end = start + nperseg
        if end > n:
            break
        
        seg = data_gpu[start:end] * window
        fft_seg = cp.fft.rfft(seg)
        psd_sum += cp.abs(fft_seg) ** 2
    
    psd = psd_sum / max(1, n_segs)
    psd /= (cp.sum(window ** 2) * fs)
    
    freqs = cp.fft.rfftfreq(nperseg, d=1.0/fs)
    
    return freqs, psd


def trapezoid_gpu(y_gpu, x_gpu):
    """Intégration trapézoïdale sur GPU"""
    dx = x_gpu[1:] - x_gpu[:-1]
    y_avg = (y_gpu[1:] + y_gpu[:-1]) / 2.0
    return float(cp.sum(dx * y_avg))


# ============================================================================
# Main GPU Function
# ============================================================================

@dataclass
class Bands:
    tau_band: Tuple[float, float] = (35.0, 250.0)
    nu_band: Tuple[float, float] = (30.0, 350.0)


def analyze_event_gpu(
    event: str,
    params: Dict[str, Any],
    flow: Optional[float] = None,
    fhigh: Optional[float] = None,
    signal_win: Optional[float] = None,
    noise_pad: Optional[float] = None,
    distance_mpc: Optional[float] = None,
    bands: Optional[Bands] = None,
    npz_dir: str = "data/npz",
    peak_quantile: float = 99.5,
    tau_scale: float = 1.0,
    window_scale: float = 1.0,
    scale_ej_in: float = 1.0,
    peak_scale: float = 1.0,
    return_internals: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Analyse GPU d'un événement LIGO - Version STANDALONE
    
    Args:
        event: Nom événement (GW150914)
        params: event_params.json
        flow, fhigh: Bande fréquence [Hz]
        signal_win: Fenêtre signal [s]
        noise_pad: Padding bruit [s]
        distance_mpc: Distance [Mpc]
        bands: Bandes tau/nu
        npz_dir: Répertoire NPZ
        peak_quantile: Quantile peak
        tau_scale, scale_ej_in, peak_scale: Facteurs calibration
        return_internals: Retourner spectres
        verbose: Mode verbeux
    
    Returns:
        Dict avec energy_J, msun_c2, tau_hl_s, nu_eff, etc.
    """
    
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU non disponible - installer: pip install cupy-cuda12x")
    
    # Paramètres
    evp = params.get(event, {}) if isinstance(params.get(event), dict) else {}
    defaults = params.get("default", {}) if isinstance(params.get("default"), dict) else {}
    
    # GPS
    gps = None
    for k in ("gps", "GPS", "gps_time"):
        if k in evp and evp[k] is not None:
            gps = float(evp[k])
            break
    if gps is None:
        try:
            gps = float(datasets.event_gps(event))
        except:
            raise RuntimeError(f"GPS time missing for {event}")
    
    # Distance
    if distance_mpc is None:
        for k in ("distance_mpc", "distance_Mpc"):
            if k in evp and evp[k]:
                distance_mpc = float(evp[k])
                break
    if distance_mpc is None:
        raise RuntimeError(f"distance_mpc missing for {event}")
    
    # Analysis params avec fallback
    flow = flow or evp.get("flow") or defaults.get("flow") or 20.0
    fhigh = fhigh or evp.get("fhigh") or defaults.get("fhigh") or 350.0
    signal_win = signal_win or evp.get("signal_win") or defaults.get("signal_win") or 1.2
    noise_pad = noise_pad or evp.get("noise_pad") or defaults.get("noise_pad") or 1200.0
    
    flow, fhigh = float(flow), float(fhigh)
    signal_win, noise_pad = float(signal_win), float(noise_pad)
    # --- Fenêtre effective (DOIT être définie UNE fois) ---
    window_scale = float(window_scale) if window_scale is not None else 1.0

    signal_win_eff = signal_win * window_scale

    # Garde-fous physiques
    signal_win_eff = max(0.05, min(signal_win_eff, 4.0))

    if bands is None:
        tau_band = evp.get("tau_band") or defaults.get("tau_band") or [35.0, 250.0]
        nu_band = evp.get("nu_band") or defaults.get("nu_band") or [30.0, 350.0]
        bands = Bands(tau_band=tuple(tau_band), nu_band=tuple(nu_band))
    
    if verbose:
        print(f"🚀 GPU: {event} | flow={flow} fhigh={fhigh} signal_win={signal_win}")
        import time as _time
        _t0 = _time.time()
    
    # Load NPZ → GPU
    pH = get_npz_path(event, "H1", npz_dir)
    pL = get_npz_path(event, "L1", npz_dir)
    
    hH_cpu, fsH, t0H, t1H = load_npz(pH)
    hL_cpu, fsL, t0L, t1L = load_npz(pL)
    
    if fsH != fsL:
        raise RuntimeError("Sample rate mismatch")
    
    fs = fsH
    
    # Transférer GPU UNE FOIS
    hH_full = cp.array(hH_cpu, dtype=cp.float64)
    hL_full = cp.array(hL_cpu, dtype=cp.float64)
    del hH_cpu, hL_cpu
    
    if verbose:
        mem_mb = (hH_full.nbytes + hL_full.nbytes) / 1e6
        _t1 = _time.time()
        print(f"   GPU transfer: {mem_mb:.1f} MB in {_t1-_t0:.3f}s")
    
    # PSD sur bruit
    n0 = gps - noise_pad
    n1 = gps - 10.0
    
    idx_start_H = int((n0 - t0H) * fs)
    idx_end_H = int((n1 - t0H) * fs)
    idx_start_L = int((n0 - t0L) * fs)
    idx_end_L = int((n1 - t0L) * fs)
    
    noise_H = hH_full[max(0, idx_start_H):min(len(hH_full), idx_end_H)]
    noise_L = hL_full[max(0, idx_start_L):min(len(hL_full), idx_end_L)]
    
    # Welch GPU
    f_psd_H, S1 = welch_psd_gpu(noise_H, fs, nperseg=min(4096, max(256, len(noise_H)//4)))
    f_psd_L, S2 = welch_psd_gpu(noise_L, fs, nperseg=min(4096, max(256, len(noise_L)//4)))
    
    if verbose:
        _t2 = _time.time()
        print(f"   PSD computed: {_t2-_t1:.3f}s")
    
    # Tau estimation
    
    tau_half = 0.5 * signal_win_eff
    tau_start, tau_end = gps - tau_half, gps + tau_half
    
    idx_tau_H_start = int((tau_start - t0H) * fs)
    idx_tau_H_end = int((tau_end - t0H) * fs)
    idx_tau_L_start = int((tau_start - t0L) * fs)
    idx_tau_L_end = int((tau_end - t0L) * fs)
    
    seg_tau_H = hH_full[max(0, idx_tau_H_start):min(len(hH_full), idx_tau_H_end)]
    seg_tau_L = hL_full[max(0, idx_tau_L_start):min(len(hL_full), idx_tau_L_end)]
    
    x_tau = bandpass_gpu(seg_tau_H, fs, bands.tau_band[0], bands.tau_band[1])
    y_tau = bandpass_gpu(seg_tau_L, fs, bands.tau_band[0], bands.tau_band[1])
    
    tau_hl = -abs(estimate_delay_gpu(x_tau, y_tau, fs))
    tau_hl_s = float(tau_scale) * tau_hl
    
    if verbose:
        _t3 = _time.time()
        print(f"   Tau estimation: {_t3-_t2:.3f}s")
    
    # Signal segment
    # Signal segment (window_scale appliqué)
    signal_win_eff = signal_win * window_scale

    s0 = gps - signal_win_eff / 2.0
    s1 = gps + signal_win_eff / 2.0

    idx_sig_H_start = int((s0 - t0H) * fs)
    idx_sig_H_end = int((s1 - t0H) * fs)
    idx_sig_L_start = int((s0 - t0L) * fs)
    idx_sig_L_end = int((s1 - t0L) * fs)
    
    hH = hH_full[max(0, idx_sig_H_start):min(len(hH_full), idx_sig_H_end)] * float(peak_scale)
    hL = hL_full[max(0, idx_sig_L_start):min(len(hL_full), idx_sig_L_end)] * float(peak_scale)
    
    # Bandpass
    hH_f = bandpass_gpu(hH, fs, flow, fhigh)
    hL_f = bandpass_gpu(hL, fs, flow, fhigh)
    
    N = min(len(hH_f), len(hL_f))
    
    # FFT
    win = cp_tukey(N, 0.2)
    H1 = cp.fft.rfft(hH_f[:N] * win)
    H2 = cp.fft.rfft(hL_f[:N] * win)
    f = cp.fft.rfftfreq(N, d=1.0/fs)
    
    # Masque
    mask = (f >= flow) & (f <= min(fhigh, 0.95 * 0.5 * fs))
    f_use = f[mask]
    H1, H2 = H1[mask], H2[mask]
    
    # PSD interpolé
    S1i = cp.interp(f_use, f_psd_H, S1)
    S2i = cp.interp(f_use, f_psd_L, S2)
    S1i = cp.maximum(S1i, 1e-60)
    S2i = cp.maximum(S2i, 1e-60)
    
    # Phase shift
    phi = 2.0 * cp.pi * f_use * tau_hl_s
    
    # Whitening + coherent sum
    Hw1 = H1 / cp.sqrt(S1i)
    Hw2 = H2 / cp.sqrt(S2i)
    Hc = Hw1 + Hw2 * cp.exp(-1j * phi)
    
    # Edge taper
    bw = float(f_use[-1] - f_use[0])
    edge = max(2.0, min(20.0, 0.10 * bw))
    
    w = cp.ones_like(f_use, dtype=cp.float64)
    lo = f_use < (f_use[0] + edge)
    hi = f_use > (f_use[-1] - edge)
    
    if cp.any(lo):
        w[lo] = 0.5 - 0.5 * cp.cos(cp.pi * (f_use[lo] - f_use[0]) / edge)
    if cp.any(hi):
        w[hi] = 0.5 - 0.5 * cp.cos(cp.pi * (f_use[-1] - f_use[hi]) / edge)
    
    # Spectre énergétique
    dEdf = (cp.abs(Hc) ** 2) * (w * w)
    
    # Intégration
    E_internal = trapezoid_gpu(dEdf, f_use) * SCALE_EJ_NORM
    energy_J = E_internal * scale_ej_in
    
    # nu_eff
    nb0, nb1 = bands.nu_band
    mask_nu = (f_use >= nb0) & (f_use <= nb1)
    
    if cp.any(mask_nu):
        f_nu = f_use[mask_nu]
        dE_nu = dEdf[mask_nu]
        E_nu_total = trapezoid_gpu(dE_nu, f_nu)
        nu_eff = trapezoid_gpu(f_nu * dE_nu, f_nu) / E_nu_total if E_nu_total > 0 else 0.0
    else:
        nu_eff = 0.0
    
    # Résultats
    result = {
        'event': event,
        'energy_J': float(energy_J),
        'E_internal': float(E_internal),
        'msun_c2': float(energy_J / (M_sun * c ** 2)),
        'tau_hl_s': float(tau_hl_s),
        'nu_eff': {'nu_eff_energy': float(nu_eff)},
        'peak_scale': float(peak_scale),
        'tau_scale': float(tau_scale),
        'scale_ej': float(scale_ej_in),
        'distance_mpc': float(distance_mpc),
        'gpu_used': True,
    }
    
    if return_internals:
        result['f_use'] = cp.asnumpy(f_use).tolist()
        result['dEdf'] = cp.asnumpy(dEdf).tolist()
    
    # Libérer GPU
    del hH_full, hL_full, H1, H2, Hc, dEdf
    cp.get_default_memory_pool().free_all_blocks()
    
    return result


# ============================================================================
# Test
# ============================================================================


# Alias pour compatibilité
analyze_event = analyze_event_gpu

if __name__ == '__main__':
    import time
    import json
    
    if not GPU_AVAILABLE:
        print("❌ GPU non disponible")
        exit(1)
    
    print("="*70)
    print("🧪 TEST GPU STANDALONE")
    print("="*70)
    
    with open('event_params.json', 'r') as f:
        params = json.load(f)
    
    event = 'GW150914'
    
    print(f"\nTest: {event}")
    start = time.time()
    result = analyze_event_gpu(event=event, params=params, verbose=True)
    gpu_time = time.time() - start
    
    print(f"\n✅ GPU: {gpu_time:.3f}s")
    print(f"   E = {result['energy_J']:.3e} J")
    print(f"   M = {result['msun_c2']:.4f} M☉")
    print(f"   ν = {result['nu_eff']['nu_eff_energy']:.1f} Hz")
    
    print("\n💡 Pour comparer avec CPU:")
    print("   python -c \"import ligo_spectral_planck as lsp; import json; import time;\"")
    print("   \"params=json.load(open('event_params.json')); start=time.time();\"")
    print("   \"r=lsp.analyze_event('GW150914', params); print(f'CPU: {time.time()-start:.3f}s')\"")
