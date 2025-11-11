#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ligo_xai_net.py — Analyse cohérente réseau (H1/L1/V1) inspirée de ligo66.py

- Corrélation spectrale (fenêtre Tukey, bande limitée)
- Énergie cohérente γ² pondérée
- ν_eff, h_eff, v_eff
- Correction PSD-like adoucie
- Calibration auto ou externe
"""

import os, json, argparse
import numpy as np
from numpy import trapezoid
from scipy.constants import c, pi, G
from scipy.signal import butter, sosfiltfilt
from gwpy.timeseries import TimeSeries
from gwosc import datasets

# --- Fenêtre Tukey
try:
    from scipy.signal.windows import tukey as _tukey
    def tukey(M, alpha=0.2): return _tukey(M, alpha)
except Exception:
    def tukey(M, alpha=0.2):
        n = np.arange(M, dtype=float)
        w = np.ones(M)
        if alpha <= 0: return w
        if alpha >= 1: return np.hanning(M)
        width = int(alpha*(M-1)/2)
        if width > 0:
            ramp = 0.5*(1 - np.cos(pi*np.arange(width)/width))
            w[:width] = ramp
            w[-width:] = ramp[::-1]
        return w

# --- Constantes
Mpc = 3.085677581491367e22
ANTENNA_AVG = 5.0
REF_E_J = {"GW150914": 5.4e47, "GW151226": 1.0e47, "GW170104": 3.5e47, "GW170814": 5.0e47}

# --- Signal utils
def bandpass(x, fs, f1, f2, order=4):
    sos = butter(order, [f1, f2], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)

def ensure_fs(ts, fs_target):
    fs = float(ts.sample_rate.value)
    return ts if abs(fs - fs_target) < 1e-6 else ts.resample(fs_target)

# --- Délai spectral (type ligo66)
def estimate_delay_spec(h1, h2, fs, search_ms=12.0):
    N = min(len(h1), len(h2))
    if N < 16:
        return 0.0
    w = tukey(N, 0.2)
    X = np.fft.rfft(h1[:N] * w)
    Y = np.fft.rfft(h2[:N] * w)
    r = np.fft.irfft(X * np.conj(Y), n=N)
    r = np.concatenate([r[N//2:], r[:N//2]])
    lags = np.arange(-N//2, N//2) / fs
    mask = (lags >= -search_ms/1000) & (lags <= search_ms/1000)
    return float(lags[mask][np.argmax(r[mask])]) if np.any(mask) else 0.0

# --- Énergie cohérente
def compute_pair_energy(A, B, tau_s, flow, fhigh, r_m):
    xA = np.asarray(A.value, float)
    xB = np.asarray(B.value, float)
    fs = float(A.sample_rate.value)
    N = min(len(xA), len(xB))
    if N < 16:
        return 0.0, np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1, bool)
    dt = 1/fs
    w = tukey(N, 0.2)
    U = np.sum(w**2)
    Aft = np.fft.rfft(xA*w)
    Bft = np.fft.rfft(xB*w)
    f = np.fft.rfftfreq(N, dt)
    phase = np.exp(-1j*2*pi*f*tau_s)
    Btau = Bft * phase
    eps = 1e-300
    SAA = (2/(fs*U)) * np.abs(Aft)**2
    SBB = (2/(fs*U)) * np.abs(Btau)**2
    C = (2/(fs*U)) * Aft * np.conj(Btau)
    ReC = np.clip(np.real(C), 0, np.sqrt(SAA*SBB + eps))
    gamma2 = np.clip((np.abs(C)**2)/(SAA*SBB + eps), 0, 1)
    K = (c**3)/(8*pi*G)*(r_m**2)*ANTENNA_AVG
    dEdf = K*(2*pi*f)**2*ReC*gamma2
    band = (f >= max(flow, 1e-9)) & (f <= fhigh)
    # marge anti-bords (évite les sur-pondérations au bord de la FFT)
    f_lo = flow + 0.05*(fhigh - flow)
    f_hi = fhigh - 0.05*(fhigh - flow)
    band &= (f >= f_lo) & (f <= f_hi)
    E = float(trapezoid(dEdf[band], f[band])) if np.any(band) else 0.0
    return E, f, dEdf, gamma2, band

# --- Analyse réseau
def analyze_network(event, distance_mpc, flow, fhigh, signal_win,
                    auto_calib=False, apply_calib=None,
                    tpad=1024.0, search_ms=12.0):

    gps = float(datasets.event_gps(event))
    print(f"Chargement des données LIGO : {event} @ GPS {gps:.1f}")
    os.makedirs("results", exist_ok=True)

    fs = 4096.0
    H1 = ensure_fs(TimeSeries.fetch_open_data("H1", gps - tpad, gps + tpad, cache=True), fs)
    L1 = ensure_fs(TimeSeries.fetch_open_data("L1", gps - tpad, gps + tpad, cache=True), fs)
    try:
        V1 = ensure_fs(TimeSeries.fetch_open_data("V1", gps - tpad, gps + tpad, cache=True), fs)
    except Exception:
        V1 = None

    half = signal_win/2
    def seg(ts): 
        return bandpass(np.asarray(ts.crop(gps - half, gps + half).value, float), fs, flow, fhigh)

    tsH, tsL = TimeSeries(seg(H1), sample_rate=fs), TimeSeries(seg(L1), sample_rate=fs)
    pairs = [("H1","L1",tsH,tsL)]
    if V1 is not None:
        tsV = TimeSeries(seg(V1), sample_rate=fs)
        pairs += [("H1","V1",tsH,tsV), ("L1","V1",tsL,tsV)]

    r_m = distance_mpc * Mpc
    pair_out = []
    for a,b,A,B in pairs:
        tau0 = estimate_delay_spec(A.value, B.value, fs, search_ms)
        Epos,f,dEdf_pos,g2_pos,band = compute_pair_energy(A,B,+abs(tau0),flow,fhigh,r_m)
        Eneg,_,dEdf_neg,g2_neg,_   = compute_pair_energy(A,B,-abs(tau0),flow,fhigh,r_m)
        if Epos >= Eneg: tau,Epair,dEdf,g2 = +abs(tau0),Epos,dEdf_pos,g2_pos
        else: tau,Epair,dEdf,g2 = -abs(tau0),Eneg,dEdf_neg,g2_neg
        print(f"{a}-{b}: τ={tau*1e3:+.3f} ms ; E={Epair:.3e} J")
        pair_out.append({"pair":f"{a}-{b}","tau_ms":tau*1e3,"E_J":Epair,
                         "f":f,"dEdf":dEdf,"gamma2":g2})

    if not pair_out: return

    dEdf_stack = np.stack([p["dEdf"] for p in pair_out])
    g2_stack = np.stack([p["gamma2"] for p in pair_out])
    f = pair_out[0]["f"]
    band = (f >= max(flow,1e-9)) & (f <= fhigh)
    W = g2_stack.sum(axis=0)+1e-12
    dEdf_net = (g2_stack*dEdf_stack).sum(axis=0)/W

    E_net = float(trapezoid(dEdf_net[band], f[band])) if np.any(band) else 0.0
    denom = max(float(trapezoid(dEdf_net[band], f[band])),1e-300)
    nu_eff = float(trapezoid(f[band]*dEdf_net[band], f[band]) / denom)
    h_eff = E_net / max(nu_eff,1e-300)
    v_eff = c*(nu_eff/300.0)

    # --- Correction PSD-like adoucie
    nu0 = 120.0
    if nu_eff < nu0:
        # bas de bande : booster modérément
        psd_corr = (nu0 / nu_eff)**0.9
    else:
        # haut de bande : freiner davantage + cut doux autour de 220–300 Hz
        beta = 0.8
        soft_roll = 1.0 / (1.0 + ((nu_eff - 220.0)/60.0)**2)  # ~1 près de 220, <1 vers 280–300
        psd_corr = (nu0 / nu_eff)**beta * soft_roll

    E_net *= psd_corr
    h_eff  = E_net / max(nu_eff, 1e-300)

    # --- Calibration
    calib_factor = 1.0
    if apply_calib is not None:
        calib_factor = float(apply_calib)
    elif auto_calib and event in REF_E_J and E_net>0:
        calib_factor = REF_E_J[event]/E_net
    E_net *= calib_factor
    h_eff *= calib_factor

    print("\n=== RÉSULTAT RÉSEAU (γ² pondéré) ===")
    print(f"E_net = {E_net:.3e} J")
    print(f"ν_eff = {nu_eff:.1f} Hz ; h_eff = {h_eff:.3e} J·s ; v_eff = {v_eff:.3e} m/s")
    print(f"[calib_factor appliqué = {calib_factor:.3e}]")
    print("=====================================\n")

    out = {
        "event":event,
        "distance_mpc":distance_mpc,
        "flow_Hz":flow,
        "fhigh_Hz":fhigh,
        "signal_win_s":signal_win,
        "tpad_s":tpad,
        "search_ms":search_ms,
        "E_net_J":E_net,
        "nu_eff_Hz":nu_eff,
        "h_eff_Js":h_eff,
        "v_eff_m_per_s":v_eff,
        "psd_corr":psd_corr,
        "calib_factor":calib_factor,
        "pairs":[{"pair":p["pair"],"tau_ms":p["tau_ms"],"E_J":p["E_J"]} for p in pair_out]
    }
    with open(f"results/{event}_network.json","w") as fjs:
        json.dump(out,fjs,indent=2)
    print(f"[saved] results/{event}_network.json")
    return out

# --- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event",required=True)
    ap.add_argument("--distance-mpc",type=float,required=True)
    ap.add_argument("--flow",type=float,default=35.0)
    ap.add_argument("--fhigh",type=float,default=250.0)
    ap.add_argument("--signal-win",type=float,default=1.0)
    ap.add_argument("--auto-calib",action="store_true")
    ap.add_argument("--apply-calib",type=float,default=None)
    ap.add_argument("--tpad",type=float,default=1024.0)
    ap.add_argument("--search-ms",type=float,default=12.0)
    args = ap.parse_args()
    analyze_network(args.event,args.distance_mpc,args.flow,args.fhigh,
                    args.signal_win,args.auto_calib,args.apply_calib,
                    args.tpad,args.search_ms)

if __name__ == "__main__":
    main()
