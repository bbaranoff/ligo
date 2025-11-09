#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GWOSC → H1 & L1 cohérents → dE/df, E, m, nu_eff, h_eff
- Auto-correction du décalage H1↔L1 (choix du signe max(Re{C}))
- Lissage optionnel de dE/df et intégrale cumulée E(<f)
- Export .npz (f, dEdf, E_cum, cohérence)

Exemple:
python ligo_energy_coherent.py --event GW150914 --distance-mpc 410 \
  --flow 20 --fhigh 512 --signal-win 0.3 --noise-pad 50 --smooth-sigma 3 \
  --plot --export out_gw150914.npz
"""

import argparse, os, numpy as np, matplotlib.pyplot as plt
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
    gaussian_filter1d = None  # lissage désactivé si indisponible

# Constantes SI
c = 299792458.0
G = 6.67430e-11
Mpc = 3.085677581491367e22
M_sun = 1.98847e30

# -------------------- utilitaires --------------------
def fetch(det, t0, t1, outdir="data"):
    os.makedirs(outdir, exist_ok=True)
    ts = TimeSeries.fetch_open_data(det, t0, t1, cache=True)
    try: ts.write(os.path.join(outdir, f"{det}_{t0:.0f}_{t1:.0f}.hdf5"), format="hdf5")
    except Exception: pass
    return ts

def bandpass(x, fs, f1=20.0, f2=1024.0, order=4):
    sos = butter(order, [f1, f2], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x)

def psd_welch(ts, seglen=4.0, overlap=2.0, fmin=10.0, fmax=2048.0):
    from numpy.fft import rfft, rfftfreq
    x = np.asarray(ts.value, float)
    fs = ts.sample_rate.value
    Nseg = int(seglen*fs); Nhop = int((seglen-overlap)*fs)
    win = tukey(Nseg, 0.2); U = (win**2).sum()
    specs = []
    for i in range(0, x.size-Nseg+1, Nhop):
        seg = bandpass(x[i:i+Nseg], fs, fmin, fmax)
        Xk = rfft(seg*win)
        Pxx = (2.0/(fs*U))*np.abs(Xk)**2  # 1/Hz
        specs.append(Pxx)
    S = np.median(np.stack(specs), axis=0)
    f = rfftfreq(Nseg, d=1.0/fs)
    return f, S

def estimate_delay(h1, h2, fs, search_ms=10.0):
    """Délai H1→L1 (s) par corrélation rapide (signe arbitraire initial)."""
    N = min(h1.size, h2.size)
    w = tukey(N, 0.2)
    x = (h1[:N]*w); y = (h2[:N]*w)
    X = np.fft.rfft(x); Y = np.fft.rfft(y)
    R = X*np.conj(Y)
    r = np.fft.irfft(R, n=N)
    r = np.concatenate([r[N//2:], r[:N//2]])
    lags = (np.arange(-N//2, N//2))/fs
    mask = (lags >= -search_ms/1000.0) & (lags <= search_ms/1000.0)
    i = np.argmax(r[mask]); tau = lags[mask][i]
    return tau

# -------------------- analyse cohérente --------------------
def analyze_coherent(tsH, tsL, gps, distance_mpc, flow=20.0, fhigh=512.0,
                     noise_pad=50.0, signal_win=0.3, smooth_sigma=None,
                     plot=False, title="", export_path=None):
    if distance_mpc is None: raise SystemExit("--distance-mpc requis.")
    r = distance_mpc*Mpc
    fsH = tsH.sample_rate.value; fsL = tsL.sample_rate.value
    if abs(fsH - fsL) > 1e-6: raise SystemExit("H1 et L1 doivent avoir la même Fs.")
    fs = fsH

    # --- PSD bruit hors-signal
    noiseH = tsH.crop(gps-noise_pad-40, gps-10)
    noiseL = tsL.crop(gps-noise_pad-40, gps-10)
    fH, S1 = psd_welch(noiseH, fmin=flow, fmax=fhigh)
    fL, S2 = psd_welch(noiseL, fmin=flow, fmax=fhigh)
    if fH.size != fL.size or np.max(np.abs(fH-fL))>1e-9:
        S2 = np.interp(fH, fL, S2); f = fH
    else:
        f = fH

    # --- Fenêtre courte centrée sur le chirp
    half = signal_win/2.0
    wH = tsH.crop(gps-half, gps+half); wL = tsL.crop(gps-half, gps+half)
    hH = bandpass(np.asarray(wH.value,float), fs, flow, fhigh)
    hL = bandpass(np.asarray(wL.value,float), fs, flow, fhigh)

    # --- Estimation du délai et auto-correction du signe
    tau_guess = estimate_delay(hH, hL, fs, search_ms=10.0)
    N = min(hH.size, hL.size); dt = 1.0/fs; T = N*dt
    w = tukey(N, 0.2)
    H1 = np.fft.rfft(hH[:N]*w); H2 = np.fft.rfft(hL[:N]*w)
    f_short = np.fft.rfftfreq(N, d=dt)

    def energy_with_tau(tau):
        ph = np.exp(-1j*2*np.pi*f_short*tau)
        H2_al = H2 * ph
        S1_tot = (2.0/T)*np.abs((dt*H1))**2
        S2_tot = (2.0/T)*np.abs((dt*H2_al))**2
        C = (dt*H1) * np.conj(dt*H2_al) * (2.0/T)
        S1n = np.interp(f_short, f, S1)
        S2n = np.interp(f_short, f, S2)
        S1_sig = np.clip(S1_tot - S1n, 0.0, np.inf)
        S2_sig = np.clip(S2_tot - S2n, 0.0, np.inf)
        ReC = np.real(C)
        ReC = np.clip(ReC, 0.0, np.sqrt(S1_sig*S2_sig + 1e-300))
        dEdf = (np.pi * c**3 * r**2 / (2.0 * G)) * (f_short**2) * ReC / (4*np.pi**2)
        band = (f_short >= max(flow,1e-9)) & (f_short <= fhigh)
        if band.sum()<2: return 0.0
        df = f_short[1]-f_short[0]
        return float(np.trapz(dEdf[band], dx=df)), ReC, S1_tot, S2_tot, dEdf

    E_pos, ReC_pos, S1_tot_pos, S2_tot_pos, dEdf_pos = energy_with_tau(+tau_guess)
    E_neg, ReC_neg, S1_tot_neg, S2_tot_neg, dEdf_neg = energy_with_tau(-tau_guess)

    if E_neg > E_pos:
        tau = -tau_guess
        ReC, S1_tot, S2_tot, dEdf = ReC_neg, S1_tot_neg, S2_tot_neg, dEdf_neg
    else:
        tau = +tau_guess
        ReC, S1_tot, S2_tot, dEdf = ReC_pos, S1_tot_pos, S2_tot_pos, dEdf_pos

    # --- Bande utile et lissage optionnel
    band = (f_short >= max(flow,1e-9)) & (f_short <= fhigh)
    f_use = f_short[band]
    dEdf_use = dEdf[band]
    # --- Nettoyage et lissage plus fort ---
    dEdf_use = np.nan_to_num(dEdf_use, nan=0.0, posinf=0.0, neginf=0.0)
    if smooth_sigma and gaussian_filter1d is not None:
        dEdf_use = gaussian_filter1d(dEdf_use, sigma=float(smooth_sigma))
    else:
    # moyenne glissante basique
        kernel = 5
        dEdf_use = np.convolve(dEdf_use, np.ones(kernel)/kernel, mode='same')

    # --- Intégration
    df = f_use[1]-f_use[0]
    E = np.trapz(dEdf_use, dx=df)
    E_cum = np.cumsum(dEdf_use) * df
    m = E/c**2; m_sun = m/M_sun
    nu_eff = np.trapz(f_use*dEdf_use, dx=df)/np.trapz(dEdf_use, dx=df)
    h_eff = E/nu_eff

    print("\n=== RÉSULTATS COHÉRENTS H1–L1 (auto-corrigés) ===")
    print(f"délai H1→L1 estimé : {tau*1e3:.3f} ms (signe choisi pour max(E))")
    print(f"E ~ {E:.3e} J")
    print(f"m = E/c^2 ~ {m:.3e} kg  (~ {m_sun:.2f} M_sun)")
    print(f"nu_eff ~ {nu_eff:.1f} Hz")
    print(f"h_eff ~ {h_eff:.3e} J*s")
    print("==============================================\n")

    # --- Export
    if export_path:
        np.savez_compressed(
            export_path,
            f=f_use, dEdf=dEdf_use, E_cum=E_cum,
            ReC=ReC[band], S1_tot=S1_tot[band], S2_tot=S2_tot[band]
        )
        print(f"[export] écrit : {export_path}")

    # --- Plots (facultatif)
    if plot:
        # cohérence gamma^2
        gamma2 = np.abs(ReC)**2 / (S1_tot * S2_tot + 1e-300)
        plt.figure(figsize=(9,4))
        plt.semilogx(f_use[1:], np.clip(gamma2[band][1:],0,1), lw=1)
        plt.axvspan(flow, fhigh, color="gray", alpha=0.2)
        plt.ylim(0,1.05); plt.xlabel("Fréquence (Hz)"); plt.ylabel(r"$\gamma^2(f)$")
        plt.title("Cohérence H1–L1 (fenêtre courte)"); plt.tight_layout(); plt.show()

        # dE/df et E_cum
        plt.figure(figsize=(9,4))
        plt.loglog(f_use, dEdf_use, lw=1)
        plt.xlabel("Fréquence (Hz)"); plt.ylabel("dE/df (J/Hz)")
        plt.title(f"{title} — dE/df cohérent (Re{{C}})"); plt.tight_layout(); plt.show()

        plt.figure(figsize=(9,4))
        plt.semilogx(f_use, E_cum, lw=1)
        plt.xlabel("Fréquence (Hz)"); plt.ylabel("E(<f) (J)")
        plt.title(f"{title} — Énergie cumulée"); plt.tight_layout(); plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--tpad", type=float, default=128.0)
    ap.add_argument("--distance-mpc", type=float, required=True)
    ap.add_argument("--flow", type=float, default=20.0)
    ap.add_argument("--fhigh", type=float, default=512.0)
    ap.add_argument("--signal-win", type=float, default=0.3)
    ap.add_argument("--noise-pad", type=float, default=50.0)
    ap.add_argument("--smooth-sigma", type=float, default=None, help="σ Gauss pour lisser dE/df")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--export", default=None, help="Chemin .npz d’export (facultatif)")
    args = ap.parse_args()

    gps = datasets.event_gps(args.event)
    t0, t1 = gps - args.tpad, gps + args.tpad
    H1 = fetch("H1", t0, t1); L1 = fetch("L1", t0, t1)

    analyze_coherent(
        H1, L1, gps, args.distance_mpc,
        flow=args.flow, fhigh=args.fhigh,
        noise_pad=args.noise_pad, signal_win=args.signal_win,
        smooth_sigma=args.smooth_sigma,
        plot=args.plot, title=f"{args.event}",
        export_path=args.export
    )

if __name__ == "__main__":
    main()
