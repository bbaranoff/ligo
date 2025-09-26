#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GWOSC → PSD (Welch) → matched-filter (NumPy/PyCBC) + Couche Floquet A_k(ν, n̂, p; t)
Compat avec le cadre "lunettes" s(t) (rescaling des étalons).

Ajouts majeurs par rapport à gwosc_fetch_and_infer.py :
- Calcul explicite de populations de bandes latérales A_{k=±1}(f) (proxy, up-to-scale)
- Export JSON des spectres A_k(f) et des densités d'énergie relatives ρ_rad^k
- Option de rescaling s ("lunettes") appliquée aux fréquences et aux densités (ρ → s^-4 ρ)
- Garde l'inférence SNR(t) & grille mchirp/eta pour rester compatible

Usage de base (comme avant) :
  python gwosc_floquet_ak.py --event GW150914 --detectors H1 L1 --plot --out results.json

Avec A_k + export :
  python gwosc_floquet_ak.py --event GW150914 --detectors H1 L1 --ak --ak-file ak_GW150914.json

Avec rescaling d'étalons (s) :
  python gwosc_floquet_ak.py --event GW150914 --detectors H1 L1 --ak --s-rescale 0.8

Interprétation :
- On ne modélise pas l'optique détaillée (cavités, SRM). On fournit A_{±1}(f) proportionnel
  au noyau de filtrage apparié |h(f)|^2 / S_n(f) (population convertie par modulation GW).
- A_0 est un proxy (N_c) constant par détecteur, utile pour ratios et bilans relatifs.
- ρ_rad^k est reporté en unités arbitraires, invariantement au facteur commun de normalisation.
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from gwosc import datasets
from gwpy.timeseries import TimeSeries

from pycbc import waveform, psd, types
from pycbc.psd.estimate import welch
from pycbc.conversions import mass1_from_mchirp_eta, mass2_from_mchirp_eta


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------

def fetch_open_data(det, t0, t1, cache_dir="data"):
    os.makedirs(cache_dir, exist_ok=True)
    ts = TimeSeries.fetch_open_data(det, t0, t1, cache=True)
    out = os.path.join(cache_dir, f"{det}_{t0}_{t1}.hdf5")
    if not os.path.exists(out):
        try:
            ts.write(out, format="hdf5")
        except OSError:
            pass
    return ts


# ---------------------------------------------------------------------
# Conversions & PSD
# ---------------------------------------------------------------------

def to_pycbc_timeseries(gwpy_ts):
    arr = np.asarray(gwpy_ts.value, dtype=np.float64, order="C")
    dt = float(gwpy_ts.dt.value)
    return types.TimeSeries(arr, delta_t=dt)

def build_psd(gwpy_ts, seglen=8, overlap=4, hp_hz=15.0):
    sr = gwpy_ts.sample_rate.value
    arr = gwpy_ts.value
    from scipy.signal import detrend, butter, filtfilt
    arr = detrend(arr, type="linear")
    if hp_hz and hp_hz > 0.0:
        b, a = butter(4, hp_hz / (0.5 * sr), btype="highpass")
        arr = filtfilt(b, a, arr)
    ts_pycbc = types.TimeSeries(np.asarray(arr, dtype=np.float64, order="C"), delta_t=1.0 / sr)
    seg = int(seglen * sr)
    stride = int((seglen - overlap) * sr)
    P = welch(ts_pycbc, seg_len=seg, seg_stride=stride, avg_method="median")
    lfc = hp_hz or 20.0
    P = psd.inverse_spectrum_truncation(P, int(sr), low_frequency_cutoff=lfc)
    return P


# ---------------------------------------------------------------------
# Outils FFT / Template
# ---------------------------------------------------------------------

def numpy_fft_to_freqseries(ts_pycbc: types.TimeSeries) -> types.FrequencySeries:
    arr = np.asarray(ts_pycbc.numpy(), dtype=np.float64, order="C")
    n = arr.shape[0]
    dt = float(ts_pycbc.delta_t)
    df = 1.0 / (n * dt)
    spec = np.fft.rfft(arr).astype(np.complex128, copy=False)
    return types.FrequencySeries(spec, delta_f=df)

def template_fd(df, flow, fhigh, mchirp, eta, approx="IMRPhenomD"):
    m1 = mass1_from_mchirp_eta(mchirp, eta)
    m2 = mass2_from_mchirp_eta(mchirp, eta)
    hp, hc = waveform.get_fd_waveform(approximant=approx,
                                      mass1=m1, mass2=m2,
                                      f_lower=flow, f_final=fhigh,
                                      delta_f=df)
    return hp  # PyCBC FrequencySeries


# ---------------------------------------------------------------------
# Couche Floquet A_k (proxy up-to-scale)
# ---------------------------------------------------------------------

def ak_populations_from_strain(stilde, P, htilde, flow, fhigh, Nc=1.0, norm="L1"):
    """
    Construit des A_k(f) proportionnels à la conversion de population
    induite par la modulation GW :
        beta(f) ∝ |h(f)|^2 / S_n(f)
    puis A_{±1}(f) = C * beta(f), A_0 = N_c (porteuse).

    Entrées :
      - stilde : FrequencySeries (rFFT) du strain (non utilisé pour A_k "source-driven")
      - P      : PSD FrequencySeries (même delta_f que stilde)
      - htilde : template FD (FrequencySeries) aligné en delta_f
      - flow, fhigh : bande utile
      - Nc     : population "porteuse" arbitraire (échelle de référence)
      - norm   : "L1" (∑ beta df = 1) ou "L2" (∑ beta^2 df = 1) pour C

    Sorties :
      - dict {"f": f, "A0": A0, "Ap1": Aplus, "Am1": Aminus}
      - densités relatives (ρ_rad^k) en unités arbitraires (dict)
    """
    df = stilde.delta_f
    n = len(stilde)
    dt = 1.0 / (2.0 * df * (n - 1)) * 2  # non utilisé, juste mémo
    freqs = np.fft.rfftfreq((n - 1) * 2, d=dt)  # mais on préfère calcul direct ci-dessous
    freqs = np.arange(n) * df

    # Alignements et masques
    P_np = np.asarray(P.numpy(), dtype=np.float64, order="C")
    h_np = np.asarray(htilde.numpy(), dtype=np.complex128, order="C")
    if len(P_np) < n:
        P_np = np.pad(P_np, (0, n - len(P_np)), constant_values=np.inf)
    elif len(P_np) > n:
        P_np = P_np[:n]
    if len(h_np) < n:
        h_np = np.pad(h_np, (0, n - len(h_np)))
    elif len(h_np) > n:
        h_np = h_np[:n]

    band = (freqs >= flow) & (freqs <= fhigh)
    beta = np.zeros_like(freqs, dtype=np.float64)
    safe = band & np.isfinite(P_np) & (P_np > 0)
    beta[safe] = (np.abs(h_np[safe]) ** 2) / P_np[safe]

    # Normalisation up-to-scale
    if norm.upper() == "L2":
        C = 1.0 / np.sqrt(np.sum(beta[safe] ** 2) * df + 1e-30)
    else:
        C = 1.0 / (np.sum(beta[safe]) * df + 1e-30)

    Ap1 = C * beta
    Am1 = C * beta  # symétrique par défaut ; asymétrie peut être injectée si besoin

    A0 = np.full_like(freqs, float(Nc), dtype=np.float64)

    # Densités d'énergie relatives ρ_rad^k ~ ∫ h(ν+kΩ_m) (ν+kΩ_m) A_k dν
    # Ici, ν ≡ f (Hz) pour les modulations (Ω_m/2π), et h est constant (Planck) → facteur relatif ∝ ∫ (f) A_k(f) df
    # On reporte des valeurs utiles pour ratios / comparaisons.
    rho_Ap1 = float(np.sum(freqs[safe] * Ap1[safe]) * df)
    rho_Am1 = float(np.sum(freqs[safe] * Am1[safe]) * df)
    # A0 n'est pas résolu en f (porteuse optique), on donne une valeur de référence sur la bande :
    rho_A0 = float(Nc * np.sum(freqs[band]) * df)

    ak = {
        "f": freqs.tolist(),
        "A0": A0.tolist(),
        "Ap1": Ap1.tolist(),
        "Am1": Am1.tolist()
    }
    rhos = {"rho_A0_rel": rho_A0, "rho_Ap1_rel": rho_Ap1, "rho_Am1_rel": rho_Am1}
    return ak, rhos


def apply_s_rescale(ak, rhos, s):
    """
    Applique le cadre "lunettes":
        f_mes = s f_phys, A_mes = s^3 A_phys, ρ_mes = s^-4 ρ_phys
    On transforme ak in-place style "mesure" et on met à jour rhos.
    """
    if s is None or s == 1.0:
        return ak, rhos

    f = np.array(ak["f"], dtype=float)
    A0 = np.array(ak["A0"], dtype=float)
    Ap1 = np.array(ak["Ap1"], dtype=float)
    Am1 = np.array(ak["Am1"], dtype=float)

    f_m = s * f
    A0_m = (s ** 3) * A0
    Ap1_m = (s ** 3) * Ap1
    Am1_m = (s ** 3) * Am1

    ak_m = {
        "f": f_m.tolist(),
        "A0": A0_m.tolist(),
        "Ap1": Ap1_m.tolist(),
        "Am1": Am1_m.tolist()
    }
    rhos_m = {k: (s ** -4) * v for k, v in rhos.items()}
    return ak_m, rhos_m


# ---------------------------------------------------------------------
# SNR pipeline (repris, compact)
# ---------------------------------------------------------------------

def snr_time_and_template(strain_ts, P, mchirp, eta, flow, fhigh, approx="IMRPhenomD"):
    stilde = numpy_fft_to_freqseries(strain_ts)
    df = stilde.delta_f
    n = len(strain_ts)
    dt = float(strain_ts.delta_t)
    freqs = np.fft.rfftfreq(n, d=dt)

    htilde = template_fd(df, flow, fhigh, mchirp, eta, approx)
    h_np = np.asarray(htilde.numpy(), dtype=np.complex128, order="C")
    if h_np.shape[0] < len(freqs):
        h_np = np.pad(h_np, (0, len(freqs) - h_np.shape[0]))
    elif h_np.shape[0] > len(freqs):
        h_np = h_np[:len(freqs)]

    Pk = P if P.delta_f == df else psd.interpolate(P, df)
    if len(Pk) < len(freqs):
        Pk = Pk.resize(len(freqs))
    elif len(Pk) > len(freqs):
        Pk = Pk[:len(freqs)]
    P_np = np.asarray(Pk.numpy(), dtype=np.float64, order="C")

    band = (freqs >= flow) & (freqs <= fhigh)
    st_np = np.asarray(stilde.numpy(), dtype=np.complex128, order="C")
    st_np[~band] = 0.0
    h_np[~band] = 0.0
    P_np[~band] = np.inf

    Q = np.conj(h_np) * st_np / P_np
    z_t = 4.0 * np.fft.irfft(Q, n=n) * df
    numer = np.abs(z_t)
    denom_sq = 4.0 * np.sum((np.abs(h_np) ** 2) / P_np) * df
    snr_t = numer / np.sqrt(denom_sq + 1e-30)
    snr_ts = types.TimeSeries(snr_t.astype(np.float64, copy=False), delta_t=dt)
    return snr_ts, types.FrequencySeries(h_np, delta_f=df), stilde, Pk


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--detectors", nargs="+", default=["H1", "L1"])
    ap.add_argument("--tpad", type=float, default=128.0)
    ap.add_argument("--flow", type=float, default=20.0)
    ap.add_argument("--fhigh", type=float, default=1024.0)
    ap.add_argument("--model", default="IMRPhenomD")
    ap.add_argument("--mass1", type=float, required=True, help="masse 1 (Msol) — detector-frame")
    ap.add_argument("--mass2", type=float, required=True, help="masse 2 (Msol) — detector-frame")
    ap.add_argument("--resample", type=float, default=None)
    ap.add_argument("--refine", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot-file", default=None)
    ap.add_argument("--z", type=float, default=None, help="redshift (uniquement pour info masses)")
    ap.add_argument("--out", default="results.json")

    # A_k layer
    ap.add_argument("--ak", action="store_true", help="exporter A_k(f) et densités ρ_rad relatives")
    ap.add_argument("--ak-file", default=None, help="fichier JSON A_k (par détecteur)")
    ap.add_argument("--ak-plot", action="store_true", help="tracer A_k(f) et sauvegarder PNG")
    ap.add_argument("--ak-plot-prefix", default=None, help="préfixe fichier PNG (défaut: ak_<event>_<det>.png)")
    ap.add_argument("--Ncar", type=float, default=1.0, help="population porteuse proxy A_0 = Ncar")
    ap.add_argument("--ak-norm", choices=["L1", "L2"], default="L1", help="normalisation de beta(f)")
    ap.add_argument("--s-rescale", type=float, default=None, help="facteur s(t) des 'lunettes' (facultatif)")

    args = ap.parse_args()

    gps = datasets.event_gps(args.event)
    t0, t1 = gps - args.tpad, gps + args.tpad

    out = {"event": args.event, "gps": float(gps), "detectors": {}, "model": args.model}
    ak_export = {"event": args.event, "gps": float(gps), "A_k": {}}
    snr_store = {}

    for det in args.detectors:
        ts = fetch_open_data(det, t0, t1)
        if args.resample:
            try:
                ts = ts.resample(args.resample)
            except Exception as ex:
                print(f"[{det}] resample {args.resample} Hz skipped: {ex}")

        P = build_psd(ts)

        center = ts.crop(gps - 8.0, gps + 8.0)
        strain_ts = to_pycbc_timeseries(center)

        # Grille simple pour approx du template
        # Paramètres requis: (mchirp, eta) ou (mass1, mass2)
        if args.mass1 is not None and args.mass2 is not None:
            from pycbc.conversions import mchirp_from_mass1_mass2, eta_from_mass1_mass2
            mb = float(mchirp_from_mass1_mass2(args.mass1, args.mass2))
            eb = float(eta_from_mass1_mass2(args.mass1, args.mass2))
        elif args.mchirp_val is not None and args.eta_val is not None:
            mb = float(args.mchirp_val)
            eb = float(args.eta_val)
        else:
            raise SystemExit("Erreur: fournir soit (--mchirp-val ET --eta-val) soit (--mass1 ET --mass2).")

        snr_ts, h_fd, st_fd, Pk = snr_time_and_template(
            strain_ts, P, mb, eb, args.flow, args.fhigh, args.model
        )
        best_snr = float(np.nanmax(np.asarray(snr_ts.numpy())))

        out["detectors"][det] = {
            "mass1": float(args.mass1),
            "mass2": float(args.mass2),
            "mchirp": float(mb),
            "eta": float(eb),
            "snr_max": float(best_snr)
        }
        snr_store[det] = snr_ts

        # Couche A_k (optionnelle)
        if args.ak:
            ak, rhos = ak_populations_from_strain(
                stilde=st_fd, P=Pk, htilde=h_fd,
                flow=args.flow, fhigh=args.fhigh,
                Nc=args.Ncar, norm=args.ak_norm
            )
            if args.s_rescale is not None:
                ak, rhos = apply_s_rescale(ak, rhos, s=float(args.s_rescale))

            ak_export["A_k"][det] = {"ak": ak, "rhos_rel": rhos}

            # Also reflect in main out for quick visibility
            out.setdefault("ak_summary", {})
            out["ak_summary"].setdefault(det, {})
            out["ak_summary"][det]["rhos_rel"] = rhos
            if args.s_rescale is not None:
                out.setdefault("s_rescale", float(args.s_rescale))

            # Optionnel: trace Ak
            if args.ak_plot:
                import matplotlib.pyplot as plt
                f = np.array(ak["f"], dtype=float)
                A0 = np.array(ak["A0"], dtype=float)
                Ap1 = np.array(ak["Ap1"], dtype=float)
                Am1 = np.array(ak["Am1"], dtype=float)
                plt.figure(figsize=(9,4))
                plt.plot(f, Ap1, label="A+1")
                plt.plot(f, Am1, label="A-1")
                plt.plot(f, A0, linestyle="--", label="A0 (ref)")
                plt.xlabel("Fréquence de modulation f (Hz)")
                plt.ylabel("Population A_k (u.a.)")
                plt.title(f"{args.event} — {det} — A_k(f)")
                plt.legend()
                plt.tight_layout()
                outpng = args.ak_plot_prefix or f"ak_{args.event}_{det}.png"
                plt.savefig(outpng, dpi=150)
                plt.close()
                ak_export["A_k"][det]["plot"] = outpng


    # Plot SNR(t)
    if args.plot and snr_store:
        plt.figure(figsize=(9, 4))
        for det, snrts in snr_store.items():
            arr = np.asarray(snrts.numpy(), dtype=float)
            t = np.arange(arr.size) * float(snrts.delta_t) - 8.0
            plt.plot(t, arr, label=det)
        plt.xlabel("Temps relatif (s)")
        plt.ylabel("SNR(t)")
        plt.title(f"{args.event} — SNR(t)")
        plt.legend()
        plt.tight_layout()
        outpng = args.plot_file or f"snr_{args.event}.png"
        plt.savefig(outpng, dpi=150)
        out["plot"] = outpng

    # Export A_k si demandé
    if args.ak and ak_export["A_k"]:
        ak_path = args.ak_file or f"ak_{args.event}.json"
        with open(ak_path, "w") as f:
            json.dump(ak_export, f, indent=2)
        out["ak_file"] = ak_path

    # Sortie JSON principale
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
