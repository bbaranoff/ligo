"""Extract observables from LIGO NPZ files for benchmark pipeline.

Lit les NPZ (data/npz/{event}_{H1,L1}.npz), applique le signal processing
de `ligo_signal.py` pour mesurer :
- f_peak : fréquence du pic spectral cohérent H1-L1 (whitened)
- f_ringdown : fréquence dominante dans la queue post-merger
- h_peak : amplitude strain crête dans la fenêtre signal
- tau_HL : délai H1-L1 (cross-corr sub-sample)
- snr_matched : SNR matched-filter contre template chirp de la classe

Produit un JSON utilisable par `ligo_bench.py --observed observed.json`.

Usage :
    python ligo_extract.py --npz-dir data/npz \\
                           --event-params event_params.json \\
                           --refs ligo_refs.json \\
                           --out observed.json
"""

import os
import sys
import json
import argparse
import numpy as np

from ligo_signal import (
    get_npz_path, load_npz, crop_array,
    psd_welch_median, bandpass, estimate_delay,
    matched_filter_snr, coherent_energy_density,
    CLASS_PARAMS,
)
from ligo_refs import get_class

M_sun = 1.989e30
c = 2.998e8


def find_peak_frequency(f_use, dEdf, inner_margin=0.2):
    """f_peak via barycentre énergétique sur la bande INNER.

    Avant : argmax(dEdf) — attrape les bords de bandpass où |Hc|²=|H/√S|²
    explose parce que S→0 au cutoff. Donne des piles à fhigh ou flow exactement.

    Maintenant : on retire `inner_margin` × bande des deux côtés, et on prend
    le barycentre ∫ f·dEdf df / ∫ dEdf df sur cette bande inner.
    Robuste aux artefacts d'edge ET aux glitches isolés.

    Avec inner_margin=0.2, sur bande 20-350 : on cherche dans 86-284 Hz.
    """
    if len(dEdf) == 0:
        return 0.0
    f_min = float(f_use[0])
    f_max = float(f_use[-1])
    bw = f_max - f_min
    f_lo_inner = f_min + inner_margin * bw
    f_hi_inner = f_max - inner_margin * bw
    mask = (f_use >= f_lo_inner) & (f_use <= f_hi_inner)
    if not np.any(mask):
        return 0.0
    f_m = f_use[mask]
    d_m = np.maximum(dEdf[mask], 0.0)
    total = float(np.sum(d_m))
    if total <= 0:
        return 0.0
    return float(np.sum(f_m * d_m) / total)


def find_ringdown_frequency(strain_post_merger, fs, flow=100.0, fhigh=500.0):
    """Estime f_ringdown depuis FFT de la fenêtre post-merger.

    On prend les ~50 ms après le pic. Le QNM dominant émerge comme la
    fréquence centrale dans la queue. Pour BBH typiques : 200-300 Hz.
    """
    if len(strain_post_merger) < 32:
        return 0.0
    # Window the post-merger ringdown
    seg = strain_post_merger - np.mean(strain_post_merger)
    N = len(seg)
    if N < 64:
        return 0.0
    win = np.hanning(N)
    S = np.fft.rfft(seg * win)
    f = np.fft.rfftfreq(N, d=1.0 / fs)
    psd = np.abs(S) ** 2
    mask = (f >= flow) & (f <= fhigh)
    if not np.any(mask):
        return 0.0
    f_m = f[mask]
    psd_m = psd[mask]
    # Barycentre énergétique dans la bande (plus robuste qu'argmax)
    total = float(np.sum(psd_m))
    if total <= 0:
        return 0.0
    return float(np.sum(f_m * psd_m) / total)


def estimate_h_peak(strain_filtered, signal_win_s, fs, quantile=99.9):
    """Estime h_peak via quantile élevé du strain filtré (robuste aux glitches).

    On ne prend pas max() car un glitch ponctuel domine. Quantile 99.9% capture
    l'amplitude réelle du signal sans être dominé par 1 sample.
    """
    abs_h = np.abs(strain_filtered)
    if len(abs_h) == 0:
        return 0.0
    return float(np.percentile(abs_h, quantile))


def extract_event_observables(event: str, gps: float, distance_Mpc: float,
                               class_params: dict, npz_dir: str,
                               verbose: bool = False):
    """Mesure tous les observables pour un event depuis ses NPZ."""
    try:
        pH = get_npz_path(event, "H1", npz_dir)
        pL = get_npz_path(event, "L1", npz_dir)
    except FileNotFoundError as e:
        return None, f"NPZ missing: {e}"

    hH, fsH, t0H, _ = load_npz(pH)
    hL, fsL, t0L, _ = load_npz(pL)
    if fsH != fsL:
        return None, "Sample rate mismatch H1/L1"
    fs = fsH

    cp = class_params
    flow, fhigh = cp['flow'], cp['fhigh']
    signal_win = cp['signal_win']
    noise_pad = cp['noise_pad']
    tau_band = cp['tau_band']

    # PSD sur segment bruit (pré-event, padding noise_pad)
    n0 = gps - noise_pad - 400.0
    n1 = gps - noise_pad - 200.0
    if n0 < t0H:
        n0 = t0H
        n1 = min(n0 + 200.0, t0H + len(hH) / fs)
    noiseH = crop_array(hH, fs, t0H, n0, n1)
    noiseL = crop_array(hL, fs, t0L, n0, n1)

    f_psd_H, psd_H = psd_welch_median(noiseH, fs)
    f_psd_L, psd_L = psd_welch_median(noiseL, fs)

    # τ estimation sur fenêtre tau-band
    tau_half = 0.5 * signal_win
    seg_tau_H = crop_array(hH, fs, t0H, gps - tau_half, gps + tau_half)
    seg_tau_L = crop_array(hL, fs, t0L, gps - tau_half, gps + tau_half)
    x_tau = bandpass(seg_tau_H, fs, *tau_band)
    y_tau = bandpass(seg_tau_L, fs, *tau_band)
    tau_HL = estimate_delay(x_tau, y_tau, fs)

    # Signal segment et bandpass full analysis band
    s0 = gps - signal_win / 2.0
    s1 = gps + signal_win / 2.0
    hH_sig = bandpass(crop_array(hH, fs, t0H, s0, s1), fs, flow, fhigh)
    hL_sig = bandpass(crop_array(hL, fs, t0L, s0, s1), fs, flow, fhigh)

    # Coherent H1-L1 spectrum
    f_use, dEdf, E_internal = coherent_energy_density(
        hH_sig, hL_sig, fs, tau_HL, flow, fhigh,
        f_psd_H, psd_H, f_psd_L, psd_L,
    )

    # Observables
    f_peak = find_peak_frequency(f_use, dEdf)
    h_peak = estimate_h_peak(hH_sig, signal_win, fs)

    # Ringdown frequency : on prend les 50 ms post-pic (post-merger)
    post_start = gps
    post_end = gps + 0.05
    post_H = bandpass(crop_array(hH, fs, t0H, post_start, post_end),
                      fs, max(flow, 100), fhigh)
    f_ringdown = find_ringdown_frequency(post_H, fs)

    # NOTE : on ne calcule plus le matched filter SNR ici — la fonction
    # `matched_filter_snr` dans ligo_signal.py a un bug dimensionnel non résolu
    # (facteur 1/fs manquant + normalisation template) qui donnait des SNR ~1e12.
    # Aucun path du bench n'utilise le SNR, donc on s'en passe ici.

    # Matched filter SNR (formule corrigée : facteur 2/fs au lieu de 4)
    M_chirp_template_kg = cp['M_chirp_template_sun'] * M_sun
    try:
        snr_mf, t_mf = matched_filter_snr(
            hH_sig, fs, M_chirp_template_kg, f_psd_H, psd_H, flow, fhigh
        )
    except Exception:
        snr_mf, t_mf = 0.0, 0.0

    obs = {
        'event': event,
        'gps': float(gps),
        'distance_Mpc': float(distance_Mpc),
        'fs': float(fs),
        'tau_HL_s': float(tau_HL),
        'f_peak_Hz_obs': float(f_peak),
        'f_ringdown_Hz_obs': float(f_ringdown),
        'h_peak_obs': float(h_peak),
        'E_internal_spectral': float(E_internal),
        'snr_matched_H1': float(snr_mf),
        't_matched_s': float(t_mf),
    }

    if verbose:
        print(f"  [{event}] f_peak={f_peak:.1f} f_rd={f_ringdown:.1f} "
              f"τ={tau_HL*1000:.2f}ms h_peak={h_peak:.2e} SNR_MF={snr_mf:.2f}")

    return obs, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="data/npz",
                    help="Dossier contenant les fichiers _H1.npz et _L1.npz")
    ap.add_argument("--event-params", default="event_params.json",
                    help="event_params.json du repo (avec GPS, distance, _adaptive_category)")
    ap.add_argument("--refs", default="ligo_refs.json",
                    help="ligo_refs.json du repo (avec msun_c2 référence)")
    ap.add_argument("--out", default="observed.json",
                    help="JSON sortie avec observables mesurés")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--only", nargs='*', default=None,
                    help="N'extraire que ces events (par défaut tous)")
    args = ap.parse_args()

    if not os.path.exists(args.event_params):
        print(f"[FATAL] {args.event_params} not found")
        sys.exit(1)
    if not os.path.exists(args.refs):
        print(f"[FATAL] {args.refs} not found")
        sys.exit(1)

    with open(args.event_params) as f:
        event_params = json.load(f)
    with open(args.refs) as f:
        refs = json.load(f)

    events = [k for k in event_params.keys()
              if k not in ("_meta", "default") and isinstance(event_params[k], dict)]
    if args.only:
        events = [e for e in events if e in args.only]

    print(f"Extraction NPZ pour {len(events)} events depuis {args.npz_dir}")

    extracted = {}
    failures = []

    for ev in events:
        evp = event_params[ev]
        gps = evp.get('gps')
        distance = evp.get('distance_mpc')
        if gps is None or distance is None:
            failures.append((ev, "no gps/distance in event_params"))
            continue

        # Get class from ref (via msun_c2 = ΔM)
        ref = refs.get(ev, {})
        msun_c2 = ref.get('msun_c2')
        if msun_c2 is None:
            failures.append((ev, "no msun_c2 in refs"))
            continue

        cls = get_class(float(msun_c2))
        cp = CLASS_PARAMS[cls]

        obs, err = extract_event_observables(
            ev, float(gps), float(distance), cp, args.npz_dir, args.verbose,
        )
        if obs is None:
            failures.append((ev, err))
            continue

        # Add class and reference for downstream bench
        obs['class'] = cls
        obs['msun_c2_ref'] = float(msun_c2)
        obs['energy_J_ref'] = float(msun_c2) * M_sun * c**2
        # M_initial et M_final : depuis refs si dispo, sinon estimé
        if 'M_initial' in ref:
            obs['M_initial'] = float(ref['M_initial'])
        elif 'total_mass_source' in ref:
            obs['M_initial'] = float(ref['total_mass_source'])
        else:
            # Estimer M_initial depuis f_peak observé : M_tot ≈ 0.1·c³/(πG·f_peak)
            from ligo_paths import G as G_const
            if obs['f_peak_Hz_obs'] > 0:
                M_kg = 0.1 * c**3 / (np.pi * G_const * obs['f_peak_Hz_obs'])
                obs['M_initial'] = M_kg / M_sun
            else:
                obs['M_initial'] = 0.0

        obs['M_final'] = obs['M_initial'] - float(msun_c2)
        extracted[ev] = obs

    print(f"\n  Extraits  : {len(extracted)}")
    print(f"  Échecs    : {len(failures)}")
    if failures:
        for ev, err in failures[:10]:
            print(f"    - {ev}: {err}")
        if len(failures) > 10:
            print(f"    ... +{len(failures)-10} autres")

    with open(args.out, 'w') as f:
        json.dump(extracted, f, indent=2)
    print(f"\n  ✅ {args.out} écrit ({len(extracted)} events)")


if __name__ == '__main__':
    main()
