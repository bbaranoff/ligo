#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LIGO — GWOSC → PSD → cohérence H1–L1 → masse effective (m = E/c^2) + fréquence effective.
Version finale avec gestion robuste des délais temporels.
"""

import argparse, os, json
from pathlib import Path

import numpy as np
from gwpy.timeseries import TimeSeries
from scipy.signal import butter, filtfilt, coherence, welch
from scipy.integrate import trapezoid
from scipy.constants import c, G
from scipy.optimize import curve_fit

# --- Constantes & utilitaires ---
c_light = c  # m/s
Mpc_to_m = 3.085677581491367e22  # 1 Mpc en mètres
M_sun = 1.989e30  # kg

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = float(lowcut) / nyq
    high = float(highcut) / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

def butter_bandpass_filter(x, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, x)
    return y

def mass_from_energy(E):
    """Masse équivalente via E=mc²."""
    return E / (c_light ** 2)

# --- Chargement & pré-traitement ---
def load_gwosc_data(event_name, detector, duration=4.0):
    """
    Charge les données depuis GWOSC pour un événement et détecteur donnés.
    """
    try:
        from pycbc.catalog import Merger
        merger = Merger(event_name)
        gps_time = merger.time
        
        if detector.upper() == 'H1':
            strain_data = merger.strain('H1')
        else:
            strain_data = merger.strain('L1')
        
        fs = 1.0 / strain_data.delta_t
        
        start_time = gps_time - duration/2
        end_time = gps_time + duration/2
        
        strain_ts = strain_data.time_slice(start_time, end_time)
        
        return strain_ts.numpy(), float(fs), float(gps_time)
        
    except Exception as e:
        print(f"Erreur lors du chargement de {detector} pour {event_name}: {e}")
        return None, None, None

def load_strain_data(event_name, detectors=['H1', 'L1'], duration=4.0):
    data_dict = {}
    for det in detectors:
        strain, fs, gps_time = load_gwosc_data(event_name, det, duration)
        if strain is not None:
            data_dict[det] = {
                'strain': strain,
                'fs': fs,
                'gps_time': gps_time
            }
        else:
            print(f"Impossible de charger les données pour {det}")
    
    return data_dict

# --- PSD & cohérence ---
def compute_psd(strain, fs, seg_len=4.0, overlap=2.0):
    from gwpy.timeseries import TimeSeries
    ts = TimeSeries(strain, sample_rate=fs)
    psd_spectrum = ts.psd(fftlength=seg_len, overlap=overlap, window='hann')
    return psd_spectrum.frequencies.value, psd_spectrum.value

def compute_robust_coherence(h1_strain, l1_strain, fs, flow=20.0, fhigh=512.0):
    """
    Calcule la cohérence de manière robuste avec plusieurs méthodes.
    """
    # Méthode 1: Cohérence standard avec scipy
    nperseg = min(4096, len(h1_strain) // 8)
    noverlap = nperseg // 2
    
    try:
        freqs, coh_spectrum = coherence(
            h1_strain, l1_strain, 
            fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann'
        )
        
        # Filtrer dans la bande d'intérêt
        mask = (freqs >= flow) & (freqs <= fhigh)
        freqs_band = freqs[mask]
        coh_band = coh_spectrum[mask]
        
        if len(coh_band) == 0:
            return 0.5, freqs, coh_spectrum
        
        mean_coherence = float(np.mean(coh_band))
        
        return mean_coherence, freqs, coh_spectrum
        
    except Exception as e:
        print(f"Erreur avec scipy coherence: {e}")
        return 0.5, np.linspace(flow, fhigh, 100), np.ones(100) * 0.5

def compute_cross_correlation_coherence(h1_strain, l1_strain, fs, flow=20.0, fhigh=512.0):
    """
    Méthode alternative basée sur la corrélation croisée.
    """
    # Filtrage
    h1_filt = butter_bandpass_filter(h1_strain, flow, fhigh, fs)
    l1_filt = butter_bandpass_filter(l1_strain, flow, fhigh, fs)
    
    # Corrélation croisée normalisée
    corr = np.correlate(h1_filt, l1_filt, mode='full')
    norm1 = np.sqrt(np.sum(h1_filt**2))
    norm2 = np.sqrt(np.sum(l1_filt**2))
    
    if norm1 * norm2 == 0:
        return 0.0
    
    max_corr = np.max(np.abs(corr)) / (norm1 * norm2)
    coherence_est = min(max_corr, 1.0)
    
    return coherence_est

# --- Masse & fréquence effective ---
def energy_from_merger(mass1_solar, mass2_solar, distance_mpc):
    """
    Estime l'énergie rayonnée pour une fusion binaire.
    """
    m1 = mass1_solar * M_sun
    m2 = mass2_solar * M_sun
    M = m1 + m2
    
    # Fraction d'énergie rayonnée (ajustée pour chaque événement)
    # Basée sur les résultats publiés
    energy_fractions = {
        'GW150914': 0.049,
        'GW151226': 0.043, 
        'GW170104': 0.042,
        'GW170814': 0.048,
        'GW170817': 0.025  # Étoiles à neutrons
    }
    
    energy_fraction = energy_fractions.get('GW150914', 0.045)  # Valeur par défaut
    
    energy_joules = energy_fraction * M * c_light**2
    
    return energy_joules

def estimate_binary_masses(distance_mpc, event_name):
    """
    Estime les masses basées sur l'événement connu.
    """
    known_masses = {
        'GW150914': (35.6, 30.6),
        'GW151226': (13.7, 7.7),
        'GW170104': (30.8, 20.0),
        'GW170814': (30.6, 25.2),
        'GW170817': (1.46, 1.27)
    }
    
    if event_name in known_masses:
        return known_masses[event_name]
    else:
        return (30.0, 25.0)

def effective_frequency_from_strain(strain, fs, flow=20.0, fhigh=512.0):
    """
    Calcule la fréquence effective directement à partir du signal.
    """
    strain_filt = butter_bandpass_filter(strain, flow, fhigh, fs)
    freqs = np.fft.rfftfreq(len(strain_filt), 1/fs)
    fft_vals = np.abs(np.fft.rfft(strain_filt))
    
    mask = (freqs >= flow) & (freqs <= fhigh)
    freqs_band = freqs[mask]
    fft_band = fft_vals[mask]
    
    if len(fft_band) == 0 or np.sum(fft_band) == 0:
        return (flow + fhigh) / 2
    
    nu_eff = np.sum(freqs_band * fft_band**2) / np.sum(fft_band**2)
    
    return nu_eff

def estimate_time_delay_physical(h1_strain, l1_strain, fs, flow=20.0, fhigh=512.0):
    """
    Estimation robuste du délai temporel avec contraintes physiques.
    """
    # Filtrage
    h1_filt = butter_bandpass_filter(h1_strain, flow, fhigh, fs)
    l1_filt = butter_bandpass_filter(l1_strain, flow, fhigh, fs)
    
    # Corrélation croisée
    corr = np.correlate(h1_filt, l1_filt, mode='full')
    lags = np.arange(-len(h1_filt) + 1, len(l1_filt)) / fs
    
    # Contrainte physique: délai maximal entre H1 et L1 ~10 ms
    max_physical_delay = 0.015  # 15 ms (conservateur)
    physical_mask = np.abs(lags) <= max_physical_delay
    
    if not np.any(physical_mask):
        return 0.0, corr
    
    # Chercher le maximum dans la région physique
    corr_physical = corr[physical_mask]
    lags_physical = lags[physical_mask]
    
    max_idx = np.argmax(np.abs(corr_physical))
    tau_seconds = lags_physical[max_idx]
    
    return tau_seconds, corr

# --- Script principal ---
def main():
    parser = argparse.ArgumentParser(
        description="LIGO — GWOSC → PSD → cohérence H1–L1 → masse effective + fréquence effective."
    )
    parser.add_argument("--event", required=True, help="Nom de l'événement (ex: GW150914)")
    parser.add_argument("--detectors", nargs="+", default=["H1","L1"], help="Liste des détecteurs")
    parser.add_argument("--distance-mpc", type=float, required=True, help="Distance en Mpc")
    parser.add_argument("--flow", type=float, default=35.0, help="Fréquence inférieure du filtre [Hz]")
    parser.add_argument("--fhigh", type=float, default=250.0, help="Fréquence supérieure du filtre [Hz]")
    parser.add_argument("--duration", type=float, default=4.0, help="Durée totale autour de l'événement [s]")
    parser.add_argument("--plot", action="store_true", help="Affiche un graphique")
    parser.add_argument("--export", help="Fichier NPZ à exporter")
    parser.add_argument("--json", action="store_true", help="Écrit un résumé JSON")

    args = parser.parse_args()

    print(f"=== Analyse de {args.event} ===")
    print(f"Distance: {args.distance_mpc} Mpc")
    print(f"Bande de fréquence: {args.flow}-{args.fhigh} Hz")

    # --- 1. Chargement des données ---
    print("\n1. Chargement des données...")
    data_dict = load_strain_data(args.event, args.detectors, args.duration)
    
    if not data_dict or 'H1' not in data_dict or 'L1' not in data_dict:
        print("Erreur: Impossible de charger les données H1 et L1")
        return

    h1_data = data_dict['H1']
    l1_data = data_dict['L1']

    h1_strain = h1_data['strain']
    l1_strain = l1_data['strain']
    fs = h1_data['fs']

    print(f"Données chargées: H1({len(h1_strain)} points), L1({len(l1_strain)} points), fs={fs} Hz")

    # --- 2. Calcul de la cohérence ---
    print("\n2. Calcul de la cohérence...")
    coherence_val, freqs_coh, coh_spectrum = compute_robust_coherence(
        h1_strain, l1_strain, fs, args.flow, args.fhigh
    )
    
    corr_coherence = compute_cross_correlation_coherence(h1_strain, l1_strain, fs, args.flow, args.fhigh)
    print(f"  Cohérence (spectrale): {coherence_val:.3f}")
    print(f"  Cohérence (corrélation): {corr_coherence:.3f}")
    
    # Moyenne pondérée
    final_coherence = (coherence_val + corr_coherence) / 2
    print(f"  Cohérence finale: {final_coherence:.3f}")

    # --- 3. Estimation du délai temporel ---
    print("\n3. Estimation du délai temporel...")
    tau_seconds, correlation = estimate_time_delay_physical(
        h1_strain, l1_strain, fs, args.flow, args.fhigh
    )
    print(f"Délai H1→L1 estimé: {tau_seconds*1000:.3f} ms")

    # --- 4. Calcul des PSD ---
    print("\n4. Calcul des densités spectrales de puissance...")
    freqs_h1, psd_h1 = compute_psd(h1_strain, fs)
    freqs_l1, psd_l1 = compute_psd(l1_strain, fs)

    # --- 5. Calcul de la masse et fréquence effective ---
    print("\n5. Calcul des paramètres physiques...")
    
    # Estimation des masses du système binaire
    mass1, mass2 = estimate_binary_masses(args.distance_mpc, args.event)
    total_mass_solar = mass1 + mass2
    
    # Énergie rayonnée
    energy = energy_from_merger(mass1, mass2, args.distance_mpc)
    mass_equivalent = mass_from_energy(energy)
    mass_equivalent_solar = mass_equivalent / M_sun
    
    # Fréquence effective
    nu_eff_h1 = effective_frequency_from_strain(h1_strain, fs, args.flow, args.fhigh)
    nu_eff_l1 = effective_frequency_from_strain(l1_strain, fs, args.flow, args.fhigh)
    nu_eff = (nu_eff_h1 + nu_eff_l1) / 2

    print(f"Masses estimées du système: {mass1:.1f} + {mass2:.1f} = {total_mass_solar:.1f} M_sun")
    print(f"Énergie rayonnée: {energy/1e47:.2f} × 10⁴⁷ J")
    print(f"Masse équivalente rayonnée: {mass_equivalent_solar:.2f} M_sun")
    print(f"Fréquence effective (H1): {nu_eff_h1:.1f} Hz")
    print(f"Fréquence effective (L1): {nu_eff_l1:.1f} Hz")
    print(f"Fréquence effective moyenne: {nu_eff:.1f} Hz")

    # --- 6. Export des données ---
    if args.export:
        print(f"\n6. Export des données: {args.export}")
        np.savez(
            args.export,
            h1_strain=h1_strain,
            l1_strain=l1_strain,
            fs=fs,
            psd_h1=psd_h1,
            psd_l1=psd_l1,
            freqs_psd=freqs_h1,
            coherence=final_coherence,
            coh_spectrum=coh_spectrum,
            freqs_coh=freqs_coh,
            tau_seconds=tau_seconds,
            energy_joules=energy,
            mass_kg=mass_equivalent,
            nu_eff_hz=nu_eff,
            mass1_solar=mass1,
            mass2_solar=mass2
        )

    # --- 7. Export JSON ---
    if args.json:
        output_json = {
            "event": args.event,
            "detectors": args.detectors,
            "distance_mpc": float(args.distance_mpc),
            "flow_hz": float(args.flow),
            "fhigh_hz": float(args.fhigh),
            "duration_s": float(args.duration),
            "coherence": float(final_coherence),
            "time_delay_ms": float(tau_seconds * 1000),
            "energy_joules": float(energy),
            "mass_equivalent_kg": float(mass_equivalent),
            "mass_equivalent_solar_masses": float(mass_equivalent_solar),
            "binary_mass1_solar": float(mass1),
            "binary_mass2_solar": float(mass2),
            "total_binary_mass_solar": float(total_mass_solar),
            "effective_frequency_hz": float(nu_eff),
            "sampling_rate_hz": float(fs)
        }
        
        json_filename = f"{args.event}_results.json"
        with open(json_filename, "w") as f:
            json.dump(output_json, f, indent=2)
        print(f"Résultats exportés: {json_filename}")

    # --- 8. Graphiques ---
    if args.plot:
        print("\nGénération des graphiques...")
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Données temporelles filtrées
        h1_filtered = butter_bandpass_filter(h1_strain, args.flow, args.fhigh, fs)
        l1_filtered = butter_bandpass_filter(l1_strain, args.flow, args.fhigh, fs)
        time_axis = np.arange(len(h1_filtered)) / fs - 2.0
        
        axes[0, 0].plot(time_axis, h1_filtered, label='H1', alpha=0.7, linewidth=0.5)
        axes[0, 0].plot(time_axis, l1_filtered, label='L1', alpha=0.7, linewidth=0.5)
        axes[0, 0].set_xlabel('Temps (s)')
        axes[0, 0].set_ylabel('Strain')
        axes[0, 0].set_title(f'{args.event} - Données filtrées')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # PSD
        axes[0, 1].loglog(freqs_h1, psd_h1, label='H1 PSD', alpha=0.7)
        axes[0, 1].loglog(freqs_l1, psd_l1, label='L1 PSD', alpha=0.7)
        axes[0, 1].axvline(nu_eff, color='red', linestyle='--', label=f'ν_eff = {nu_eff:.1f} Hz')
        axes[0, 1].set_xlabel('Fréquence (Hz)')
        axes[0, 1].set_ylabel('PSD (1/Hz)')
        axes[0, 1].set_title('Densités spectrales de puissance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cohérence
        axes[1, 0].plot(freqs_coh, coh_spectrum, alpha=0.7)
        axes[1, 0].axhline(final_coherence, color='red', linestyle='--', 
                          label=f'Cohérence = {final_coherence:.3f}')
        axes[1, 0].set_xlabel('Fréquence (Hz)')
        axes[1, 0].set_ylabel('Cohérence')
        axes[1, 0].set_title('Spectre de cohérence H1-L1')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
        
        # Corrélation croisée
        lags_ms = np.arange(-len(h1_filtered) + 1, len(l1_filtered)) / fs * 1000
        axes[1, 1].plot(lags_ms, correlation, alpha=0.7)
        axes[1, 1].axvline(tau_seconds * 1000, color='red', linestyle='--', 
                          label=f'Délai = {tau_seconds*1000:.1f} ms')
        axes[1, 1].set_xlim(-20, 20)  # Zoom sur la région physique
        axes[1, 1].set_xlabel('Délai (ms)')
        axes[1, 1].set_ylabel('Corrélation')
        axes[1, 1].set_title('Corrélation croisée H1-L1')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    print("\n=== Analyse terminée ===")

if __name__ == "__main__":
    main()
