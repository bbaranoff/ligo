#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import trapezoid
import matplotlib.cm as cm

# Crée le dossier pour sauvegarder les plots
os.makedirs("plots", exist_ok=True)

# Configuration des graphiques
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 12,
    "figure.max_open_warning": False
})

def load_spec(path):
    """Charge les données spectrales depuis un fichier JSON."""
    with open(path, "r") as f:
        d = json.load(f)
    if "freq_Hz" in d and "dEdf_internal" in d:
        return d["freq_Hz"], d["dEdf_internal"], d.get("distance_mpc", None)
    else:
        raise ValueError(f"Clés 'freq_Hz' ou 'dEdf_internal' manquantes dans {path}.")

def plot_and_save_combined(json_files, plot_type, filename):
    """Génère un plot combiné pour tous les événements d'un type donné."""
    fig = plt.figure()
    colors = cm.rainbow(np.linspace(0, 1, len(json_files)))

    for json_file, color in zip(json_files, colors):
        freq, dEdf, _ = load_spec(json_file)
        event_name = os.path.basename(json_file).replace('.json', '')

        if plot_type == "spectrum":
            plt.loglog(freq, gaussian_filter1d(dEdf, sigma=2), color=color, label=event_name)
            plt.xlabel("Fréquence (Hz)")
            plt.ylabel(r"dE/df (J/Hz)")
            plt.title("Spectres d'énergie (tous les événements)")
        elif plot_type == "cumulative":
            cumulative_energy = np.cumsum(dEdf)
            plt.semilogx(freq, cumulative_energy, color=color, label=event_name)
            plt.xlabel("Fréquence (Hz)")
            plt.ylabel("Énergie cumulée (J)")
            plt.title("Énergie cumulée (tous les événements)")
        elif plot_type == "normalized":
            total_energy = trapezoid(dEdf, freq)
            normalized_dEdf = dEdf / total_energy
            plt.loglog(freq, normalized_dEdf, color=color, label=event_name)
            plt.xlabel("Fréquence (Hz)")
            plt.ylabel("dE/df normalisé")
            plt.title("Spectres normalisés (tous les événements)")
        elif plot_type == "distribution":
            plt.hist(freq, weights=dEdf, bins=50, color=color, alpha=0.5, label=event_name, density=True)
            plt.xlabel("Fréquence (Hz)")
            plt.ylabel("Densité d'énergie")
            plt.title("Distribution des fréquences (tous les événements)")

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{filename}", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_peaks_and_distance(json_files):
    """Génère les graphiques des pics d'énergie et énergie vs distance."""
    # Comparaison des pics d'énergie
    fig_pics = plt.figure()
    for json_file in json_files:
        freq, dEdf, _ = load_spec(json_file)
        event_name = os.path.basename(json_file).replace('.json', '')
        peak_idx = np.argmax(dEdf)
        peak_freq = freq[peak_idx]
        plt.scatter(peak_freq, dEdf[peak_idx], label=f"{event_name} (pic à {peak_freq:.1f} Hz)")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Énergie maximale (J/Hz)")
    plt.title("Comparaison des pics d'énergie (tous les événements)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/pics_energie.png", dpi=300, bbox_inches='tight')
    plt.close(fig_pics)

    # Énergie vs distance
    fig_distance = plt.figure()
    for json_file in json_files:
        freq, dEdf, distance = load_spec(json_file)
        if distance is not None:
            total_energy = trapezoid(dEdf, freq)
            event_name = os.path.basename(json_file).replace('.json', '')
            plt.scatter(distance, total_energy, color='green', label=event_name)
    plt.xlabel("Distance (Mpc)")
    plt.ylabel("Énergie totale (J)")
    plt.title("Énergie en fonction de la distance (tous les événements)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/energie_vs_distance.png", dpi=300, bbox_inches='tight')
    plt.close(fig_distance)

def main(json_dir="results"):
    """Génère tous les graphiques combinés pour les fichiers JSON dans le répertoire spécifié."""
    json_files = glob.glob(os.path.join(json_dir, "*.json"))
    if not json_files:
        raise FileNotFoundError(f"Aucun fichier JSON trouvé dans {json_dir}.")

    # Génère un plot combiné pour chaque type de courbe
    plot_and_save_combined(json_files, "spectrum", "spectres_combines.png")
    plot_and_save_combined(json_files, "cumulative", "energie_cumulee_combinee.png")
    plot_and_save_combined(json_files, "normalized", "spectres_normalises_combines.png")
    plot_and_save_combined(json_files, "distribution", "distribution_frequences_combinee.png")

    # Graphiques comparatifs
    plot_peaks_and_distance(json_files)

if __name__ == "__main__":
    main()
