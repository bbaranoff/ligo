#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import matplotlib.cm as cm

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 12
})
"""
def load_spec(path):
    with open(path, "r") as f:
        d = json.load(f)
    if "freq_Hz" in d and "dEdf_J_Hz" in d:
        return np.array(d["freq_Hz"], float), np.array(d["dEdf_J_Hz"], float)
    return None, None
"""
def load_spec(path):
    with open(path, "r") as f:
        d = json.load(f)

    if "freq_Hz" not in d or "dEdf_J_Hz" not in d:
        return None, None

    f = np.array(d["freq_Hz"], float)
    dEdf = np.array(d["dEdf_J_Hz"], float)

    # --- normalisation par énergie totale ---
    E_total = trapezoid(dEdf, f)
    if E_total == 0 or np.isnan(E_total):
        return f, dEdf * 0  # cas degueu / vide

    dEdf_norm = dEdf / E_total

    return f, dEdf_norm

files = sorted(glob.glob(os.path.join("results", "*.json")))
if not files:
    print("⚠️  Aucun spectre trouvé dans ./results/")
    raise SystemExit(1)

plt.figure()
colors = cm.viridis(np.linspace(0, 1, len(files)))
plotted = 0

for i, jf in enumerate(files):
    f_Hz, dEdf = load_spec(jf)
    if f_Hz is None or np.all(dEdf <= 0):
        continue
        
    # Normalisation et lissage
    E = trapezoid(np.maximum(dEdf, 0), f_Hz)
    spec_norm = np.maximum(dEdf / max(E, 1e-30), 1e-50)
    sigma = max(1, len(spec_norm)//300)
    smooth = gaussian_filter1d(np.log10(spec_norm), sigma=sigma)
    
    plt.loglog(f_Hz, 10**smooth, color=colors[i], lw=1.5, alpha=0.8)
    plotted += 1

plt.xlabel("Fréquence (Hz)")
plt.ylabel(r"$dE/df\,/\,E_{tot}$ (1/Hz)")
plt.title(f"Spectres d'énergie normalisés - {plotted} événements")
plt.tight_layout()
plt.savefig("plots/all_spectra_simple.png", dpi=200, bbox_inches='tight')

print(f"✅ {plotted} spectres tracés")
