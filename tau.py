#!/usr/bin/env python3
import json
from datetime import datetime, timedelta
import numpy as np
import cmath

CLUSTERS_FILE = "spectral_invariants.json"
ANCHOR_EVENT = "GW151226"
ANCHOR_DATE  = datetime(2015, 12, 26)  # date LIGO officielle : 2017-08-23

# Facteurs optimisés pour avoir une échelle temporelle réaliste
K_FACTOR = 500.0  # Augmenté pour étaler sur plusieurs années
I_FACTOR = 0.001  # Réduit pour éviter les oscillations trop rapides
J_FACTOR = 0.0    # Terme constant

# ----------------------------------------------------
# Chargement + reconstruction τ
# ----------------------------------------------------
def load_tau():
    with open(CLUSTERS_FILE,"r") as f:
        data = json.load(f)

    tau = {}
    for ev, event_data in data.items():
        # Utilise tau_s et neff depuis le JSON
        tau_raw = event_data["tau_s"]
        neff = event_data.get("nu_eff_Hz", 1.0)  # Si neff n'existe pas, utilise 1.0 par défaut
        
        # Transformation complexe améliorée
        # Au lieu de K * cos(neff * tau_raw * I) + J, nous utilisons une échelle logarithmique
        complex_exponent = cmath.exp(neff * tau_raw * I_FACTOR * 1j)
        real_part = complex_exponent.real
        
        # Transformation non-linéaire pour étaler sur plusieurs années
        # Nous utilisons une combinaison de la partie réelle et de tau_raw lui-même
        scaled_tau = K_FACTOR * tau_raw + J_FACTOR * real_part
        
        tau[ev] = scaled_tau

    return tau

# ----------------------------------------------------
def tau_to_date(t, t_ref):
    return ANCHOR_DATE + timedelta(days=(t - t_ref))

# ----------------------------------------------------
# Version alternative avec échelle temporelle linéaire
# ----------------------------------------------------
def load_tau_linear():
    """Version avec transformation linéaire simple pour comparaison"""
    with open(CLUSTERS_FILE,"r") as f:
        data = json.load(f)

    tau = {}
    for ev, event_data in data.items():
        tau_raw = event_data["tau_s"]
        
        # Échelle linéaire : 1 jour de τ = 365 jours réels
        scaling_factor = 365.0
        tau[ev] = scaling_factor * tau_raw

    return tau

# ----------------------------------------------------
# Détection géodésique par pentes
# ----------------------------------------------------
def find_slope_chains(ev_sorted, tolerance=0.15):
    """
    On clusterise par SLOPE : 
    slope_i = tau[i+1] - tau[i]
    Deux gaps appartiennent à la même géodésique si
    gap[i+1]/gap[i] ≈ 1 (± tolerance)
    """
    chains = []
    chain = [ev_sorted[0]]

    prev_gap = None

    for i in range(1, len(ev_sorted)):
        ev_prev, τ_prev = ev_sorted[i-1]
        ev_curr, τ_curr = ev_sorted[i]

        gap = τ_curr - τ_prev

        if prev_gap is None:
            prev_gap = gap
            chain.append(ev_sorted[i])
            continue

        ratio = gap / prev_gap if prev_gap != 0 else 1e9

        if (1 - tolerance) <= ratio <= (1 + tolerance):
            # même pente
            chain.append(ev_sorted[i])
        else:
            chains.append(chain)
            chain = [ev_sorted[i]]

        prev_gap = gap

    chains.append(chain)
    return chains

# ----------------------------------------------------
# Prédiction future par extrapolation du dernier chain
# ----------------------------------------------------
def predict_future(chain):
    if len(chain) < 3:
        return None

    τ1 = chain[-3][1]
    τ2 = chain[-2][1]
    τ3 = chain[-1][1]

    slope = τ3 - τ2
    τ_next = τ3 + slope

    return τ_next, slope

# ----------------------------------------------------
if __name__ == "__main__":

    print("=== TAU ORDRE - TRANSFORMATION COMPLEXE ===\n")
    print(f"Paramètres: K={K_FACTOR}, I={I_FACTOR}, J={J_FACTOR}")
    
    tau = load_tau()
    tau_ref = tau[ANCHOR_EVENT]
    
    ev_sorted = sorted(tau.items(), key=lambda x: x[1])
    for ev, τ in ev_sorted:
        print(f"{ev:20s} τ={τ:10.6f} → {tau_to_date(τ, tau_ref)}")
    
    print(f"\n=== TAU ORDRE - TRANSFORMATION LINÉAIRE (comparaison) ===\n")
    tau_linear = load_tau_linear()
    tau_linear_ref = tau_linear[ANCHOR_EVENT]
    
    ev_sorted_linear = sorted(tau_linear.items(), key=lambda x: x[1])
    for ev, τ in ev_sorted_linear:
        print(f"{ev:20s} τ={τ:10.6f} → {tau_to_date(τ, tau_linear_ref)}")
    
    # Statistiques
    print(f"\n=== STATISTIQUES ===")
    tau_values = list(tau.values())
    print(f"Plage τ (complexe): {min(tau_values):.3f} à {max(tau_values):.3f} jours")
    print(f"Étendue: {max(tau_values) - min(tau_values):.3f} jours")
    
    tau_linear_values = list(tau_linear.values())
    print(f"Plage τ (linéaire): {min(tau_linear_values):.3f} à {max(tau_linear_values):.3f} jours")
    print(f"Étendue: {max(tau_linear_values) - min(tau_linear_values):.3f} jours")
