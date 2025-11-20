#!/usr/bin/env python3
import json
from datetime import datetime, timedelta
import numpy as np

CLUSTERS_FILE = "spectral_clusters_with_T0.json"
ANCHOR_EVENT = "GW190521"
ANCHOR_DATE  = datetime(2019,5,21,9,50,44,400000)

# ----------------------------------------------------
# Chargement + reconstruction τ
# ----------------------------------------------------
def load_tau():
    with open(CLUSTERS_FILE,"r") as f:
        data = json.load(f)

    tau = {}
    for cid, cluster in data.items():
        T0 = cluster["T0_days"]
        for ev in cluster["members"]:
            tau[ev] = T0
        for p in cluster["pairs"]:
            A, B = p["A"], p["B"]
            dt = p["t_pred_days"]
            tau[B] = T0
            tau[A] = T0 + dt

    # normalisation
    base = min(tau.values())
    for k in tau:
        tau[k] -= base

    return tau

# ----------------------------------------------------
def tau_to_date(t, t_ref):
    return ANCHOR_DATE + timedelta(days=(t - t_ref))

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

    tau = load_tau()
    tau_ref = tau[ANCHOR_EVENT]

    print("=== TAU ORDRE ===\n")
    ev_sorted = sorted(tau.items(), key=lambda x: x[1])
    for ev, τ in ev_sorted:
        print(f"{ev:20s} τ={τ:10.3f} → {tau_to_date(τ, tau_ref)}")

    print("\n=== CHAÎNES PAR PENTE ===")
    chains = find_slope_chains(ev_sorted)

    for i, ch in enumerate(chains):
        print(f"\nChaîne {i}:")
        for ev, τ in ch:
            print(f"  {ev:20s} τ={τ:.3f}")

    print("\n=== PRÉDICTION FUTURE ===")
    last = chains[-1]
    pred = predict_future(last)

    if pred is None:
        print("Chaîne trop courte, impossible de prédire.")
    else:
        τ_next, slope = pred
        date_next = tau_to_date(τ_next, tau_ref)
        print(f"τ_next = {τ_next:.3f}, slope={slope:.3f}")
        print(f"Date future prédite : {date_next}")
