#!/usr/bin/env python3
import json
from datetime import timedelta
from tau import load_tau, ANCHOR_EVENT, ANCHOR_DATE

# ----------------------------------------------------
# Charger les données (clusters + τ global)
# ----------------------------------------------------
with open("spectral_clusters_with_T0.json", "r") as f:
    clusters = json.load(f)

tau = load_tau()
tau_ref = tau[ANCHOR_EVENT]

# ----------------------------------------------------
def tau_to_date(t):
    return ANCHOR_DATE + timedelta(days=(t - tau_ref))

# ----------------------------------------------------
# Préparer la liste globale de tous les évènements
# ----------------------------------------------------
all_events = []
for cid, c in clusters.items():
    for e in c["members"]:
        if e not in all_events:
            all_events.append(e)

# ----------------------------------------------------
print("\n=== GÉODÉSIQUES COMPLÈTES (Δτ | Δt) ===\n")

for cid, cluster in clusters.items():
    members = cluster["members"]
    print(f"--- Cluster {cid} ---")

    # Pour chaque A dans ce cluster
    for A in members:
        τA = tau[A]
        tA = tau_to_date(τA)

        # On calcule vers tous les B possibles (tous clusters)
        for B in all_events:
            if A == B:
                continue

            τB = tau[B]
            tB = tau_to_date(τB)

            dτ = τB - τA
            dt = (tB - tA).total_seconds() / 86400.0

            print(f"{A:15s} → {B:15s} | Δτ={dτ:10.3f}  Δt={dt:10.3f}")

    print()  # séparation clusters
