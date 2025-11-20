#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import numpy as np

DAY = 86400.0  # secondes → jours
FACTOR = 90e8
# ------------------------------------------------------------
# Charger les événements (uniquement tau & nu)
# ------------------------------------------------------------
def load_all_events(path="results/*.json"):
    events = {}

    for file in glob.glob(path):
        with open(file, "r") as f:
            data = json.load(f)

        if "tau_s" not in data or "nu_eff_Hz" not in data:
            continue

        tau = float(data["tau_s"])
        nu  = float(data["nu_eff_Hz"])
        I  = tau * nu

        events[data["event"]] = {
            "tau": tau,
            "nu": nu,
            "I": I
        }

    return events


# ------------------------------------------------------------
# Calcul de T0 pour un cluster
# ------------------------------------------------------------
def compute_T0(members, events):
    """
    Calcule T0 pour un cluster :
    - λ = |I_A - I_B|
    - Δτ_obs = |τA - τB|
    - T0 = Δτ_obs / (exp(λ) - 1)
    """
    T0_list = []

    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            A = members[i]
            B = members[j]

            IA = events[A]["I"]
            IB = events[B]["I"]
            lam = abs(IA - IB)

            dtau = abs(events[A]["tau"] - events[B]["tau"])
            denom = np.exp(lam) - 1

            if denom <= 0:
                continue

            T0 = dtau / denom
            if T0 > 0:
                T0_list.append(T0)

    if not T0_list:
        return None, None

    T0_final = float(np.median(T0_list)) * FACTOR
    return T0_final, T0_final / DAY


# ------------------------------------------------------------
# Prédictions avec T0 calculé
# ------------------------------------------------------------
def deltaT_pred(T0, lam):
    dt = T0 * (np.exp(lam) - 1.0)
    return dt, dt / DAY


def fill_cluster_predictions(events, clusters):
    for lab, C in clusters.items():
        T0 = C["T0"]
        members = C["members"]

        for i in range(len(members)):
            for j in range(i+1, len(members)):
                A = members[i]
                B = members[j]

                lam = abs(events[A]["I"] - events[B]["I"])
                _, dt_pred_d = deltaT_pred(T0, lam)

                C["pairs"].append({
                    "A": A,
                    "B": B,
                    "lam": lam,
                    "t_pred_days": dt_pred_d
                })


# ------------------------------------------------------------
# MAIN avec clusters fixes
# ------------------------------------------------------------
if __name__ == "__main__":
    print("📡 Chargement des événements...\n")
    events = load_all_events()

    if len(events) < 3:
        print("❌ Pas assez d'événements.")
        exit(1)

    # Clusters prédéfinis (remplace par tes clusters réels)
    CLUSTERS = {
        "0": ['GW190403_051519', 'GW170608', 'GW170814', 'GW190503_185404', 'GW170823', 'GW190413_052954', 'GW190413_134308'],
        "1": ['GW190521', 'GW190519_153544', 'GW190412', 'GW190421_213856', 'GW170104', 'GW170809', 'GW151226', 'GW190517_055101'],
        "2": ['GW170729', 'GW190514_065416']
    }

    print("=== Calcul de T0 par cluster ===")
    clusters_output = {}

    for cid, members in CLUSTERS.items():
        print(f"Cluster {cid}: {members}")

        T0, T0_days = compute_T0(members, events)

        if T0 is None:
            print("  ❌ Pas assez de paires exploitables\n")
            continue

        print(f"  → T0 = {T0:.9f} s")
        print(f"  → T0_days = {T0_days:.9f} jours\n")

        clusters_output[cid] = {
            "members": members,
            "T0": T0,
            "T0_days": T0_days,
            "pairs": []
        }

    # Remplir les prédictions
    fill_cluster_predictions(events, clusters_output)

    # Export JSON
    with open("spectral_clusters_with_T0.json", "w") as f:
        json.dump(clusters_output, f, indent=2)

    print("📦 Export JSON → spectral_clusters_with_T0.json")
    print("🎉 Terminé.")
