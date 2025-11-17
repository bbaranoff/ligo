#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN

DAY = 86400.0  # secondes → jours

# ---------------------------------------------------------------------
# Chargement des événements
# ---------------------------------------------------------------------

def load_all_events(path="results/*.json"):
    events = {}

    for file in glob.glob(path):
        with open(file, "r") as f:
            data = json.load(f)

        # Il faut τ et ν
        if "tau_s" not in data or "nu_eff_Hz" not in data:
            continue

        # GPS : soit dans le JSON, soit via GWOSC
        if "gps" in data:
            gps = float(data["gps"])
            date = datetime.utcfromtimestamp(gps)
        else:
            try:
                import gwosc.datasets as ds
                gps = ds.event_gps(data["event"])
                date = datetime.utcfromtimestamp(gps)
            except Exception:
                continue

        events[data["event"]] = {
            "file": file,
            "tau_s": float(data["tau_s"]),
            "nu": float(data["nu_eff_Hz"]),
            "I": float(data["tau_s"]) * float(data["nu_eff_Hz"]),
            "date": date,
            "gps": gps
        }

    return events


# ---------------------------------------------------------------------
# ΔT observé et ΔT prédit
# ---------------------------------------------------------------------

def deltaT_obs(evA, evB):
    dt = abs(evB["gps"] - evA["gps"])
    return dt, dt / DAY

def deltaT_pred(T0, lam):
    dt = T0 * (np.exp(lam) - 1.0)
    return dt, dt / DAY


# ---------------------------------------------------------------------
# Matrice λ
# ---------------------------------------------------------------------

def build_lambda_matrix(events):
    names = list(events.keys())
    n = len(names)
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            L[i, j] = abs(events[names[i]]["I"] - events[names[j]]["I"])

    return names, L


# ---------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------

def clusterize(L, eps=0.35):
    db = DBSCAN(eps=eps, min_samples=2, metric="precomputed")
    labels = db.fit_predict(L)
    return labels


# ---------------------------------------------------------------------
# Calibration T₀ par cluster
# ---------------------------------------------------------------------

def calibrate_cluster_T0(events, names, labels):
    clusters = {}

    for lab in set(labels):
        if lab == -1:
            continue  # outliers

        idx = [i for i, x in enumerate(labels) if x == lab]
        members = [names[i] for i in idx]

        T0_list = []

        for i in range(len(members)):
            for j in range(i+1, len(members)):
                A = events[members[i]]
                B = events[members[j]]

                lam = abs(A["I"] - B["I"])
                dt_obs_s, _ = deltaT_obs(A, B)

                denom = np.exp(lam) - 1
                if denom <= 0:
                    continue

                T0_est = dt_obs_s / denom
                if T0_est > 0:
                    T0_list.append(T0_est)

        if len(T0_list) == 0:
            continue

        T0_final = np.median(T0_list)

        # IMPORTANT : clé → string (PATCH JSON)
        clusters[str(lab)] = {
            "members": members,
            "T0": T0_final,
            "T0_days": T0_final / DAY,
            "pairs": []
        }

    return clusters


# ---------------------------------------------------------------------
# Remplissage des prédictions ΔT
# ---------------------------------------------------------------------

def fill_cluster_predictions(events, clusters):
    for lab, C in clusters.items():
        T0 = C["T0"]
        members = C["members"]

        for i in range(len(members)):
            for j in range(i+1, len(members)):
                A = members[i]
                B = members[j]

                evA = events[A]
                evB = events[B]

                lam = abs(evA["I"] - evB["I"])
                dt_obs_s, dt_obs_d = deltaT_obs(evA, evB)
                dt_pred_s, dt_pred_d = deltaT_pred(T0, lam)

                C["pairs"].append({
                    "A": A,
                    "B": B,
                    "lam": lam,
                    "dt_obs_days": dt_obs_d,
                    "dt_pred_days": dt_pred_d
                })


# ---------------------------------------------------------------------
# Export JSON
# ---------------------------------------------------------------------

def export_json(clusters, out="spectral_clusters.json"):
    with open(out, "w") as f:
        json.dump(clusters, f, indent=2)
    print(f"\n📦 Export JSON → {out}")


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------

def plot_lambda_matrix(names, L):
    plt.figure(figsize=(7,6))
    plt.imshow(L, cmap="magma", interpolation="nearest")
    plt.colorbar(label="λ = |I_A - I_B|")
    plt.xticks(range(len(names)), names, rotation=90)
    plt.yticks(range(len(names)), names)
    plt.tight_layout()
    plt.savefig("lambda_matrix.png")
    plt.close()
    print("📈 lambda_matrix.png généré.")

def plot_clusters(names, labels, events):
    x = [events[n]["I"] for n in names]
    y = [events[n]["nu"] for n in names]

    plt.figure(figsize=(7,6))
    sc = plt.scatter(x, y, c=labels, cmap="tab20", s=90)
    for i, n in enumerate(names):
        plt.text(x[i], y[i], n, fontsize=8)
    plt.xlabel("Invariant I = τ × ν")
    plt.ylabel("ν_effective (Hz)")
    plt.title("Clusters spectraux")
    plt.colorbar(sc)
    plt.tight_layout()
    plt.savefig("clusters.png")
    plt.close()
    print("📈 clusters.png généré.")

def plot_dt_validation(clusters):
    obs = []
    pred = []

    for C in clusters.values():
        for p in C["pairs"]:
            obs.append(p["dt_obs_days"])
            pred.append(p["dt_pred_days"])

    plt.figure(figsize=(6,6))
    plt.scatter(obs, pred, s=20, alpha=0.7)
    if len(obs) > 0:
        M = max(max(obs), max(pred))
        plt.plot([0, M], [0, M], "r--", label="y = x")

    plt.xlabel("ΔT_obs (jours)")
    plt.ylabel("ΔT_pred (jours)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("deltaT_validation.png")
    plt.close()
    print("📈 deltaT_validation.png généré.")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("📡 Chargement des événements...\n")
    events = load_all_events()

    if len(events) < 3:
        print("❌ Pas assez d'événements.")
        exit(1)

    names, L = build_lambda_matrix(events)
    labels = clusterize(L, eps=0.35)

    print(f"{len(events)} événements chargés.\n")
    print("=== Clusters trouvés ===")

    for lab in set(labels):
        subset = [names[i] for i,k in enumerate(labels) if k == lab]
        print(f"Cluster {lab}: {subset}")

    clusters = calibrate_cluster_T0(events, names, labels)
    fill_cluster_predictions(events, clusters)

    export_json(clusters)

    plot_lambda_matrix(names, L)
    plot_clusters(names, labels, events)
    plot_dt_validation(clusters)

    print("\n🎉 Terminé !")
