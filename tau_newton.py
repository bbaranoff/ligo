#!/usr/bin/env python3
import json
import math
from datetime import datetime, timedelta

CLUSTERS_FILE = "spectral_clusters.json"

# -----------------------------------------------------------
# Paramètres du modèle spectral interne τ(t)
# τ(t) = t - (α/ω) cos(ω t) + C
# -----------------------------------------------------------
ALPHA = 0.40
OMEGA = 1.0

# La constante C est choisie pour forcer τ(GW150914) à correspondre
# exactement à t = date_LIGO(GW150914)
ANCHOR_EVENT = "GW150914"
ANCHOR_DATE  = datetime(2015, 9, 14)  # date LIGO officielle : 2015-09-14
ANCHOR_T_LIGO = 0.0  # on définit t=0 pour l'ancre

# -----------------------------------------------------------
# Reconstruction brute des τ depuis spectral_clusters.json
# -----------------------------------------------------------
def load_tau():
    with open(CLUSTERS_FILE, "r") as f:
        data = json.load(f)

    tau = {}

    for cid, cluster in data.items():
        T0 = cluster["T0_days"]

        # Tous les membres = T0
        for ev in cluster["members"]:
            tau[ev] = T0

        # Ajoute dt_pred_days pour chaque paire
        for p in cluster["pairs"]:
            A_ev, B_ev = p["A"], p["B"]
            dt = p["dt_pred_days"]

            tau[B_ev] = T0
            tau[A_ev] = T0 + dt

    # Normalisation : min(tau) → 0
    base = min(tau.values())
    for k in tau:
        tau[k] -= base

    return tau

# -----------------------------------------------------------
# τ(t) = t - (α/ω) cos(ω t) + C
# -----------------------------------------------------------
def tau_model(t, C):
    return t - (ALPHA / OMEGA) * math.cos(OMEGA * t) + C

# Dérivée dτ/dt (pour Newton)
def dtau_dt(t):
    return 1 + ALPHA * math.sin(OMEGA * t)

# -----------------------------------------------------------
# Inversion numérique τ → t via Newton-Raphson
# -----------------------------------------------------------
def invert_tau_to_t(tau_value, C, max_iter=50, tol=1e-12):
    # Initial guess : t ≈ τ
    t = tau_value

    for _ in range(max_iter):
        f = tau_model(t, C) - tau_value
        fp = dtau_dt(t)

        if abs(fp) < 1e-14:
            break

        t_new = t - f / fp
        if abs(t_new - t) < tol:
            return t_new
        t = t_new

    return t  # fallback

# -----------------------------------------------------------
# Calcule la constante C en imposant :
# τ_model(t=0) = τ(GW150914)
# -----------------------------------------------------------
def compute_C(tau_anchor):
    # τ(0) = C - (α/ω) cos(0)  = C - α/ω
    # donc :
    # tau_anchor = C - α/ω
    return tau_anchor + (ALPHA / OMEGA)

# -----------------------------------------------------------
# Conversion t (jours depuis ancre) → date réelle
# -----------------------------------------------------------
def t_to_date(t_value):
    return ANCHOR_DATE + timedelta(days=t_value)

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
if __name__ == "__main__":

    tau_dict = load_tau()
    tau_anchor = tau_dict[ANCHOR_EVENT]

    # Calcule constante C pour que τ_model(0) = τ(GW150914)
    C = compute_C(tau_anchor)

    print("=== TAU ORDRE (INVERSION NEWTON, MODÈLE SPECTRAL) ===\n")

    # Trie par τ croissant
    ev_sorted = sorted(tau_dict.items(), key=lambda x: x[1])

    for ev, tau_value in ev_sorted:

        # inversion τ → t réel
        t_real = invert_tau_to_t(tau_value, C)

        # conversion t → date réelle
        date_real = t_to_date(t_real)

        print(f"{ev:20s}  τ={tau_value:10.3f}  t={t_real:12.6f} j  →  {date_real}")
