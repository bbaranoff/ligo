#!/usr/bin/env python3
from datetime import datetime, timezone
import numpy as np
from scipy.interpolate import PchipInterpolator

import tau              # pour load_tau()
import tau_newton       # pour invert_tau_to_t()

# ------------------------------------------------------
# Dates LIGO (ancres)
# ------------------------------------------------------
LIGO_DATES = {
    "GW150914":        "2015-09-14 09:50:45",
    "GW151226":        "2015-12-26 03:38:53",
    "GW170104":        "2017-01-04 10:11:58",
    "GW170608":        "2017-06-08 02:01:16",
    "GW170729":        "2017-07-29 18:56:29",
    "GW170809":        "2017-08-09 08:28:21",
    "GW170814":        "2017-08-14 10:30:43",
    "GW170817":        "2017-08-17 12:41:04",
    "GW170818":        "2017-08-18 02:25:09",
    "GW170823":        "2017-08-23 13:13:37",
    "GW190412":        "2019-04-12 05:30:44",
    "GW190403_051519": "2019-04-03 05:15:19",
    "GW190413_052954": "2019-04-13 05:29:54",
    "GW190413_134308": "2019-04-13 13:43:08",
    "GW190421_213856": "2019-04-21 21:38:56",
    "GW190503_185404": "2019-05-03 18:54:04",
    "GW190514_065416": "2019-05-14 06:54:16",
    "GW190517_055101": "2019-05-17 05:51:01",
    "GW190519_153544": "2019-05-19 15:35:44",
    "GW190521":        "2019-05-21 03:02:29",
    "GW190521_074359": "2019-05-21 07:43:59",
    "GW190602_175927": "2019-06-02 17:59:27",
    "GW190828_063405": "2019-08-28 06:34:05",
    "GW190828_065509": "2019-08-28 06:55:09",
}

def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()

# ------------------------------------------------------
# 1. Charger τ
# ------------------------------------------------------
tau_vals = tau.load_tau()

# ------------------------------------------------------
# 2. Calculer t_newton(τ) pour chaque event
# ------------------------------------------------------
tau_anchor = tau_vals[tau_newton.ANCHOR_EVENT]
C = tau_newton.compute_C(tau_anchor)

t_newton_vals = {}

for ev, τ in tau_vals.items():
    t_newton_vals[ev] = tau_newton.invert_tau_to_t(τ, C)

# ------------------------------------------------------
# 3. Construire les jeux d’ancres
# ------------------------------------------------------
tau_anchor_x = []
t_ligo_y    = []

tnewton_anchor_x = []
t_ligo_y2        = []

for ev, date_str in LIGO_DATES.items():
    if ev not in tau_vals:
        continue
    tL = parse_date(date_str)

    tau_anchor_x.append(tau_vals[ev])
    tnewton_anchor_x.append(t_newton_vals[ev])
    t_ligo_y.append(tL)
    t_ligo_y2.append(tL)

# Tri par τ pour g₁
tau_anchor_x = np.array(tau_anchor_x)
t_ligo_y = np.array(t_ligo_y)

order1 = np.argsort(tau_anchor_x)
tau_anchor_x = tau_anchor_x[order1]
t_ligo_y = t_ligo_y[order1]

# Tri par t_newton pour g₂
tnewton_anchor_x = np.array(tnewton_anchor_x)
t_ligo_y2 = np.array(t_ligo_y2)

order2 = np.argsort(tnewton_anchor_x)
tnewton_anchor_x = tnewton_anchor_x[order2]
t_ligo_y2 = t_ligo_y2[order2]

# ------------------------------------------------------
# 4. Construire les interpolations monotones
# ------------------------------------------------------
g1 = PchipInterpolator(tau_anchor_x, t_ligo_y)      # τ → t_LIGO
g2 = PchipInterpolator(tnewton_anchor_x, t_ligo_y2) # t_newton → t_LIGO
print(t_ligo_y)
print(t_ligo_y2)
# ------------------------------------------------------
# 5. Moyenne croisée : g(τ) = ½[g₁(τ) + g₂(t_newton(τ))]
# ------------------------------------------------------
print("=== CALIBRATION CROISÉE τ & τ_newton ===\n")
print(f"{'Event':20s} {'τ':>10s} {'t_newton':>12s} {'t_pred(UTC)':>25s}")

for ev, τ in sorted(tau_vals.items(), key=lambda x: x[1]):
    tN = t_newton_vals[ev]
    t_pred = 0.5*(g1(τ) + g2(tN))
    dt = datetime.fromtimestamp(float(t_pred), tz=timezone.utc)
    print(f"{ev:20s} {τ:10.3f} {tN:12.3f} {str(dt):>25s}")
