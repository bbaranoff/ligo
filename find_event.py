import numpy as np
from datetime import datetime, timedelta, timezone
from math import cos
import json
import sys,os
# Importer les fonctions de tau.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tau import load_tau

TAU_FULL = load_tau()

# ---------- LIGO dates ----------
def to_dt(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)

DATES_LIGO = {
    'GW190517_055101': to_dt("2019-05-17T05:51:01"),
    'GW170608': to_dt("2017-06-08T02:01:16"),
    'GW190828_063405': to_dt("2019-08-28T06:34:05"),
    'GW170104': to_dt("2017-01-04T10:11:58"),
    'GW190519_153544': to_dt("2019-05-19T15:35:44"),
    'GW190403_051519': to_dt("2019-04-03T05:51:19"),
    'GW190521': to_dt("2019-05-21T03:02:29"),
    'GW190828_065509': to_dt("2019-08-28T06:55:09"),
    'GW190412': to_dt("2019-04-12T05:30:44"),
    'GW190521_074359': to_dt("2019-05-21T07:43:59"),
    'GW170814': to_dt("2017-08-14T10:30:43"),
    'GW190814': to_dt("2019-08-14T21:10:39"),
    'GW190503_185404': to_dt("2019-05-03T18:54:04"),
    'GW170818': to_dt("2017-08-18T02:25:09"),
    'GW170729': to_dt("2017-07-29T18:56:29"),
    'GW190602_175927': to_dt("2019-06-02T17:59:27"),
    'GW190421_213856': to_dt("2019-04-21T21:38:56"),
    'GW150914': to_dt("2015-09-14T09:50:45"),
    'GW170809': to_dt("2017-08-09T08:28:21"),
    'GW190413_052954': to_dt("2019-04-13T05:29:54"),
    'GW190413_134308': to_dt("2019-04-13T13:43:08"),
    'GW170817': to_dt("2017-08-17T12:41:04"),
    'GW151226': to_dt("2015-12-26T03:38:53"),
    'GW170823': to_dt("2017-08-23T13:13:58"),
}

TAU = {ev: TAU_FULL[ev] for ev in TAU_FULL if ev in DATES_LIGO}

# ---------- Paramètres fixes ----------
A = 62375532.0       # amplitude interne
φ = 0.0              # phase brute
Q = 108.931          # FEUILLETAGE CORRECT
t0 = datetime(2017,5,25,7,18,49,tzinfo=timezone.utc)

# ---------- Temps brut complexe ----------
def t_raw(omega, tau):
    return t0 + timedelta(seconds=float(A * cos(omega*tau + φ)))

# ---------- Unwrap secteur Q ----------
def unwrap(omega, event):
    raw = t_raw(omega, TAU[event])
    target = DATES_LIGO[event]
    best = raw
    best_err = abs((raw - target).total_seconds())
    for k in range(-50, 51):
        cand = raw + timedelta(days=Q*k)
        err  = abs((cand - target).total_seconds())
        if err < best_err:
            best = cand
            best_err = err
    return best, best_err

# ---------- Fonction de coût ----------
def cost(omega):
    tot = 0.0
    for e in TAU:
        _, err = unwrap(omega, e)
        tot += err*err
    return tot

# ---------- Recherche ω optimale ----------
search = np.linspace(0.041, 0.051, 6000)
vals   = [cost(w) for w in search]
omega_opt = search[np.argmin(vals)]

print("ω optimal trouvé =", omega_opt)

# ---------- Calcul final ----------
print("\n=== MODELE COMPLEXE + ω OPTIMISE + FEUILLETAGE ===\n")
errs=[]
for e in sorted(TAU, key=lambda x: TAU[x]):
    pred, err = unwrap(omega_opt, e)
    real = DATES_LIGO[e]
    dt   = (pred - real).total_seconds()/86400
    errs.append(abs(dt))
    print(f"{e:20s} τ={TAU[e]:8.3f} → {pred.isoformat()}  Δt={dt:7.3f} j")

print("\nErreur moyenne :", np.mean(errs), "j")
print("Erreur max     :", np.max(errs), "j")
