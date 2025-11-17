import numpy as np
from datetime import datetime, timedelta, timezone
from math import cos

# ---------- Données τ ----------
TAU = {
    'GW190517_055101': 0.000,
    'GW170608': 81.818,
    'GW190828_063405': 87.515,
    'GW170104': 99.704,
    'GW190519_153544': 104.847,
    'GW190403_051519': 106.700,
    'GW190521': 108.931,
    'GW190828_065509': 137.816,
    'GW190412': 143.338,
    'GW190521_074359': 145.592,
    'GW170814': 155.619,
    'GW190814': 180.542,
    'GW190503_185404': 207.598,
    'GW170818': 251.727,
    'GW170729': 288.929,
    'GW190602_175927': 296.247,
    'GW190421_213856': 298.370,
    'GW150914': 420.614,
    'GW170809': 490.064,
    'GW190413_052954': 630.680,
    'GW190413_134308': 670.454,
    'GW170817': 690.999,
    'GW151226': 1502.594,
    'GW170823': 2108.994,
}

# ---------- DATES LIGO ----------
def dt(x):
    return datetime.fromisoformat(x).replace(tzinfo=timezone.utc)

DATES_LIGO = {
    'GW190517_055101': dt("2019-05-17T05:51:01"),
    'GW170608': dt("2017-06-08T02:01:16"),
    'GW190828_063405': dt("2019-08-28T06:34:05"),
    'GW170104': dt("2017-01-04T10:11:58"),
    'GW190519_153544': dt("2019-05-19T15:35:44"),
    'GW190403_051519': dt("2019-04-03T05:51:19"),
    'GW190521': dt("2019-05-21T03:02:29"),
    'GW190828_065509': dt("2019-08-28T06:55:09"),
    'GW190412': dt("2019-04-12T05:30:44"),
    'GW190521_074359': dt("2019-05-21T07:43:59"),
    'GW170814': dt("2017-08-14T10:30:43"),
    'GW190814': dt("2019-08-14T21:10:39"),
    'GW190503_185404': dt("2019-05-03T18:54:04"),
    'GW170818': dt("2017-08-18T02:25:09"),
    'GW170729': dt("2017-07-29T18:56:29"),
    'GW190602_175927': dt("2019-06-02T17:59:27"),
    'GW190421_213856': dt("2019-04-21T21:38:56"),
    'GW150914': dt("2015-09-14T09:50:45"),
    'GW170809': dt("2017-08-09T08:28:21"),
    'GW190413_052954': dt("2019-04-13T05:29:54"),
    'GW190413_134308': dt("2019-04-13T13:43:08"),
    'GW170817': dt("2017-08-17T12:41:04"),
    'GW151226': dt("2015-12-26T03:38:53"),
    'GW170823': dt("2017-08-23T13:13:58"),
}

# ---------- Paramètres du modèle complexe ----------
A = 62375532.0
Q = 108.931  # feuilletage
t0 = datetime(2017, 5, 25, 7, 18, 49, tzinfo=timezone.utc)

# ω déjà trouvé comme quasi-optimal auparavant
OMEGA_FIXED = 0.04558576429404901


def t_raw(omega, phi, tau):
    """Temps brut (sans feuilletage)."""
    return t0 + timedelta(seconds=float(A * cos(omega * tau + phi)))


def unwrap(omega, phi, event):
    """
    Applique le feuilletage en multiples de Q jours
    pour rapprocher t_model de la date LIGO.
    """
    tau = TAU[event]
    target = DATES_LIGO[event]
    base = t_raw(omega, phi, tau)

    best = base
    best_err = abs((base - target).total_seconds())

    # on teste quelques feuillets autour (−40..+40 par ex.)
    for k in range(-40, 41):
        cand = base + timedelta(days=Q * k)
        err = abs((cand - target).total_seconds())
        if err < best_err:
            best = cand
            best_err = err

    return best, best_err


def cost_phi(phi):
    """Erreur quadratique globale pour un φ donné (ω fixé)."""
    tot = 0.0
    for e in TAU:
        _, err = unwrap(OMEGA_FIXED, phi, e)
        tot += err * err
    return tot


def main():
    # --------- recherche 1D sur φ ----------
    # On sait déjà que φ est petit, on scanne par ex. [-0.5, 0.5]
    phi_vals = np.linspace(-0.5, 0.5, 400)

    best_cost = 1e99
    best_phi = None

    for phi in phi_vals:
        c = cost_phi(phi)
        if c < best_cost:
            best_cost = c
            best_phi = phi

    print("ω fixé =", OMEGA_FIXED)
    print("φ_opt =", best_phi)
    print()

    # --------- Affichage des prédictions ----------
    print("=== MODELE COMPLEXE (ω fixé, φ optimal) ===\n")
    errs_days = []

    for e in sorted(TAU, key=lambda x: TAU[x]):
        pred, err_sec = unwrap(OMEGA_FIXED, best_phi, e)
        real = DATES_LIGO[e]
        dt_days = (pred - real).total_seconds() / 86400.0
        errs_days.append(abs(dt_days))
        print(
            f"{e:20s}  τ={TAU[e]:8.3f} → {pred.isoformat()}  "
            f"Δt={dt_days:8.3f} j"
        )

    print("\nErreur moyenne :", float(np.mean(errs_days)), "j")
    print("Erreur max     :", float(np.max(errs_days)), "j")


if __name__ == "__main__":
    main()
