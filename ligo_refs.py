"""Références LIGO/Virgo extraites des catalogues publics GWTC-1/2/3.

Valeurs point-estimate (médianes des distributions postérieures publiées).
Source: GWTC-1 (PRX 9, 031040), GWTC-2 (PRX 11, 021053), GWTC-3 (PRX 13, 011048).

Le but n'est pas d'avoir les valeurs exactes au 4e chiffre — c'est d'avoir
un benchmark suffisamment représentatif pour tester nos paths.
"""

import numpy as np

# Constantes
M_sun = 1.989e30
c = 2.998e8
G = 6.674e-11

# Format: (event, M_initial, M_final, distance_Mpc, f_peak_Hz, f_ringdown_Hz)
# où M_initial = M_total source frame, M_final = mass du remnant
LIGO_EVENTS = [
    # ===== ULTRA_LIGHT (ΔM < 1.0 M☉) =====
    ("GW170608",      18.0,  17.15,    340,  450,  280),
    ("GW190924_021846", 13.9, 13.6,    570,  500,  330),
    ("GW190707_093326", 19.7, 18.9,    770,  400,  300),
    ("GW190720_000836", 16.1, 15.6,    760,  430,  310),
    ("GW190728_064510", 20.6, 19.7,    870,  410,  295),
    # ===== LIGHT (1.0 ≤ ΔM < 1.5 M☉) =====
    ("GW151226",      21.5,  20.5,     440,  450,  320),
    ("GW190412",      37.3,  35.8,     740,  280,  220),
    ("GW190512_180714", 23.4, 22.1,    1430, 330,  240),
    ("GW190708_232457", 30.4, 29.0,    870,  290,  225),
    # ===== MEDIUM (1.5 ≤ ΔM < 4.0 M☉) =====
    ("GW150914",      65.3,  62.3,     430,  250,  250),
    ("GW170104",      51.0,  48.7,     960,  230,  225),
    ("GW170809",      59.7,  57.0,     1030, 240,  235),
    ("GW170814",      55.9,  53.2,     600,  240,  235),
    ("GW170823",      68.7,  65.6,     1850, 220,  215),
    ("GW190408_181802", 47.4, 45.1,    1540, 250,  240),
    ("GW190503_185404", 71.7, 68.2,    1410, 215,  205),
    ("GW190630_185205", 59.0, 56.0,    870,  235,  225),
    # ===== MASSIVE (ΔM ≥ 4.0 M☉) =====
    ("GW190521",      150.0, 142.0,    5300, 80,   75),
    ("GW170729",      80.0,  75.3,     2840, 200,  190),
    ("GW190602_175927", 116.0, 110.0,  2840, 110,  100),
    ("GW190519_153544", 106.0, 100.5,  2400, 120,  115),
    ("GW190929_012149", 102.0, 96.0,   2050, 130,  120),
]


def get_class(delta_m: float) -> str:
    """Classification adaptative basée sur la masse rayonnée."""
    if delta_m < 1.0:
        return "ULTRA_LIGHT"
    elif delta_m < 1.5:
        return "LIGHT"
    elif delta_m < 4.0:
        return "MEDIUM"
    else:
        return "MASSIVE"


def get_reference_data():
    """Retourne dict[event] -> dict avec valeurs computées."""
    data = {}
    for (ev, M_i, M_f, D, f_pk, f_rd) in LIGO_EVENTS:
        delta_m = M_i - M_f
        # Énergie de référence LIGO (E = ΔM·c²)
        E_J_ref = delta_m * M_sun * c**2
        msun_c2_ref = delta_m  # par définition
        data[ev] = {
            'M_initial': M_i,
            'M_final': M_f,
            'delta_m': delta_m,
            'distance_Mpc': D,
            'f_peak_Hz': f_pk,
            'f_ringdown_Hz': f_rd,
            'energy_J_ref': E_J_ref,
            'msun_c2_ref': msun_c2_ref,
            'class': get_class(delta_m),
        }
    return data


def summary():
    """Affiche la distribution par classe."""
    data = get_reference_data()
    from collections import defaultdict
    classes = defaultdict(list)
    for ev, d in data.items():
        classes[d['class']].append(d['delta_m'])
    print("Distribution par classe adaptative :")
    for cls in ["ULTRA_LIGHT", "LIGHT", "MEDIUM", "MASSIVE"]:
        if cls in classes:
            arr = classes[cls]
            print(f"  {cls:12s} : n={len(arr):2d}  ΔM range=[{min(arr):.2f}, {max(arr):.2f}] M☉  "
                  f"E range=[{min(arr)*M_sun*c**2:.2e}, {max(arr)*M_sun*c**2:.2e}] J")


if __name__ == '__main__':
    summary()
