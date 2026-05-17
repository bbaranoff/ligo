"""Paths physiques pour estimer l'énergie rayonnée d'une fusion compacte.

Chaque path utilise un SOUS-ENSEMBLE DIFFÉRENT d'observables et a une erreur
intrinsèque que la calibration par classe peut corriger.

Path A est la référence (E = ΔM·c²), les autres utilisent des observables
indirectes : f_ringdown, f_peak, distance, masse totale.

Chaque path est rattaché à une transformation de notre orchestrateur :
- B : g09_thermality (Hawking T from QNM)
- C : phenom_inspiral (g05_interferometry conceptuel)
- D : g04_dispersion (peak frequency → mass)
- E : g03_propagation + g10_cosmological_T (luminosity at distance)
- F : g08_entropy (Bekenstein-Hawking area)
- G : y11_cosmological_holography (info-theoretic bound)
"""

import numpy as np

# Constantes
M_sun = 1.989e30
c = 2.998e8
G = 6.674e-11
hbar = 1.055e-34
k_B = 1.381e-23
l_P = 1.616e-35

# η canonique inspiral non-spinant q=1 (Buonanno-Cook-Pretorius 2007)
ETA_CANONICAL = 0.048


def path_A_reference(M_initial, M_final, **kw):
    """A — RÉFÉRENCE : E = ΔM · c² depuis (M_i, M_f). Ground truth, pas calibrable."""
    return (M_initial - M_final) * M_sun * c**2


def path_B_qnm_ringdown(f_ringdown_Hz, M_initial, **kw):
    """B — Via QNM ringdown : M_f depuis f_ringdown, puis ΔM = M_i - M_f.

    Berti-Cardoso-Will 2009 : ω_QNM(l=m=2, n=0) ≈ 0.3737 c³/(G·M_f) [a*=0].
    Erreur intrinsèque ~10-20% selon spin réel du remnant.

    Observable : f_ringdown
    Transformation pierre : g09_thermality
    """
    if f_ringdown_Hz <= 0:
        return 0.0
    omega_QNM_re = 0.3737
    M_f_kg = omega_QNM_re * c**3 / (G * 2 * np.pi * f_ringdown_Hz)
    M_f_sun = M_f_kg / M_sun
    delta_m = M_initial - M_f_sun
    return delta_m * M_sun * c**2 if delta_m > 0 else 0.0


def path_C_eta_phenomenological(M_initial, **kw):
    """C — Fraction phénoménologique : E = η · M_initial · c².

    η_canonical = 4.8% (BBH médian non-spinant q=1).
    Calibration par classe :
    - ULTRA_LIGHT/LIGHT → η réel petit (BNS) → facteur ~0.05-0.5
    - MEDIUM → η ≈ canonical, facteur ~1
    - MASSIVE → η réel plus élevé (spins), facteur ~1.2-1.6

    Observable : M_initial seulement
    """
    return ETA_CANONICAL * M_initial * M_sun * c**2


def path_D_chirp_from_peak(f_peak_Hz, **kw):
    """D — Masse totale depuis f_peak puis E = η · M_tot · c².

    f_peak,GW ≈ 0.1 c³/(π G M_tot) (phénoménologique).
    Erreur ~30-50% avant calibration (compounded f_peak ↔ M et η).

    Observable : f_peak seulement
    Transformation pierre : g04_dispersion
    """
    if f_peak_Hz <= 0:
        return 0.0
    M_tot_kg = 0.1 * c**3 / (np.pi * G * f_peak_Hz)
    return ETA_CANONICAL * M_tot_kg * c**2


def path_E_luminosity_distance(f_peak_Hz, distance_Mpc, M_initial, **kw):
    """E — Luminosité GW depuis amplitude à distance D.

    h_peak ≈ 4(G M_chirp/c²)² ω²_GW/(D c²),  M_chirp ≈ 0.435·M_total (q=1).
    L_peak ≈ (c⁵/G) (h_peak D)² × π/4 (corr. géométrique).
    E ≈ L_peak × (5/f_peak).

    Observables : f_peak, distance, M_initial
    Transformations pierre : g03_propagation + g10_cosmological_T
    """
    if f_peak_Hz <= 0 or distance_Mpc <= 0 or M_initial <= 0:
        return 0.0
    D_m = distance_Mpc * 3.086e22
    omega_GW = 2 * np.pi * f_peak_Hz
    M_chirp_kg = 0.435 * M_initial * M_sun
    h_peak = 4.0 * (G * M_chirp_kg / c**2)**2 * omega_GW**2 / (D_m * c**2)
    L_peak = (c**5 / G) * h_peak**2 * D_m**2 * np.pi / 4
    T_sig = 5.0 / f_peak_Hz
    return L_peak * T_sig


def path_F_bekenstein_area(f_ringdown_Hz, M_initial, **kw):
    """F — Différence d'aire d'horizon via Bekenstein-Hawking.

    On infère M_f via QNM, puis : E_rad = ΔM·c² avec
    ΔM ∝ (√A_i − √A_f) puisque A ∝ M².

    Variante du path B mais on passe par les aires/entropie. Algébriquement
    identique pour Schwarzschild isolé, mais la SCALE LSQ correctrice peut
    différer si on calibre différemment (la formule d'aire pèse plus les BH
    massifs car A ∝ M²).

    Observable : f_ringdown, M_initial
    Transformation pierre : g08_entropy
    """
    if f_ringdown_Hz <= 0:
        return 0.0
    omega_QNM_re = 0.3737
    M_f_kg = omega_QNM_re * c**3 / (G * 2 * np.pi * f_ringdown_Hz)
    M_i_kg = M_initial * M_sun
    if M_f_kg >= M_i_kg:
        return 0.0
    A_i = 16 * np.pi * G**2 * M_i_kg**2 / c**4
    A_f = 16 * np.pi * G**2 * M_f_kg**2 / c**4
    # ΔM via aire (équivalent à path B mais explicite via S)
    sqrt_factor = np.sqrt(c**4 / (16 * np.pi * G**2))
    delta_M = (np.sqrt(A_i) - np.sqrt(A_f)) * sqrt_factor
    return delta_M * c**2


def path_G_holographic_info(M_initial, M_final, **kw):
    """G — Borne holographique de Bekenstein.

    Borne : S ≤ 2π R E /(ℏc) pour système de taille R, énergie E.
    Pour BH : saturé avec S = A/(4ℓ_P²), R = r_s.

    E_holographic_radié = S_i · ℏc/(2π R_i) - S_f · ℏc/(2π R_f)
    qui se simplifie à E = (M_i - M_f) c² × (facteur géométrique) → coïncide
    avec ΔM·c² au facteur de saturation près.

    Le facteur "holographique" devient une CONSTANTE par classe en calibration.

    Observable : M_initial, M_final
    Transformation pierre : y11_cosmological_holography
    """
    M_i_kg = M_initial * M_sun
    M_f_kg = M_final * M_sun if M_final > 0 else M_i_kg * 0.95
    R_i = 2 * G * M_i_kg / c**2
    R_f = 2 * G * M_f_kg / c**2
    S_i = 4 * np.pi * R_i**2 / (4 * l_P**2)
    S_f = 4 * np.pi * R_f**2 / (4 * l_P**2)
    E_i = S_i * hbar * c / (2 * np.pi * R_i)
    E_f = S_f * hbar * c / (2 * np.pi * R_f)
    return abs(E_i - E_f)


# Registre
PATHS = {
    'A_reference':            path_A_reference,
    'B_qnm_ringdown':         path_B_qnm_ringdown,
    'C_eta_phenom':           path_C_eta_phenomenological,
    'D_chirp_from_peak':      path_D_chirp_from_peak,
    'E_luminosity':           path_E_luminosity_distance,
    'F_bekenstein_area':      path_F_bekenstein_area,
    'G_holographic':          path_G_holographic_info,
}

CALIBRATABLE_PATHS = [k for k in PATHS if k != 'A_reference']


def estimate_all_paths(event_data: dict) -> dict:
    """Applique tous les paths à un événement. Retourne dict[path_name] -> E_J."""
    results = {}
    kw = dict(
        M_initial=event_data['M_initial'],
        M_final=event_data['M_final'],
        distance_Mpc=event_data['distance_Mpc'],
        f_peak_Hz=event_data['f_peak_Hz'],
        f_ringdown_Hz=event_data['f_ringdown_Hz'],
    )
    for name, fn in PATHS.items():
        try:
            val = float(fn(**kw))
            if np.isfinite(val) and val > 0:
                results[name] = val
            else:
                results[name] = 0.0
        except Exception:
            results[name] = 0.0
    return results
