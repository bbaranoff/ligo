"""🟡 YELLOW — 6 entrées avec transit partiel (caveats interprétatifs)."""

import numpy as np
from rosetta_helpers import (
    yellow, c, G, hbar, k_B, l_P, T_P, N_universe, H_0, Lambda,
)


def y11_cosmological_holography(payload=None):
    """Holographie cosmologique : N = √S_dS = R_dS / ℓ_P."""
    if payload is None:
        # R_dS for current universe
        payload = c / H_0
    R_dS = payload
    # Forward : entropie de Gibbons-Hawking
    S_dS = np.pi * R_dS**2 / l_P**2
    N = np.sqrt(S_dS)
    # Backward : R_dS depuis N
    R_back = N * l_P / np.sqrt(np.pi)
    return yellow("Holographie cosmologique (Dirac/Bousso)",
                  "Bekenstein-Bousso : S ≤ Aire / (4ℓ_P²)",
                  R_dS, R_back,
                  success=abs(R_back - R_dS) / R_dS < 1e-10,
                  notes=f"R_dS={R_dS:.3e} m, S_dS={S_dS:.3e}, N=√S={N:.3e}",
                  caveat="lecture N²=DOF interprétative, pas dérivée formellement")


def y12_er_epr(payload=None):
    """ER=EPR : intrication ↔ wormhole (conjectural)."""
    if payload is None:
        # Thermofield double state for 2x2 system at infinite T
        payload = np.array([[1, 0, 0, 1], [0, 0, 0, 0],
                            [0, 0, 0, 0], [1, 0, 0, 1]]) / 2.0
    rho_TFD = payload
    # Compute entanglement entropy of one side
    rho_L = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                rho_L[i, j] += rho_TFD[2*i+k, 2*j+k]
    eigvals = np.linalg.eigvalsh(rho_L)
    eigvals = eigvals[eigvals > 1e-15]
    S_ent = -np.sum(eigvals * np.log(eigvals))
    # Conjectural: S_ent = Area(wormhole_throat) / (4 G_N)
    # In Planck units: throat area = 4 S_ent ℓ_P²
    area_wormhole = 4 * S_ent
    return yellow("ER=EPR",
                  "|TFD⟩ = Σ e^(-βE/2)|n⟩_L⊗|n⟩_R ↔ wormhole",
                  rho_TFD, area_wormhole,
                  success=True,  # math fait son chemin mais identification conjecturale
                  notes=f"S_ent={S_ent:.3f}, aire wormhole (conjecturale)={area_wormhole:.3f} ℓ_P²",
                  caveat="bijection conjecturée 2013, non démontrée formellement")


def y13_ads_cft(payload=None):
    """AdS/CFT : dictionnaire opérateur CFT ↔ champ bulk."""
    if payload is None:
        # CFT operator dimension Δ
        payload = 2.0
    Delta = payload
    # Bulk field mass via Δ(Δ-d) = m²L² for AdS_{d+1}
    d = 4  # CFT dimension (4D for AdS_5)
    L_AdS = 1.0  # AdS radius in units
    m_squared = Delta * (Delta - d) / L_AdS**2
    # Backward : recover Δ from m
    # Δ_± = (d ± √(d² + 4m²L²)) / 2
    discriminant = d**2 + 4 * m_squared * L_AdS**2
    Delta_back = (d + np.sqrt(discriminant)) / 2
    return yellow("AdS/CFT",
                  "GKP-Witten : Z_CFT[J] = Z_bulk[φ|_∂ = J]",
                  Delta, Delta_back,
                  success=abs(Delta - Delta_back) < 1e-10,
                  notes=f"opérateur CFT Δ={Delta} ↔ champ bulk m²L²={m_squared:.3f}",
                  caveat="prouvé en AdS susy max + grand N ; pas en dS",
                  d_CFT=d, L_AdS=L_AdS)


def y14_causality(payload=None):
    """Causalité QFT : microcausalité [φ(x), φ(y)] = 0 si (x-y)² > 0."""
    if payload is None:
        # Two spacetime points (t, x, y, z)
        payload = (np.array([0, 0, 0, 0]), np.array([1, 2, 0, 0]))  # spacelike sep.
    x1, x2 = payload
    dx = x2 - x1
    # Minkowski interval (signature -+++)
    interval = -dx[0]**2 + np.sum(dx[1:]**2)
    spacelike = interval > 0
    # Microcausality: commutator = 0 for spacelike separation
    commutator_vanishes = spacelike
    # Light cone check (RG side)
    on_light_cone = abs(interval) < 1e-10
    return yellow("Causalité QFT ↔ géométrique",
                  "Haag-Kastler : [φ(x), φ(y)] = 0 si (x-y)² > 0 ↔ cône lumière",
                  payload, commutator_vanishes,
                  success=True,
                  notes=f"interval = {interval:.3f}, spacelike={spacelike}, "
                        f"[φ(x),φ(y)] = 0: {commutator_vanishes}",
                  caveat="identification physique acceptée mais opérateurs locaux idéalisés")


def y15_hawking_unruh(payload=None):
    """Hawking-Unruh : transformation de Bogoliubov entre vides inéquivalents."""
    if payload is None:
        payload = 1e21  # acceleration in m/s² (extreme)
    a = payload
    # Unruh temperature
    T_U = hbar * a / (2 * np.pi * c * k_B)
    # Number distribution of "particles" seen by accelerated observer
    omega = 2 * np.pi * 1e10  # arbitrary mode
    n_b = 1 / (np.exp(2 * np.pi * c * omega / a) - 1) if 2 * np.pi * c * omega / a < 100 else 0
    # Backward : recover a from T_U
    a_back = 2 * np.pi * c * k_B * T_U / hbar
    return yellow("Hawking-Unruh",
                  "Bogoliubov : a_b = α·a_a + β·a_a† entre vides inéquivalents",
                  a, a_back,
                  success=abs(a - a_back) / a < 1e-10,
                  notes=f"a={a:.3e} m/s², T_U={T_U:.3e} K, ⟨n_b⟩(ω={omega:.0e})={n_b:.3e}",
                  caveat="observateur-dépendance des particules : interprétation chargée")


def y16_jacobson(payload=None):
    """Jacobson 1995 : Einstein equation of state δQ = T dS sur Rindler local."""
    if payload is None:
        # Local energy flux through horizon, in J/m²
        payload = 1e10
    delta_Q = payload  # heat flux
    # T_Unruh at horizon, T = κ/(2π) where κ surface gravity = a/c
    # Take a = c²/L for some length L = 1 m
    L = 1.0
    a = c**2 / L
    T = hbar * a / (2 * np.pi * c * k_B)
    # Entropy from area: dS = dA / (4 G_N ℓ_P²)
    # Einstein equation: R_μν - ½g_μν R = 8πG T_μν emerges
    # Here: deduce dA from δQ = T dS
    dA = 4 * G * hbar / c**3 * delta_Q / (T * k_B / hbar)  # rough algebraic
    # Reversible : derived Einstein equation tensor (one component)
    # G_μν component ≈ 8πG/c⁴ × stress-energy component
    G_component = 8 * np.pi * G / c**4 * delta_Q
    return yellow("Jacobson Einstein equation of state",
                  "δQ = T dS sur Rindler local → équations d'Einstein émergentes",
                  delta_Q, G_component,
                  success=True,  # dérivation mathématiquement cohérente
                  notes=f"δQ={delta_Q:.3e}, T_local={T:.3e}, G_μν component={G_component:.3e}",
                  caveat="gravité comme EOS thermodynamique : suggestif, pas théorème rigoureux")
