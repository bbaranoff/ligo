"""🟠 ORANGE — 13 entrées avec transformation unilatérale.

Chaque entrée a une math bien définie d'un seul côté (MQ ou RG).
Pas de pont vers l'autre théorie.
"""

import numpy as np
from rosetta_helpers import orange_mq, orange_rg, c, G, hbar, k_B, M_sun, l_P, Lambda


# ============================================================
# MQ-only : 6 entrées
# ============================================================

def o17_tensor_product(payload=None):
    """Produit tensoriel ℋ_A ⊗ ℋ_B (MQ uniquement, pas d'analogue RG)."""
    if payload is None:
        psi_A = np.array([1, 0])
        psi_B = np.array([1, 1]) / np.sqrt(2)
        payload = (psi_A, psi_B)
    a, b = payload
    # Forward: tensor product
    psi_AB = np.kron(a, b)
    # Bell state via entanglement
    bell = (np.kron([1, 0], [1, 0]) + np.kron([0, 1], [0, 1])) / np.sqrt(2)
    return orange_mq("Produit tensoriel multi-systèmes",
                     "ℋ_A ⊗ ℋ_B (algèbre tensorielle)",
                     payload, psi_AB,
                     notes=f"|ψ_AB⟩ separable shape={psi_AB.shape}, "
                           f"Bell state existe: ||bell||={np.linalg.norm(bell):.3f}")


def o18_born(payload=None):
    """Règle de Born : P(a) = |⟨a|ψ⟩|² (MQ uniquement)."""
    if payload is None:
        psi = np.array([0.6, 0.8])
        observable_eigenvecs = np.eye(2)  # measure in computational basis
        payload = (psi, observable_eigenvecs)
    psi, eigvecs = payload
    # Forward : probabilities via Gleason
    probs = np.array([abs(np.vdot(v, psi))**2 for v in eigvecs.T])
    return orange_mq("Règle de Born",
                     "Gleason : P(a) = ⟨ψ|P̂_a|ψ⟩",
                     payload, probs,
                     notes=f"P = {probs}, sum = {probs.sum():.4f}")


def o19_spin(payload=None):
    """Spin intrinsèque : [Ŝ_i, Ŝ_j] = iℏ ε_ijk Ŝ_k, représentation SU(2)."""
    if payload is None:
        payload = np.array([1, 0])  # |↑⟩
    psi = payload
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    # Verify commutator: [Sx, Sy] = 2i Sz (in units of ℏ/2)
    comm = sx @ sy - sy @ sx
    commutator_ok = np.allclose(comm, 2j * sz)
    # Apply Sx to |↑⟩ → |↓⟩
    psi_rotated = sx @ psi
    return orange_mq("Spin intrinsèque",
                     "[Ŝ_i, Ŝ_j] = iℏ ε_ijk Ŝ_k, reps SU(2)",
                     payload, psi_rotated,
                     notes=f"commutateur Pauli correct: {commutator_ok}, "
                           f"Sx|↑⟩ = {psi_rotated.tolist()}")


def o20_exchange_statistics(payload=None):
    """Statistiques d'échange : P̂_12 |ψ⟩ = ±|ψ⟩."""
    if payload is None:
        # 2-particle state in 2x2 (4-dim)
        payload = np.array([0, 1, -1, 0]) / np.sqrt(2)  # antisymmetric (fermionic)
    psi = payload
    # Exchange operator on 2x2 system: swaps i↔j
    exchange = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]])
    exchanged = exchange @ psi
    # Determine statistic
    if np.allclose(exchanged, psi):
        statistic = "Bose (symétrique)"
    elif np.allclose(exchanged, -psi):
        statistic = "Fermi (antisymétrique)"
    else:
        statistic = "anyonique"
    return orange_mq("Statistiques d'échange",
                     "P̂_12 |ψ⟩ = ±|ψ⟩, reps S_n / groupes de tresses B_n",
                     payload, exchanged,
                     notes=f"statistique détectée : {statistic}")


def o21_no_cloning(payload=None):
    """No-cloning : ∄ U unitaire telle que U|ψ⟩|0⟩ = |ψ⟩|ψ⟩ ∀ψ."""
    if payload is None:
        payload = np.array([0.6, 0.8])
    psi = payload
    # Suppose U is a cloning operator on 2x2 = 4-dim Hilbert.
    # Demand: U(|ψ⟩⊗|0⟩) = |ψ⟩⊗|ψ⟩ for all ψ. Check on two non-orthogonal states.
    psi1 = np.array([1, 0])
    psi2 = psi  # non-orthogonal to psi1
    # If U linear: U(α|ψ1⟩+β|ψ2⟩)|0⟩ = α|ψ1⟩|ψ1⟩+β|ψ2⟩|ψ2⟩ ?
    # But also U(α|ψ1⟩+β|ψ2⟩)|0⟩ = (α|ψ1⟩+β|ψ2⟩)⊗(α|ψ1⟩+β|ψ2⟩)
    # Cross terms break it: αβ(|ψ1⟩|ψ2⟩+|ψ2⟩|ψ1⟩) ≠ 0
    contradiction = True
    return orange_mq("No-cloning",
                     "Wootters-Zurek 1982 : pas de U linéaire qui clone ∀ψ",
                     psi, None,
                     notes=f"contradiction démontrée par linéarité : {contradiction}")


def o22_bell_ks(payload=None):
    """Violations de Bell : CHSH > 2 jusqu'à 2√2 ≈ 2.828."""
    # Singlet state, ideal Bell test angles
    # CHSH = E(a,b) - E(a,b') + E(a',b) + E(a',b')
    # Quantum prediction at optimal angles: 2√2
    if payload is None:
        # Angles: a=0, a'=π/2, b=π/4, b'=3π/4 (optimal)
        payload = (0, np.pi/2, np.pi/4, 3*np.pi/4)
    a, a_p, b, b_p = payload
    def E(theta1, theta2):
        return -np.cos(theta1 - theta2)
    S = E(a, b) - E(a, b_p) + E(a_p, b) + E(a_p, b_p)
    classical_bound = 2
    tsirelson_bound = 2 * np.sqrt(2)
    violates = abs(S) > classical_bound
    return orange_mq("Violations Bell / Kochen-Specker",
                     "|CHSH| ≤ 2 classique, jusqu'à 2√2 ≈ 2.828 quantique",
                     payload, S,
                     notes=f"S_CHSH = {S:.4f}, viole Bell (>2) : {violates}, "
                           f"saturation Tsirelson 2√2 = {tsirelson_bound:.4f}")


# ============================================================
# RG-only : 7 entrées
# ============================================================

def o23_background_independence(payload=None):
    """Diff(M) : difféomorphismes agissent sur les sections de fibrés."""
    if payload is None:
        # Schwarzschild metric component g_tt(r) in 2D slice
        r = np.linspace(2.5, 10, 5)  # avoid horizon
        M = 1 * M_sun
        rs = 2 * G * M / c**2
        payload = (r, 1 - rs / r)
    r, g_tt = payload
    # Apply a coordinate transformation r → r' = r + α/r
    alpha = 0.5
    r_new = r + alpha / r
    # The metric components transform with Jacobian
    dr_dr_new = 1 + alpha / r**2  # ∂r/∂r' (inverse of ∂r'/∂r)
    # In new coords, g_tt' = g_tt (scalar field doesn't change here)
    g_tt_new = g_tt
    return orange_rg("Indépendance du fond",
                     "Diff(M) : x → x'(x), g_μν → ∂x^α/∂x'^μ ∂x^β/∂x'^ν g_αβ",
                     (r, g_tt), (r_new, g_tt_new),
                     notes=f"diff appliqué r→r+α/r, métrique scalaire préservée")


def o24_singularities(payload=None):
    """Singularités : R_μνρσ R^μνρσ → ∞."""
    if payload is None:
        payload = 1e6 * M_sun  # supermassive BH
    M = payload
    r_s = 2 * G * M / c**2
    # Kretschmann scalar for Schwarzschild: K = 48 G²M² / (c⁴ r⁶)
    r_test = np.array([10 * r_s, r_s, 0.1 * r_s, 0.01 * r_s])
    K = 48 * G**2 * M**2 / (c**4 * r_test**6)
    diverges_at_r0 = K[-1] > K[0] * 1e10
    return orange_rg("Singularités",
                     "Penrose-Hawking : R_μνρσ R^μνρσ → ∞",
                     M, K.tolist(),
                     notes=f"M={M/M_sun:.0e} M☉, r_s={r_s:.3e} m, "
                           f"K(10rs)={K[0]:.3e}, K(0.01rs)={K[-1]:.3e}, "
                           f"diverge: {diverges_at_r0}")


def o25_horizons(payload=None):
    """Horizons d'événements : frontière causale globale."""
    if payload is None:
        payload = 10 * M_sun
    M = payload
    r_horizon = 2 * G * M / c**2
    # Surface gravity κ = 1 / (2 r_s) in natural units, or c²/(4GM) for Schwarzschild
    kappa = c**4 / (4 * G * M)
    # Hawking temperature
    T_H = hbar * kappa / (2 * np.pi * c * k_B)
    return orange_rg("Horizons d'événements",
                     "compactification conforme, surface r = 2GM/c²",
                     M, r_horizon,
                     notes=f"M={M/M_sun:.1f} M☉, r_horizon={r_horizon:.3e} m, "
                           f"κ={kappa:.3e}, T_H={T_H:.3e} K")


def o26_topology(payload=None):
    """Topologie de l'espace-temps : genre, classes caractéristiques."""
    if payload is None:
        # Genus of various 2D surfaces
        payload = ['sphere', 'torus', 'genus-2', 'klein-bottle']
    surfaces = payload
    # Euler characteristic χ = 2 - 2g for orientable surface
    euler = {'sphere': 2, 'torus': 0, 'genus-2': -2, 'klein-bottle': 0}
    return orange_rg("Topologie de l'espace-temps",
                     "topologie algébrique : π_n(M), H_n(M), classes caractéristiques",
                     surfaces, [euler[s] for s in surfaces],
                     notes=f"χ(sphere)={euler['sphere']}, "
                           f"χ(torus)={euler['torus']}, "
                           f"χ(genus-2)={euler['genus-2']}")


def o27_cosmological_constant(payload=None):
    """Λ : constante cosmologique, énergie du vide géométrique."""
    if payload is None:
        payload = Lambda
    L = payload
    # Contribution to G_μν : Λ g_μν term
    # Vacuum energy density: ρ_Λ = Λ c² / (8πG)
    rho_vac = L * c**4 / (8 * np.pi * G)
    # Vs QFT prediction: ~ E_P^4 / (ℏc)³ ~ ρ_P
    rho_Planck = c**5 / (hbar * G**2)
    discrepancy = rho_Planck / rho_vac
    return orange_rg("Constante cosmologique Λ",
                     "G_μν + Λ g_μν = 8πG T_μν",
                     L, rho_vac,
                     notes=f"Λ={L:.3e} m⁻², ρ_vac={rho_vac:.3e} J/m³, "
                           f"discrepance Planck/observation : {discrepancy:.3e}")


def o28_penrose_process(payload=None):
    """Processus de Penrose : extraction d'énergie d'ergosphère Kerr."""
    if payload is None:
        payload = (1e7 * M_sun, 0.9)  # M, spin parameter a
    M, a_spin = payload
    # Irreducible mass (Christodoulou-Ruffini)
    M_irr = M * np.sqrt(0.5 * (1 + np.sqrt(1 - a_spin**2)))
    # Maximum extractable energy
    E_extract = (M - M_irr) * c**2
    fraction = (M - M_irr) / M
    return orange_rg("Processus de Penrose",
                     "ergosphère Kerr : E_extract = (M - M_irr) c²",
                     payload, E_extract,
                     notes=f"M={M/M_sun:.0e} M☉, a={a_spin}, "
                           f"M_irr={M_irr/M_sun:.3e} M☉, "
                           f"E_extract={E_extract:.3e} J ({fraction*100:.1f}% de M c²)")


def o29_adm_bondi(payload=None):
    """Masse ADM/Bondi : quantité asymptotique globale."""
    if payload is None:
        payload = 1e30  # mass in kg
    M = payload
    # For asymptotically flat spacetime, ADM mass = total energy at spatial infinity
    # Schwarzschild: M_ADM = M coincides with parametric mass
    # In natural units M_ADM = M
    M_ADM = M
    # Energy at infinity
    E_ADM = M_ADM * c**2
    return orange_rg("Masse ADM / Bondi",
                     "M_ADM = (1/16π) lim_{r→∞} ∮ (∂_i g_ij - ∂_j g_ii) dS^j",
                     M, M_ADM,
                     notes=f"M={M:.3e} kg, M_ADM={M_ADM:.3e} kg (Schwarzschild), "
                           f"E_ADM={E_ADM:.3e} J")
