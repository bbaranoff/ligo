"""Infrastructure partagée pour la pierre de Rosette MQ ↔ Hilbert ↔ RG."""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ============================================================
# Constantes physiques (SI)
# ============================================================
c       = 2.998e8        # vitesse lumière, m/s
G       = 6.674e-11      # constante de Newton, m³/(kg·s²)
hbar    = 1.055e-34      # ℏ, J·s
h       = 2 * np.pi * hbar
k_B     = 1.381e-23      # Boltzmann, J/K
M_sun   = 1.989e30       # masse solaire, kg

# Unités de Planck
m_P     = 2.176e-8       # kg
l_P     = 1.616e-35      # m
t_P     = 5.391e-44      # s
T_P     = 1.417e32       # K
E_P     = 1.956e9        # J

# Cosmologie
N_universe = 1e61         # R_U / l_P
H_0     = 2.2e-18         # Hubble actuel, s⁻¹
Lambda  = 1.1e-52         # constante cosmologique, m⁻²
T_CMB   = 2.7255          # K

# ============================================================
# Statut d'une entrée de la pierre
# ============================================================
class Status(Enum):
    GREEN     = "🟢"
    YELLOW    = "🟡"
    ORANGE_MQ = "🟠←MQ"   # transformation existe côté MQ uniquement
    ORANGE_RG = "🟠←RG"   # transformation existe côté RG uniquement
    RED       = "🔴"

# ============================================================
# Résultat d'un transit
# ============================================================
@dataclass
class TransitResult:
    name: str
    status: Status
    transformation: str
    payload_in: Any = None
    payload_out: Any = None
    success: bool = False
    notes: str = ""
    metadata: dict = field(default_factory=dict)

    def short(self):
        marks = {
            Status.GREEN: "✓" if self.success else "✗",
            Status.YELLOW: "≈" if self.success else "?",
            Status.ORANGE_MQ: "→",
            Status.ORANGE_RG: "←",
            Status.RED: "?",
        }
        return f"{self.status.value} {marks[self.status]} {self.name}"

    def report(self):
        lines = [self.short()]
        lines.append(f"   transfo : {self.transformation}")
        if self.notes:
            lines.append(f"   note    : {self.notes}")
        if self.metadata:
            mds = ", ".join(f"{k}={v}" for k, v in self.metadata.items())
            lines.append(f"   data    : {mds}")
        return "\n".join(lines)

# ============================================================
# Constructeurs utilitaires
# ============================================================
def green(name, transfo, p_in, p_out, success=True, notes="", **md):
    return TransitResult(name, Status.GREEN, transfo, p_in, p_out, success, notes, md)

def yellow(name, transfo, p_in, p_out, success=True, notes="", **md):
    return TransitResult(name, Status.YELLOW, transfo, p_in, p_out, success, notes, md)

def orange_mq(name, transfo, p_in, p_out, notes="", **md):
    return TransitResult(name, Status.ORANGE_MQ, transfo, p_in, p_out, False,
                         notes + " [côté RG vide]", md)

def orange_rg(name, transfo, p_in, p_out, notes="", **md):
    return TransitResult(name, Status.ORANGE_RG, transfo, p_in, p_out, False,
                         notes + " [côté MQ vide]", md)

def red(name, transfo, p_in, p_out, notes="", **md):
    return TransitResult(name, Status.RED, transfo, p_in, p_out, False,
                         notes + " [pas de transformation consensuelle]", md)
