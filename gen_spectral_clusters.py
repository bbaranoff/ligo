#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EXTRACT SPECTRAL INVARIANTS (NO GPS, NO T0)
-------------------------------------------
Pour chaque fichier JSON dans results/, retourne :

• nu_eff_Hz
• tau_s
• Lambda = nu_eff_Hz * tau_s
• E_total_J  (énergie totale)
"""

import os
import json

RESULTS = "results"


def load_events():
    events = []
    for fn in os.listdir(RESULTS):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(RESULTS, fn)
        with open(path, "r") as f:
            try:
                events.append(json.load(f))
            except Exception as e:
                print("Erreur lecture:", fn, e)
    return events


def extract(ev):
    """
    Renvoie les invariants spectraux d’un événement JSON
    """
    name = ev["event"]

    nu_eff = float(ev.get("nu_eff_Hz", 0.0))
    tau_s  = float(ev.get("tau_s", 0.0))
    e_norm = float(ev.get("E_total_J", 0.0))

    # Invariant Λ = ν_eff × τ
    lam = nu_eff * tau_s

    return name, {
        "nu_eff_Hz": nu_eff,
        "tau_s": tau_s,
        "Lambda": lam,
        "E_total_J": e_norm
    }


def main():
    events = load_events()
    out = {}

    for ev in events:
        name, data = extract(ev)
        out[name] = data

    with open("spectral_invariants.json", "w") as f:
        json.dump(out, f, indent=4)

    print("✓ spectral_invariants.json généré.")
    print("Événements chargés :", len(out))
    print("-------------------------------------")

    for name, d in out.items():
        print(
            name.ljust(20),
            f"ν={d['nu_eff_Hz']:8.3f} Hz |",
            f"τ={d['tau_s']*1000:8.3f} ms |",
            f"Λ={d['Lambda']:10.4f} |",
            f"E_total_J={d['E_total_J']:.3e}"
        )


if __name__ == "__main__":
    main()
