#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json

INPUT = "geodesique_residual.py"     # on lit TON fichier
OUTPUT = "geodesiques.json"          # output propre

# -------------------------------------------------------
# EXTRACTION PAR REGEX
# -------------------------------------------------------
PATTERN = re.compile(
    r"(GW\d{6}(?:_[0-9A-Za-z]+)?)\s*→\s*(GW\d{6}(?:_[0-9A-Za-z]+)?)"
    r".*?Δτ=\s*([-+]?\d+\.\d+)"
    r".*?Δt=\s*([-+]?\d+\.\d+)"
)

def extract_pairs(text):
    pairs = []
    for line in text.splitlines():
        m = PATTERN.search(line)
        if m:
            A, B, dTau, dT = m.groups()
            pairs.append({
                "A": A,
                "B": B,
                "dTau": float(dTau),
                "dT_days": float(dT)
            })
    return pairs

# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    # On lit le fichier ENTIER même si c’est du Python
    with open(INPUT, "r") as f:
        raw = f.read()

    pairs = extract_pairs(raw)

    # Supprime les doublons A→B / B→A
    unique = {}
    for p in pairs:
        key = tuple(sorted([p["A"], p["B"]]))
        if key not in unique:
            unique[key] = p

    final = list(unique.values())

    with open(OUTPUT, "w") as f:
        json.dump(final, f, indent=2)

    print(f"✔️ {OUTPUT} généré ({len(final)} paires extraites).")

if __name__ == "__main__":
    main()
