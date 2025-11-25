#!/bin/bash
set -euo pipefail

PY=python3
SCRIPT="ligo_spectral_planck.py"
REFS="ligo_refs.json"

EVENTS=(
  GW150914
  GW151226
  GW170104
  GW170608
  GW170729
  GW170809
  GW170814
  GW170817
  GW190403_051519
  GW190412
  GW190413_052954
  GW190413_134308
  GW190421_213856
  GW190503_185404
  GW190514_065416
  GW190517_055101
  GW190519_153544
  GW190521
  GW190828_063405
  GW190828_065509
)

echo "============================================================="
echo "      🌌  RUN GLOBAL – Nouveau Pipeline Spectral τ "
echo "============================================================="
echo
echo "====================== SYNTHÈSE PROGRESSIVE =========================================================================="
printf "Event                 |   ν_eff | ν_LIGO |   τ[s]   | τ_LIGO  |  M⊙c²  | M⊙c²_L |  Energie[J] |  E_LIGO[J] | Notes\n"
echo "----------------------------------------------------------------------------------------------------------------------"

for ev in "${EVENTS[@]}"; do

    # distance depuis ton script Python
    dist=$($PY - <<EOF
import ligo_spectral_planck as L
print(L.EVENT_PARAMS["$ev"]["distance_mpc"])
EOF
)

    # run analyse
    $PY "$SCRIPT" --event "$ev" >> events.log

    # lecture résultats
    read -r nu tau msun energy < <(
        $PY - <<EOF
import json
with open("results/$ev.json") as f:
    R=json.load(f)
print(R["nu_eff_Hz"], R["tau_s"], R["m_sun"], R["E_total_J"])
EOF
    )

    # lecture LIGO JSON
    read -r nuL tauL msunL energyL notes< <(
        $PY - <<EOF
import json
with open("$REFS") as f:
    X=json.load(f)
R=X["$ev"]
print(R["nu_eff"], R["tau"], R["msun_c2"], R["energy_J"], R["notes"])
EOF
    )

LC_NUMERIC=C printf "%-20s | %7.1f | %7.1f | %8.5f | %8.5f | %6.2f | %7.2f | %11.3E | %11.3E | %s\n" \
        "$ev" "$nu" "$nuL" "$tau" "$tauL" "$msun" "$msunL" "$energy" "$energyL" "$notes"
done

echo
echo "============================================================="
echo "                    🛰️  FIN DU RUN GLOBAL "
echo "============================================================="
