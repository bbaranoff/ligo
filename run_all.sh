#!/usr/bin/env bash
set -Eeuo pipefail

# ================================
#   CONFIG PIPELINE SPECTRAL
# ================================
PY=python3
SCRIPT="ligo_spectral_planck.py"

# Distances Mpc officielles (GWOSC)
declare -A DIST=(
  [GW150914]=410
  [GW151226]=440
  [GW170104]=880
  [GW170608]=320
  [GW170729]=2840
  [GW170809]=990
  [GW170814]=540
  [GW170817]=40
  
  [GW190403_051519]=1230
  [GW190412]=740
  [GW190413_052954]=1200
  [GW190413_134308]=1500
  [GW190421_213856]=1130
  [GW190503_185404]=2100
  [GW190514_065416]=1480
  [GW190517_055101]=1800
  [GW190519_153544]=2540
  [GW190521]=5300
  [GW190828_063405]=720
  [GW190828_065509]=891
)

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

# ================================
#   RUN GLOBAL
# ================================
echo ""
echo "============================================================="
echo "      🌌  RUN GLOBAL – Nouveau Pipeline Spectral τ "
echo "============================================================="
echo ""

printf "====================== SYNTHÈSE PROGRESSIVE ======================\n"
printf "Event                 |   ν_eff[Hz] |     τ[s]   |   m_sun   |  Energie[J]\n"
echo   "============================================================="

for EV in "${EVENTS[@]}"; do
    D=${DIST[$EV]:-500}   # fallback sécurité

    OUT=$( $PY "$SCRIPT" --event "$EV" --distance-mpc "$D" 2>/dev/null )
    
    # Lecture du JSON généré
    J="results/$EV.json"
    if [[ ! -f "$J" ]]; then
        printf "%-20s |   ERROR JSON MISSING\n" "$EV"
        continue
    fi

    # Extraction valeurs
    nu=$(jq '.nu_eff_Hz' "$J")
    tau=$(jq '.tau_s' "$J")
    m=$(jq '.m_sun' "$J")
    E=$(jq '.E_total_J' "$J")

    LC_NUMERIC=C printf "%-20s | %10.1f | %10.5f | %9.3f | %12.3E\n" \
            "$EV" "$nu" "$tau" "$m" "$E"
done

echo ""
echo "🎯 FIN PIPELINE"
