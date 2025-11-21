#!/usr/bin/env bash
set -e

# --------------------------------------------
#   RUN GLOBAL – Nouveau Pipeline Spectral τ
# --------------------------------------------

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

# Distances (les vraies, en Mpc)
declare -A DIST
DIST[GW150914]=410
DIST[GW151226]=440
DIST[GW170104]=880
DIST[GW170608]=320
DIST[GW170729]=2750
DIST[GW170809]=990
DIST[GW170814]=540
DIST[GW170817]=40
DIST[GW190403_051519]=900
DIST[GW190412]=740
DIST[GW190413_052954]=700
DIST[GW190413_134308]=700
DIST[GW190421_213856]=1500
DIST[GW190503_185404]=1890
DIST[GW190514_065416]=1500
DIST[GW190517_055101]=1400
DIST[GW190519_153544]=1500
DIST[GW190521]=5300
DIST[GW190828_063405]=900
DIST[GW190828_065509]=900

echo ""
echo "============================================================="
echo "      🌌  RUN GLOBAL – Nouveau Pipeline Spectral τ "
echo "============================================================="

echo ""
echo "====================== SYNTHÈSE PROGRESSIVE ======================"
echo -e "Event                 |   ν_eff[Hz] |     τ[s]   |   m_sun   |  Energie[J]"
echo "--------------------------------------------------------------------------"

for EV in "${EVENTS[@]}"; do
    python3 ligo_spectral_planck.py --event "$EV" --distance-mpc "${DIST[$EV]}" > ligo.log
    # Récupération immédiate du JSON
    JSON="results/${EV}.json"
    if [[ -f "$JSON" ]]; then
        nu=$(jq -r '.nu_eff_Hz' "$JSON")
        tau=$(jq -r '.tau_s' "$JSON")
        m=$(jq -r '.m_sun' "$JSON")
        e=$(jq -r '.E_total_J' "$JSON")
        LC_NUMERIC=C printf "%-20s | %10.1f | %10.5f | %9.3f | %12.3e\n" \
            "$EV" "$nu" "$tau" "$m" "$e"
    else
        echo "❌ Impossible de lire $JSON"
    fi
done

echo ""
echo "🎯 Pipeline terminé."
