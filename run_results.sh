#!/usr/bin/env bash
set -euo pipefail

PYTHON=python3
SCRIPT=ligo_spectral_planck.py

EVENT_PARAMS=event_params.json
REFS=ligo_refs.json
OUTDIR=results

mkdir -p "$OUTDIR"

EVENTS=(
  GW150914
  GW151226
  GW170104
  GW170608
  GW170809
  GW170814
  GW170818
  GW170823
  GW190403_051519
  GW190408_181802
  GW190412
  GW190413_052954
  GW190413_134308
  GW190421_213856
  GW190503_185404
  GW190514_065416
  GW190517_055101
  GW190519_153544
  GW190521
  GW190521_074359
  GW190527_092055
  GW190602_175927
  GW190701_203306
  GW190706_222641
  GW190707_093326
  GW190720_000836
  GW190727_060333
  GW190728_064510
  GW190731_140936
  GW190803_022701
  GW190814
  GW190828_063405
  GW190828_065509
)

for ev in "${EVENTS[@]}"; do
  echo "▶ Processing $ev"


  python "$SCRIPT" \
    --event "$ev" \
    --event-params "$EVENT_PARAMS" \
    --refs "$REFS" \
    --ref-key msun_c2 \
    --flow 30 \
    --fhigh 500 \
    --signal-win 0.6 \
    --noise-pad 800 \
    --peak-quantile 0.9


done
echo
echo "============================================================"
echo "📊 STATS GLOBALES — RESULTS/"
echo "============================================================"

N_TOTAL=$(ls results/*.json 2>/dev/null | wc -l)
echo "Nombre de fichiers résultats : $N_TOTAL"

if [ "$N_TOTAL" -eq 0 ]; then
  echo "⚠️ Aucun résultat trouvé"
  exit 0
fi

# Moyennes / médianes avec jq + awk
jq -r '
  select(.msun_c2 != null) |
  [.event, .tau, .nu_eff, .msun_c2] | @tsv
' results/*.json > /tmp/results_stats.tsv

N_VALID=$(wc -l < /tmp/results_stats.tsv)
echo "Résultats exploitables      : $N_VALID"

# Masse
echo
echo "M☉c² (masse radiée)"
awk '{print $4}' /tmp/results_stats.tsv | sort -n > /tmp/msun.txt
awk '
  {a[NR]=$1}
  END {
    print "  min     :", a[1]
    print "  median  :", a[int((NR+1)/2)]
    print "  max     :", a[NR]
  }
' /tmp/msun.txt

# Tau
echo
echo "τ (délai inter-détecteurs)"
awk '{print $2}' /tmp/results_stats.tsv | sort -n > /tmp/tau.txt
awk '
  {a[NR]=$1}
  END {
    print "  min     :", a[1]
    print "  median  :", a[int((NR+1)/2)]
    print "  max     :", a[NR]
  }
' /tmp/tau.txt

# Nu_eff
echo
echo "ν_eff (Hz)"
awk '{print $3}' /tmp/results_stats.tsv | sort -n > /tmp/nu.txt
awk '
  {a[NR]=$1}
  END {
    print "  min     :", a[1]
    print "  median  :", a[int((NR+1)/2)]
    print "  max     :", a[NR]
  }
' /tmp/nu.txt

# Détection valeurs aberrantes simples
echo
echo "⚠️ ÉVÉNEMENTS POTENTIELLEMENT PROBLÉMATIQUES"
awk '$4 < 0.5 || $4 > 10.0 {print "  ", $1, "(M☉c² =", $4 ")"}' /tmp/results_stats.tsv || true

echo
echo "============================================================"
echo " Fin des stats results/"
echo "============================================================"
