#!/usr/bin/env bash
set -u
LC_ALL=C
export LC_ALL

# ===================== CONFIG =====================
CALIB_JSON="cluster_calibrations_iterative.json"
CLUSTERS_JSON="clusters.json"
REFS_JSON="ligo_refs.json"
EVENT_PARAMS="event_params.json"
RESULTS_DIR="results"              # <-- vrai dossier utilisé par le script python
# ==================================================

mkdir -p "$RESULTS_DIR"

echo "------------------------------------------------------------"
echo "📊 COMPARAISON ÉNERGIE PIPELINE vs LIGO (stream)"
echo "EVENT                C   E_PIPE[J]        E_LIGO[J]       ERR[%]"
echo "------------------------------------------------------------"

# boucle sur les événements connus du clustering
jq -r '.event_to_cluster | to_entries[] | "\(.key) \(.value)"' "$CLUSTERS_JSON" |
while read -r EVENT CLUSTER; do
  # ignorer totalement les clusters -1
  if [[ "$CLUSTER" == "-1" ]]; then
    continue
  fi

  # énergie LIGO de référence
  E_LIGO=$(jq -r --arg ev "$EVENT" '.[$ev].energy_J // empty' "$REFS_JSON")

  # paramètres de calibration (si cluster connu)
  PEAK=$(jq -r ".\"$CLUSTER\".peak_scale // empty" "$CALIB_JSON")
  TAU=$(jq -r ".\"$CLUSTER\".tau_scale  // empty" "$CALIB_JSON")
  SCALE=$(jq -r ".\"$CLUSTER\".scale_ej   // empty" "$CALIB_JSON")
  NU=$(jq -r ".\"$CLUSTER\".nu   // empty" "$CALIB_JSON")

  # si pas de vérité LIGO → affichage neutre
  if [[ -z "$E_LIGO" ]]; then
    printf "%-20s %2s   --               --               --\n" "$EVENT" "$CLUSTER"
    continue
  fi

  # si cluster sans calibration (ex: -1)
  if [[ -z "$PEAK" || -z "$TAU" || -z "$SCALE" ]]; then
    printf "%-20s %2s   --               %.3e        --\n" \
      "$EVENT" "$CLUSTER" "$E_LIGO"
    continue
  fi

  # exécution silencieuse
  python ligo_spectral_planck.py \
    --event "$EVENT" \
    --event-params "$EVENT_PARAMS" \
    --clusters-json "$CLUSTERS_JSON" \
    --tau-scale "$TAU" \
    --nu-scale "$NU" \
    --peak-scale "$PEAK" \
    --scale-ej "$SCALE" \
    >/dev/null 2>&1
    
  RES="$RESULTS_DIR/$EVENT.json"
  if [[ ! -f "$RES" ]]; then
    printf "%-20s %2s   --               %.3e        --\n" \
      "$EVENT" "$CLUSTER" "$E_LIGO"
    continue
  fi

  E_PIPE=$(jq -r '.energy_J // empty' "$RES")
  if [[ -z "$E_PIPE" ]]; then
    printf "%-20s %2s   --               %.3e        --\n" \
      "$EVENT" "$CLUSTER" "$E_LIGO"
    continue
  fi

  ERR=$(python - <<EOF
e_p = float("$E_PIPE")
e_l = float("$E_LIGO")
print(f"{100*(e_p-e_l)/e_l:+.2f}")
EOF
)

  printf "%-20s %2s   %.3e        %.3e        %6s\n" \
    "$EVENT" "$CLUSTER" "$E_PIPE" "$E_LIGO" "$ERR"

done

echo "------------------------------------------------------------"
