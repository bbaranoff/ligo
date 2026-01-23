#!/usr/bin/env bash
set -euo pipefail

# ===============================
# CONFIG
# ===============================
PYTHON=python3
SCRIPT=ligo_spectral_planck.py
EVENT_PARAMS=event_params.json
CLUSTER_CALIB_JSON=cluster_calibrations_final.json

# Liste des événements
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
  GW190915_235702
)

# ===============================
# UTILS
# ===============================

# ⬅️ CORRECTION : Lire le cluster depuis le JSON
get_event_cluster () {
  local event="$1"
  jq -r ".event_to_cluster[\"$event\"] // \"0\"" "$CLUSTER_CALIB_JSON"
}

get_cluster_param () {
  local cluster="$1"
  local key="$2"
  jq -r ".calibrations[\"$cluster\"].$key" "$CLUSTER_CALIB_JSON"
}

# ===============================
# SANITY CHECKS
# ===============================
command -v jq >/dev/null 2>&1 || {
  echo "[ERR] jq est requis (sudo apt install jq)"
  exit 1
}

[[ -f "$CLUSTER_CALIB_JSON" ]] || {
  echo "[ERR] calibration cluster introuvable: $CLUSTER_CALIB_JSON"
  exit 1
}

# ===============================
# RUN
# ===============================
echo "============================================================="
echo " RUN ALL — CLUSTER CALIBRATION MODE"
echo " Source calib : $CLUSTER_CALIB_JSON"
echo "============================================================="

for ev in "${EVENTS[@]}"; do
  cluster="$(get_event_cluster "$ev")"
  HSTAR="$(get_cluster_param "$cluster" "H_STAR")"
  SCALE_EJ="$(get_cluster_param "$cluster" "SCALE_EJ")"
  
  echo
  echo "-------------------------------------------------------------"
  echo "[EVENT] $ev"
  echo "[CAL]   cluster=$cluster | H_STAR=$HSTAR | SCALE_EJ=$SCALE_EJ"
  echo "-------------------------------------------------------------"
  
  "$PYTHON" "$SCRIPT" \
    --event "$ev" \
    --event-params "$EVENT_PARAMS" \
    --hstar "$HSTAR" \
    --scale-ej "$SCALE_EJ"
done

echo
echo "============================================================="
echo " ✅ TOUS LES ÉVÉNEMENTS ANALYSÉS"
echo "============================================================="
