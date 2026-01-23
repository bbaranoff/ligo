#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# run_all.sh — Spectral τ pipeline (GLOBAL or CLUSTER calibration aware)
# =============================================================================

PY="${PY:-python3}"
SCRIPT="${SCRIPT:-ligo_spectral_planck.py}"
PARAMS="${PARAMS:-event_params.json}"
REFS="${REFS:-ligo_refs.json}"

CALIB_CLUSTER="cluster_calibrations.json"
CALIB_GLOBAL="calibrated.json"

NO_VIRGO="${NO_VIRGO:-}"
DEBUG="${DEBUG:-}"

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
die() { echo "[FATAL] $*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "commande manquante: $1"; }

num_or_nan() {
  local s="${1:-}"
  local v
  v="$(printf '%s\n' "$s" | grep -Eo '[+-]?[0-9]*([.][0-9]+)?([eE][+-]?[0-9]+)?' | tail -n 1 || true)"
  [[ -n "$v" ]] && printf '%s' "$v" || printf 'nan'
}

fmt_fixed() {
  local v="${1:-nan}" fmt="$2"
  LC_NUMERIC=C awk -v v="$v" -v fmt="$fmt" 'BEGIN{
    if (v=="" || v=="nan" || v=="NaN" || v=="null") printf "nan";
    else printf fmt, v
  }'
}

# -----------------------------------------------------------------------------
# Sanity
# -----------------------------------------------------------------------------
need_cmd jq
need_cmd "$PY"
[[ -f "$SCRIPT" ]] || die "introuvable: $SCRIPT"
[[ -f "$PARAMS" ]] || die "introuvable: $PARAMS"

# -----------------------------------------------------------------------------
# Calibration detection
# -----------------------------------------------------------------------------
CAL_MODE="none"
HSTAR="1.0"
SCALE_EJ="1.0"

if [[ -f "$CALIB_CLUSTER" ]]; then
  CAL_MODE="cluster"
  echo "[CAL] mode=cluster ($CALIB_CLUSTER)"

elif [[ -f "$CALIB_GLOBAL" ]]; then
  CAL_MODE="global"
  HSTAR="$(jq -r '.H_STAR' "$CALIB_GLOBAL")"
  SCALE_EJ="$(jq -r '.SCALE_EJ' "$CALIB_GLOBAL")"
  echo "[CAL] mode=global ($CALIB_GLOBAL)"
  echo "[CAL] H_STAR=$HSTAR  SCALE_EJ=$SCALE_EJ"

else
  echo "[CAL] no calibration found → running global LSQ calibration"
  "$PY" "$SCRIPT" \
    --calibrate-lsq \
    --event-params "$PARAMS" \
    --refs "$REFS" \
    --ref-key energy_J \
    --exclude-cls BNS \
    --cal-out "$CALIB_GLOBAL"

  [[ -f "$CALIB_GLOBAL" ]] || die "échec calibration globale"

  CAL_MODE="global"
  HSTAR="$(jq -r '.H_STAR' "$CALIB_GLOBAL")"
  SCALE_EJ="$(jq -r '.SCALE_EJ' "$CALIB_GLOBAL")"
  echo "[CAL] global calibration done"
fi

echo

# -----------------------------------------------------------------------------
# Banner
# -----------------------------------------------------------------------------
echo "============================================================="
echo "      RUN GLOBAL – Pipeline Spectral τ"
echo "============================================================="
echo "[INFO] mode:   $CAL_MODE"
echo "[INFO] python: $PY"
echo "[INFO] script: $SCRIPT"
echo "[INFO] params: $PARAMS"
[[ -f "$REFS" ]] && echo "[INFO] refs:   $REFS"
echo

printf "Event                 |   ν_eff | ν_ref  |   τ[s]   | τ_ref   |  M⊙c²  | M⊙c²_r |  Energie[J] |  E_ref[J]  | Notes\n"
echo "----------------------------------------------------------------------------------------------------------------------"

# -----------------------------------------------------------------------------
# Events list
# -----------------------------------------------------------------------------
mapfile -t EVENTS < <(jq -r 'keys[] | select(. != "_meta" and . != "default")' "$PARAMS")
[[ "${#EVENTS[@]}" -gt 0 ]] || die "aucun event dans $PARAMS"

mkdir -p results
LOG="results/events.log"
: > "$LOG"

# ---- build list of events from JSON (ignore meta/default) ----
mapfile -t EVENTS < <(jq -r 'keys[] | select(. != "_meta" and . != "default")' "$PARAMS")
[[ "${#EVENTS[@]}" -gt 0 ]] || die "Aucun event trouve dans $PARAMS (hors _meta/default)."

# ----------------------------- output dirs -----------------------------------
mkdir -p results
LOG="results/events.log"
: > "$LOG"


# ------------------------------- main loop -----------------------------------
for ev in "${EVENTS[@]}"; do
  CMD=(
    "$PY" "$SCRIPT"
    --event "$ev"
    --event-params "$PARAMS"
    --no-virgo
  )


  # ⚠️ calibration UNIQUEMENT en mode global
  if [[ "$CAL_MODE" == "global" ]]; then
    CMD+=(--hstar "$HSTAR" --scale-ej "$SCALE_EJ")
  fi

  [[ -n "${DEBUG:-}" ]] && echo "[CMD] ${CMD[*]}" >&2

  OUT="$("${CMD[@]}" 2>&1 | tee -a "$LOG")"

# nu_eff (Hz)
nu="$(num_or_nan "$(
  awk -F'[= ]+' '/nu_eff/{print $(NF-1)}' <<<"$OUT" | tail -n 1
)")"

# tau H1-L1 (s)
tau="$(num_or_nan "$(
  awk -F'[: ]+' '/Tau \(H1-L1\)/{print $(NF-1)}' <<<"$OUT" | tail -n 1
)")"

# Energie en Joules
eJ="$(num_or_nan "$(
  awk -F'[= ]+' '/E_total/{print $(NF-3)}' <<<"$OUT" | tail -n 1
)")"

# Masse équivalente en M_sun
msun="$(num_or_nan "$(
  sed -nE 's/.*\(([0-9.eE+-]+) *M_sun\).*/\1/p' <<<"$OUT" | tail -n 1
)")"

  nuR="nan"; tauR="nan"; msunR="nan"; eJR="nan"; notes=""
  if [[ -f "$REFS" ]]; then
    nuR="$(num_or_nan "$(jq -r --arg ev "$ev" '.[$ev].nu_eff // "nan"' "$REFS")")"
    tauR="$(num_or_nan "$(jq -r --arg ev "$ev" '.[$ev].tau // "nan"' "$REFS")")"
    msunR="$(num_or_nan "$(jq -r --arg ev "$ev" '.[$ev].msun_c2 // "nan"' "$REFS")")"
    eJR="$(num_or_nan "$(jq -r --arg ev "$ev" '.[$ev].energy_J // "nan"' "$REFS")")"
    notes="$(jq -r --arg ev "$ev" '.[$ev].notes // ""' "$REFS")"
  fi

  printf "%-20s | %7s | %7s | %8s | %8s | %6s | %7s | %11s | %11s | %s\n" \
    "$ev" \
    "$(fmt_fixed "$nu" "%.1f")" \
    "$(fmt_fixed "$nuR" "%.1f")" \
    "$(fmt_fixed "$tau" "%.5f")" \
    "$(fmt_fixed "$tauR" "%.5f")" \
    "$(fmt_fixed "$msun" "%.2f")" \
    "$(fmt_fixed "$msunR" "%.2f")" \
    "$eJ" "$eJR" "$notes"
done
