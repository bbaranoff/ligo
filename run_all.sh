#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# run_all.sh
# -----------------------------------------------------------------------------
# Run spectral pipeline for all events described in a JSON parameter file.
# Prints a progressive summary table and appends raw per-event logs to events.log
# (in ./results).
#
# Config via env:
#   PY=python3
#   SCRIPT=ligo_spectral_planck.py
#   PARAMS=event_params.json
#   REFS=ligo_refs.json
#
# Optional knobs:
#   NO_VIRGO=1
#   DEBUG=1
#
# Optional calibrations:
#   CAL_H_STAR_EVENT=GW170817  HPEAK_TARGET=3e-22
#   CAL_EVENT=GW150914        MSUN_TARGET=3.0
# -----------------------------------------------------------------------------

PY="${PY:-python3}"
SCRIPT="${SCRIPT:-ligo_spectral_planck.py}"
PARAMS="${PARAMS:-event_params.json}"
REFS="${REFS:-ligo_refs.json}"
# Valeurs par défaut (évite set -u qui explose)
H_STAR="${H_STAR:-1.0}"
EJ_SCALE="${EJ_SCALE:-1.0}"
CAL_FILE="${CAL_FILE:-calibrated.json}"

# --- Calibration LSQ (une seule fois) ---
python3 "$SCRIPT" \
  --calibrate-lsq \
  --event-params "$PARAMS" \
  --refs "$REFS" \
  --ref-key energy_J \
  --exclude-cls BNS \
  --cal-out calibrated.json


# ------------------------------- utils ---------------------------------------
abspath() {
  "$PY" - <<'PY' "$1"
import os,sys
print(os.path.abspath(sys.argv[1]))
PY
}

die() {
  echo "[FATAL] $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "commande manquante: $1"
}

parse_keyline() {
  # parse_keyline "KEY" reads stdin and extracts: KEY = <value>
  local key="$1"
  sed -n "s/^${key} = //p"
}

num_or_nan() {
  # Extract the last float-ish token (supports sci notation) from a string.
  # If none found -> "nan".
  local s="${1:-}"
  local v
  v="$(printf '%s\n' "$s" | grep -Eo '[+-]?[0-9]*([.][0-9]+)?([eE][+-]?[0-9]+)?' | tail -n 1 || true)"
  [[ -n "$v" ]] && printf '%s' "$v" || printf 'nan'
}

fmt_fixed() {
  # fmt_fixed <value> <printf_format>
  # Uses awk for formatting so "nan" doesn't make bash printf error.
  local v="${1:-nan}"
  local fmt="$2"
  LC_NUMERIC=C awk -v v="$v" -v fmt="$fmt" 'BEGIN{
    if (v=="" || v=="nan" || v=="NaN" || v=="null") {
      printf "nan"
    } else {
      printf fmt, v
    }
  }'
}

# ----------------------- params + option building ----------------------------
get_param() {
  # get_param <event> <key>
  local ev="$1" key="$2"
  jq -r --arg ev "$ev" --arg key "$key" '(.[$ev][$key] // .default[$key] // empty)' "$PARAMS"
}

get_arr2() {
  # get_arr2 <event> <key>  => prints "a b" if array length>=2
  local ev="$1" key="$2"
  jq -r --arg ev "$ev" --arg key "$key" '
    (.[ $ev ][ $key ] // .default[ $key ] // empty)
    | if type=="array" and length>=2 then "\(.[0]) \(.[1])" else empty end
  ' "$PARAMS"
}

build_opts() {
  # build_opts <event>
  # Prints a NUL-separated list suitable for: mapfile -d '' -t arr < <(build_opts ev)
  local ev="$1"
  local opts=()

  local flow fhigh noise_pad signal_win
  flow="$(get_param "$ev" flow)"
  fhigh="$(get_param "$ev" fhigh)"
  noise_pad="$(get_param "$ev" noise_pad)"
  signal_win="$(get_param "$ev" signal_win)"

  local tau_band nu_band
  tau_band="$(get_arr2 "$ev" tau_band)"
  nu_band="$(get_arr2 "$ev" nu_band)"

  [[ -n "${flow:-}" ]] && opts+=(--flow "$flow")
  [[ -n "${fhigh:-}" ]] && opts+=(--fhigh "$fhigh")
  [[ -n "${noise_pad:-}" ]] && opts+=(--noise-pad "$noise_pad")
  [[ -n "${signal_win:-}" ]] && opts+=(--signal-win "$signal_win")

  if [[ -n "${tau_band:-}" ]]; then
    local tb
    # shellcheck disable=SC2206
    tb=(${tau_band})
    opts+=(--tau-band "${tb[0]}" "${tb[1]}")
  fi
  if [[ -n "${nu_band:-}" ]]; then
    local nb
    # shellcheck disable=SC2206
    nb=(${nu_band})
    opts+=(--nu-band "${nb[0]}" "${nb[1]}")
  fi

  printf '%s\0' "${opts[@]}"
}

# ------------------------------ sanity ---------------------------------------
need_cmd jq
need_cmd "$PY"
[[ -f "$PARAMS" ]] || die "introuvable: $PARAMS"
[[ -f "$SCRIPT" ]] || die "introuvable: $SCRIPT"


# ------------------------------- banner --------------------------------------
echo "============================================================="
echo "      RUN GLOBAL – Pipeline Spectral τ"
echo "============================================================="
echo "[INFO] script:  $(abspath "$0")"
echo "[INFO] python:  $PY"
echo "[INFO] analyse: $(abspath "$SCRIPT")"
echo "[INFO] params:  $(abspath "$PARAMS")"
if [[ -f "$REFS" ]]; then
  echo "[INFO] refs:    $(abspath "$REFS")"
else
  echo "[INFO] refs:    (absent) $REFS"
fi

echo

echo "====================== SYNTHÈSE PROGRESSIVE =========================================================================="
printf "Event                 |   ν_eff | ν_ref  |   τ[s]   | τ_ref   |  M⊙c²  | M⊙c²_r |  Energie[J] |  E_ref[J]  | Notes\n"
echo "----------------------------------------------------------------------------------------------------------------------"

# ---- build list of events from JSON (ignore meta/default) ----
mapfile -t EVENTS < <(jq -r 'keys[] | select(. != "_meta" and . != "default")' "$PARAMS")
[[ "${#EVENTS[@]}" -gt 0 ]] || die "Aucun event trouve dans $PARAMS (hors _meta/default)."

# ----------------------------- output dirs -----------------------------------
mkdir -p results
LOG="results/events.log"
: > "$LOG"


# ------------------------------- main loop -----------------------------------
count=0
for ev in "${EVENTS[@]}"; do
  mapfile -d '' -t EV_OPTS < <(build_opts "$ev")

  CMD=(
    "$PY" "$SCRIPT"
    --event "$ev"
    --event-params "$PARAMS"
    "${EV_OPTS[@]}"
    ${NO_VIRGO:+--no-virgo}
    --hstar "$H_STAR"
    --scale "$EJ_SCALE"
  )

  [[ -n "${DEBUG:-}" ]] && echo "[CMD] ${CMD[*]}" >&2

  OUT="$("${CMD[@]}" 2>&1 | tee -a "$LOG")"

  # ---- Parse from stdout (robust) ----
  nu_line="$(printf '%s\n' "$OUT" | awk '/nu_eff \(energy\)/{print}' | tail -n 1)"
  tau_line="$(printf '%s\n' "$OUT" | awk '/Tau \(H1-L1\)/{print}' | tail -n 1)"
  e_line="$(printf '%s\n' "$OUT" | awk '/Energie intrins/{print}' | tail -n 1)"

  nu="$(num_or_nan "$nu_line")"
  tau="$(num_or_nan "$tau_line")"

  # Energy: take the FIRST number after ':' (works whether it says (J) or (EJ) etc.)
  energy_raw="$(printf '%s\n' "$e_line" | sed -nE 's/.*:[[:space:]]*([+-]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?).*/\1/p' | head -n 1)"
  energy_J="$(num_or_nan "${energy_raw:-}")"

  # Mass in M_sun: explicitly from "(... M_sun)"
  msun_raw="$(printf '%s\n' "$e_line" | sed -nE 's/.*\(([+-]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?) *M_sun\).*/\1/p')"
  msun="$(num_or_nan "${msun_raw:-}")"

  # ---- Refs (optional) ----
  # ligo_refs.json keys expected:
  #   nu_eff, tau, msun_c2, energy_J, notes
  # ---- Refs (optional) ----
  nuR="nan"; tauR="nan"; msunR="nan"; energyR_J="nan"; notes=""
  if [[ -f "$REFS" ]]; then
    nuR="$(jq -r --arg ev "$ev" '.[$ev].nu_eff // "nan"' "$REFS")"
    tauR="$(jq -r --arg ev "$ev" '.[$ev].tau // "nan"' "$REFS")"
    msunR="$(jq -r --arg ev "$ev" '.[$ev].msun_c2 // "nan"' "$REFS")"
    energyR_J="$(jq -r --arg ev "$ev" '.[$ev].energy_J // "nan"' "$REFS")"
    notes="$(jq -r --arg ev "$ev" '.[$ev].notes // ""' "$REFS")"

    nuR="$(num_or_nan "$nuR")"
    tauR="$(num_or_nan "$tauR")"
    msunR="$(num_or_nan "$msunR")"
    energyR_J="$(num_or_nan "$energyR_J")"
  fi

  # ---- nan-safe formatting ----
  nu_s="$(fmt_fixed "$nu" "%.1f")"
  nuR_s="$(fmt_fixed "$nuR" "%.1f")"
  tau_s="$(fmt_fixed "$tau" "%.5f")"
  tauR_s="$(fmt_fixed "$tauR" "%.5f")"
  msun_s="$(fmt_fixed "$msun" "%.2f")"
  msunR_s="$(fmt_fixed "$msunR" "%.2f")"
  energy_s="$energy_J"
  energyR_s="$energyR_J"

  printf "%-20s | %7s | %7s | %8s | %8s | %6s | %7s | %11s | %11s | %s\n" \
    "$ev" "$nu_s" "$nuR_s" "$tau_s" "$tauR_s" "$msun_s" "$msunR_s" "$energy_s" "$energyR_s" "$notes"

  count=$((count+1))
done

[[ "$count" -gt 0 ]] || die "Aucun event traite (boucle vide). Verifie $PARAMS et jq."

echo
echo "============================================================="
echo "                    FIN DU RUN GLOBAL"
echo "============================================================="
echo "[INFO] log: $(abspath "$LOG")"

