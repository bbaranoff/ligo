#!/usr/bin/env bash
# Pipeline LIGO Energy Estimation via Pierre de Rosette
# ======================================================
# Workflow complet :
#   1. Extraction des observables depuis les NPZ
#   2. Benchmark E_J vs LIGO ground truth (ligo_refs)
#   3. Tests statistiques SNR par classe (signal vs bruit)
#   4. Régression E_pred vs E_ref par classe + global
#
# Prérequis : data/npz/{event}_{H1,L1}.npz, event_params.json, ligo_refs.json
# Sortie    : observed.json + tous les rapports stdout
#
# Usage :
#   ./run.sh               # tout le pipeline
#   ./run.sh --quick       # skip extraction si observed.json existe
#   ./run.sh --extract     # juste extraction
#   ./run.sh --stats       # juste stats (suppose observed.json présent)

set -euo pipefail

# ----------------------------------------------------------
# Config
# ----------------------------------------------------------
NPZ_DIR="data/npz"
EVENT_PARAMS="event_params.json"
LIGO_REFS="ligo_refs.json"
OBSERVED="observed.json"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# ----------------------------------------------------------
# Sanity checks
# ----------------------------------------------------------
check_prereqs() {
    local missing=()
    for f in "$EVENT_PARAMS" "$LIGO_REFS"; do
        [ ! -f "$f" ] && missing+=("$f")
    done
    [ ! -d "$NPZ_DIR" ] && missing+=("$NPZ_DIR/")

    if [ ${#missing[@]} -gt 0 ]; then
        echo -e "${RED}❌ Fichiers/dossiers manquants :${NC}"
        printf '   - %s\n' "${missing[@]}"
        echo -e "\n${YELLOW}Génère event_params.json + ligo_refs.json via write_events.py"
        echo -e "puis télécharge les NPZ via ligo_npz_downloader.py.${NC}"
        exit 1
    fi

    for script in ligo_extract.py ligo_bench.py ligo_snr_stats.py ligo_ej_stats.py; do
        if [ ! -f "$script" ]; then
            echo -e "${RED}❌ Script absent : $script${NC}"
            exit 1
        fi
    done
}

# ----------------------------------------------------------
# Section banners
# ----------------------------------------------------------
section() {
    echo
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════════${NC}"
}

# ----------------------------------------------------------
# Étape 1 : extraction NPZ → observed.json
# ----------------------------------------------------------
step_extract() {
    section "ÉTAPE 1 — EXTRACTION DES OBSERVABLES DEPUIS LES NPZ"
    echo "  PSD Welch médian + bandpass SOS + matched filter + f_peak/f_ringdown barycentriques"
    echo "  Sortie : $OBSERVED"
    echo

    local t0=$(date +%s)
    python3 ligo_extract.py \
        --npz-dir "$NPZ_DIR" \
        --event-params "$EVENT_PARAMS" \
        --refs "$LIGO_REFS" \
        --out "$OBSERVED" \
        --verbose
    local t1=$(date +%s)
    echo -e "\n${GREEN}✓ Extraction terminée en $((t1 - t0))s${NC}"
}

# ----------------------------------------------------------
# Étape 2 : benchmark E_J vs LIGO ground truth
# ----------------------------------------------------------
step_bench() {
    section "ÉTAPE 2 — BENCHMARK E_J vs LIGO ground truth"
    echo "  7 paths physiques (orchestrateur pierre de Rosette)"
    echo "  Calibration LSQ par classe ΔM-based (ULTRA_LIGHT/LIGHT/MEDIUM/MASSIVE)"
    echo

    python3 ligo_bench.py --observed "$OBSERVED"
}

# ----------------------------------------------------------
# Étape 3 : tests statistiques SNR
# ----------------------------------------------------------
step_snr_stats() {
    section "ÉTAPE 3 — TESTS STATISTIQUES SUR LE SNR (signal vs bruit)"
    echo "  ANOVA inter-classe + régression SNR vs log(M_initial)"
    echo

    python3 ligo_snr_stats.py --observed "$OBSERVED"
}

# ----------------------------------------------------------
# Étape 4 : régression E_pred vs E_ref
# ----------------------------------------------------------
step_ej_stats() {
    section "ÉTAPE 4 — RÉGRESSION E_pred vs E_ref"
    echo "  slope, R², p-value, MAE par path et par classe"
    echo

    python3 ligo_ej_stats.py --observed "$OBSERVED"
}

# ----------------------------------------------------------
# Résumé final
# ----------------------------------------------------------
summary() {
    section "RÉSUMÉ"
    echo "  Fichiers produits :"
    echo "    - $OBSERVED  (observables mesurés depuis NPZ)"
    echo
    echo "  Pour relancer une étape seule :"
    echo "    ./run.sh --extract   (extraction NPZ)"
    echo "    ./run.sh --stats     (bench + stats sans ré-extraire)"
}

# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
main() {
    check_prereqs

    case "${1:-all}" in
        --quick)
            if [ -f "$OBSERVED" ]; then
                echo -e "${YELLOW}⚡ $OBSERVED existe déjà — skip extraction${NC}"
            else
                step_extract
            fi
            step_bench
            step_snr_stats
            step_ej_stats
            ;;
        --extract)
            step_extract
            ;;
        --stats)
            if [ ! -f "$OBSERVED" ]; then
                echo -e "${RED}❌ $OBSERVED absent — lance --extract d'abord${NC}"
                exit 1
            fi
            step_bench
            step_snr_stats
            step_ej_stats
            ;;
        all|"")
            step_extract
            step_bench
            step_snr_stats
            step_ej_stats
            summary
            ;;
        -h|--help)
            head -n 18 "$0" | tail -n 17 | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Argument inconnu : $1${NC}"
            echo "Usage : ./run.sh [--quick|--extract|--stats|--help]"
            exit 1
            ;;
    esac
}

main "$@"
