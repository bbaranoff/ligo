#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# OFFICIAL LIGO ENERGY VALUES (Joules)
# ============================================================

#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# OFFICIAL LIGO ENERGY VALUES (Joules)
# ============================================================
declare -A E_LIGO=(
    [GW150914]=5.3e47
    [GW151226]=1.8e46
    [GW170104]=2.0e47
    [GW170608]=2.3e47
    [GW170729]=4.8e47
    [GW170809]=2.7e47
    [GW170814]=2.7e47
    [GW170817]=3.6e46
    [GW190412]=1.4e47
    [GW190521]=8.0e47
    [GW170823]=3.0e47
    [GW170818]=3.5e47
    [GW190403_051519]=1.0e47
    [GW190413_052954]=1.5e47
    [GW190413_134308]=2.0e47
    [GW190421_213856]=2.0e47
    [GW190503_185404]=2.0e47
    [GW190514_065416]=3.0e47
    [GW190517_055101]=3.0e47
    [GW190519_153544]=4.0e47
    [GW190521_074359]=7.0e47
    [GW190602_175927]=2.0e47
    [GW190620_030421]=2.0e47
    [GW190630_185205]=2.0e47
    [GW190814]=2.0e47
    [GW190828_063405]=2.5e47
    [GW190828_065509]=2.5e47
)

# ============================================================
# DISTANCES (Mpc)
# ============================================================
declare -A DIST_MPC=(
    [GW150914]=410
    [GW151226]=440
    [GW170104]=880
    [GW170608]=320
    [GW170729]=2750
    [GW170809]=1030
    [GW170814]=540
    [GW170817]=40
    [GW190412]=740
    [GW190521]=5400
    [GW170823]=1850
    [GW170818]=1060
    [GW190403_051519]=1100
    [GW190413_052954]=1100
    [GW190413_134308]=800
    [GW190421_213856]=900
    [GW190503_185404]=1400
    [GW190514_065416]=1500
    [GW190517_055101]=1500
    [GW190519_153544]=2500
    [GW190521_074359]=5300
    [GW190602_175927]=1000
    [GW190620_030421]=1350
    [GW190630_185205]=1300
    [GW190814]=240
    [GW190828_063405]=850
    [GW190828_065509]=900
)

# ============================================================
# EVENT LIST
# ============================================================
EVENTS=(
    GW150914 GW151226 GW170104 GW170608 GW170729 GW170809 GW170814 GW170817
    GW190412 GW190521
    GW170823 GW170818
    GW190403_051519 GW190413_052954 GW190413_134308 GW190421_213856
    GW190503_185404 GW190514_065416 GW190517_055101 GW190519_153544
    GW190521_074359 GW190602_175927 GW190620_030421
    GW190630_185205 GW190814 GW190828_063405 GW190828_065509
)

# ============================================================
# REGIME (classification simple par fréquence)
# ============================================================
regime_from_freq() {
    local f=$1
    if (( $(echo "$f < 80" | bc -l) )); then
        echo "inspiral"
    elif (( $(echo "$f < 200" | bc -l) )); then
        echo "merger"
    else
        echo "ringdown"
    fi
}

mkdir -p results

# ============================================================
# TABLE HEADER (ν_eff & τ added)
# ============================================================
LC_NUMERIC=C printf "%-12s | %-10s | %-10s | %-10s | %-10s | %-12s | %-12s | %-8s\n" \
       "Événement" "régime" "Distance" "ν_eff[Hz]" "tau[ms]" "E_LIGO[J]" "E_TOI[J]" "Δ[%]"
echo "-------------------------------------------------------------------------------------------------------------------"

# ============================================================
# MAIN LOOP
# ============================================================
for ev in "${EVENTS[@]}"; do

    DIST="${DIST_MPC[$ev]:-???}"

    # ---- EXECUTE PYTHON ANALYSIS ----
    python ligo_spectral_planck.py --event "$ev" --distance-mpc "$DIST" > /dev/null || continue

    JSON="results/${ev}.json"
    [[ -f "$JSON" ]] || { echo "[ERR] $ev: JSON manquant"; continue; }

    # ---- READ RESULT ----
    NU=$(jq -r '.nu_eff_Hz // 0' "$JSON")
    TAU=$(jq -r '.tau_s // 0' "$JSON")

    # tau en millisecondes
    TAUMS=$(python3 - <<EOF
print(($TAU)*1000)
EOF
)

    ELIGO="${E_LIGO[$ev]:-nan}"
    ETOI=$(jq -r '.E_total_J // .E_J // 0' "$JSON")

    # regime spectral basé sur nu_eff
    REG=$(regime_from_freq "$NU")

    # différence %
    DELTA=$(python3 - <<EOF
import math
el=$ELIGO
et=$ETOI
print(100*(et-el)/el if el>0 else float("nan"))
EOF
)

    # ---- PRINT LINE (robuste locale FR) ----
    LC_NUMERIC=C printf "%-12s | %-10s | %-10s | %-10.1f | %-10.3f | %-12.3e | %-12.3e | %8.2f\n" \
           "$ev" "$REG" "${DIST}Mpc" "$NU" "$TAUMS" "$ELIGO" "$ETOI" "$DELTA"

done
python3 plot_all_spectra.py || echo "⚠️  Impossible de tracer le graphe."
