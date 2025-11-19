#!/usr/bin/env bash
set -euo pipefail

# Forcer une locale POSIX (sinon printf FR casse tout)
export LC_ALL=C
export LANG=C

# ------------------------------------------------------------
# Distances (Mpc)
# ------------------------------------------------------------
declare -A DIST=(
    [GW150914]=410
    [GW151226]=440
    [GW170104]=880
    [GW170608]=320
    [GW170729]=2840
    [GW170809]=990
    [GW170814]=540
    [GW170817]=40
    [GW170823]=1850
    [GW190412]=740
    [GW190521]=5400
    [GW190403_051519]=1100
    [GW190413_052954]=1100
    [GW190413_134308]=800
    [GW190421_213856]=900
    [GW190503_185404]=1400
    [GW190514_065416]=1500
    [GW190517_055101]=1500
    [GW190519_153544]=2500
)

# ------------------------------------------------------------
# Régimes simplifiés (juste pour affichage)
# ------------------------------------------------------------
declare -A REGIME=(
    [GW150914]=merger
    [GW151226]=ringdown
    [GW170104]=ringdown
    [GW170608]=merger
    [GW170729]=ringdown
    [GW170809]=ringdown
    [GW170814]=ringdown
    [GW170817]=merger
    [GW170823]=ringdown
    [GW190412]=ringdown
    [GW190521]=ringdown
    [GW190403_051519]=merger
    [GW190413_052954]=merger
    [GW190413_134308]=merger
    [GW190421_213856]=merger
    [GW190503_185404]=merger
    [GW190514_065416]=merger
    [GW190517_055101]=ringdown
    [GW190519_153544]=ringdown
)

printf "Événement  | régime    | Distance | ν_eff[Hz] | tau[ms]  | E_total[J] | m_sun\n"
printf "===========|===========|==========|===========|==========|============|========\n"

for ev in "${!DIST[@]}"; do
    d="${DIST[$ev]}"

    # Créer le dossier results s'il n'existe pas
    mkdir -p results

    # Lance l'analyse (JSON écrit dans results/)
    python ligo_spectral_planck.py --event "$ev" --distance-mpc "$d" >/dev/null 2>&1 || {
        echo "⚠️  Erreur avec $ev, continuation..."
        continue
    }

    json="results/$ev.json"

    # Vérifier que le fichier JSON existe et n'est pas vide
    if [[ ! -f "$json" || ! -s "$json" ]]; then
        echo "❌ Fichier JSON manquant ou vide pour $ev"
        continue
    fi

    # Extraction des champs JSON (avec valeurs par défaut en cas d'erreur)
    nu=$(jq -r '.nu_eff_Hz // 0' "$json" 2>/dev/null || echo "0")
    tau=$(jq -r '.tau_s // 0' "$json" 2>/dev/null || echo "0")  # Corrigé: tau_s au lieu de tau_event_s
    E_total=$(jq -r '.E_total_J // 0' "$json" 2>/dev/null || echo "0")  # Corrigé: E_total_J au lieu de E_norm
    m_sun=$(jq -r '.m_sun // 0' "$json" 2>/dev/null || echo "0")

    # Conversion tau en millisecondes
    tau_ms=$(awk -v t="$tau" 'BEGIN {printf "%.3f", t*1000}' 2>/dev/null || echo "0.000")

    # Formattage scientifique pour les grandes valeurs
    E_total_formatted=$(printf "%.2e" "$E_total" 2>/dev/null || echo "0.00e0")
    m_sun_formatted=$(printf "%.4f" "$m_sun" 2>/dev/null || echo "0.0000")

    printf "%-12s | %-9s | %4s Mpc | %9.1f | %7s | %11s | %7.4f\n" \
        "$ev" "${REGIME[$ev]}" "$d" "$nu" "$tau_ms" "$E_total_formatted" "$m_sun"
done

echo ""
echo "=== SYNTHÈSE TERMINÉE ==="
