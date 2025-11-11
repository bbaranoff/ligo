#!/usr/bin/env bash
set -euo pipefail

C0="\033[0m"; C1="\033[1;36m"; C2="\033[1;33m"; C3="\033[1;32m"

mkdir -p results
rm -f results_raw.txt

echo -e "${C2}=== Étape 1 : exécutions brutes (aucune calibration) ===${C0}"

# ---- Fonction de lancement d’un événement ----
run_event() {
  local ev="$1" dist="$2" flow="$3" fhigh="$4" win="$5"
  echo -e "${C1}=== $ev ===${C0}\n"
  python3 ligo_net.py \
    --event "$ev" \
    --distance-mpc "$dist" \
    --flow "$flow" \
    --fhigh "$fhigh" \
    --signal-win "$win" > "tmp_${ev}.log"

  local E=$(grep -m1 "E_net =" "tmp_${ev}.log" | awk '{print $3}')
  local NU=$(grep -m1 "ν_eff =" "tmp_${ev}.log" | awk '{print $3}')
  local TAU=$(grep -m1 "τ=" "tmp_${ev}.log" | sed -E 's/.*τ=([+-]?[0-9.]+).*/\1/')
  local V=$(grep -m1 "v_eff =" "tmp_${ev}.log" | sed -E 's/.*v_eff = ([0-9.eE+-]+).*/\1/')

  echo "$ev $TAU $E $NU $V" >> results_raw.txt
}

# ---- Paramètres des événements ----
run_event GW150914 410 35 250 0.8
run_event GW151226 440 80 280 1.8
run_event GW170104 880 35 250 1.4
run_event GW170814 540 35 250 1.8

# ---- Résumé global ----
echo -e "\n${C3}=== RÉSUMÉ GLOBAL (sans calibration) ===${C0}\n"

# En-tête
printf "%-10s │ %9s │ %8s │ %8s │ %10s │ %8s │ %13s │ %13s │ %12s │ %7s\n" \
  "Event" "τ(ms)" "τ_ref" "Δ|τ|" "ν_eff(Hz)" "ν_ref" "E_net(J)" "E_LIGO(J)" "v_eff" "ΔE(%)"
printf "──────────┼──────────┼────────┼────────┼────────────┼────────┼───────────────┼───────────────┼──────────────┼────────\n"

# Lecture et mise en forme
awk -v OFMT="%.3e" '
function refE(ev) {
  return (ev=="GW150914")?5.4e47:(ev=="GW151226")?1.0e47:(ev=="GW170104")?3.5e47:(ev=="GW170814")?5.0e47:0;
}
function refTau(ev) {
  return (ev=="GW150914")?6.9:(ev=="GW151226")?8.9:(ev=="GW170104")?0.0:(ev=="GW170814")?3.7:0;
}
function refNu(ev) {
  return (ev=="GW150914")?108.0:(ev=="GW151226")?220.0:(ev=="GW170104")?160.0:(ev=="GW170814")?80.0:0;
}
{
  ev=$1; tau=$2+0; E=$3+0; nu=$4+0; v=$5+0;
  refE0=refE(ev); refT=refTau(ev); refN=refNu(ev);

  dE=(refE0>0)?100*(E-refE0)/refE0:0;
  dTau=((tau>=0)?tau:-tau) - refT; if(dTau<0)dTau=-dTau;

  printf("%-10s │ %9.3f │ %8.2f │ %8.2f │ %10.1f │ %8.1f │ %13.3e │ %13.3e │ %12.3e │ %7.1f\n",
         ev, tau, refT, dTau, nu, refN, E, refE0, v, dE);
}' results_raw.txt

echo -e "\n${C0}"
