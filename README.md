# Spectral–Coherent Energy Calibration Pipeline (LIGO)

Ce dépôt implémente un pipeline **non-template**, basé sur l’analyse spectrale cohérente H1–L1, visant à :

* estimer une **énergie globale robuste** des événements GW,
* identifier automatiquement les **événements bien contraints**,
* calibrer les résultats **sans sur-ajustement astrophysique**,
* et expliciter **ce qui n’est pas estimable** avec ces observables.

Le pipeline est volontairement conservateur :
il sépare **estimation** et **ignorance mesurée**.

---

## Vue d’ensemble du pipeline

Le workflow complet est le suivant :

```bash
# 1. Préparation de l’environnement
source go.sh
python write_events.py

# 2. Analyse spectrale cohérente de tous les événements
bash run_results.sh

# 3. Clustering latent (énergie, tau, nu_eff)
python cluster_latent_kmeans.py   --results-glob "results/GW*.json"   --method hdbscan+kmeans   --k 4   --min-cluster-size 3   --min-samples 5   --cluster-selection-epsilon 0.0   --use-logE   --export clusters.json

# 4. Recalcul des résultats en excluant les outliers
mv results results.bak
bash run_results_excl.sh

# 5. Calibration itérative par grille exhaustive
python run_iterative_calibration.py \
  --refs ligo_refs.json \
  --event-params event_params.json \
  --results-glob "results/GW*.json" \
  --method hdbscan+kmeans \
  --k 4 \
  --min-cluster-size 3 \
  --min-samples 6 \
  --cluster-selection-epsilon 0.0 \
  --exclude-cluster-minus1 \
  --exclude-cls BNS \
  --peak-min 0 --peak-max 2 --peak-step 1 \
  --tau-min 0  --tau-max 2  --tau-step 1 \
  --peak-quantile 0.9 \
  --use-logE \
  --out calibration_report_astrophysical.txt \
  --calib-json cluster_calibrations_astrophysical.json
```

---

## Principe méthodologique

### 1. Estimation (ce qui est mesurable)

Pour chaque événement GW :

* construction d’un signal **cohérent H1–L1**,
* pondération par le **PSD de bruit réel**,
* intégration spectrale de l’énergie,
* calcul d’observables secondaires (τ, ν_eff),
* **sans templates**, sans paramètres morphologiques.

Cette étape produit une estimation **fermée** :
l’information provient uniquement des données observées.

---

### 2. Clustering latent (ce qui est stable)

Les événements sont regroupés automatiquement à partir de l’espace latent :

* énergie (log),
* délais temporels,
* signatures spectrales.

Objectif :

* identifier les événements **cohérents entre eux**,
* isoler les **outliers** et les cas non contraints,
* éviter toute sélection manuelle a posteriori.

---

### 3. Calibration par grille exhaustive (sans triche)

Pour chaque cluster stable :

* exploration **discrète** des paramètres `(PEAK_SCALE, TAU_SCALE)`
  ici volontairement limités à `{0, 1, 2}`,
* calcul analytique de `SCALE_EJ`,
* sélection par erreur relative globale (MAE / médiane),
* **aucun ajustement continu**, aucun fit caché.

Cette approche empêche le sur-ajustement et rend la calibration traçable.

---

## Résultats

```
======================================================================
✅ CALIBRATION PAR GRILLE EXHAUSTIVE TERMINÉE
======================================================================
MAE globale : 19.92%
Médiane     : 16.16%

✨ STATS CLEAN (sans outliers ni clusters 1-event):
MAE clean   : 19.92%
Médiane     : 16.16%
N clean     : 23
======================================================================
```

* **23 événements** bien contraints,
* **≈ 16 % d’erreur médiane** sur l’énergie / masse équivalente,
* sans templates,
* sans hypothèses astrophysiques fortes,
* avec identification explicite des événements non estimables.

Rapport détaillé :
`calibration_report_astrophysical.txt`

Paramètres calibrés par cluster :
`cluster_calibrations_astrophysical.json`

---

## Ce que ce pipeline montre

* Une estimation globale robuste est possible **sans modèles d’ondes**.
* Tous les événements ne sont **pas également contraints** — et c’est mesuré.
* Les paramètres τ et peak ont un **impact limité** sur l’énergie globale.
* La performance vient de la **cohérence spectrale**, pas du tuning.

Ce travail ne remplace pas les pipelines bayésiens existants.
Il montre autre chose :

> **où l’information est réellement présente dans les données,
> et où elle ne l’est pas.**

---

## Statut

* Approche méthodologique exploratoire
* Résultats reproductibles
* Zéro claim astrophysique fort
* Prêt pour analyse comparative ou stress-testing
