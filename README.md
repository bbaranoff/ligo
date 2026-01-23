# LIGO Spectral Calibration — Pipeline expérimental

Ce dépôt implémente un pipeline expérimental pour analyser des événements LIGO/Virgo à partir des **strain time series**, produire des observables spectro-temporelles homogènes, puis étudier leur **calibrabilité mutuelle** via une calibration itérative non supervisée.

L’approche est volontairement **agnostique aux paramètres astrophysiques publiés** (masses, spins, inclinaison) pour :

* la sélection des événements,
* le clustering,
* la calibration.

Ces paramètres ne sont utilisés **qu’a posteriori**, uniquement pour l’évaluation quantitative des erreurs.

---

## Principe général

1. Télécharger les données strain LIGO (H1/L1, Virgo optionnel).
2. Extraire des observables spectro-temporelles à partir d’une **fenêtre temporelle contrôlée**.
3. Générer des résultats homogènes par événement (`results/GW*.json`).
4. Effectuer une calibration itérative par clustering :

   * détection d’outliers,
   * calibration globale puis locale,
   * calcul de statistiques *clean* (hors outliers et clusters triviaux).

---

## Installation

### Script d’installation (`go.sh`)

```bash
#!/bin/bash

# 1. créer l’environnement seulement si absent
if [ ! -d ".env" ]; then
    python3 -m venv .env
fi

# 2. ACTIVER l'env
source .env/bin/activate

# 3. installer les dépendances
pip install --upgrade pip
pip3 install scipy gwosc numpy matplotlib gwpy numba
# ou : pip install -r requirements.txt
```

---

## Utilisation du pipeline

### 1️⃣ Activer l’environnement

```bash
source go.sh
```

---

### 2️⃣ Télécharger les données LIGO (NPZ)

Télécharge les fichiers strain nécessaires pour les événements définis :

```bash
python ligo_npz_downloader.py
```

Les fichiers sont stockés localement et réutilisés par la suite.

---

### 3️⃣ Générer les résultats spectro-temporels

Analyse chaque événement et produit un fichier JSON par événement :

```bash
bash run_results.sh
```

Résultat :

```
results/
 ├── GW150914.json
 ├── GW151226.json
 ├── GW170104.json
 └── ...
```

Chaque fichier contient uniquement des observables dérivées du **strain**.

---

### 4️⃣ Calibration itérative par clustering

Lance la calibration itérative globale + locale :

```bash
python run_iterative_calibration.py \
  --refs ligo_refs.json \
  --event-params event_params.json \
  --signal-win 0.6 \
  --noise-pad 800 \
  --peak-quantile 0.9 \
  --k 4 \
  --db-eps 0.7 \
  --db-min-samples 2 \
  --exclude-cls BNS \
  --exclude-cluster-minus1 \
  --flow 30 \
  --fhigh 500
```

### Effets de cette étape

* clustering non supervisé des événements,
* détection automatique des outliers (cluster `-1`),
* calibration globale puis par cluster,
* calcul des **stats clean** (hors outliers et clusters à 1 événement).

Sorties typiques :

* `calibration_iterative.txt` : rapport détaillé lisible,
* `cluster_calibrations_iterative.json` : paramètres de calibration.

---

## Interprétation des résultats

* **MAE globale** : dominée par les événements non comparables.
* **Stats clean** :

  * calculées uniquement sur des événements cohérents entre eux,
  * sélection indépendante des paramètres astrophysiques publiés,
  * reflètent un régime physique commun capturé par le modèle.

Les événements exclus (outliers) correspondent généralement à :

* systèmes NSBH,
* rapports de masse extrêmes,
* géométries fortement dégénérées.

---

## Cadre et limites

* Ce pipeline **ne cherche pas** une loi universelle.
* Il met en évidence l’existence de **sous-populations calibrables** à partir du strain seul.
* Toute extension nécessite l’introduction explicite de paramètres supplémentaires (spin effectif, q, inclinaison…).

---

## Résumé en une phrase

> Ce pipeline montre qu’un sous-ensemble d’événements LIGO est mutuellement calibrable à partir de la seule structure spectro-temporelle du strain, via un choix contrôlé de fenêtre temporelle et une mise à l’échelle énergétique globale.

