# Spectral–Coherent Energy Calibration Pipeline (LIGO)

Ce dépôt implémente un pipeline **spectral cohérent H1–L1**, sans templates, structuré en trois phases strictement séparées :

1. **Estimation spectrale brute** (données uniquement)
2. **Clustering latent optimisé automatiquement**
3. **Calibration itérative par cluster (GPU/CUDA)**

Chaque étape est traçable, reproductible, et ne dépend que des sorties de l’étape précédente.

---

## Exécution complète du pipeline

### 0. Environnement

```bash
git checkout 995c70c26e9428f326fa26afde23447c834e3b7b
source go.sh
```

Le pipeline utilise **CUDA par défaut** lorsqu’un GPU compatible est disponible
(`ligo_spectral_gpu`, via CuPy).
Le mode CPU n’est utilisé qu’en fallback explicite.

---

### 1. Analyse spectrale cohérente (tous événements)

```bash
bash run_results.sh
```

Cette étape :

* charge les données LIGO/Virgo,
* construit un signal cohérent H1–L1,
* intègre l’énergie spectrale,
* calcule `τ` et `ν_eff`,
* écrit un fichier JSON par événement dans `results/`.

Aucun clustering, aucune calibration à ce stade.

---

### 2. Optimisation automatique du clustering latent

```bash
python optimize_clustering.py --min-clean 20
```

Cette étape :

* teste automatiquement plusieurs configurations de clustering,
* évalue chaque configuration sur :

  * MAE CLEAN,
  * médiane,
  * proportion d’événements conservés,
* sélectionne **la meilleure configuration globale**.

#### Résultat (exemple réel)

```
📊 Résultats finaux:
   MAE CLEAN : 27.3%
   Médiane   : 22.8%
   N clean   : 29/63 (46.0%)
   Score     : 21.21
```

Les paramètres optimaux sont sauvegardés dans :

```
best_clustering_params.json
```

#### Commande de reproduction (émise automatiquement)

```bash
python cluster_latent_kmeans.py \
  --results-glob 'results/GW*.json' \
  --method dbscan+kmeans \
  --k 3 \
  --eps 0.6 \
  --min-samples 3 \
  --use-logE \
  --export clusters.json
```

---

### 3. Reproduction explicite du clustering optimal

La commande fournie par `optimize_clustering.py` doit ensuite être **rejouée telle quelle** :

```bash
python cluster_latent_kmeans.py \
  --results-glob 'results/GW*.json' \
  --method dbscan+kmeans \
  --k 3 \
  --eps 0.6 \
  --min-samples 3 \
  --use-logE \
  --export clusters.json
```

Cette étape produit :

* `clusters.json`
* l’affectation finale des événements aux clusters
* le cluster `-1` correspondant aux **outliers**

---

### 4. Calibration itérative par cluster (GPU / CUDA)

```bash
python run_iterative_calibration.py \
  --refs ligo_refs.json \
  --event-params event_params.json \
  --clusters clusters.json \
  --exclude-cluster-minus1 \
  --peak-scale 1.0 \
  --k-target 10.0 \
  --nu-min 0.1 --nu-max 1.5 --nu-step 0.2 \
  --max-iter 10
```

#### Points clés

* **CUDA activé par défaut**
* le cluster `-1` est explicitement exclu
* `PEAK_SCALE` est fixé
* `K = PEAK² × TAU × SCALE_EJ` est imposé
* `NU_SCALE` est exploré **sur une grille discrète**
* `SCALE_EJ` et `TAU_SCALE` sont calculés analytiquement
* arrêt par convergence ou stagnation

Tout paramètre passé en ligne de commande **écrase les valeurs du JSON**
(`event_params.json` ne fournit que des valeurs par défaut).

---

## Résultats de calibration (exemple réel)

### Cluster 0 (14 événements)

* `NU_SCALE = 1.5`
* `TAU_SCALE = 23.52`
* `SCALE_EJ = 0.425`
* **MAE = 28.31 %**

---

### Cluster 1 (5 événements)

* `NU_SCALE = 1.5`
* `TAU_SCALE = 6.64`
* `SCALE_EJ = 1.51`
* **MAE = 17.57 %**

---

### Cluster 2 (10 événements)

* `NU_SCALE = 1.3`
* `TAU_SCALE = 8.72`
* `SCALE_EJ = 1.15`
* **MAE = 22.20 %**

---

### Fichiers produits

* `clusters.json`
* `best_clustering_params.json`
* `cluster_calibrations_iterative.json`
* `calibration_iterative.txt`

---

## Philosophie du pipeline

* Pas de templates
* Pas de fit continu caché
* Pas d’hypothèses astrophysiques fortes
* Séparation stricte :
  **estimation → sélection → calibration**
* Identification explicite de ce qui est **non contraint**

Ce pipeline ne cherche pas à remplacer les analyses bayésiennes LIGO.
Il répond à une autre question :

> **où est réellement l’information mesurable dans les données,
> et où elle ne l’est pas.**

