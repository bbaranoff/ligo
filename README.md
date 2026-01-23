# ligo

### *Testing LIGO research in Python*

Ce dépôt contient un **pipeline de traitement et d’analyse des signaux gravitationnels** détectés par les observatoires LIGO/Virgo. Il propose une approche expérimentale basée sur l’analyse spectrale brute, un clustering aveugle des événements, et une calibration par classe pour estimer l’énergie rayonnée, **sans utiliser directement les résultats d’inférence officiels**.

---

## 🚀 Objectif

L’objectif est d’explorer **la relation entre la morphologie spectrale des signaux LIGO et l’énergie radiative estimée**, en utilisant **des outils de traitement du signal**, une méthode de **clustering non supervisée**, et une **calibration par cluster** pour rapprocher les estimations des valeurs publiées.

Ce projet est purement exploratoire et vise à fournir une preuve de concept pour des approches alternatives d’analyse des signaux d’ondes gravitationnelles.

---

## 🧠 Pipeline d’analyse

Le pipeline se déroule en cinq grandes étapes :

1. **Traitement du signal brut**
   Extraction d’observables spectrales (PSD, énergie spectrale, fréquence moyenne, asymétrie) à partir des *strains bruts H1/L1/V1* des événements LIGO.

2. **Clustering spectral aveugle**
   Regroupement des événements par similarité spectrale, sans utiliser les énergies ou paramètres astrophysiques fournis par LIGO.

3. **Calibration par cluster**
   Pour chaque cluster, ajustement de deux paramètres effectifs :

   * `H_STAR` : correction de délai H1–L1 / effet géométrique
   * `SCALE_EJ` : facteur d’échelle énergétique
     Ces paramètres sont optimisés pour réduire l’écart entre les énergies calculées et les valeurs de référence officielles LIGO.

4. **Évaluation a posteriori**
   Calcul des erreurs relatives par événement et par cluster par rapport aux valeurs officielles publiées.

5. **Analyse des performances**
   Classement des meilleurs et pires ajustements pour interpréter la qualité de la calibration selon la morphologie des signaux.

---

## 📊 Résultats attendus

Le pipeline génère notamment :

* Un **classement des ajustements** par erreur relative.
* Des **statistiques par cluster** qui montrent quelles classes d’événements sont bien modélisées (erreur moyenne faible) et lesquelles ne le sont pas.
* Une **synthèse CSV/JSON** des paramètres calibrés et des erreurs.

Typiquement :

* Certains clusters atteignent des erreurs moyennes **~3–5 %**, ce qui indique une bonne cohérence entre l’approche spectrale et les valeurs officielles.
* D’autres clusters montrent des erreurs plus élevées (**~10–40 %+**), révélant les limites d’un modèle à deux paramètres pour ces morphologies.

---

## 🔧 Comment utiliser

### Pré-requis

Installer les dépendances :

```bash
pip install -r requirements.txt
```

### Téléchargement des données

Avant d’analyser, télécharge les fichiers NPZ LIGO/Virgo :

```bash
python ligo_npz_downloader.py
```

### Exécuter tout le pipeline

```bash
bash run_all.sh
```

### Calibration itérative par cluster

```bash
python run_iterative_calibration.py \
  --refs ligo_refs.json \
  --event-params event_params.json \
  --max-iter 10 \
  --tol 1e-4 \
  --k 4
```

Options utiles :

* `--exclude-cluster-minus1` : exclut les outliers (cluster -1)
* `--exclude-cls BNS` : exclut les événements BNS (neutron stars)

---

## 📁 Structure du dépôt

* `ligo_spectral_planck.py` — Extraction d’observables spectrales
* `cluster_latent_kmeans.py` — Clustering des événements
* `run_iterative_calibration.py` — Calibration et optimisation par cluster
* `plot_all_spectra.py` — Visualisation des spectres
* `results/` — Dossiers de résultats générés
* `event_params.json`, `ligo_refs.json` — Données d’entrée

---

## 🧪 Exemple de sortie

Le pipeline génère des classements comme :

```
🏆 TOP 10 MEILLEURS FITS
 1. GW190412 (Cluster 0)      Erreur: +2.37%
 2. GW170104 (Cluster 2)      Erreur: -3.01%
 ...
💀 TOP 10 PIRES FITS
 1. GW170817 (Cluster -1)     Erreur: +665.85%
 2. GW170608 (Cluster -1)     Erreur: +269.94%
 ...
```

Ce classement met en lumière les événements bien modélisés et ceux qui ne le sont pas, permettant une **interprétation physique et méthodologique**.

---

## 💡 Interprétation

Ce projet n’a pas vocation à remplacer les pipelines d’inférence officiels des collaborations LIGO/Virgo, mais à explorer **des approches complémentaires** basées sur des caractéristiques spectrales et des calibrations simples. Il met en évidence des classes d’événements compatibles avec une faible erreur (indiquant un invariant énergétique localisable par cluster) et d’autres hors du domaine de validité de ce modèle.

---

## 📜 Licence

Ce dépôt est en open-source. Pour les détails de licence, voir le fichier `LICENSE.md`.
