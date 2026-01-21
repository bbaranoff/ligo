#!/bin/bash

# 1. créer l’environnement seulement si absent
if [ ! -d ".env" ]; then
    python3 -m venv .env
fi

# 2. ACTIVER l'env
source .env/bin/activate

# 3. installer
pip install --upgrade pip
pip3 install scipy gwosc numpy matplotlib gwpy numba
# ou pip install -r requirements.txt

# 4. lancer pipeline
bash run_all.sh
python3 plot_all_spectra.py
python3 cluster_kmeans_pca.py
cat clusters_kmeans.txt
xdg-open plots/spectres_normalises.png
