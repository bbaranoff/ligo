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
