#!/bin/bash
sudo apt install python3-venv
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
bash run_all.sh
python plot_all_spectra.py
