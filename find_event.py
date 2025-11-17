#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
import sys
import os

# Importer les fonctions de tau.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tau import load_tau, tau_to_date

# Charger les données
tau_dict = load_tau()
ANCHOR_EVENT = "GW190521"
ANCHOR_DATE = datetime(2017, 9, 27, 9, 50, 44, 400000)
tau_ref = tau_dict[ANCHOR_EVENT]

# Préparer les données
tau_values = []
gps_timestamps = []

for event, tau_val in tau_dict.items():
    date_calc = tau_to_date(tau_val, tau_ref)
    tau_values.append(tau_val)
    gps_timestamps.append(date_calc.timestamp())

# Régression linéaire
tau_array = np.array(tau_values)
gps_array = np.array(gps_timestamps)
slope, intercept = np.polyfit(tau_array, gps_array, 1)

# Fonction de prédiction
def predict_gps(tau):
    return datetime.fromtimestamp(slope * tau + intercept)

# Créer le plot
plt.figure(figsize=(12, 8))

# Points observés
dates_obs = [datetime.fromtimestamp(ts) for ts in gps_timestamps]
plt.plot(dates_obs, tau_values, 'bo', markersize=6, label='Événements')

# Droite de régression
tau_fit = np.linspace(min(tau_values), max(tau_values), 100)
dates_fit = [datetime.fromtimestamp(slope * t + intercept) for t in tau_fit]
plt.plot(dates_fit, tau_fit, 'r-', linewidth=2, label='Régression linéaire')

# Formatage
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gcf().autofmt_xdate()

plt.xlabel('Date GPS LIGO')
plt.ylabel('τ (jours)')
plt.title('Droite de Régression: τ → GPS LIGO')
plt.legend()
plt.grid(True, alpha=0.3)

# Annotations
for event, tau_val, date in zip(tau_dict.keys(), tau_values, dates_obs):
    plt.annotate(event, (date, tau_val), xytext=(5,5), textcoords='offset points', fontsize=7)

plt.tight_layout()
plt.show()

print(f"\nÉquation: GPS = {slope:.6f}·τ + {intercept:.1f}")
print("Utilisez predict_gps(tau) pour prédire des dates GPS")
