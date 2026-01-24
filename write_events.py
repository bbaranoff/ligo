#!/usr/bin/env python3
"""
Script pour récupérer les paramètres d'événements LIGO depuis GWOSC API
et générer event_params.json et ligo_refs.json avec paramètres adaptatifs

NOUVEAU : Génère automatiquement des paramètres optimisés par masse
- ULTRA_LIGHT (ΔM < 1.0 M☉) : signal_win=3.0s, flow=15Hz
- LIGHT (1.0-1.5 M☉)        : signal_win=2.0s, flow=18Hz  
- MEDIUM (1.5-4.0 M☉)       : signal_win=1.2s, flow=20Hz (standard)
- MASSIVE (ΔM ≥ 4.0 M☉)     : signal_win=0.6s, flow=25Hz, fhigh=500Hz
"""

import requests
import json
import time
import math

# --- Constantes physiques ---
C = 299792458.0
M_SUN = 1.98847e30

# --- Paramètres par défaut pour l'analyse ---
DEFAULT_PARAMS = {
    "flow": 20.0,
    "fhigh": 350.0,
    "tau_band": [30.0, 350.0],
    "nu_band": [20.0, 500.0],
    "signal_win": 1.2,
    "noise_pad": 1200.0
}

# --- Paramètres adaptatifs par catégorie de masse ---
ADAPTIVE_PARAMS = {
    "ULTRA_LIGHT": {  # ΔM < 1.0 M☉
        "flow": 15.0,
        "fhigh": 400.0,
        "tau_band": [20.0, 400.0],
        "nu_band": [15.0, 450.0],
        "signal_win": 3.0,      # Signal très long
        "noise_pad": 600.0      # PSD proche
    },
    "LIGHT": {  # 1.0 ≤ ΔM < 1.5 M☉
        "flow": 18.0,
        "fhigh": 380.0,
        "tau_band": [25.0, 380.0],
        "nu_band": [18.0, 450.0],
        "signal_win": 2.0,
        "noise_pad": 800.0
    },
    "MEDIUM": {  # 1.5 ≤ ΔM < 4.0 M☉
        "flow": 20.0,
        "fhigh": 350.0,
        "tau_band": [30.0, 350.0],
        "nu_band": [20.0, 500.0],
        "signal_win": 1.2,
        "noise_pad": 1200.0
    },
    "MASSIVE": {  # ΔM ≥ 4.0 M☉
        "flow": 25.0,
        "fhigh": 500.0,
        "tau_band": [40.0, 500.0],
        "nu_band": [30.0, 600.0],
        "signal_win": 0.6,      # Signal court
        "noise_pad": 1500.0     # PSD loin
    }
}

EVENTS = [
    "GW150914", "GW151226", "GW170104", "GW170809",
    "GW170814", "GW170818", "GW170823", "GW190403_051519", "GW190408_181802",
    "GW190412", "GW190413_052954", "GW190413_134308", "GW190421_213856",
    "GW190426_190642", "GW190503_185404",
    "GW190512_180714", "GW190513_205428", "GW190514_065416", "GW190517_055101",
    "GW190519_153544", "GW190521", "GW190521_074359", "GW190527_092055",
    "GW190602_175927", "GW190701_203306",
    "GW190706_222641", "GW190707_093326", "GW190719_215514",
    "GW190720_000836", "GW190725_174728", "GW190727_060333", "GW190728_064510",
    "GW190731_140936", "GW190803_022701", "GW190805_211137",
    "GW190828_063405", "GW190828_065509", "GW190910_112807",
    "GW190915_235702", "GW190916_200658", "GW190924_021846",
    "GW190929_012149", "GW190930_133541", "GW191109_010717",
    "GW191126_115259", "GW191127_050227", "GW191129_134029", "GW191204_171526",
    "GW191215_223052", "GW191222_033537", "GW191230_180458",
    "GW200128_022011", "GW200129_065458", "GW200202_154313",
    "GW200208_222617", "GW200219_094415", "GW200220_061928",
    "GW200220_124850", "GW200224_222234", "GW200225_060421", "GW200306_093714",
    "GW200308_173609", "GW200311_115853", "GW200316_215756",
    "GW200322_091133"
]

RELEASES = ["GWTC-3-confident", "GWTC-2.1-confident", "GWTC-1-confident"]
VERSIONS = ["v4", "v3", "v2", "v1"]

# Mapping des catalogues vers DOIs
CATALOG_DOIS = {
    "GWTC-1-confident": "https://doi.org/10.1103/PhysRevX.9.031040",
    "GWTC-2.1-confident": "https://doi.org/10.1103/PhysRevX.11.021053",
    "GWTC-3-confident": "https://doi.org/10.1103/PhysRevX.13.011048",
}


def get_adaptive_category(delta_m):
    """Détermine la catégorie adaptative basée sur la masse rayonnée."""
    if delta_m < 1.0:
        return "ULTRA_LIGHT"
    elif delta_m < 1.5:
        return "LIGHT"
    elif delta_m < 4.0:
        return "MEDIUM"
    else:
        return "MASSIVE"


def fetch_event_json(event):
    """Récupère le JSON d'un événement depuis GWOSC API."""
    for rel in RELEASES:
        for ver in VERSIONS:
            url = f"https://gwosc.org/eventapi/json/{rel}/{event}/{ver}/"
            try:
                r = requests.get(url, timeout=15)
                if r.status_code != 200:
                    continue
                if "application/json" not in r.headers.get("Content-Type", ""):
                    continue
                data = r.json()
                # Retourner avec metadata de source
                return {
                    "data": data,
                    "release": rel,
                    "version": ver,
                    "url": url
                }
            except Exception as e:
                continue
    return None


def main():
    print("="*70)
    print("RÉCUPÉRATION DES PARAMÈTRES D'ÉVÉNEMENTS DEPUIS GWOSC")
    print("AVEC ADAPTATION AUTOMATIQUE PAR MASSE")
    print("="*70)
    
    event_params = {
        "_meta": {
            "description": "GWOSC event parameters for the spectral-tau pipeline with adaptive parameters based on radiated mass. Categories: ULTRA_LIGHT (<1.0 M☉), LIGHT (1.0-1.5 M☉), MEDIUM (1.5-4.0 M☉), MASSIVE (≥4.0 M☉).",
            "sources": {
                "gwtc21_table": "https://gwosc.org/eventapi/html/GWTC-2.1-confident/?pagesize=all",
                "query_lastver_page5": "https://gwosc.org/eventapi/html/query/show?lastver=true&page=5&pagesize=50&release=GWTC-1-confident%2CGWTC-2.1-confident%2CGWTC-3-confident%2CGWTC-4.0",
                "gwtc1_paper_doi": "https://doi.org/10.1103/PhysRevX.9.031040",
                "gwtc2_paper_doi": "https://doi.org/10.1103/PhysRevX.11.021053"
            },
            "adaptive_categories": ADAPTIVE_PARAMS
        },
        "default": DEFAULT_PARAMS.copy()
    }
    
    ligo_refs = {}
    
    success_count = 0
    fail_count = 0
    
    # Stats par catégorie
    category_counts = {cat: 0 for cat in ADAPTIVE_PARAMS.keys()}
    category_counts["DEFAULT"] = 0  # Pour événements sans masse

    for ev in EVENTS:
        print(f"\n[INFO] Récupération de {ev}...")
        result = fetch_event_json(ev)

        if not result or "data" not in result or "events" not in result["data"]:
            print(f"[WARN] Pas de JSON utilisable pour {ev}")
            fail_count += 1
            time.sleep(0.8)
            continue

        data = result["data"]
        release = result["release"]
        version = result["version"]
        source_url = result["url"]
        
        _, evdata = next(iter(data["events"].items()))

        # ----------------------------------------------------
        # Calcul des masses pour déterminer la catégorie
        # ----------------------------------------------------
        m_tot = evdata.get("total_mass_source")
        m_fin = evdata.get("final_mass_source")
        
        delta_m = None
        category = "DEFAULT"
        
        if m_tot is not None and m_fin is not None:
            delta_m = max(m_tot - m_fin, 0.0)
            category = get_adaptive_category(delta_m)
        
        # ----------------------------------------------------
        # event_params.json (paramètres adaptatifs)
        # ----------------------------------------------------
        gps = evdata.get("GPS")
        dist = evdata.get("luminosity_distance")

        if gps is None:
            print(f"[WARN] Pas de GPS pour {ev}")
            fail_count += 1
            time.sleep(0.8)
            continue

        # Créer l'entrée avec GPS et distance
        event_entry = {
            "gps": gps,
            "distance_mpc": round(dist) if dist is not None else None,
        }
        
        # Ajouter les paramètres adaptatifs selon la catégorie
        if category in ADAPTIVE_PARAMS:
            event_entry.update(ADAPTIVE_PARAMS[category].copy())
            category_counts[category] += 1
        else:
            event_entry.update(DEFAULT_PARAMS.copy())
            category_counts["DEFAULT"] += 1
        
        # Ajouter la catégorie pour diagnostic
        event_entry["_adaptive_category"] = category
        
        # Ajouter les sources
        event_entry["sources"] = {
            "gwosc_event_html": source_url.replace("/json/", "/html/"),
            "gwosc_event_json": source_url,
            "gwosc_table_or_query": f"https://gwosc.org/eventapi/html/{release}/?pagesize=all",
            "catalog_paper_doi": CATALOG_DOIS.get(release, "")
        }
        
        event_params[ev] = event_entry

        # ----------------------------------------------------
        # ligo_refs.json (énergies de référence)
        # ----------------------------------------------------
        if delta_m is not None:
            energy_J = delta_m * M_SUN * C**2

            # Classification
            cls = "BBH"
            if m_tot < 6:
                cls = "BNS"
            elif m_tot < 10:
                cls = "NSBH"

            ligo_refs[ev] = {
                "msun_c2": round(delta_m, 6),
                "energy_J": energy_J,
                "cls": cls,
                "source": f"GWOSC EventAPI ({release}/{version})"
            }
            
            success_count += 1
            print(f"[OK] {ev}: GPS={gps}, dist={dist} Mpc, ΔM={delta_m:.4f} M☉ → {category}")
        else:
            print(f"[WARN] Masses manquantes pour {ev}, utilise paramètres par défaut")
            success_count += 1
        
        time.sleep(0.8)

    # Sauvegarder les fichiers
    print("\n" + "="*70)
    print("SAUVEGARDE DES FICHIERS")
    print("="*70)
    
    with open("event_params.json", "w") as f:
        json.dump(event_params, f, indent=2)
    print(f"✅ event_params.json: {len(event_params)-2} événements (+ _meta + default)")
    
    with open("ligo_refs.json", "w") as f:
        json.dump(ligo_refs, f, indent=2)
    print(f"✅ ligo_refs.json: {len(ligo_refs)} événements")
    
    # Stats par catégorie
    print("\n" + "="*70)
    print("DISTRIBUTION PAR CATÉGORIE ADAPTATIVE")
    print("="*70)
    for cat in ["ULTRA_LIGHT", "LIGHT", "MEDIUM", "MASSIVE", "DEFAULT"]:
        count = category_counts.get(cat, 0)
        if count > 0:
            emoji = {"ULTRA_LIGHT": "🔴", "LIGHT": "🟠", "MEDIUM": "🟢", "MASSIVE": "🔵", "DEFAULT": "⚪"}
            print(f"{emoji[cat]} {cat:15} : {count:3} événements")
    
    print("\n" + "="*70)
    print("RÉSUMÉ")
    print("="*70)
    print(f"Total événements traités : {len(EVENTS)}")
    print(f"Succès                   : {success_count}")
    print(f"Échecs                   : {fail_count}")
    print("="*70)
    
    # Afficher quelques exemples
    print("\n" + "="*70)
    print("EXEMPLES DE PARAMÈTRES ADAPTATIFS")
    print("="*70)
    
    examples = {
        "GW190924_021846": "ULTRA_LIGHT",
        "GW151226": "LIGHT", 
        "GW150914": "MEDIUM",
        "GW190426_190642": "MASSIVE"
    }
    
    for event, expected_cat in examples.items():
        if event in event_params:
            params = event_params[event]
            cat = params.get("_adaptive_category", "?")
            print(f"\n{event} ({cat}):")
            print(f"  signal_win = {params['signal_win']:.1f} s")
            print(f"  noise_pad  = {params['noise_pad']:.0f} s")
            print(f"  flow       = {params['flow']:.0f} Hz")
            print(f"  fhigh      = {params['fhigh']:.0f} Hz")


if __name__ == "__main__":
    main()
