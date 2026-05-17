"""Orchestrateur : exécute les 38 transformations de la pierre de Rosette
MQ ↔ Hilbert/Signal ↔ RG, et présente le rapport agrégé.

Usage: python orchestrator.py [--verbose] [--filter green|yellow|orange|red]
"""

import sys
import time
import traceback
from rosetta_helpers import Status, TransitResult
import rosetta_green as G
import rosetta_yellow as Y
import rosetta_orange as O
import rosetta_red as R


# Liste des 38 entrées ordonnées
ENTRIES = [
    # 🟢 GREEN
    (G.g01_amplitude_phase,           "amplitude-phase"),
    (G.g02_spectrum,                  "spectre"),
    (G.g03_propagation,               "propagation"),
    (G.g04_dispersion,                "dispersion"),
    (G.g05_interferometry,            "interférométrie"),
    (G.g06_coherence,                 "cohérence"),
    (G.g07_squeezing,                 "squeezing"),
    (G.g08_entropy,                   "entropie"),
    (G.g09_thermality,                "thermalité"),
    (G.g10_cosmological_T,            "T cosmologique"),
    # 🟡 YELLOW
    (Y.y11_cosmological_holography,   "holo cosmologique (Dirac)"),
    (Y.y12_er_epr,                    "ER=EPR"),
    (Y.y13_ads_cft,                   "AdS/CFT"),
    (Y.y14_causality,                 "causalité QFT↔géo"),
    (Y.y15_hawking_unruh,             "Hawking-Unruh"),
    (Y.y16_jacobson,                  "Jacobson"),
    # 🟠 ORANGE (MQ-only)
    (O.o17_tensor_product,            "produit tensoriel"),
    (O.o18_born,                      "Born"),
    (O.o19_spin,                      "spin"),
    (O.o20_exchange_statistics,       "stats d'échange"),
    (O.o21_no_cloning,                "no-cloning"),
    (O.o22_bell_ks,                   "Bell/KS"),
    # 🟠 ORANGE (RG-only)
    (O.o23_background_independence,   "indép. fond"),
    (O.o24_singularities,             "singularités"),
    (O.o25_horizons,                  "horizons"),
    (O.o26_topology,                  "topologie"),
    (O.o27_cosmological_constant,     "Λ"),
    (O.o28_penrose_process,           "Penrose process"),
    (O.o29_adm_bondi,                 "ADM/Bondi"),
    # 🔴 RED
    (R.r30_measurement_collapse,      "mesure/collapse"),
    (R.r31_ds_holography,             "dS holography"),
    (R.r32_quantum_metric,            "quantif. g_μν"),
    (R.r33_time_emergence,            "émergence du temps"),
    (R.r34_hierarchy,                 "hiérarchie"),
    (R.r35_dark_sector,               "DM/DE"),
    (R.r36_inflation,                 "inflation"),
    (R.r37_uv_renormalization,        "renorm. UV gravité"),
    (R.r38_geometry_from_entanglement, "ent. → géométrie"),
]


def run_all(verbose=False, filter_status=None):
    """Exécute toutes les entrées et collecte les résultats."""
    results = []
    failed = []
    t_start = time.time()
    print(f"\n{'='*70}")
    print(f"  PIERRE DE ROSETTE MQ ↔ HILBERT/SIGNAL ↔ RG")
    print(f"  Exécution de {len(ENTRIES)} transformations")
    print(f"{'='*70}\n")

    for fn, short_name in ENTRIES:
        try:
            r = fn()
        except Exception as e:
            r = TransitResult(short_name, Status.RED,
                              transformation="ERROR",
                              notes=f"Exception : {e}")
            failed.append((short_name, traceback.format_exc()))

        if filter_status and r.status not in filter_status:
            continue

        results.append(r)
        if verbose:
            print(r.report())
            print()
        else:
            print(r.short())

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  RAPPORT FINAL ({elapsed*1000:.1f} ms)")
    print(f"{'='*70}\n")

    by_status = {s: [r for r in results if r.status == s] for s in Status}
    counts = {s.name: len(by_status[s]) for s in Status}

    print(f"  🟢 GREEN     : {counts['GREEN']:2d} entrées — transit bidirectionnel prouvé")
    n_ok = sum(1 for r in by_status[Status.GREEN] if r.success)
    print(f"     dont {n_ok}/{counts['GREEN']} avec payload récupéré intact")
    print()
    print(f"  🟡 YELLOW    : {counts['YELLOW']:2d} entrées — transit partiel avec caveats")
    print()
    print(f"  🟠 ORANGE-MQ : {counts['ORANGE_MQ']:2d} entrées — MQ-only (RG vide)")
    print(f"  🟠 ORANGE-RG : {counts['ORANGE_RG']:2d} entrées — RG-only (MQ vide)")
    print()
    print(f"  🔴 RED       : {counts['RED']:2d} entrées — transformation non-consensuelle")
    print()
    print(f"  TOTAL        : {len(results):2d}/{len(ENTRIES)} entrées exécutées")

    # Coverage breakdown
    total = len(ENTRIES)
    bidirectional = counts['GREEN'] + counts['YELLOW']
    unilateral = counts['ORANGE_MQ'] + counts['ORANGE_RG']
    open_ = counts['RED']
    print()
    print(f"  Couverture pierre :")
    print(f"    pont opérationnel  : {bidirectional}/{total} ({100*bidirectional/total:.0f}%)")
    print(f"    pont unilatéral    : {unilateral}/{total} ({100*unilateral/total:.0f}%)")
    print(f"    chantier ouvert    : {open_}/{total} ({100*open_/total:.0f}%)")

    if failed:
        print(f"\n  ⚠ {len(failed)} exception(s) :")
        for name, tb in failed:
            print(f"    - {name}")
            if verbose:
                print(tb)

    return results


def main():
    args = sys.argv[1:]
    verbose = '--verbose' in args or '-v' in args
    filter_status = None
    if '--filter' in args:
        idx = args.index('--filter')
        if idx + 1 < len(args):
            kind = args[idx + 1].lower()
            mapping = {
                'green': [Status.GREEN],
                'yellow': [Status.YELLOW],
                'orange': [Status.ORANGE_MQ, Status.ORANGE_RG],
                'red': [Status.RED],
            }
            filter_status = mapping.get(kind)

    run_all(verbose=verbose, filter_status=filter_status)


if __name__ == '__main__':
    main()
