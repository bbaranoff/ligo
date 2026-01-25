#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_clustering.py
======================

Script d'orchestration pour trouver les meilleurs paramètres de clustering
qui minimisent l'erreur MAE sur les événements CLEAN (hors outliers).

Le script teste différentes combinaisons de paramètres et sélectionne
celle qui donne le meilleur compromis entre:
  - MAE faible sur les événements CLEAN
  - Nombre raisonnable d'événements conservés (> 20%)

Usage:
    python optimize_clustering.py
    python optimize_clustering.py --quick      # Test rapide (moins de combinaisons)
    python optimize_clustering.py --verbose    # Affiche détails pour chaque test
"""

import json
import subprocess
import sys
import os
from typing import Dict, List, Tuple, Any
import itertools
import re

# Configuration
RESULTS_GLOB = "results/GW*.json"
REFS_PATH = "ligo_refs.json"
CLUSTERS_PATH = "clusters.json"


class ClusteringTest:
    """Représente un test de clustering avec ses paramètres et résultats"""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.n_outliers = None
        self.n_clean = None
        self.mae_all = None
        self.mae_clean = None
        self.median_all = None
        self.median_clean = None
        self.n_problematic_clean = None
        self.improvement = None
        self.score = None  # Score composite
        
    def __repr__(self):
        return (f"ClusteringTest(method={self.params['method']}, "
                f"MAE={self.mae_clean:.1f}%, n_clean={self.n_clean})")


def run_clustering(params: Dict[str, Any], verbose: bool = False) -> bool:
    """
    Lance cluster_latent_kmeans.py avec les paramètres donnés
    
    Returns:
        True si succès, False sinon
    """
    cmd = [
        "python", "cluster_latent_kmeans.py",
        "--results-glob", RESULTS_GLOB,
        "--method", params["method"],
        "--k", str(params["k"]),
        "--use-logE",
        "--export", CLUSTERS_PATH,
    ]
    
    # Ajouter paramètres spécifiques selon méthode
    if "hdbscan" in params["method"]:
        cmd.extend([
            "--min-cluster-size", str(params.get("min_cluster_size", 3)),
            "--min-samples", str(params.get("min_samples", 5)),
            "--cluster-selection-epsilon", str(params.get("cluster_selection_epsilon", 0.0)),
        ])
    
    if "dbscan" in params["method"]:
        cmd.extend([
            "--eps", str(params.get("eps", 0.8)),
            "--min-samples", str(params.get("min_samples", 3)),
        ])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if verbose:
            print(result.stdout)
        
        if result.returncode != 0:
            print(f"❌ Clustering échoué: {result.stderr}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False


def run_stats(exclude_clusters: List[int] = [-1], verbose: bool = False) -> Dict[str, Any]:
    import glob
    import numpy as np

    if not os.path.exists(CLUSTERS_PATH):
        print(f"❌ {CLUSTERS_PATH} non trouvé")
        return {}

    with open(CLUSTERS_PATH, 'r') as f:
        cluster_data = json.load(f)

    event_to_cluster = cluster_data.get('event_to_cluster', {})

    if not os.path.exists(REFS_PATH):
        print(f"❌ {REFS_PATH} non trouvé")
        return {}

    with open(REFS_PATH, 'r') as f:
        refs = json.load(f)

    mass_refs = {
        ev: d["msun_c2"]
        for ev, d in refs.items()
        if isinstance(d, dict) and "msun_c2" in d
    }

    results_files = glob.glob(RESULTS_GLOB)
    results = {}

    for fpath in results_files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
                if "event" in data:
                    results[data["event"]] = data
        except:
            pass

    errors_all = []
    errors_clean = []

    for event, result in results.items():
        if event not in mass_refs:
            continue

        cluster_id = event_to_cluster.get(event, -999)
        if cluster_id in exclude_clusters:
            continue

        mass_pred = result.get("msun_c2", 0)
        mass_ref = mass_refs[event]

        if mass_pred <= 0 or mass_ref <= 0:
            continue

        err = abs(mass_pred - mass_ref) / mass_ref * 100
        errors_all.append(err)

        if cluster_id != -1:
            errors_clean.append(err)

    n_total = len(event_to_cluster)
    n_outliers = sum(1 for c in event_to_cluster.values() if c == -1)
    n_clean = len(errors_clean)

    metrics = {
        "n_total": n_total,
        "n_outliers": n_outliers,
        "n_clean": n_clean,
        "pct_clean": 100 * n_clean / n_total if n_total else 0,
    }

    if errors_all:
        metrics["mae_all"] = float(np.mean(errors_all))
        metrics["median_all"] = float(np.median(errors_all))

    if errors_clean:
        metrics["mae_clean"] = float(np.mean(errors_clean))
        metrics["median_clean"] = float(np.median(errors_clean))
        metrics["n_problematic_clean"] = sum(e > 50 for e in errors_clean)

        if errors_all:
            metrics["improvement"] = (
                (metrics["mae_all"] - metrics["mae_clean"])
                / metrics["mae_all"] * 100
            )

    return metrics


def compute_score(test: ClusteringTest) -> float:
    """
    Calcule un score composite pour évaluer la qualité du clustering
    
    Score = f(MAE_clean, n_clean, n_problematic)
    - Minimiser MAE_clean (poids 60%)
    - Maximiser n_clean (poids 30%, mais pénalité si < 15% ou > 40%)
    - Minimiser n_problematic_clean (poids 10%)
    """
    if test.mae_clean is None or test.n_clean is None:
        return float('inf')

    # Sécurité supplémentaire
    if test.n_clean < 8:
        return float('inf')
    
    # Composante 1: MAE (lower is better)
    mae_component = test.mae_clean
    
    # Composante 2: Nombre d'événements conservés
    pct_clean = test.n_clean / 63.0 * 100  # Assuming 63 total events
    
    # Pénalité si trop peu d'événements (< 15% = très sélectif)
    # Pénalité si trop d'événements (> 40% = peu sélectif)
    if pct_clean < 15:
        coverage_penalty = (15 - pct_clean) * 2  # Forte pénalité
    elif pct_clean > 40:
        coverage_penalty = (pct_clean - 40) * 1  # Pénalité modérée
    else:
        coverage_penalty = 0  # Sweet spot: 15-40%
    
    # Composante 3: Nombre d'événements problématiques (> 50% erreur)
    problematic_component = test.n_problematic_clean or 0
    
    # Score composite (lower is better)
    score = (
        0.6 * mae_component +
        0.3 * coverage_penalty +
        0.1 * problematic_component * 5  # Chaque événement problématique coûte 5 points
    )
    
    return score


def test_configuration(
    params: Dict[str, Any],
    verbose: bool = False,
    min_clean: int = 8
) -> ClusteringTest:
    """
    Teste une configuration de clustering complète
    """
    test = ClusteringTest(params)
    
    # 1. Clustering
    success = run_clustering(params, verbose=verbose)
    if not success:
        return test
    
    # 2. Stats
    metrics = run_stats(exclude_clusters=[-1], verbose=verbose)

    # ⛔ Rejet DUR : pas assez d'événements CLEAN
    if metrics.get("n_clean", 0) < min_clean:
        if verbose:
            print(
                f"   ⛔ Rejeté: n_clean={metrics.get('n_clean', 0)} < min_clean={min_clean}"
            )
        return test
    
    # 3. Remplir les résultats
    test.n_outliers = metrics.get('n_outliers')
    test.n_clean = metrics.get('n_clean')
    test.mae_all = metrics.get('mae_all')
    test.mae_clean = metrics.get('mae_clean')
    test.median_all = metrics.get('median_all')
    test.median_clean = metrics.get('median_clean')
    test.n_problematic_clean = metrics.get('n_problematic_clean')
    test.improvement = metrics.get('improvement')
    
    # 4. Score
    test.score = compute_score(test)
    
    return test


def generate_parameter_grid(quick: bool = False) -> List[Dict[str, Any]]:
    """
    Génère la grille de paramètres à tester
    """
    grid = []
    
    if quick:
        # Configuration rapide (9 tests)
        methods = ["hdbscan+kmeans", "dbscan+kmeans"]
        k_values = [4]
        min_samples_values = [4, 6]
        
        for method in methods:
            for k in k_values:
                for min_samples in min_samples_values:
                    if method == "hdbscan+kmeans":
                        grid.append({
                            "method": method,
                            "k": k,
                            "min_cluster_size": 3,
                            "min_samples": min_samples,
                            "cluster_selection_epsilon": 0.0,
                        })
                    else:  # dbscan+kmeans
                        grid.append({
                            "method": method,
                            "k": k,
                            "eps": 0.8,
                            "min_samples": min_samples,
                        })
    else:
        # Configuration complète (36 tests)
        methods = ["hdbscan+kmeans", "dbscan+kmeans"]
        k_values = [3, 4, 5]
        
        for method in methods:
            for k in k_values:
                if method == "hdbscan+kmeans":
                    for min_samples in [3, 4, 5, 6, 7, 8]:
                        grid.append({
                            "method": method,
                            "k": k,
                            "min_cluster_size": 3,
                            "min_samples": min_samples,
                            "cluster_selection_epsilon": 0.0,
                        })
                else:  # dbscan+kmeans
                    for eps in [0.6, 0.8, 1.0]:
                        for min_samples in [3, 4, 5, 6]:
                            grid.append({
                                "method": method,
                                "k": k,
                                "eps": eps,
                                "min_samples": min_samples,
                            })
    
    return grid


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Optimisation automatique des paramètres de clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python optimize_clustering.py              # Test complet (~36 configs)
  python optimize_clustering.py --quick      # Test rapide (~9 configs)
  python optimize_clustering.py --verbose    # Avec détails
        """
    )
    
    parser.add_argument("--quick", action="store_true", help="Test rapide avec moins de combinaisons")
    parser.add_argument("--verbose", action="store_true", help="Afficher les détails de chaque test")
    parser.add_argument("--top", type=int, default=10, help="Nombre de meilleurs résultats à afficher")

    parser.add_argument(
        "--min-clean",
        type=int,
        default=8,
        help="Nombre minimum d'événements CLEAN requis (non-outliers)"
    )

    args = parser.parse_args()
    
    # Vérifications préliminaires
    if not os.path.exists(REFS_PATH):
        print(f"❌ Erreur: {REFS_PATH} non trouvé")
        sys.exit(1)
    
    if not os.path.exists("results"):
        print(f"❌ Erreur: répertoire results/ non trouvé")
        sys.exit(1)
    
    # Générer la grille
    param_grid = generate_parameter_grid(quick=args.quick)
    
    print("="*80)
    print("🔍 OPTIMISATION DES PARAMÈTRES DE CLUSTERING")
    print("="*80)
    print(f"Nombre de configurations à tester : {len(param_grid)}")
    print(f"Mode : {'RAPIDE' if args.quick else 'COMPLET'}")
    print("="*80)
    
    # Tester toutes les configurations
    results = []
    
    for i, params in enumerate(param_grid, 1):
        print(f"\n[{i}/{len(param_grid)}] Test: {params['method']}, k={params['k']}", end="")
        
        if params['method'] == "hdbscan+kmeans":
            print(f", min_samples={params['min_samples']}")
        else:
            print(f", eps={params['eps']}, min_samples={params['min_samples']}")
        
        test = test_configuration(
            params,
            verbose=args.verbose,
            min_clean=args.min_clean
        )
        
        if test.mae_clean is not None:
            print(f"   → MAE={test.mae_clean:.1f}%, n_clean={test.n_clean}, score={test.score:.1f}")
            results.append(test)
        else:
            print(f"   → ÉCHEC")
    
    # Trier par score (lower is better)
    results.sort(key=lambda t: t.score)
    
    # Afficher le top N
    print("\n" + "="*80)
    print(f"🏆 TOP {args.top} MEILLEURES CONFIGURATIONS")
    print("="*80)
    
    for i, test in enumerate(results[:args.top], 1):
        print(f"\n{i}. Score: {test.score:.2f}")
        print(f"   Méthode    : {test.params['method']}")
        print(f"   Paramètres : k={test.params['k']}", end="")
        
        if test.params['method'] == "hdbscan+kmeans":
            print(f", min_samples={test.params['min_samples']}")
        else:
            print(f", eps={test.params['eps']}, min_samples={test.params['min_samples']}")
        
        print(f"   Résultats  :")
        print(f"      MAE ALL   : {test.mae_all:.1f}%")
        print(f"      MAE CLEAN : {test.mae_clean:.1f}%")
        print(f"      Médiane CLEAN : {test.median_clean:.1f}%")
        print(f"      Amélioration  : {test.improvement:+.1f}%")
        print(f"      N outliers    : {test.n_outliers}")
        print(f"      N clean       : {test.n_clean} ({test.n_clean/63*100:.1f}%)")
        print(f"      N problématiques : {test.n_problematic_clean}")
    
    # Sauvegarder la meilleure configuration
    if results:
        best = results[0]
        
        print("\n" + "="*80)
        print("✅ MEILLEURE CONFIGURATION TROUVÉE")
        print("="*80)
        
        # Ré-exécuter avec la meilleure config pour sauvegarder clusters.json
        print("\nRé-exécution avec la meilleure configuration...")
        run_clustering(best.params, verbose=True)
        
        # Afficher la commande
        print("\n📋 Commande pour reproduire:")
        print("-"*80)
        
        cmd = f"python cluster_latent_kmeans.py \\\n"
        cmd += f"  --results-glob '{RESULTS_GLOB}' \\\n"
        cmd += f"  --method {best.params['method']} \\\n"
        cmd += f"  --k {best.params['k']} \\\n"
        
        if best.params['method'] == "hdbscan+kmeans":
            cmd += f"  --min-cluster-size {best.params['min_cluster_size']} \\\n"
            cmd += f"  --min-samples {best.params['min_samples']} \\\n"
            cmd += f"  --cluster-selection-epsilon {best.params['cluster_selection_epsilon']} \\\n"
        else:
            cmd += f"  --eps {best.params['eps']} \\\n"
            cmd += f"  --min-samples {best.params['min_samples']} \\\n"
        
        cmd += f"  --use-logE \\\n"
        cmd += f"  --export clusters.json"
        
        print(cmd)
        
        print("\n📊 Résultats finaux:")
        print(f"   MAE CLEAN : {best.mae_clean:.1f}%")
        print(f"   Médiane   : {best.median_clean:.1f}%")
        print(f"   N clean   : {best.n_clean}/63 ({best.n_clean/63*100:.1f}%)")
        print(f"   Score     : {best.score:.2f}")
        
        # Sauvegarder les paramètres
        with open("best_clustering_params.json", "w") as f:
            json.dump({
                "params": best.params,
                "metrics": {
                    "mae_clean": best.mae_clean,
                    "median_clean": best.median_clean,
                    "n_clean": best.n_clean,
                    "n_outliers": best.n_outliers,
                    "improvement": best.improvement,
                    "score": best.score,
                }
            }, f, indent=2)
        
        print("\n💾 Paramètres sauvegardés dans: best_clustering_params.json")
        print("="*80)


if __name__ == "__main__":
    main()
