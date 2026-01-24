#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
cluster_latent_kmeans.py
========================

Module de clustering intelligent pour les événements d'ondes gravitationnelles LIGO.
Supporte HDBSCAN, DBSCAN et KMeans avec sélection automatique et API unifiée.

Features:
- Extraction automatique de caractéristiques depuis les résultats JSON
- Normalisation StandardScaler pour une meilleure stabilité
- Support multi-algorithme : HDBSCAN (par défaut), DBSCAN, KMeans
- Détection automatique des outliers
- Visualisation optionnelle (matplotlib)
- API backward-compatible

Author: Enhanced version
Date: 2025
"""

from __future__ import annotations

import glob
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from sklearn.preprocessing import StandardScaler

# Imports conditionnels pour les algorithmes de clustering
try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    warnings.warn("hdbscan non disponible. Utiliser DBSCAN ou KMeans à la place.")

try:
    from sklearn.cluster import DBSCAN, KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    raise ImportError("scikit-learn est requis pour ce module")

# Imports optionnels pour visualisation
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================
# Structure de données pour les caractéristiques
# ============================================================

@dataclass
class EventFeatures:
    """
    Conteneur pour les caractéristiques spectrales et temporelles d'un événement GW.
    
    Attributes:
        event: Nom de l'événement (ex: GW150914)
        logE: Log10 de l'énergie [J] (optionnel, non utilisé par défaut)
        nu_mean: Fréquence effective moyenne [Hz]
        tau: Délai temporel H1-L1 [s]
        extra: Dictionnaire pour caractéristiques additionnelles
    """
    event: str
    logE: float = 0.0
    nu_mean: float = 0.0
    tau: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def row(self, use_logE: bool = False) -> np.ndarray:
        """
        Retourne les caractéristiques sous forme de vecteur numpy.
        
        Args:
            use_logE: Si True, inclut logE dans le vecteur (3D: [logE, nu_mean, tau])
                     Si False, retourne [nu_mean, tau] (2D, par défaut)
        
        Returns:
            Vecteur numpy de caractéristiques
        """
        if use_logE:
            return np.array([self.logE, self.nu_mean, self.tau], dtype=float)
        else:
            return np.array([self.nu_mean, self.tau], dtype=float)
    
    def __repr__(self) -> str:
        return (f"EventFeatures(event={self.event!r}, "
                f"nu_mean={self.nu_mean:.2f} Hz, tau={self.tau:.6e} s)")


# ============================================================
# Extraction des caractéristiques depuis JSON
# ============================================================

def compute_features(
    path: str,
    f_split: float = 150.0,
    verbose: bool = False,
) -> Optional[EventFeatures]:
    """
    Extrait les caractéristiques spectrales d'un fichier JSON d'analyse.
    
    Args:
        path: Chemin vers le fichier JSON
        f_split: Fréquence de séparation [Hz] (legacy, non utilisé actuellement)
        verbose: Afficher les warnings
    
    Returns:
        EventFeatures si succès, None si échec
    """
    try:
        with open(path, "r") as f:
            d = json.load(f)
        
        # Nom de l'événement (obligatoire)
        ev = d.get("event")
        if ev is None:
            if verbose:
                print(f"⚠️  {path}: pas de champ 'event'")
            return None
        
        # Énergie (optionnel, pour visualisation)
        energy_J = d.get("energy_J", 0.0)
        if np.isfinite(energy_J) and energy_J > 0:
            logE = float(np.log10(energy_J))
        else:
            logE = 0.0
        
        # Fréquence effective (plusieurs formats possibles)
        nu_eff = 0.0
        if "nu_eff" in d:
            nu_obj = d["nu_eff"]
            if isinstance(nu_obj, dict):
                # Essayer plusieurs clés possibles
                for key in ["nu_eff_energy", "mean", "median", "value"]:
                    if key in nu_obj:
                        nu_eff = float(nu_obj[key])
                        break
            elif isinstance(nu_obj, (int, float)):
                nu_eff = float(nu_obj)
        
        if not np.isfinite(nu_eff):
            nu_eff = 0.0
        
        # Délai temporel H1-L1
        tau = d.get("tau_hl_s", 0.0)
        if not np.isfinite(tau):
            tau = 0.0
        tau = float(tau)
        
        # Caractéristiques additionnelles (optionnel)
        extra = {}
        for key in ["m_sun", "distance_mpc", "gps"]:
            if key in d:
                val = d[key]
                if np.isfinite(val):
                    extra[key] = float(val)
        
        return EventFeatures(
            event=ev,
            logE=logE,
            nu_mean=nu_eff,
            tau=tau,
            extra=extra,
        )
        
    except json.JSONDecodeError as e:
        if verbose:
            print(f"⚠️  {path}: JSON invalide - {e}")
        return None
    except Exception as e:
        if verbose:
            print(f"⚠️  {path}: erreur - {e}")
        return None


# ============================================================
# Clustering unifié
# ============================================================

def cluster_events(
    X: np.ndarray,
    method: str = "hdbscan",
    **kwargs,
) -> np.ndarray:
    """
    Applique un algorithme de clustering sur les données.
    
    Args:
        X: Matrice de features (n_samples, n_features), déjà normalisée
        method: Algorithme ('hdbscan', 'dbscan', 'kmeans', 'hdbscan+kmeans')
        **kwargs: Paramètres spécifiques à l'algorithme
    
    Returns:
        labels: Vecteur de labels de cluster (n_samples,)
                -1 indique un outlier pour HDBSCAN/DBSCAN
    
    Raises:
        ValueError: Si l'algorithme demandé n'est pas disponible
    """
    method = method.lower()
    
    # ========== HDBSCAN + KMeans ==========
    if method == "hdbscan+kmeans":
        if not HAS_HDBSCAN:
            raise ValueError("HDBSCAN non disponible. Installer: pip install hdbscan")
        
        # Phase 1: HDBSCAN pour détecter les outliers
        min_cluster_size = kwargs.get("min_cluster_size", 3)
        min_samples = kwargs.get("min_samples", 2)
        cluster_selection_epsilon = kwargs.get("cluster_selection_epsilon", 0.0)
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method='eom',
            metric='euclidean',
        )
        hdb_labels = clusterer.fit_predict(X)
        
        # Séparer outliers (-1) et inliers (≥0)
        outlier_mask = (hdb_labels == -1)
        n_outliers = np.sum(outlier_mask)
        
        if n_outliers == len(X):
            # Tous sont outliers, utiliser KMeans sur tout
            print(f"[INFO] HDBSCAN: tous outliers ({n_outliers}), utilisation de KMeans complet")
            k = kwargs.get("k", kwargs.get("n_clusters", 3))
            random_state = kwargs.get("seed", kwargs.get("random_state", 42))
            clusterer_km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            return clusterer_km.fit_predict(X)
        
        if n_outliers == 0:
            # Aucun outlier, retourner les clusters HDBSCAN
            print(f"[INFO] HDBSCAN: aucun outlier, {len(np.unique(hdb_labels))} clusters")
            return hdb_labels
        
        # Phase 2: KMeans sur les inliers
        print(f"[INFO] HDBSCAN: {n_outliers} outliers détectés, KMeans sur {len(X)-n_outliers} inliers")
        
        X_inliers = X[~outlier_mask]
        k = kwargs.get("k", kwargs.get("n_clusters", 3))
        random_state = kwargs.get("seed", kwargs.get("random_state", 42))
        
        clusterer_km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km_labels = clusterer_km.fit_predict(X_inliers)
        
        # Recombiner: outliers gardent -1, inliers ont leurs nouveaux labels KMeans
        final_labels = np.full(len(X), -1, dtype=int)
        final_labels[~outlier_mask] = km_labels
        
        return final_labels
    
    # ========== HDBSCAN ==========
    if method == "hdbscan":
        if not HAS_HDBSCAN:
            raise ValueError("HDBSCAN non disponible. Installer: pip install hdbscan")
        
        min_cluster_size = kwargs.get("min_cluster_size", 3)
        min_samples = kwargs.get("min_samples", 2)
        cluster_selection_epsilon = kwargs.get("cluster_selection_epsilon", 0.0)
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method='eom',  # Excess of Mass
            metric='euclidean',
        )
        labels = clusterer.fit_predict(X)
        return labels
    
    # ========== DBSCAN ==========
    elif method == "dbscan":
        eps = kwargs.get("eps", kwargs.get("db_eps", 0.5))
        min_samples = kwargs.get("min_samples", kwargs.get("db_min_samples", 3))
        
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = clusterer.fit_predict(X)
        return labels
    
    # ========== KMeans ==========
    elif method == "kmeans":
        k = kwargs.get("k", kwargs.get("n_clusters", 3))
        random_state = kwargs.get("seed", kwargs.get("random_state", 42))
        
        clusterer = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = clusterer.fit_predict(X)
        return labels
    
    else:
        raise ValueError(f"Méthode inconnue: {method}. Utiliser 'hdbscan', 'dbscan' ou 'kmeans'")


# ============================================================
# API principale - Backward compatible
# ============================================================

def load_cluster_assignments(
    results_glob: str,
    f_split: float = 150.0,
    method: str = "hdbscan",
    use_logE: bool = False,
    normalize: bool = True,
    verbose: bool = False,
    seed: int | None = 42,
    **kwargs,
) -> Tuple[Dict[str, int], List[EventFeatures]]:
    """
    Charge les résultats JSON, extrait les features et effectue le clustering.
    
    Args:
        results_glob: Pattern glob pour les fichiers JSON (ex: "results/GW*.json")
        f_split: Fréquence de séparation [Hz] (legacy, non utilisé)
        method: Algorithme de clustering ('hdbscan', 'dbscan', 'kmeans')
        use_logE: Inclure logE dans les features (3D au lieu de 2D)
        normalize: Normaliser les features avec StandardScaler
        verbose: Afficher les informations de debug
        seed: Random seed pour reproductibilité (KMeans seulement)
        **kwargs: Paramètres additionnels pour l'algorithme de clustering
            - HDBSCAN: min_cluster_size, min_samples, cluster_selection_epsilon
            - DBSCAN: eps (ou db_eps), min_samples (ou db_min_samples)
            - KMeans: k (ou n_clusters)
    
    Returns:
        event_to_cluster: Dict[event_name -> cluster_id]
        feats: Liste des EventFeatures extraites
    
    Examples:
        >>> # HDBSCAN avec paramètres par défaut
        >>> mapping, feats = load_cluster_assignments("results/GW*.json")
        
        >>> # DBSCAN avec paramètres personnalisés
        >>> mapping, feats = load_cluster_assignments(
        ...     "results/GW*.json",
        ...     method="dbscan",
        ...     eps=1.2,
        ...     min_samples=3
        ... )
        
        >>> # KMeans avec 4 clusters
        >>> mapping, feats = load_cluster_assignments(
        ...     "results/GW*.json",
        ...     method="kmeans",
        ...     k=4
        ... )
    """
    # 1) Charger les fichiers JSON
    paths = sorted(glob.glob(results_glob))
    if len(paths) == 0:
        raise RuntimeError(f"Aucun fichier trouvé pour pattern: {results_glob}")
    
    if verbose:
        print(f"📂 {len(paths)} fichiers JSON trouvés")
    
    # 2) Extraire les features
    feats = []
    for p in paths:
        f = compute_features(p, f_split=f_split, verbose=verbose)
        if f is not None:
            feats.append(f)
    
    if len(feats) == 0:
        raise RuntimeError("Aucun événement valide après extraction des features")
    
    if verbose:
        print(f"✅ {len(feats)} événements avec features valides")
    
    # 3) Cas particuliers : très peu d'événements
    if len(feats) == 1:
        if verbose:
            print("⚠️  1 seul événement → cluster 0")
        return {feats[0].event: 0}, feats
    
    if len(feats) == 2:
        if verbose:
            print("⚠️  2 événements → cluster 0")
        return {f.event: 0 for f in feats}, feats
    
    # 4) Construire la matrice de features
    X = np.array([f.row(use_logE=use_logE) for f in feats], dtype=float)
    
    if verbose:
        print(f"📊 Matrice de features: {X.shape}")
        print(f"   Features: {'[logE, nu_mean, tau]' if use_logE else '[nu_mean, tau]'}")
    
    # 5) Normalisation (recommandé pour DBSCAN/HDBSCAN)
    if normalize:
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)
        if verbose:
            print(f"🔧 Normalisation StandardScaler appliquée")
            print(f"   Mean: {scaler.mean_}")
            print(f"   Std:  {scaler.scale_}")
    else:
        X_norm = X
    
    # 6) Clustering
    if verbose:
        print(f"🔍 Clustering avec méthode: {method.upper()}")
    
    # Transmettre le seed si fourni et pas déjà dans kwargs
    if seed is not None and 'seed' not in kwargs:
        kwargs['seed'] = seed
    
    labels = cluster_events(X_norm, method=method, **kwargs)
    
    # 7) Statistiques
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels >= 0])
    n_outliers = np.sum(labels == -1)
    
    if verbose:
        print(f"📈 Résultats:")
        print(f"   Clusters trouvés: {n_clusters}")
        print(f"   Outliers (label=-1): {n_outliers}")
        for lbl in sorted(unique_labels):
            count = np.sum(labels == lbl)
            print(f"   Cluster {lbl:2}: {count} événements")
    
    # 8) Construction du dictionnaire
    event_to_cluster = {f.event: int(lbl) for f, lbl in zip(feats, labels)}
    
    return event_to_cluster, feats


# ============================================================
# Visualisation (optionnel)
# ============================================================

def plot_clusters(
    feats: List[EventFeatures],
    labels: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Visualise les clusters dans l'espace (nu_mean, tau).
    
    Args:
        feats: Liste des EventFeatures
        labels: Vecteur de labels de cluster
        save_path: Si fourni, sauvegarde la figure à ce chemin
        show: Afficher la figure interactivement
    """
    if not HAS_MATPLOTLIB:
        warnings.warn("matplotlib non disponible. Installer: pip install matplotlib")
        return
    
    # Extraire les coordonnées
    nu_vals = np.array([f.nu_mean for f in feats])
    tau_vals = np.array([f.tau for f in feats])
    
    # Préparer les couleurs
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels[unique_labels >= 0])
    
    # Colormap
    cmap = plt.cm.get_cmap('tab10', n_clusters)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot chaque cluster
    for lbl in unique_labels:
        mask = labels == lbl
        if lbl == -1:
            # Outliers en noir avec 'x'
            ax.scatter(
                nu_vals[mask],
                tau_vals[mask],
                c='black',
                marker='x',
                s=100,
                alpha=0.6,
                label=f'Outliers (n={np.sum(mask)})'
            )
        else:
            ax.scatter(
                nu_vals[mask],
                tau_vals[mask],
                c=[cmap(lbl)],
                marker='o',
                s=80,
                alpha=0.7,
                edgecolors='black',
                linewidths=0.5,
                label=f'Cluster {lbl} (n={np.sum(mask)})'
            )
    
    ax.set_xlabel('Fréquence effective νₑff [Hz]', fontsize=12)
    ax.set_ylabel('Délai τ(H1-L1) [s]', fontsize=12)
    ax.set_title('Clustering des événements GW', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Figure sauvegardée: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
# Utilitaires
# ============================================================

def get_cluster_summary(
    event_to_cluster: Dict[str, int],
    feats: List[EventFeatures],
) -> Dict[int, Dict[str, Any]]:
    """
    Génère un résumé statistique par cluster.
    
    Args:
        event_to_cluster: Mapping event -> cluster_id
        feats: Liste des EventFeatures
    
    Returns:
        Dict[cluster_id -> stats]
        où stats contient: n_events, mean_nu, std_nu, mean_tau, std_tau, events
    """
    from collections import defaultdict
    
    # Grouper par cluster
    clusters = defaultdict(list)
    feat_dict = {f.event: f for f in feats}
    
    for event, cid in event_to_cluster.items():
        if event in feat_dict:
            clusters[cid].append(feat_dict[event])
    
    # Calculer les stats
    summary = {}
    for cid, cluster_feats in clusters.items():
        nu_vals = [f.nu_mean for f in cluster_feats]
        tau_vals = [f.tau for f in cluster_feats]
        
        summary[cid] = {
            'n_events': len(cluster_feats),
            'mean_nu': float(np.mean(nu_vals)),
            'std_nu': float(np.std(nu_vals)),
            'mean_tau': float(np.mean(tau_vals)),
            'std_tau': float(np.std(tau_vals)),
            'events': [f.event for f in cluster_feats],
        }
    
    return summary


def export_clusters_json(
    event_to_cluster: Dict[str, int],
    feats: List[EventFeatures],
    output_path: str,
) -> None:
    """
    Exporte les assignments de clusters et le résumé en JSON.
    
    Args:
        event_to_cluster: Mapping event -> cluster_id
        feats: Liste des EventFeatures
        output_path: Chemin du fichier JSON de sortie
    """
    summary = get_cluster_summary(event_to_cluster, feats)
    
    data = {
        'event_to_cluster': event_to_cluster,
        'cluster_summary': summary,
        'n_events': len(event_to_cluster),
        'n_clusters': len([cid for cid in summary.keys() if cid >= 0]),
        'n_outliers': len([cid for cid in event_to_cluster.values() if cid == -1]),
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"💾 Clusters exportés: {output_path}")


# ============================================================
# Point d'entrée CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Clustering intelligent pour événements LIGO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # HDBSCAN par défaut
  python cluster_latent_kmeans.py --results-glob "results/GW*.json"
  
  # HDBSCAN puis KMeans sur les inliers (hybride)
  python cluster_latent_kmeans.py --results-glob "results/GW*.json" \\
      --method hdbscan+kmeans --k 5 --min-cluster-size 3
  
  # DBSCAN avec paramètres personnalisés
  python cluster_latent_kmeans.py --results-glob "results/GW*.json" \\
      --method dbscan --eps 1.2 --min-samples 3
  
  # KMeans avec 4 clusters
  python cluster_latent_kmeans.py --results-glob "results/GW*.json" \\
      --method kmeans --k 4
  
  # Avec visualisation et export
  python cluster_latent_kmeans.py --results-glob "results/GW*.json" \\
      --plot --plot-save clusters.png --export clusters.json
        """
    )
    
    # Fichiers
    parser.add_argument(
        "--results-glob",
        required=True,
        help="Pattern glob pour les fichiers JSON (ex: 'results/GW*.json')"
    )
    
    # Méthode de clustering
    parser.add_argument(
        "--method",
        choices=["hdbscan", "dbscan", "kmeans", "hdbscan+kmeans"],
        default="hdbscan",
        help="Algorithme de clustering (défaut: hdbscan)"
    )
    
    # HDBSCAN params
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--cluster-selection-epsilon", type=float, default=0.0)
    
    # DBSCAN params
    parser.add_argument("--eps", "--db-eps", type=float, default=0.5, dest="eps")
    parser.add_argument("--db-min-samples", type=int, default=3)
    
    # KMeans params
    parser.add_argument("--k", "--n-clusters", type=int, default=3, dest="k")
    
    # Options générales
    parser.add_argument("--use-logE", action="store_true", help="Inclure logE dans les features")
    parser.add_argument("--no-normalize", action="store_true", help="Désactiver la normalisation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")
    
    # Visualisation
    parser.add_argument("--plot", action="store_true", help="Afficher un plot des clusters")
    parser.add_argument("--plot-save", type=str, help="Sauvegarder le plot à ce chemin")
    parser.add_argument("--no-show", action="store_true", help="Ne pas afficher le plot interactivement")
    
    # Export
    parser.add_argument("--export", type=str, help="Exporter les clusters en JSON")
    
    args = parser.parse_args()
    
    # Préparer les kwargs pour le clustering
    kwargs = {}
    if args.method == "hdbscan":
        kwargs['min_cluster_size'] = args.min_cluster_size
        kwargs['min_samples'] = args.min_samples
        kwargs['cluster_selection_epsilon'] = args.cluster_selection_epsilon
    elif args.method == "hdbscan+kmeans":
        kwargs['min_cluster_size'] = args.min_cluster_size
        kwargs['min_samples'] = args.min_samples
        kwargs['cluster_selection_epsilon'] = args.cluster_selection_epsilon
        kwargs['k'] = args.k
        # Ne pas ajouter seed ici, il est passé explicitement à load_cluster_assignments
    elif args.method == "dbscan":
        kwargs['eps'] = args.eps
        kwargs['min_samples'] = args.db_min_samples
    elif args.method == "kmeans":
        kwargs['k'] = args.k
        # Ne pas ajouter seed ici, il est passé explicitement à load_cluster_assignments
    
    try:
        # Clustering
        mapping, features = load_cluster_assignments(
            results_glob=args.results_glob,
            method=args.method,
            use_logE=args.use_logE,
            normalize=not args.no_normalize,
            verbose=args.verbose,
            seed=args.seed,
            **kwargs,
        )
        
        # Affichage des résultats
        print("\n" + "="*70)
        print("📊 RÉSULTATS DU CLUSTERING")
        print("="*70)
        
        summary = get_cluster_summary(mapping, features)
        for cid in sorted(summary.keys()):
            stats = summary[cid]
            print(f"\nCluster {cid:2} ({stats['n_events']} événements):")
            print(f"  νₑff: {stats['mean_nu']:.2f} ± {stats['std_nu']:.2f} Hz")
            print(f"  τ:    {stats['mean_tau']:.3e} ± {stats['std_tau']:.3e} s")
            print(f"  Events: {', '.join(sorted(stats['events'])[:5])}" + 
                  (f" ... (+{len(stats['events'])-5})" if len(stats['events']) > 5 else ""))
        
        # Visualisation
        if args.plot or args.plot_save:
            labels = np.array([mapping[f.event] for f in features])
            plot_clusters(
                feats=features,
                labels=labels,
                save_path=args.plot_save,
                show=(not args.no_show),
            )
        
        # Export JSON
        if args.export:
            export_clusters_json(mapping, features, args.export)
        
        print("\n✅ Clustering terminé avec succès!")
        
    except RuntimeError as e:
        print(f"❌ Erreur: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        exit(1)
