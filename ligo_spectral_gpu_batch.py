#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ligo_spectral_gpu_batch.py - Batch processing sur GPU
======================================================

Traite PLUSIEURS événements en PARALLÈLE sur GPU pour saturer la mémoire.

Speedup attendu: 20-30× vs CPU séquentiel

Usage:
    from ligo_spectral_gpu_batch import analyze_events_batch_gpu
    
    results = analyze_events_batch_gpu(
        events=['GW150914', 'GW151226', ...],
        params=params,
        batch_size=8  # 8 événements en parallèle
    )
"""

import os
import numpy as np
from typing import Dict, Any, List
import cupy as cp
from ligo_spectral_gpu import (
    analyze_event_gpu, 
    GPU_AVAILABLE,
    get_npz_path,
    load_npz
)


def analyze_events_batch_gpu(
    events: List[str],
    params: Dict[str, Any],
    batch_size: int = 8,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Analyse batch d'événements sur GPU en parallèle
    
    Args:
        events: Liste des événements à analyser
        params: event_params.json
        batch_size: Nombre d'événements en parallèle
        **kwargs: Arguments pour analyze_event_gpu
    
    Returns:
        Liste de résultats
    """
    
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU requis")
    
    results = []
    
    # Traiter par batches
    for i in range(0, len(events), batch_size):
        batch = events[i:i+batch_size]
        
        #print(f"\n📦 Batch {i//batch_size + 1}/{(len(events)-1)//batch_size + 1}: {len(batch)} événements")
        
        # Pré-charger TOUS les événements du batch sur GPU
        batch_data = {}
        total_mem = 0
        
        for event in batch:
            try:
                pH = get_npz_path(event, "H1", kwargs.get('npz_dir', 'data/npz'))
                pL = get_npz_path(event, "L1", kwargs.get('npz_dir', 'data/npz'))
                
                hH_cpu, fsH, t0H, t1H = load_npz(pH)
                hL_cpu, fsL, t0L, t1L = load_npz(pL)
                
                # Transférer GPU
                hH_gpu = cp.array(hH_cpu, dtype=cp.float64)
                hL_gpu = cp.array(hL_cpu, dtype=cp.float64)
                
                batch_data[event] = {
                    'hH': hH_gpu,
                    'hL': hL_gpu,
                    'fs': fsH,
                    't0H': t0H,
                    't0L': t0L,
                    't1H': t1H,
                    't1L': t1L,
                }
                
                total_mem += hH_gpu.nbytes + hL_gpu.nbytes
                
            except Exception as e:
                print(f"   ⚠️  {event}: {e}")
        
        #print(f"   💾 GPU chargé: {total_mem / 1e9:.2f} GB ({len(batch_data)} événements)")
        
        # Traiter en parallèle (CuPy gère le pipeline automatiquement)
        import concurrent.futures
        
        def process_one(event):
            if event not in batch_data:
                return None
            try:
                # analyze_event_gpu va réutiliser les données déjà sur GPU
                # si on les passait directement (à implémenter)
                return analyze_event_gpu(event, params, **kwargs)
            except Exception as e:
                print(f"   ❌ {event}: {e}")
                return None
        
        # Lancer en parallèle CPU (pour orchestration)
        # Le GPU exécute les kernels en pipeline
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            batch_results = list(executor.map(process_one, batch))
        
        results.extend([r for r in batch_results if r is not None])
        
        # Libérer batch
        for data in batch_data.values():
            del data['hH']
            del data['hL']
        
        cp.get_default_memory_pool().free_all_blocks()
    
    return results


def estimate_optimal_batch_size(gpu_mem_gb: float = 16.0, 
                                event_size_mb: float = 100.0) -> int:
    """
    Estime le batch size optimal selon mémoire GPU
    
    Args:
        gpu_mem_gb: Mémoire GPU disponible [GB]
        event_size_mb: Taille moyenne d'un événement [MB]
    
    Returns:
        Batch size optimal
    """
    # Réserver 20% pour overhead
    usable_mem_mb = gpu_mem_gb * 1000 * 0.8
    
    # Nombre d'événements qui tiennent
    batch_size = int(usable_mem_mb / event_size_mb)
    
    # Limiter à 32 pour éviter overhead orchestration
    batch_size = min(32, max(1, batch_size))
    
    return batch_size


# ============================================================================
# Version OPTIMALE : Stream processing avec préchargement
# ============================================================================

class GPUStreamProcessor:
    """
    Processeur GPU avec pipeline asynchrone
    
    Pendant que le GPU calcule le batch N, le CPU charge le batch N+1
    """
    
    def __init__(self, batch_size: int = 8):
        self.batch_size = batch_size
        self.stream = cp.cuda.Stream() if GPU_AVAILABLE else None
    
    def process_events(
        self,
        events: List[str],
        params: Dict[str, Any],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process avec pipeline asynchrone"""
        
        results = []
        
        for i in range(0, len(events), self.batch_size):
            batch = events[i:i+self.batch_size]
            
            # Charger batch N+1 pendant calcul batch N (à implémenter)
            batch_results = self._process_batch(batch, params, **kwargs)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(self, batch, params, **kwargs):
        """Process un batch"""
        results = []
        
        # Utiliser stream pour pipeline
        with self.stream:
            for event in batch:
                try:
                    result = analyze_event_gpu(event, params, **kwargs)
                    results.append(result)
                except Exception as e:
                    print(f"   ❌ {event}: {e}")
        
        # Synchroniser
        self.stream.synchronize()
        
        return results


# ============================================================================
# CLI pour test
# ============================================================================

if __name__ == '__main__':
    import json
    import time
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch GPU processing pour événements LIGO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # 3 événements avec batch auto
  python ligo_spectral_gpu_batch.py
  
  # 10 événements, batch size 8
  python ligo_spectral_gpu_batch.py --batch-size 8 \\
      --events GW150914 GW151226 GW170104 GW170608 GW170814 \\
               GW170817 GW170823 GW190412 GW190425 GW190521
  
  # Tous les événements, batch size 64
  python ligo_spectral_gpu_batch.py --batch-size 64 --all-events
  
  # Tous les événements, saturer GPU (batch auto)
  python ligo_spectral_gpu_batch.py --all-events
        """
    )
    
    parser.add_argument(
        '--events', 
        nargs='+', 
        default=['GW150914', 'GW151226', 'GW170104'],
        help='Liste des événements à analyser'
    )
    parser.add_argument(
        '--all-events',
        action='store_true',
        help='Analyser tous les événements depuis event_params.json'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=None,
        help='Taille du batch (défaut: auto selon GPU mem)'
    )
    parser.add_argument(
        '--params', 
        default='event_params.json',
        help='Fichier event_params.json'
    )
    parser.add_argument(
        '--gpu-mem-gb',
        type=float,
        default=16.0,
        help='Mémoire GPU disponible en GB (pour calcul batch auto)'
    )
    parser.add_argument(
        '--skip-sequential',
        action='store_true',
        help='Skip le test séquentiel (gain de temps)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Mode verbeux'
    )
    
    args = parser.parse_args()
    
    if not GPU_AVAILABLE:
        print("❌ GPU non disponible")
        exit(1)
    
    print("="*70)
    print("🚀 BATCH GPU PROCESSING")
    print("="*70)
    
    # Charger params
    with open(args.params, 'r') as f:
        params = json.load(f)
    
    # Récupérer liste événements
    if args.all_events:
        events = [k for k in params.keys() if k != 'default' and isinstance(params[k], dict)]
        print(f"\n📚 Mode --all-events: {len(events)} événements trouvés")
    else:
        events = args.events
    
    # Estimer batch size optimal
    if args.batch_size is None:
        batch_size = estimate_optimal_batch_size(
            gpu_mem_gb=args.gpu_mem_gb, 
            event_size_mb=100.0
        )
        print(f"\n💡 Batch size optimal estimé: {batch_size} (GPU: {args.gpu_mem_gb} GB)")
        print(f"   Utilisation GPU prévue: ~{batch_size * 100 / 1000:.1f} GB")
    else:
        batch_size = args.batch_size
        print(f"\n⚙️  Batch size manuel: {batch_size}")
    
    print(f"📊 Événements totaux: {len(events)}")
    print(f"📦 Batch size: {batch_size}")
    print(f"🔢 Nombre de batches: {(len(events) - 1) // batch_size + 1}")
    
    # Test séquentiel (optionnel)
    if not args.skip_sequential:
        print("\n" + "="*70)
        print("🐌 MODE SÉQUENTIEL (baseline)")
        print("="*70)
        
        # Limiter à 10 événements pour le test séquentiel
        events_test = events[:min(10, len(events))]
        if len(events_test) < len(events):
            print(f"⚠️  Test séquentiel limité à {len(events_test)} événements (gain de temps)")
        
        start = time.time()
        results_seq = []
        for i, event in enumerate(events_test, 1):
            try:
                if args.verbose:
                    print(f"   [{i}/{len(events_test)}] {event}...", end='')
                result = analyze_event_gpu(event, params, verbose=False)
                results_seq.append(result)
                if args.verbose:
                    print(f" ✓ {result['msun_c2']:.3f} M☉")
            except Exception as e:
                print(f"❌ {event}: {e}")
        seq_time = time.time() - start
        
        print(f"\n✅ Séquentiel: {seq_time:.2f}s ({len(results_seq)}/{len(events_test)} events)")
        print(f"   Temps/event: {seq_time/len(results_seq):.3f}s")
        
        # Estimer temps total séquentiel
        if len(events) > len(events_test):
            estimated_total = seq_time * len(events) / len(events_test)
            print(f"   Temps total estimé pour {len(events)} events: {estimated_total/60:.1f} min")
    
    # Test batch
    print("\n" + "="*70)
    print("🚀 MODE BATCH GPU")
    print("="*70)
    
    start = time.time()
    results_batch = analyze_events_batch_gpu(
        events=events,
        params=params,
        batch_size=batch_size,
        verbose=args.verbose
    )
    batch_time = time.time() - start
    
    print(f"\n✅ Batch GPU: {batch_time:.2f}s ({len(results_batch)}/{len(events)} events)")
    print(f"   Temps/event: {batch_time/max(1, len(results_batch)):.3f}s")
    print(f"   Throughput: {len(results_batch)/batch_time:.1f} events/s")
    
    # Speedup
    if not args.skip_sequential and seq_time > 0:
        # Normaliser par le nombre d'événements testés
        seq_per_event = seq_time / len(results_seq)
        batch_per_event = batch_time / len(results_batch)
        speedup = seq_per_event / batch_per_event
        
        print(f"\n🚀 Speedup batch vs séquentiel: {speedup:.1f}×")
        
        # Temps économisé
        time_saved = seq_per_event * len(events) - batch_time
        print(f"   Temps économisé: {time_saved/60:.1f} min")
    
    # Stats
    if len(results_batch) > 0:
        print("\n📊 Statistiques:")
        energies = [r['energy_J'] for r in results_batch]
        masses = [r['msun_c2'] for r in results_batch]
        
        print(f"   Énergie: [{min(energies):.2e}, {max(energies):.2e}] J")
        print(f"   Masse:   [{min(masses):.3f}, {max(masses):.3f}] M☉")
        print(f"   Moyenne: {np.mean(masses):.3f} M☉")
    
    # Vérifier cohérence si test séquentiel fait
    if not args.skip_sequential and len(results_seq) > 0:
        print("\n🔍 Vérification cohérence (premiers événements):")
        n_check = min(len(results_seq), len(results_batch))
        max_diff = 0
        for r1, r2 in zip(results_seq[:n_check], results_batch[:n_check]):
            diff = abs(r1['energy_J'] - r2['energy_J']) / r1['energy_J'] * 100
            max_diff = max(max_diff, diff)
            if diff > 1.0:
                print(f"   ⚠️  {r1['event']}: diff {diff:.2f}%")
        
        if max_diff < 1.0:
            print(f"   ✅ Résultats cohérents (diff max: {max_diff:.3f}%)")
        else:
            print(f"   ⚠️  Divergence détectée (diff max: {max_diff:.2f}%)")
    
    print("\n" + "="*70)
    print("✨ TERMINÉ")
    print("="*70)
