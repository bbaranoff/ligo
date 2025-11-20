#!/usr/bin/env python3
import json
from itertools import combinations
from tau import load_tau


def load_retrace():
    """
    Calcule un 'τ géodésique' via un MST sur l'espace τ,
    et renvoie:
      - retrace: dict event -> indice sur la géodésique globale
      - mst_edges: liste de (evtA, evtB, |Δτ|)
    """
    tau = load_tau()  # dict event -> τ (global)
    events = list(tau.keys())

    # -------------------------
    # 1) Construire toutes les arêtes pondérées par |Δτ|
    # -------------------------
    edges = []
    for a, b in combinations(events, 2):
        w = abs(tau[a] - tau[b])
        edges.append((w, a, b))
    edges.sort(key=lambda x: x[0])

    # -------------------------
    # 2) Kruskal pour MST
    # -------------------------
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            return True
        return False

    mst_edges = []
    for w, a, b in edges:
        if union(a, b):
            mst_edges.append((a, b, w))

    # -------------------------
    # 3) Construire l'adjacence + géodésique globale
    # -------------------------
    adj = {e: [] for e in events}
    for a, b, w in mst_edges:
        adj[a].append((b, w))
        adj[b].append((a, w))

    # point de départ = plus petit τ (comme dans tes logs)
    start = min(events, key=lambda e: tau[e])

    order = []
    visited = set()

    def dfs(u):
        visited.add(u)
        order.append(u)
        # pour la reproductibilité, on visite dans l'ordre croissant de τ
        for v, _ in sorted(adj[u], key=lambda x: tau[x[0]]):
            if v not in visited:
                dfs(v)

    dfs(start)

    # 'retrace' = paramètre le long de la géodésique (0,1,2,...)
    retrace = {evt: i for i, evt in enumerate(order)}
    return retrace, mst_edges


# Mode script : afficher MST + ordre (comme avant)
if __name__ == "__main__":
    tau = load_tau()
    retrace, mst_edges = load_retrace()

    print("\n=== ARBRE MINIMAL (MST) SUR l’ESPACE τ ===")
    for a, b, w in sorted(mst_edges, key=lambda x: x[2]):
        print(f"{a:<12} -- {b:<12} | Δτ = {w:7.3f}")

    print("\n=== ORDRE OPTIMAL (géodésique globale) ===")
    for i, evt in enumerate(sorted(retrace.keys(), key=lambda e: retrace[e])):
        print(f"{i:02d} : {evt:<10} τ = {tau[evt]:8.3f}")
