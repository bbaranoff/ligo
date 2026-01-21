#!/usr/bin/env python3
import glob, json, argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def trapz(y, x): return float(np.trapz(y, x))

def extract(d):
    f = np.asarray(d.get("freq_Hz", []), float)
    s = np.asarray(d.get("dEdf_internal", []), float)
    if f.size < 16 or s.size != f.size: return None
    m = (f > 0) & np.isfinite(f) & np.isfinite(s) & (s >= 0)
    f, s = f[m], s[m]
    if f.size < 16 or np.all(s <= 0): return None

    E = trapz(s, f)
    denom = trapz(s / f, f)
    if E <= 0 or denom <= 0: return None

    w = s / (s.sum() + 1e-30)
    nu_mean = float(np.sum(f * w))
    cdf = np.cumsum(w)
    f10 = float(np.interp(0.10, cdf, f))
    f50 = float(np.interp(0.50, cdf, f))
    f90 = float(np.interp(0.90, cdf, f))
    bw = f90 - f10
    frac_bw = bw / max(f50, 1e-9)
    Q = f50 / max(bw, 1e-9)

    ipeak = int(np.argmax(s))
    nu_peak = float(f[ipeak])
    peak_rel = float(s[ipeak] / (np.mean(s) + 1e-30))

    return [np.log10(E), nu_mean, nu_peak, frac_bw, Q, peak_rel]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/GW*.json")
    ap.add_argument("--k", type=int, default=4, help="nb de régimes")
    ap.add_argument("--out", default="clusters_kmeans.txt")
    args = ap.parse_args()

    names, rows = [], []
    for p in sorted(glob.glob(args.glob)):
        d = json.load(open(p))
        ev = d.get("event") or p.split("/")[-1].replace(".json","")
        r = extract(d)
        if r is None: continue
        names.append(ev)
        rows.append(r)

    X = np.asarray(rows, float)
    Xs = StandardScaler().fit_transform(X)
    Z = PCA(n_components=2, random_state=0).fit_transform(Xs)
    lab = KMeans(n_clusters=args.k, n_init=50, random_state=0).fit_predict(Z)

    clusters = {}
    for ev, c, z, x in zip(names, lab, Z, X):
        clusters.setdefault(int(c), []).append((ev, z, x))

    with open(args.out, "w") as f:
        f.write("features: logE, nu_mean, nu_peak, frac_bw, Q, peak_rel\n\n")
        for c in sorted(clusters):
            f.write(f"=== CLUSTER {c} ({len(clusters[c])}) ===\n")
            # tri par proximité du centre PCA
            zz = np.vstack([z for _, z, _ in clusters[c]])
            center = zz.mean(axis=0)
            items = []
            for ev, z, x in clusters[c]:
                dist = float(np.linalg.norm(z - center))
                items.append((dist, ev, z, x))
            items.sort()

            for dist, ev, z, x in items:
                f.write(f"{ev:22s}  logE={x[0]:.3f}  nu_mean={x[1]:6.1f}  frac_bw={x[3]:.3f}  Q={x[4]:.2f}\n")
            f.write("\n")

    print("[OK] wrote", args.out)

if __name__ == "__main__":
    main()
