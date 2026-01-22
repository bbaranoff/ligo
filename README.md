# ligo  
Testing LIGO research in Python  

1st run


```
source go.sh
```

next runs 
```
bash run_all.sh
```


```
(.env) nirvana@lenovo:~/liGO$ bash run_all.sh 
[cal] refs_loaded=21 y_obs=19 events=19 ref_key=energy_J

🔧 Calibration par moindres carrés...
Événements: 19

...

📡 Telechargement des donnees H1/L1/V1 pour GW190828_065509...

=== LSQ (degenerate) : fit K = H_STAR*SCALE_EJ ===
events_used = 19
K_hat       = 4.237771e+37
-> choose H_STAR=1 ; SCALE_EJ=K
rel_MAE     = 44.00%
rel_MED     = 22.08%
[cal] H_STAR=1.000000e+00 SCALE_EJ=4.237771e+37
[cal] wrote calibrated.json
=============================================================
      RUN GLOBAL – Pipeline Spectral τ
=============================================================
[INFO] script:  /home/nirvana/liGO/run_all.sh
[INFO] python:  python3
[INFO] analyse: /home/nirvana/liGO/ligo_spectral_planck.py
[INFO] params:  /home/nirvana/liGO/event_params.json
[INFO] refs:    /home/nirvana/liGO/ligo_refs.json

====================== SYNTHÈSE PROGRESSIVE ==========================================================================
Event                 |   ν_eff | ν_ref  |   τ[s]   | τ_ref   |  M⊙c²  | M⊙c²_r |  Energie[J] |  E_ref[J]  | Notes
----------------------------------------------------------------------------------------------------------------------
GW150914             |   162.8 |   150.0 | -0.00416 | -0.00690 |   3.50 |    3.00 |   6.253e+47 |     5.3E+47 | Premier signal d'ondes gravitationnelles détecté (BBH)
GW151226             |   174.1 |   445.0 | -0.00046 | -0.00270 |   1.68 |    1.00 |   3.001e+47 |     1.8E+47 | Énergie corrigée (BBH)
GW170104             |   184.5 |   110.0 | -0.00000 | -0.00300 |   2.10 |    2.00 |   3.761e+47 |     3.5E+47 | Masse et énergie ajustées (BBH)
GW170608             |   117.3 |    30.0 | -0.00648 | -0.00400 |   3.29 |    0.90 |   5.877e+47 |     1.6E+47 | Plus petite masse de trou noir stellaire, énergie corrigée (BBH)
GW170729             |   178.5 |   130.0 | -0.00278 | -0.00600 |   2.07 |    4.80 |   3.703e+47 |     8.5E+47 | Système le plus massif d'O2, énergie corrigée (BBH)
GW170809             |   190.2 |   200.0 | -0.00509 | -0.00700 |   2.01 |    2.70 |   3.586e+47 |     4.8E+47 | Énergie corrigée (BBH)
GW170814             |   168.8 |   170.0 | -0.00370 | -0.00600 |   2.33 |    2.70 |   4.166e+47 |     4.8E+47 | Détection à trois détecteurs (LIGO+Virgo) (BBH)
GW170817             |   184.8 |    80.0 | -0.00046 |  0.00000 |   2.01 |    0.03 |   3.586e+47 |     4.4E+46 | Fusion d'étoiles à neutrons, avec contrepartie électromagnétique (BNS)
GW190403_051519      |   187.6 |   180.0 | -0.00093 | -0.00600 |   2.02 |    1.80 |   3.616e+47 |     3.2E+47 | O3a (BBH)
GW190412             |   154.9 |   150.0 | -0.00185 | -0.00600 |   2.51 |    1.70 |   4.489e+47 |       3E+47 | Rapport de masse asymétrique significatif (BBH)
GW190413_052954      |   182.4 |   170.0 | -0.00139 | -0.00700 |   1.83 |    1.30 |   3.268e+47 |     2.3E+47 | O3a (BBH)
GW190413_134308      |   183.7 |   175.0 | -0.00046 | -0.00700 |   2.08 |    1.40 |   3.712e+47 |     2.5E+47 | O3a (BBH)
GW190421_213856      |   170.4 |   160.0 | -0.00046 | -0.00800 |   2.16 |    1.60 |   3.855e+47 |     2.8E+47 | O3a (BBH)
GW190503_185404      |   168.8 |   200.0 | -0.00000 | -0.00700 |   2.03 |    2.60 |   3.627e+47 |     4.6E+47 | O3a (BBH)
GW190514_065416      |   182.1 |   210.0 | -0.01573 | -0.00600 |   2.02 |    2.20 |   3.602e+47 |     3.9E+47 | O3a (BBH)
GW190517_055101      |   183.1 |   195.0 | -0.00879 | -0.00700 |   1.07 |    2.00 |   1.916e+47 |     3.5E+47 | O3a (BBH)
GW190519_153544      |   181.3 |   140.0 | -0.00231 | -0.00900 |   1.72 |    2.10 |   3.073e+47 |     3.7E+47 | O3a, énergie corrigée (BBH)
GW190521             |   161.7 |    70.0 | -0.00694 | -0.00500 |   2.55 |    7.00 |   4.564e+47 |     1.2E+48 | Trous noirs de masse intermédiaire, masse finale ~142 Msun (BBH)
GW190828_063405      |   188.0 |   180.0 | -0.01018 | -0.00800 |   1.83 |    1.90 |   3.273e+47 |     3.4E+47 | O3b (BBH)
GW190828_065509      |   186.9 |   190.0 | -0.00833 | -0.00800 |   2.09 |    2.10 |   3.738e+47 |     3.7E+47 | O3b (BBH)

=============================================================
                    FIN DU RUN GLOBAL
=============================================================
[INFO] log: /home/nirvana/liGO/results/events.log
```

```
python3 cluster_latent_kmeans.py --glob "results/GW*.json"  --out clusters_dbscan_kmeans.txt
```
```
features: logE, nu_mean, nu_peak, nu_invf, frac_bw, Q_eff, peak_rel, R_LH | f_split=150.0 | DBSCAN(eps=1.4,min_samples=3) -> inliers=15 outliers=5 ; KMeans(k=4)

=== CLUSTER -1 (5) ===
means: logE=9.97 | nu_mean=166 | nu_peak=200 | nu_invf=100 | frac_bw=1.64 | Q_eff=0.642 | peak_rel=24.3 | R_LH=-0.0318
Event | logE | nu_mean | frac_bw | Q_eff | R_LH
-|-|-|-|-|-
GW170608 | 10.192 | 117.4 | 2.188 | 0.457 | 0.298
GW150914 | 10.152 | 153.9 | 1.798 | 0.556 | 0.082
GW151226 | 9.864 | 174.0 | 1.693 | 0.591 | -0.051
GW190517_055101 | 9.671 | 182.7 | 1.343 | 0.745 | -0.195
GW170809 | 9.978 | 199.8 | 1.159 | 0.863 | -0.294

=== CLUSTER 0 (2) ===
means: logE=10 | nu_mean=164 | nu_peak=62.5 | nu_invf=109 | frac_bw=1.53 | Q_eff=0.655 | peak_rel=8.67 | R_LH=0.00492
Event | logE | nu_mean | frac_bw | Q_eff | R_LH
-|-|-|-|-|-
GW190412 | 10.029 | 160.3 | 1.546 | 0.647 | 0.057
GW190521 | 10.016 | 167.7 | 1.509 | 0.663 | -0.047

=== CLUSTER 1 (5) ===
means: logE=9.95 | nu_mean=185 | nu_peak=243 | nu_invf=129 | frac_bw=1.32 | Q_eff=0.756 | peak_rel=7.38 | R_LH=-0.179
Event | logE | nu_mean | frac_bw | Q_eff | R_LH
-|-|-|-|-|-
GW170729 | 9.922 | 181.4 | 1.294 | 0.773 | -0.183
GW190828_065509 | 9.974 | 182.6 | 1.407 | 0.711 | -0.161
GW190413_134308 | 9.961 | 183.6 | 1.310 | 0.763 | -0.148
GW170104 | 9.962 | 184.5 | 1.325 | 0.755 | -0.171
GW190828_063405 | 9.912 | 190.7 | 1.286 | 0.778 | -0.233

=== CLUSTER 2 (4) ===
means: logE=9.97 | nu_mean=172 | nu_peak=174 | nu_invf=115 | frac_bw=1.44 | Q_eff=0.695 | peak_rel=8.2 | R_LH=-0.1
Event | logE | nu_mean | frac_bw | Q_eff | R_LH
-|-|-|-|-|-
GW190503_185404 | 9.947 | 168.8 | 1.480 | 0.676 | -0.074
GW170814 | 10.000 | 170.5 | 1.439 | 0.695 | -0.113
GW190421_213856 | 9.968 | 170.7 | 1.477 | 0.677 | -0.081
GW190514_065416 | 9.969 | 177.5 | 1.367 | 0.731 | -0.133

=== CLUSTER 3 (4) ===
means: logE=9.93 | nu_mean=187 | nu_peak=261 | nu_invf=130 | frac_bw=1.29 | Q_eff=0.778 | peak_rel=13.8 | R_LH=-0.252
Event | logE | nu_mean | frac_bw | Q_eff | R_LH
-|-|-|-|-|-
GW170817 | 9.951 | 185.6 | 1.232 | 0.812 | -0.314
GW190413_052954 | 9.924 | 187.3 | 1.337 | 0.748 | -0.211
GW190403_051519 | 9.948 | 187.4 | 1.300 | 0.769 | -0.238
GW190519_153544 | 9.914 | 189.0 | 1.277 | 0.783 | -0.246
```

