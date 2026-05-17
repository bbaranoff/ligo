"""Test stat : nos SNR matched-filter contiennent-ils du signal réel ?

Trois niveaux :
1. Comparaison par classe vs référence LIGO H1-only attendue
2. Régression linéaire SNR vs log(M_initial) — signal → corrélation positive
3. ANOVA + Kruskal-Wallis inter-classes — signal → distributions différentes

Verdict simple "heard" / "marginal" / "not heard" par classe.

Usage : python3 ligo_snr_stats.py [--observed observed.json]
"""

import json
import sys
import numpy as np

try:
    from scipy import stats
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ============================================================
# Référence LIGO (network SNR moyens par classe, depuis GWTC-1/2/3)
# Sources : Table II GWTC-3, PRX 13 011048
# Network SNR = √(SNR_H1² + SNR_L1² + SNR_V1²)
# Pour H1 seul on prend network / √2 (2 détecteurs typiques H1+L1)
# ============================================================
LIGO_NETWORK_SNR_MEAN = {
    'ULTRA_LIGHT': 11.0,   # BNS-like et BBH légers : GW170608, GW190924, GW190707
    'LIGHT':       14.0,   # BBH légers : GW151226, GW190412, GW190728
    'MEDIUM':      14.5,   # BBH typiques : GW150914 (26), GW170104, GW170814 (18)
    'MASSIVE':     11.5,   # BBH lourds, plus loin : GW190521, GW190426
}

# Conversion network → H1-only (√2 réduction pour 2 détecteurs)
LIGO_H1_SNR_MEAN = {cls: v / np.sqrt(2) for cls, v in LIGO_NETWORK_SNR_MEAN.items()}


def verdict(measured_mean, ligo_ref, n):
    """Verdict 'heard' simple."""
    if measured_mean >= ligo_ref * 0.7:
        return "✓ heard         "
    elif measured_mean >= ligo_ref * 0.4:
        return "~ marginal      "
    elif measured_mean >= ligo_ref * 0.2:
        return "✗ noise dominé "
    else:
        return "✗ not heard     "


def main():
    path = 'observed.json'
    if '--observed' in sys.argv:
        idx = sys.argv.index('--observed')
        if idx + 1 < len(sys.argv):
            path = sys.argv[idx + 1]

    with open(path) as f:
        obs = json.load(f)

    if not SCIPY_OK:
        print("⚠ scipy non disponible — tests stat partiels")

    # Group by class, collecte (event, snr, msun_c2, M_initial)
    by_class = {}
    for ev, d in obs.items():
        cls = d['class']
        snr = d.get('snr_matched_H1', 0.0)
        if snr <= 0 or not np.isfinite(snr):
            continue
        M_i = d.get('M_initial', 0.0)
        dm = d.get('msun_c2_ref', 0.0)
        by_class.setdefault(cls, []).append((ev, snr, M_i, dm))

    n_total = sum(len(v) for v in by_class.values())
    print(f"\n{'='*78}")
    print(f"  TEST STATISTIQUE SNR — {n_total} events avec SNR > 0")
    print(f"{'='*78}\n")

    # ===== 1. Par classe =====
    print(f"  {'Classe':<13} {'N':>3} {'mean_SNR':>9} {'std':>6} {'median':>7} "
          f"{'LIGO_ref':>9} {'verdict':<18}")
    print(f"  {'-'*13} {'-'*3} {'-'*9} {'-'*6} {'-'*7} {'-'*9} {'-'*18}")
    for cls in ['ULTRA_LIGHT', 'LIGHT', 'MEDIUM', 'MASSIVE']:
        if cls not in by_class:
            print(f"  {cls:<13} {0:>3} {'—':>9} {'—':>6} {'—':>7} "
                  f"{LIGO_H1_SNR_MEAN[cls]:>9.2f} {'(no data)':<18}")
            continue
        snrs = np.array([s for _, s, _, _ in by_class[cls]])
        m, sd, med = float(np.mean(snrs)), float(np.std(snrs)), float(np.median(snrs))
        ref = LIGO_H1_SNR_MEAN[cls]
        v = verdict(m, ref, len(snrs))
        print(f"  {cls:<13} {len(snrs):>3d} {m:>9.2f} {sd:>6.2f} {med:>7.2f} "
              f"{ref:>9.2f} {v}")

    # ===== 2. Régression SNR vs log(M_initial) =====
    all_snrs, all_log_M = [], []
    for cls_data in by_class.values():
        for _, snr, M_i, _ in cls_data:
            if M_i > 0:
                all_snrs.append(snr)
                all_log_M.append(np.log(M_i))
    all_snrs = np.array(all_snrs)
    all_log_M = np.array(all_log_M)

    print(f"\n  RÉGRESSION SNR vs log(M_initial)")
    print(f"  ────────────────────────────────")
    if len(all_snrs) >= 5 and SCIPY_OK:
        slope, intercept, r, p, _ = stats.linregress(all_log_M, all_snrs)
        print(f"    slope        : {slope:+.3f}  (positif si signal corrèle avec masse)")
        print(f"    R²           : {r**2:.3f}")
        print(f"    p-value      : {p:.3e}")
        print(f"    N samples    : {len(all_snrs)}")
        if p < 0.05 and slope > 0:
            verdict_reg = "✓ corrélation SNR-masse significative → SIGNAL détecté"
        elif p < 0.05 and slope < 0:
            verdict_reg = "⚠ corrélation NÉGATIVE — sur-tuning template massif possible"
        else:
            verdict_reg = "✗ pas de corrélation — on mesure du bruit ou un proxy non-physique"
        print(f"    verdict      : {verdict_reg}")
    else:
        print(f"    (scipy requis ou pas assez d'events)")

    # ===== 3. ANOVA inter-classes =====
    print(f"\n  TEST INTER-CLASSES (signal → classes statistiquement différentes)")
    print(f"  ──────────────────────────────────────────────────────────────────")
    if SCIPY_OK:
        class_arrays = [
            np.array([s for _, s, _, _ in by_class[cls]])
            for cls in by_class if len(by_class[cls]) >= 2
        ]
        class_names = [cls for cls in by_class if len(by_class[cls]) >= 2]
        if len(class_arrays) >= 2:
            f_stat, anova_p = stats.f_oneway(*class_arrays)
            k_stat, kw_p = stats.kruskal(*class_arrays)
            print(f"    Classes testées  : {class_names}")
            print(f"    ANOVA F          : {f_stat:.3f}, p = {anova_p:.3e}")
            print(f"    Kruskal-Wallis H : {k_stat:.3f}, p = {kw_p:.3e}")
            if anova_p < 0.05 or kw_p < 0.05:
                print(f"    verdict          : ✓ classes différentes — STRUCTURE SIGNAL")
            else:
                print(f"    verdict          : ✗ classes indistinguables — bruit dominant")
        else:
            print(f"    (pas assez de classes peuplées)")

    # ===== 4. Verdict global =====
    print(f"\n  {'='*72}")
    print(f"  VERDICT GLOBAL")
    print(f"  {'='*72}")
    heard_classes = sum(
        1 for cls in by_class
        if np.mean([s for _, s, _, _ in by_class[cls]]) >= LIGO_H1_SNR_MEAN[cls] * 0.7
    )
    print(f"  Classes 'heard'   : {heard_classes}/{len(by_class)}")
    print(f"  N events utiles   : {n_total}")
    if heard_classes == len(by_class):
        print(f"  → On capte le signal sur toutes les classes")
    elif heard_classes >= len(by_class) / 2:
        print(f"  → Signal capté sur la majorité des classes, bruit sur certaines")
    else:
        print(f"  → Capture faible : revoir SNR matched-filter ou bandes")


if __name__ == '__main__':
    main()
