#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIGO Spectral Unified Pipeline – Version Professionnelle
--------------------------------------------------------
Pipeline spectral basé sur :
  - τ géométrique (cross-corr)
  - τ_phase (phase log-spectrale)
  - fusion tau_final bornée ±0.02 s
  - deux modèles énergétiques : "toi" ou "gr"
  - export JSON propre, stable et lisible
"""

import os
import json
import numpy as np
from scipy.signal import butter, filtfilt, welch, hilbert
from scipy.interpolate import interp1d
from gwpy.timeseries import TimeSeries
import argparse
from gwosc import datasets
# ============================================================
# CONST. PHYSIQUES
# ============================================================

c = 299792458.0            # vitesse de la lumière
G = 6.67430e-11            # constante gravitationnelle
M_sun = 1.98847e30         # masse solaire

# ============================================================
# PARAMS GLOBAUX
# ============================================================

H_STAR = 1.0  # ton modèle h★ = 1 normalisé, on conserve
TAU_LIMIT = 0.02  # borne de sécurité sur tau_final

# modèle "toi" (énergie unifiée)
ENERGY_SCALE_TOI = 1e-35  # ton ancienne constante (conservée)

# modèle GR : dE/df = (pi^2 c^3 / G) * r^2 * f^2 |h|^2
ENERGY_SCALE_GR = (np.pi**2) * (c**3 / G)

# Fenêtres par défaut
DEFAULT_FLOW = 20.0
DEFAULT_FHIGH = 350.0

# chemins
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# PARAMÈTRES PAR DÉFAUT DES ÉVÉNEMENTS LIGO
# ============================================================
EVENT_PARAMS = {
    "default": {
        "flow": 20.0,
        "fhigh": 1024.0,
        "signal_win": 0.25,
        "noise_pad": 5.0,
        "distance_mpc": 500.0
    }
}
# ============================================================
# 2) HELPER : BUTTERWORTH BANDPASS
# ============================================================

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 0.999999)
    b, a = butter(order, [low, high], btype="band")
    return b, a


def safe_bandpass(x, fs, lowcut, highcut, order=4):
    if len(x) < 20:
        return x
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, x).astype(float)


# ============================================================
# 3) HELPER : PSD Welch propre et stable
# ============================================================

def psd_welch(ts, fmin=20.0, fmax=350.0):
    """
    ts : array-like
    renvoie f, S(f) sur [fmin, fmax]
    """
    x = np.asarray(ts, float)
    if len(x) < 32:
        return np.array([fmin, fmax], float), np.array([1e-30, 1e-30], float)

    fs = 1.0 / (ts.sample_spacing.value if hasattr(ts, "sample_spacing") else 1.0)
    f, Pxx = welch(x, fs=fs, nperseg=4096)

    # Sélection domaine utile
    mask = (f >= fmin) & (f <= fmax)
    if not np.any(mask):
        return np.array([fmin, fmax]), np.array([1e-30, 1e-30])
    return f[mask], Pxx[mask]


# ============================================================
# 4) HELPER : estimation du délai géométrique (cross-corr)
# ============================================================
def estimate_delay(h1, h2, fs, max_shift=0.01):
    """
    Délai spectral basé sur la phase :
    τ = - dφ/dω
    Signature et retour IDENTIQUES à la version temporelle.
    h1, h2 : signaux temporels
    fs : fréquence d'échantillonnage
    max_shift : borne de sécurité (secondes)
    """

    # ----------------------------------------------------------
    # 1) Mise à longueur commune
    # ----------------------------------------------------------
    n = min(len(h1), len(h2))
    h1 = h1[:n]
    h2 = h2[:n]

    # ----------------------------------------------------------
    # 2) Passage en fréquentiel
    # ----------------------------------------------------------
    H1 = np.fft.rfft(h1)
    H2 = np.fft.rfft(h2)
    freq = np.fft.rfftfreq(n, 1/fs)

    # ----------------------------------------------------------
    # 3) Coherence spectral-domain (cross-spectrum)
    # ----------------------------------------------------------
    # produit conjugué → phase relative
    HP = H1 * np.conj(H2)

    amp = np.abs(HP)
    if np.nanmax(amp) == 0:
        return 0.0

    # ----------------------------------------------------------
    # 4) Sélection des zones significatives
    # ----------------------------------------------------------
    thresh = 0.05 * np.nanmax(amp)
    mask = amp > thresh

    f = freq[mask]
    hp = HP[mask]

    if len(f) < 10:
        return 0.0

    # ----------------------------------------------------------
    # 5) Phase unwrap
    # ----------------------------------------------------------
    phi = np.unwrap(np.angle(hp))

    # ----------------------------------------------------------
    # 6) Dérivée lissée
    # ----------------------------------------------------------
    dphi = np.gradient(phi, f)
    dphi = np.convolve(dphi, np.ones(7)/7, mode="same")

    # ----------------------------------------------------------
    # 7) Tau spectral :  - dφ/dω  = - dφ/(2πf)
    # ----------------------------------------------------------
    tau = - dphi / (2 * np.pi)

    # ----------------------------------------------------------
    # 8) Trim robuste (10% - 90%)
    # ----------------------------------------------------------
    q1 = np.percentile(tau, 10)
    q2 = np.percentile(tau, 90)
    tau_clip = tau[(tau > q1) & (tau < q2)]

    if len(tau_clip) == 0:
        tau_est = float(np.nanmedian(tau))
    else:
        tau_est = float(np.nanmedian(tau_clip))

    # ----------------------------------------------------------
    # 9) Clip final identique à la version d'origine
    # ----------------------------------------------------------
    tau_est = float(np.clip(tau_est, -max_shift, max_shift))

    # ----------------------------------------------------------
    # 10) Retour identique
    # ----------------------------------------------------------
    return tau_est


# ============================================================
# 5) HELPER : Fetch TimeSeries LIGO propre
# ============================================================

def fetch(det, t0, t1, outdir="data"):
    """
    Télécharge les données publiques LIGO Open Data pour un détecteur.
    Renvoie un TimeSeries (gwpy).
    """
    os.makedirs(outdir, exist_ok=True)
    return TimeSeries.fetch_open_data(det, t0, t1, cache=True)
# ============================================================
# 6) HELPER : FFT + h_phys(f)
# ============================================================

def compute_fft(ts, fs):
    """
    ts : array flottant (signal filtré)
    fs : fréquence d'échantillonnage
    Renvoie (f_pos, H_pos) = FFT demi-raie, centrée, unilatérale.
    """
    n = len(ts)
    if n < 32:
        return np.array([1.0]), np.array([1e-30])

    H = np.fft.rfft(ts)
    f = np.fft.rfftfreq(n, 1.0 / fs)

    # éviter les zéros fantômes
    H = np.asarray(H, complex)
    return f, H


def compute_h_phys(H, S):
    """
    H : FFT du signal
    S : PSD (interpolée sur mêmes freq)
    Retourne le h_phys = H / sqrt(S)
    """
    S_safe = np.maximum(S, 1e-30)
    return H / np.sqrt(S_safe)
# ============================================================
# 7) SPECTRE D'ÉNERGIE (TOI ou GR)
# ============================================================

def compute_energy_spectrum(f, h_phys, distance_mpc, model="toi"):
    """
    f : fréquences positives
    h_phys : amplitude physique
    distance_mpc : distance événement (Mpc)
    model : "toi" | "gr"
    Renvoie : dE/df (J/Hz), E_total (J)
    """

    # Conversion distance Mpc → mètres
    r = float(distance_mpc) * (3.085677581e22)

    # amplitude quadratique
    amp2 = np.abs(h_phys)**2

    if model == "gr":
        # dE/df from GR : (pi^2 c^3 / G) r^2 f^2 |h|^2
        scale = ENERGY_SCALE_GR
    else:
        # modèle unifié (toi)
        scale = ENERGY_SCALE_TOI


    # suppression éventuelle des singularités
    dEdf = np.nan_to_num(dEdf, nan=0.0, posinf=0.0, neginf=0.0)

    # énergie totale (intégrale)
    E_total = float(np.trapz(dEdf, f))

    return dEdf, E_total
# ============================================================
# 8) SPECTRE COHÉRENT h★ — dE/df
# ============================================================

def coherent_spectrum(hH, hL, fs, f_psd, S1, S2, distance_mpc, model="toi"):
    """
    Calcule dE/df cohérent :
        dE/df = |Hc(f)|^2 * scale * r^2 * f^2
    Hc étant la combinaison cohérente pondérée par les PSD.
    """

    # ------------------------------------------------------------
    # Fréquences FFT
    # ------------------------------------------------------------
    n = len(hH)
    freq = np.fft.rfftfreq(n, 1.0/fs)

    # FFT H1 / L1
    HH = np.fft.rfft(hH)
    LL = np.fft.rfft(hL)

    # ------------------------------------------------------------
    # PSD interpolées
    # ------------------------------------------------------------
    S1i = np.interp(freq, f_psd, S1)
    S2i = np.interp(freq, f_psd, S2)

    # ------------------------------------------------------------
    # Combinaison cohérente
    # ------------------------------------------------------------
    W = 1.0 / (S1i + S2i + 1e-30)
    Hc = (HH + LL) * W

    # ------------------------------------------------------------
    # Paramètres physiques
    # ------------------------------------------------------------
    r = float(distance_mpc) * 3.085677581e22   # Mpc → mètres

    if model == "gr":
        scale = ENERGY_SCALE_GR
    else:
        scale = ENERGY_SCALE_TOI

    # ------------------------------------------------------------
    # Formule complète
    # dE/df = |Hc|^2 * scale * r^2 * f^2
    # ------------------------------------------------------------
    amp2 = np.abs(Hc)**2
    dEdf = amp2 * scale * (r**2) * (freq**2)

    # Nettoyage
    dEdf = np.nan_to_num(dEdf, nan=0.0, posinf=0.0, neginf=0.0)

    # Bande de fréquence utile
    mask = (freq >= 10) & (freq <= 2000)

    return freq[mask], dEdf[mask]

# ============================================================
# 8) EXTRACTION PHASE LOG-SPECTRALE (α, φ)
# ============================================================

def extract_phase_logsin(f, dEdf, min_pts=40):
    """
    Tente d'extraire une phase log-sin :
        dE/df = A(f) * sin(α log f + φ)
    f : array fréquences (>0)
    dEdf : array énergie spectrale (>=0)
    Renvoie (alpha, phi)

    Si extraction impossible -> (nan, nan)
    """

    # Nettoyage et domaine utile
    mask = (f > 0) & (np.isfinite(dEdf)) & (dEdf > 0)
    f2 = f[mask]
    S = dEdf[mask]

    if len(f2) < min_pts:
        return np.nan, np.nan

    # log-frequencies
    x = np.log(f2)
    y = S

    # Normalisation douce
    y = y / np.max(y)

    # On cherche des oscillations en x (log f)
    # FFT sur y(x)
    Y = np.fft.rfft(y - np.mean(y))
    k = np.fft.rfftfreq(len(y), d=(x[1] - x[0]))

    # Trouver le pic (k>0)
    if len(k) < 3:
        return np.nan, np.nan

    k_abs = np.abs(Y)
    k_abs[0] = 0  # retire le DC

    idx = np.argmax(k_abs)
    k0 = k[idx]

    if k0 <= 0:
        return np.nan, np.nan

    # alpha = 2π * k0
    alpha = 2 * np.pi * k0

    # pour phi, on approxime la phase du coefficient dominant
    phi = np.angle(Y[idx])

    return float(alpha), float(phi)


# ============================================================
# 9) FRÉQUENCE EFFECTIVE ν_eff (robuste)
# ============================================================
def compute_nu_eff(freq, dEdf):
    # égalisation des tailles
    L = min(len(freq), len(dEdf))
    freq = freq[:L]
    dEdf = dEdf[:L]

    if dEdf.max() == 0:
        return 0.0

    # bande utile BBH
    mask = (freq >= 30) & (freq <= 300)
    f = freq[mask]
    e = dEdf[mask]

    # seuil anti-bruit
    if len(e) == 0:
        return 0.0

    thresh = 0.01 * e.max()
    keep = e > thresh

    f = f[keep]
    e = e[keep]

    if len(f) < 5:
        return 0.0

    # lissage
    e_s = np.convolve(e, np.ones(7)/7, mode="same")
    E = np.trapz(e_s, f)

    if E <= 0:
        return 0.0

    return float(np.trapz(f * e_s, f) / E)

"""
def compute_nu_eff(f, dEdf):

    f = np.asarray(f, float)
    S = np.asarray(dEdf, float)

    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    den = np.trapz(S, f)

    if den <= 0:
        # fallback = max spectral
        idx = int(np.argmax(S))
        return float(f[idx])

    num = np.trapz(f * S, f)
    nu_eff = num / den

    # si aberrant, fallback f_peak
    if not np.isfinite(nu_eff) or nu_eff <= 0:
        idx = int(np.argmax(S))
        return float(f[idx])

    return float(nu_eff)
"""
# ============================================================
# 10) CALCUL DU TAU GÉOMÉTRIQUE
# ============================================================

def estimate_tau_geo(tsH, tsL, gps, fs, flow, fhigh):
    """
    Extrait un segment autour du GPS, applique bandpass,
    puis corrélation croisée.
    """
    segH = tsH.crop(gps - 0.5, gps + 0.5)
    segL = tsL.crop(gps - 0.5, gps + 0.5)

    hH = safe_bandpass(np.asarray(segH.value, float), fs, flow, fhigh)
    hL = safe_bandpass(np.asarray(segL.value, float), fs, flow, fhigh)

    return estimate_delay(hH, hL, fs)


# ============================================================
# 11) FUSION TAU : τ_geo + τ_phase → τ_final
# ============================================================

def combine_tau(tau_geo, alpha_logsin, nu_eff,
                w_geo=0.6, w_phase=0.4,
                tau_min=-0.02, tau_max=0.02):
    """
    Combine :
        τ_geo    = délai géométrique
        τ_phase  = -α / ν_eff
    avec pondération w_geo / w_phase et bornage.
    """

    # τ_phase
    if np.isfinite(alpha_logsin) and np.isfinite(nu_eff) and nu_eff > 0:
        tau_phase = - alpha_logsin / nu_eff
    else:
        tau_phase = 0.0

    # fusion
    tau_final = w_geo * tau_geo + w_phase * tau_phase

    # bornage final
    tau_final = float(np.clip(tau_final, tau_min, tau_max))

    return float(tau_final), float(tau_phase)
# ============================================================
# 12) CALCUL DE L'ÉNERGIE, MASSE, EXPORT JSON
# ============================================================

def build_event_result(event_name,
                       gps,              # ← ajouté ici
                       nu_eff,
                       tau_final,
                       tau_geo,
                       tau_phase,
                       f_use,
                       dEdf_use,
                       E_total,
                       model_energy="toi"):

    M_sun_J = M_sun * c**2

    result = {
        "event": event_name,
        "gps": float(gps),              # ← AJOUT ICI
        "nu_eff": float(nu_eff),
        "tau": float(tau_final),
        "tau_geo": float(tau_geo),
        "tau_phase": float(tau_phase),
        "alpha_logsin": float(alpha_logsin),
        "phi_logsin": float(phi_logsin),
        "f": [float(x) for x in f_use],
        "dEdf": [float(x) for x in dEdf_use],
        "E_total": float(E_total),
        "M_sun": float(E_total / M_sun_J),
        "model_energy": model_energy
    }

    return result


def export_json_result(event_name, result):
    """
    Sauvegarde propre dans results/<event>.json
    """
    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", f"{event_name}.json")

    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"📁 Résultat sauvegardé : {out}")
# ============================================================
# 13) PIPELINE COMPLET D'UN ÉVÉNEMENT
# ============================================================

def analyze_event(event_name,
                  tsH,
                  tsL,
                  gps,
                  distance_mpc,
                  flow,
                  fhigh,
                  signal_win,
                  noise_pad,
                  model_energy="toi",
                  plot=False):
    """
    Pipeline complet :
    - PSD bruit
    - Extraction spectrale
    - nu_eff
    - alpha_logsin / phi
    - tau_geo
    - tau_phase
    - tau_final
    - énergie totale
    - export JSON
    """

    print(f"\n==================== {event_name} ====================\n")

    # ------------------------------------------------------------
    # Extraction du signal utile
    # ------------------------------------------------------------
    fs = tsH.sample_rate.value
    t0 = gps - signal_win
    t1 = gps + signal_win

    segH = tsH.crop(t0, t1)
    segL = tsL.crop(t0, t1)

    hH = np.asarray(segH.value, float)
    hL = np.asarray(segL.value, float)

    # ------------------------------------------------------------
    # 1) BANDPASS
    # ------------------------------------------------------------
    hH_bp = safe_bandpass(hH, fs, flow, fhigh)
    hL_bp = safe_bandpass(hL, fs, flow, fhigh)

    # ------------------------------------------------------------
    # 2) PSD bruit local
    # ------------------------------------------------------------
    noiseH = tsH.crop(t0 - noise_pad, t0 - 1.0)
    noiseL = tsL.crop(t0 - noise_pad, t0 - 1.0)

    fH_psd, S1 = psd_welch(noiseH, fmin=flow, fmax=fhigh)
    fL_psd, S2 = psd_welch(noiseL, fmin=flow, fmax=fhigh)

    # Interpolation correcte des deux PSD sur fH_psd
    if not np.allclose(fH_psd, fL_psd):
        S2 = np.interp(fH_psd, fL_psd, S2)

    # IMPORTANT : S1 et S2 doivent TOUJOURS être interpolés sur freq FFT,
    # pas uniquement S2 !
    f_psd = fH_psd
    # ------------------------------------------------------------
    # 3) SPECTRE COHÉRENT
    # ------------------------------------------------------------
    try:
        f_use, dEdf_use = coherent_spectrum(
            hH_bp, hL_bp, fs,
            f_psd, S1, S2,
            distance_mpc,
            model_energy
        )
    except Exception as e:
        print(f"[ERREUR] Spectre impossible : {e}")
        return None
    # ------------------------------------------------------------
    # 4) Energie totale
    # ------------------------------------------------------------
    E_total = float(np.trapz(dEdf_use, f_use))

    # ------------------------------------------------------------
    # 5) nu_eff
    # ------------------------------------------------------------
    """
    denom = max(np.trapz(dEdf_use, f_use), 1e-30)
    nu_eff = float(np.trapz(f_use * dEdf_use, f_use) / denom)
    # --- Patch anti-broadcast : forcer vecteurs compatibles ---
    f_use = np.array(f_use).flatten()
    dEdf_use = np.array(dEdf_use).flatten()

    if len(f_use) != len(dEdf_use) or len(f_use) < 10:
        print("[ERREUR] Spectre trop court — skip event")
        return {"E_total": 0.0, "m_sun": 0.0, "nu_eff": 0.0}
    """
    nu_eff = compute_nu_eff(f_use,dEdf_use);
    # ------------------------------------------------------------
    # 6) Phase log-spectrale (alpha, phi)
    # ------------------------------------------------------------
    alpha_logsin, phi_logsin = extract_phase_logsin(
        np.array(f_use),
        np.array(dEdf_use)
    )

    # ------------------------------------------------------------
    # 7) tau_geo (géométrique)
    # ------------------------------------------------------------
    tau_geo = estimate_tau_geo(tsH, tsL, gps, fs, flow, fhigh)

    # ------------------------------------------------------------
    # 8) tau_final (fusion géo + phase)
    # ------------------------------------------------------------
    tau_final, tau_phase = combine_tau(
        tau_geo,
        alpha_logsin,
        nu_eff,
        w_geo=0.6,
        w_phase=0.4,
        tau_min=-0.02,
        tau_max=0.02
    )

    # ------------------------------------------------------------
    # Résultats
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # 9) Construction directe du JSON (sans build_event_result)
    # ------------------------------------------------------------
    M_sun_J = M_sun * c**2

    result = {
        "event": event_name,
        "gps": float(gps),                # ← demandé
        "nu_eff": float(nu_eff),
        "tau": float(tau_final),
        "tau_geo": float(tau_geo),
        "tau_phase": float(tau_phase),

        # Phase log-spectrale
        "alpha_logsin": float(alpha_logsin),
        "phi_logsin": float(phi_logsin),

        # Spectre d'énergie
        "f": [float(x) for x in f_use],
        "dEdf": [float(x) for x in dEdf_use],

        # Énergies
        "E_total": float(E_total),
        "M_sun": float(E_total / M_sun_J),
        "model_energy": model_energy
    }

    # ------------------------------------------------------------
    # Export JSON
    # ------------------------------------------------------------
    export_json_result(event_name, result)

    # ------------------------------------------------------------
    # Affichage console
    # ------------------------------------------------------------
    print(f"{event_name:20s} | ν_eff={nu_eff:7.1f} Hz | τ={tau_final:+.5f} s | "
          f"E={E_total:.3e} J | M={result['M_sun']:.3f} M_sun")

    return result

# ============================================================
# 14) MAIN — INTERFACE PROFESSIONNELLE
# ============================================================
# ==========================
# CLI
# ==========================
def main():
    ap = argparse.ArgumentParser(description="Analyse LIGO cohérente h★=1 (énergie intrinsèque)")
    ap.add_argument("--event", required=True)
    ap.add_argument("--distance-mpc", type=float, required=True, 
                   help="Distance de l'événement en Mpc (utilisée pour calculer l'énergie intrinsèque)")
    ap.add_argument("--flow", type=float)
    ap.add_argument("--fhigh", type=float)
    ap.add_argument("--signal-win", type=float)
    ap.add_argument("--noise-pad", type=float)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    evp = EVENT_PARAMS.get(args.event, EVENT_PARAMS["default"])
    flow = evp["flow"] if args.flow is None else args.flow
    fhigh = evp["fhigh"] if args.fhigh is None else args.fhigh
    signal_win = evp["signal_win"] if args.signal_win is None else args.signal_win
    noise_pad = evp["noise_pad"] if args.noise_pad is None else args.noise_pad

    gps = datasets.event_gps(args.event)
    t0, t1 = gps - 1200.0, gps + 1200.0
    print(f"📡 Téléchargement des données pour {args.event}...")
    H1, L1 = fetch("H1", t0, t1), fetch("L1", t0, t1)

    res = analyze_event(
        args.event,
        H1,
        L1,
        gps,
        args.distance_mpc,
        flow,
        fhigh,
        signal_win,
        noise_pad,
        model_energy="toi",
        plot=args.plot
    )

    print(f"\n🎯 SYNTHÈSE FINALE: {args.event}")
    print(f"Distance: {args.distance_mpc} Mpc")
    print(f"Énergie intrinsèque: {res['E_total']:.3e} J ({res['M_sun']:.3f} M☉)")
    print(f"Fréquence effective: {res['nu_eff']:.1f} Hz")
    print(f"NOTE: Énergie intrinsèque calculée avec correction de distance (×d²)")
    print(f"      Référence d'amplitude: h★={H_STAR:.2e}\n{'='*60}")

if __name__ == "__main__":
    main()
