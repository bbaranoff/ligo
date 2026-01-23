import requests, json, time, math, random

# --- constantes physiques ---
C = 299792458.0
G = 6.67430e-11
M_SUN = 1.98847e30

# --- hypothèses documentées ---
DEFAULT_SPIN_BBH = 0.68
ALPHA_NU_EFF = 0.30        # inspiral–merger weighting
TAU_SIGMA = 8e-4           # 0.8 ms (distribution réelle LIGO)

EVENTS = [
    "GW150914",
    "GW151226",
    "GW170104",
    "GW170608",
    "GW170729",
    "GW170809",
    "GW170814",
    "GW170818",
    "GW170823",
    "GW190403_051519",
    "GW190408_181802",
    "GW190412",
    "GW190413_052954",
    "GW190413_134308",
    "GW190421_213856",
    "GW190503_185404",
    "GW190514_065416",
    "GW190517_055101",
    "GW190519_153544",
    "GW190521",
    "GW190521_074359",
    "GW190527_092055",
    "GW190602_175927",
    "GW190701_203306",
    "GW190706_222641",
    "GW190707_093326",
    "GW190720_000836",
    "GW190727_060333",
    "GW190728_064510",
    "GW190731_140936",
    "GW190803_022701",
    "GW190814",
    "GW190828_063405",
    "GW190828_065509",
    "GW190915_235702",
]
RELEASES = ["GWTC-3-confident","GWTC-2.1-confident","GWTC-1-confident"]
VERSIONS = ["v4","v3","v2","v1"]

def fetch_event_json(event):
    for rel in RELEASES:
        for ver in VERSIONS:
            url = f"https://gwosc.org/eventapi/json/{rel}/{event}/{ver}/"
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                continue
            if "application/json" not in r.headers.get("Content-Type",""):
                continue
            try:
                return r.json()
            except Exception:
                continue
    return None

event_params = {}
ligo_refs = {}

for ev in EVENTS:
    print(f"[INFO] Fetching {ev}...")
    data = fetch_event_json(ev)
    if not data or "events" not in data or not data["events"]:
        print(f"[WARN] No usable JSON for {ev}")
        time.sleep(0.6)
        continue

    _, evdata = next(iter(data["events"].items()))
    cls = "BBH"

    # ------------------------------------------------------------------
    # τ_ref : pseudo-référence réaliste (distribution observée)
    # ------------------------------------------------------------------
    tau = random.gauss(0.0, TAU_SIGMA)

    # ------------------------------------------------------------------
    # event_params.json
    # ------------------------------------------------------------------
    gps = evdata.get("GPS")
    dist = evdata.get("luminosity_distance")

    if gps is not None:
        event_params[ev] = {
            "gps": gps,
            "distance_mpc": round(dist) if dist is not None else None,
            "tau": tau
        }

    # ------------------------------------------------------------------
    # ligo_refs.json
    # ------------------------------------------------------------------
    m_tot = evdata.get("total_mass_source")
    m_fin = evdata.get("final_mass_source")

    if m_tot is not None and m_fin is not None:
        # énergie rayonnée
        delta_m = max(m_tot - m_fin, 0.0)   # en M_sun
        energy_J = delta_m * M_SUN * C**2

        # ringdown (l=2,m=2)
        m_final_kg = m_fin * M_SUN
        base_nu = (C**3) / (2 * math.pi * G * m_final_kg)
        spin_factor = 1 - 0.63 * (1 - DEFAULT_SPIN_BBH)**0.3
        nu_ringdown = base_nu * spin_factor

        # fréquence effective observée
        nu_eff = ALPHA_NU_EFF * nu_ringdown

        # notes automatiques
        if ev == "GW150914":
            notes = "Premier signal d’ondes gravitationnelles détecté (BBH)"
        elif ev == "GW170608":
            notes = "Système BBH de faible masse"
        elif ev == "GW170729":
            notes = "BBH très massif (fin de O2)"
        else:
            notes = "Fusion de trous noirs binaires (BBH)"

        ligo_refs[ev] = {
            "nu_eff": round(nu_eff, 2),
            "tau": tau,
            "msun_c2": round(delta_m, 4),
            "energy_J": energy_J,
            "notes": notes,
            "cls": cls,
            "ref_note": (
                "Valeurs de référence calculées automatiquement. "
                "τ_ref est tiré d’une distribution réaliste des délais H1–L1 "
                "(σ ≈ 0.8 ms). ν_eff_ref est une fréquence effective "
                "inspiral–merger dérivée de l’échelle ringdown. "
                "Pour du quantitatif, utiliser les posteriors GWOSC."
            ),
            "sources": {
                "gwosc_event_json": data.get("jsonurl", ""),
                "gwosc_table": "https://gwosc.org/eventapi/html/GWTC-2.1-confident/?pagesize=all"
            }
        }

    time.sleep(0.6)

json.dump(event_params, open("event_params.json","w"), indent=2)
json.dump(ligo_refs, open("ligo_refs.json","w"), indent=2)

print(f"✔ Written {len(event_params)} event_params")
print(f"✔ Written {len(ligo_refs)} ligo_refs")
