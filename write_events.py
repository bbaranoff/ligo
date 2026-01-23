import requests, json, time, math

# --- constantes physiques ---
C = 299792458.0
M_SUN = 1.98847e30

EVENTS = [
    "GW150914","GW151226","GW170104","GW170608","GW170809",
    "GW170814","GW170818","GW170823","GW190403_051519","GW190408_181802",
    "GW190412","GW190413_052954","GW190413_134308","GW190421_213856",
    "GW190503_185404","GW190514_065416","GW190517_055101","GW190519_153544",
    "GW190521","GW190521_074359","GW190527_092055","GW190602_175927",
    "GW190701_203306","GW190706_222641","GW190707_093326","GW190720_000836",
    "GW190727_060333","GW190728_064510","GW190731_140936","GW190803_022701",
    "GW190814","GW190828_063405","GW190828_065509","GW190915_235702",
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
    print(f"[INFO] Fetching {ev}…")
    data = fetch_event_json(ev)

    if not data or "events" not in data or not data["events"]:
        print(f"[WARN] No usable JSON for {ev}")
        time.sleep(0.8)
        continue

    _, evdata = next(iter(data["events"].items()))

    # ----------------------------------------------------
    # event_params.json (données brutes utiles)
    # ----------------------------------------------------
    gps = evdata.get("GPS")
    dist = evdata.get("luminosity_distance")

    if gps is not None:
        event_params[ev] = {
            "gps": gps,
            "distance_mpc": round(dist) if dist is not None else None
        }

    # ----------------------------------------------------
    # ligo_refs.json (LIGO only)
    # ----------------------------------------------------
    m_tot = evdata.get("total_mass_source")
    m_fin = evdata.get("final_mass_source")

    if m_tot is None or m_fin is None:
        print(f"[WARN] Missing mass info for {ev}")
        time.sleep(0.8)
        continue

    delta_m = max(m_tot - m_fin, 0.0)
    energy_J = delta_m * M_SUN * C**2

    cls = "BBH"
    if m_tot < 6:
        cls = "BNS"
    elif m_tot < 10:
        cls = "NSBH"

    ligo_refs[ev] = {
        "msun_c2": round(delta_m, 4),
        "energy_J": energy_J,
        "cls": cls,
        "source": "GWOSC EventAPI"
    }

    time.sleep(0.8)

json.dump(event_params, open("event_params.json","w"), indent=2)
json.dump(ligo_refs, open("ligo_refs.json","w"), indent=2)

print(f"✔ Written {len(event_params)} event_params")
print(f"✔ Written {len(ligo_refs)} ligo_refs")
