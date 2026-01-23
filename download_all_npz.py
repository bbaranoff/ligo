#!/usr/bin/env python3
import json
from gwosc import datasets
from ligo_npz_downloader import ensure_npz

# -----------------------------
# PARAMÈTRES
# -----------------------------
EVENTS_JSON = "event_params.json"   # ou ligo_refs.json
USE_VIRGO = True

PRE  = 1200.0
POST = 200.0

# -----------------------------
# CHARGER LES EVENTS
# -----------------------------
with open(EVENTS_JSON, "r") as f:
    params = json.load(f)

events = sorted(params.keys())

print(f"📡 Download NPZ pour {len(events)} events")

# -----------------------------
# BOUCLE DOWNLOAD
# -----------------------------
for ev in events:
    try:
        gps = datasets.event_gps(ev)
    except Exception as e:
        print(f"[SKIP] {ev}: GPS introuvable ({e})")
        continue

    t0 = gps - PRE
    t1 = gps + POST

    print(f"\n=== {ev} ===")

    # SOCLE
    try:
        ensure_npz(ev, "H1", t0, t1)
        ensure_npz(ev, "L1", t0, t1)
    except Exception as e:
        print(f"[FAIL] {ev}: H1/L1 impossible ({e})")
        continue

    # BONUS V1
    if USE_VIRGO:
        try:
            dets = datasets.event_detectors(ev)
        except Exception:
            dets = []

        if "V1" in dets:
            try:
                ensure_npz(ev, "V1", t0, t1)
            except Exception as e:
                print(f"[INFO] {ev}: V1 indisponible ({e})")

print("\n✅ Download terminé.")
