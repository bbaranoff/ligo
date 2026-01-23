#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Télécharge et met en cache les données GWOSC au format NPZ
pour tous les événements listés dans event_params.json.

Aucune analyse ici. I/O propre + rate limiting.
"""

import os
import json
import time
import numpy as np
from gwpy.timeseries import TimeSeries

DATA_DIR = "data/npz"
EVENT_PARAMS = "event_params.json"

# --- réglages ---
WINDOW_PRE  = 8.0     # secondes avant GPS
WINDOW_POST = 8.0     # secondes après GPS
SLEEP_SEC   = 0.8     # anti-kick GWOSC
DETECTORS   = ["H1", "L1", "V1"]


def npz_path(event: str, det: str) -> str:
    return os.path.join(DATA_DIR, f"{event}_{det}.npz")


def download_npz(event: str, det: str, t0: float, t1: float) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = npz_path(event, det)

    print(f"⬇️  {event} {det}")
    ts = TimeSeries.fetch_open_data(det, t0, t1, cache=True)

    fs = float(ts.sample_rate.value)
    t0_ts = float(ts.t0.value)
    t1_ts = t0_ts + (len(ts) / fs)

    np.savez_compressed(
        path,
        data=np.asarray(ts.value, float),
        fs=fs,
        t0=t0_ts,
        t1=t1_ts,
        det=det,
        event=event,
    )
    return path


def ensure_npz(event: str, det: str, t0: float, t1: float):
    path = npz_path(event, det)
    if os.path.exists(path):
        return

    try:
        download_npz(event, det, t0, t1)
    except Exception as e:
        msg = str(e)
        if "Cannot find a GWOSC dataset" in msg or "absent" in msg:
            print(f"[INFO] {det} indisponible pour {event}")
        else:
            print(f"[WARN] {det} erreur pour {event}: {e}")
    finally:
        time.sleep(SLEEP_SEC)


def main():
    with open(EVENT_PARAMS, "r") as f:
        events = json.load(f)

    for event, params in events.items():
        gps = params.get("gps")
        if gps is None:
            continue

        t0 = gps - WINDOW_PRE
        t1 = gps + WINDOW_POST

        print(f"\n📡 {event}  (GPS={gps})")

        for det in DETECTORS:
            ensure_npz(event, det, t0, t1)


if __name__ == "__main__":
    main()
