#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Télécharge et met en cache les données GWOSC au format NPZ.
Aucune analyse ici. Juste I/O propre.
"""

import os
import numpy as np
from gwosc import datasets
from gwpy.timeseries import TimeSeries


DATA_DIR = "data/npz"


def npz_path(event: str, det: str) -> str:
    return os.path.join(DATA_DIR, f"{event}_{det}.npz")


def download_npz(event: str, det: str, t0: float, t1: float) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = npz_path(event, det)

    print(f"⬇️  Download {event} {det} → {path}")
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

def ensure_npz(event: str, det: str, t0: float, t1: float) -> str:
    path = npz_path(event, det)
    if os.path.exists(path):
        return path
    try:
        return download_npz(event, det, t0, t1)
    except Exception as e:
        if det == "V1":
            raise RuntimeError(f"V1 absent pour {event}") from e
        raise

