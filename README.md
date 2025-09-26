# ligo
testing ligo research in python

Example

```bash
python gwosc_floquet_ak.py --event GW150914 --detectors H1 L1 \
  --mass1 42.9436 --mass2 28.6290 \
  --ak --ak-plot --ak-file ak_GW150914.json
/root/.env/lib/python3.12/site-packages/pykerr/qnm.py:2: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
{
  "event": "GW150914",
  "gps": 1126259462.4,
  "detectors": {
    "H1": {
      "mass1": 42.9436,
      "mass2": 28.629,
      "mchirp": 30.399988384995936,
      "eta": 0.23999988822507,
      "snr_max": 51.99254718002936
    },
    "L1": {
      "mass1": 42.9436,
      "mass2": 28.629,
      "mchirp": 30.399988384995936,
      "eta": 0.23999988822507,
      "snr_max": 7.2726156200125365
    }
  },
  "model": "IMRPhenomD",
  "ak_summary": {
    "H1": {
      "rhos_rel": {
        "rho_A0_rel": 524120.625,
        "rho_Ap1_rel": 110.54021565186352,
        "rho_Am1_rel": 110.54021565186352
      }
    },
    "L1": {
      "rhos_rel": {
        "rho_A0_rel": 524120.625,
        "rho_Ap1_rel": 118.60914466985439,
        "rho_Am1_rel": 118.60914466985439
      }
    }
  },
  "ak_file": "ak_GW150914.json"
}
```

