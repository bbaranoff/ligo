# ligo  
Testing LIGO research in Python  

run

```
python3 ligo_spectral_planck.py   --calibrate-lsq   --event-params event_params.json   --refs ligo_refs.json   --ref-key energy_J   --exclude-cls BNS   --cal-out calibrated.json
bash run_all.sh
python test421_better.py results/ --glob GW*
```
