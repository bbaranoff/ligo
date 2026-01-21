#!/usr/bin/env python3
# mc2_percent.py
# Calcule erreurs en % sur M⊙c² à partir d'un tableau texte "Event | ... | M⊙c² | M⊙c²_r | ..."

import argparse
import re
import statistics as stats
from typing import List, Dict, Optional, Tuple


def _to_float(s: str) -> Optional[float]:
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def parse_table(text: str) -> Tuple[List[str], List[Dict[str, str]]]:
    lines = [ln.rstrip("\n") for ln in text.splitlines() if ln.strip()]
    if not lines:
        return [], []

    # détecte séparateur (| prioritaire)
    header_idx = None
    sep = None
    for i, ln in enumerate(lines):
        if ("|" in ln) and re.search(r"[A-Za-z_⊙τν]", ln):
            header_idx = i
            sep = "|"
            break
        if ("\t" in ln) and re.search(r"[A-Za-z_⊙τν]", ln):
            header_idx = i
            sep = "\t"
            break
    if header_idx is None:
        raise SystemExit("Header introuvable: il faut une ligne de noms de colonnes avec '|'.")

    headers = [h.strip() for h in lines[header_idx].split(sep) if h.strip()]

    start = header_idx + 1
    # saute ligne de tirets si présente
    if start < len(lines) and re.fullmatch(r"[\-\+\|\s\t]+", lines[start]):
        start += 1

    rows: List[Dict[str, str]] = []
    for ln in lines[start:]:
        if re.fullmatch(r"[\-\+\|\s\t]+", ln):
            continue
        parts = [p.strip() for p in ln.split(sep)]
        if len(parts) < len(headers):
            continue
        parts = parts[:len(headers)]
        rows.append({headers[j]: parts[j] for j in range(len(headers))})

    return headers, rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Erreur % sur M⊙c² (signée & absolue), + stats robustes.")
    ap.add_argument("--file", required=True, help="Fichier contenant le tableau (format avec '|').")
    ap.add_argument("--pred", default="M⊙c²", help="Nom colonne prédite (défaut: M⊙c²).")
    ap.add_argument("--ref", default="M⊙c²_r", help="Nom colonne référence (défaut: M⊙c²_r).")
    ap.add_argument("--event-col", default="Event", help="Nom colonne event (défaut: Event).")
    ap.add_argument("--exclude-cls", default=None, help="Exclure une classe via substring dans Notes (ex: BNS).")
    ap.add_argument("--top", type=int, default=10, help="Afficher TOP N pires |%err|.")
    args = ap.parse_args()

    with open(args.file, "r", encoding="utf-8") as f:
        text = f.read()

    headers, rows = parse_table(text)
    if not rows:
        raise SystemExit("Aucune donnée détectée.")

    # cherche colonne Notes si dispo
    notes_key = None
    for k in headers:
        if k.lower() == "notes":
            notes_key = k
            break

    out = []  # (event, pred, ref, err, err_pct, abs_err_pct)
    for r in rows:
        ev = r.get(args.event_col, "").strip()
        if not ev:
            continue

        if args.exclude_cls and notes_key:
            if args.exclude_cls in r.get(notes_key, ""):
                continue

        pred = _to_float(r.get(args.pred, ""))
        ref = _to_float(r.get(args.ref, ""))
        if pred is None or ref is None:
            continue
        if not (ref == ref) or ref == 0.0:
            continue

        err = pred - ref
        err_pct = 100.0 * (err / ref)
        out.append((ev, pred, ref, err, err_pct, abs(err_pct)))

    if not out:
        raise SystemExit("Aucune ligne exploitable. Vérifie noms de colonnes et données.")

    err_pcts = [x[4] for x in out]
    abs_err_pcts = [x[5] for x in out]

    def summarize(name: str, vals: List[float]) -> None:
        mean = sum(vals) / len(vals)
        median = stats.median(vals)
        mad = stats.median([abs(v - median) for v in vals])
        print(f"\n=== {name} ===")
        print(f"n         = {len(vals)}")
        print(f"mean%     = {mean:.6f}")
        print(f"median%   = {median:.6f}")
        print(f"MAD%      = {mad:.6f}")
        print(f"min%      = {min(vals):.6f}")
        print(f"max%      = {max(vals):.6f}")

    summarize("ERREUR % SIGNÉE (100*(pred-ref)/ref)", err_pcts)
    summarize("ERREUR % ABSOLUE", abs_err_pcts)

    # affiche tableau per-event trié par |%err|
    out_sorted = sorted(out, key=lambda x: x[5], reverse=True)

    print(f"\n=== TOP {min(args.top, len(out_sorted))} (par |%err| décroissant) ===")
    print(f"{'Event':18s}  {'pred':>8s}  {'ref':>8s}  {'err':>9s}  {'err%':>10s}")
    print("-" * 60)
    for ev, pred, ref, err, err_pct, abs_err_pct in out_sorted[: args.top]:
        print(f"{ev:18s}  {pred:8.2f}  {ref:8.2f}  {err:9.2f}  {err_pct:10.3f}")

    # (optionnel) dump complet
    print("\n=== LISTE COMPLÈTE (triée par |%err|) ===")
    for ev, pred, ref, err, err_pct, abs_err_pct in out_sorted:
        print(f"{ev}: pred={pred:.2f} ref={ref:.2f} err={err:+.2f}  err%={err_pct:+.3f}%")


if __name__ == "__main__":
    main()
