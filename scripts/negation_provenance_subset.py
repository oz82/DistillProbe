"""Polarity flips on the negation items that involve no author judgement.

Usage: python scripts/negation_provenance_subset.py

The cleaned negation sets mix two sources: hand-curated seed triplets (Turkish:
the 25 of tr_negation.v1; English: the 25 marked "hand-curated seed") and pairs
taken from a lexicon's antonym relation (KeNet / WordNet) and then filtered by the
deterministic cleaning rule (scripts/clean_negation_dataset.py). Only the seeds
rest on the author's antonymy judgement. This reports, from the canonical runs'
per-item tables, the polarity-flip counts on the lexicon-only subset next to the
full cleaned set, for raw and carrier mode.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = {
    ("tr", "raw"): "negation_v3_clean_raw_baselines",
    ("tr", "carrier"): "negation_v3_clean_carrier_baselines",
    ("en", "raw"): "negation_en_clean_raw_baselines",
    ("en", "carrier"): "negation_en_clean_carrier_baselines",
}


def run_dirs() -> dict[str, Path]:
    out = {}
    for line in (ROOT / "results" / "_canonical_runs.txt").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].startswith("results/"):
            out[parts[0]] = ROOT / parts[1]
    return out


def seed_pairs(lang: str) -> set[tuple[str, str]]:
    if lang == "tr":
        seeds = json.load(open(ROOT / "data/negation/tr_negation.v1.json", encoding="utf-8"))
        return {(t["positive"], t["antonym"]) for t in seeds["triplets"]}
    clean = json.load(open(ROOT / "data/negation/en_negation.v3_clean.json", encoding="utf-8"))
    return {
        (t["positive"], t["antonym"])
        for t in clean["triplets"]
        if t["note"] == "hand-curated seed"
    }


def main() -> None:
    dirs = run_dirs()
    for (lang, mode), cfg in RUNS.items():
        rows = list(csv.DictReader(open(dirs[cfg] / "per_category.csv", encoding="utf-8")))
        seeds = seed_pairs(lang)
        lexicon = [r for r in rows if (r["positive"], r["antonym"]) not in seeds]
        encoders = [c.split(":", 1)[1] for c in rows[0] if c.startswith("polarity_flip:")]
        cells = []
        for enc in encoders:
            col = f"polarity_flip:{enc}"
            flips_all = sum(float(r[col]) > 0 for r in rows)
            flips_lex = sum(float(r[col]) > 0 for r in lexicon)
            cells.append(f"{enc} {flips_lex}/{len(lexicon)} (all {flips_all}/{len(rows)})")
        print(f"[{lang} {mode}] lexicon-only: " + "; ".join(cells))


if __name__ == "__main__":
    main()
