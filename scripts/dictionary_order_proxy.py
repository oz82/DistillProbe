"""Dictionary sense order as a salience proxy, and the encoders' agreement with it.

Usage: python scripts/dictionary_order_proxy.py

1. Validation on the human pilot set (Turkish v3, 50 nouns): per word, Spearman rho
   between the annotators' median salience and TDK's sense order (first sense =
   most salient), and how often the human top-rated sense is TDK's first sense.
   Human ties are broken toward the lowest-numbered sense, as in the pilot analysis.
2. On the rebuilt primary sets (Turkish v4, English v2), which have no human
   ratings: per encoder, the per-word Spearman rho between its sense ranking and
   dictionary order, and how often it ranks the first dictionary sense on top,
   against the chance rate mean(1/k) with an exact one-sided p-value.

Reads the released ratings and the canonical runs' per-item tables only.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from distillprobe.metrics.stats import bootstrap_ci  # noqa: E402

RATINGS = ROOT / "annotation" / "polysemy_ratings_anon.csv"
SETS = {"Turkish v4": "polysemy_v4_primary", "English v2": "polysemy_en_primary"}


def run_dirs() -> dict[str, Path]:
    out = {}
    for line in (ROOT / "results" / "_canonical_runs.txt").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].startswith("results/"):
            out[parts[0]] = ROOT / parts[1]
    return out


def ci_text(values: list[float]) -> str:
    ci = bootstrap_ci(values)
    return f"{np.mean(values):.2f} [{ci.low:.2f}, {ci.high:.2f}]"


def tail_p(hits: int, probs: list[float]) -> float:
    """P(X >= hits) for a sum of independent Bernoulli(p_i) (Poisson binomial)."""
    dist = np.zeros(len(probs) + 1)
    dist[0] = 1.0
    for p in probs:
        dist[1:] = dist[1:] * (1 - p) + dist[:-1] * p
        dist[0] *= 1 - p
    return float(dist[hits:].sum())


def pilot() -> None:
    by_word: dict[int, dict[int, float]] = defaultdict(dict)
    with open(RATINGS, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        coders = [c for c in reader.fieldnames or [] if c.startswith("R")]
        for row in reader:
            scores = [float(row[c]) for c in coders if row[c] not in ("", None)]
            median = float(np.median(scores)) if scores else float("nan")
            by_word[int(row["kelime_no"])][int(row["anlam_no"])] = median
    rhos, hits, chance = [], 0, []
    for senses in by_word.values():
        order = sorted(senses)
        gold = np.array([senses[i] for i in order])
        mask = ~np.isnan(gold)
        rho, _ = spearmanr(gold[mask], -np.array(order)[mask])
        if not np.isnan(rho):
            rhos.append(float(rho))
        hits += int(np.argmax(np.where(mask, gold, -np.inf)) == 0)
        chance.append(1 / len(order))
    print(
        f"[pilot, Turkish v3, {len(by_word)} words] human median vs TDK order: "
        f"rho = {ci_text(rhos)}; human top sense = TDK first sense in {hits}/{len(by_word)} "
        f"(chance {np.mean(chance):.2f}, p = {tail_p(hits, chance):.1e})"
    )


def rebuilt_sets() -> None:
    dirs = run_dirs()
    for name, cfg in SETS.items():
        rows = list(csv.DictReader(open(dirs[cfg] / "per_category.csv", encoding="utf-8")))
        encoders = [c.split(":", 1)[1] for c in rows[0] if c.startswith("sense_sims:")]
        chance = [1 / int(r["n_senses"]) for r in rows]
        for enc in encoders:
            rhos, hits = [], 0
            for r in rows:
                sims = np.array(json.loads(r[f"sense_sims:{enc}"]))
                rho, _ = spearmanr(sims, -np.arange(len(sims)))
                if not np.isnan(rho):
                    rhos.append(float(rho))
                hits += int(np.argmax(sims) == 0)
            print(
                f"[{name}, {len(rows)} words] {enc} vs dictionary order: rho = {ci_text(rhos)}; "
                f"first sense on top {hits}/{len(rows)} (chance {np.mean(chance):.2f}, "
                f"p = {tail_p(hits, chance):.1e})"
            )


def main() -> None:
    pilot()
    rebuilt_sets()


if __name__ == "__main__":
    main()
