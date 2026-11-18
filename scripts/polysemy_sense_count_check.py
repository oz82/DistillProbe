"""Check that the Turkish-English polysemy gap is not an artefact of senses per word.

The two primary sets differ in polysemy degree (Turkish 4.4, English 5.9 senses per
word), and a per-word Spearman rho over k senses behaves differently for small k.
This compares the per-word student-reference rho within sense-count strata and
reweights English to the Turkish sense-count distribution, with a bootstrap CI.

Usage: python scripts/polysemy_sense_count_check.py
(reads polysemy_v4_primary and polysemy_en_primary from results/_canonical_runs.txt)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
COL = "spearman:m2v_potion_labse"
STRATA = {"3": (3, 3), "4-5": (4, 5), ">=6": (6, 999)}
CAP = 8  # sense counts >= CAP are pooled for reweighting (English has few words above it)


def run_dir(cfg: str) -> Path:
    for line in (ROOT / "results" / "_canonical_runs.txt").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0] == cfg:
            return ROOT / parts[1]
    raise SystemExit(f"{cfg} not in results/_canonical_runs.txt")


def reweighted(en: pd.DataFrame, weights: pd.Series) -> float:
    means = en.groupby("kb")[COL].mean()
    return float(sum(weights[k] * means[k] for k in weights.index))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tr", default=None, help="Turkish run dir (default: canonical)")
    p.add_argument("--en", default=None, help="English run dir (default: canonical)")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tr = pd.read_csv(Path(args.tr or run_dir("polysemy_v4_primary")) / "per_category.csv")
    en = pd.read_csv(Path(args.en or run_dir("polysemy_en_primary")) / "per_category.csv")
    print(f"mean senses/word: TR {tr.n_senses.mean():.2f}, EN {en.n_senses.mean():.2f}")
    for name, (lo, hi) in STRATA.items():
        t = tr[tr.n_senses.between(lo, hi)][COL]
        e = en[en.n_senses.between(lo, hi)][COL]
        print(
            f"  k {name:>4}: TR n={len(t):3d} rho={t.mean():.3f} | "
            f"EN n={len(e):3d} rho={e.mean():.3f}"
        )

    tr["kb"] = tr.n_senses.clip(upper=CAP)
    en["kb"] = en.n_senses.clip(upper=CAP)
    weights = tr.kb.value_counts(normalize=True)
    rng = np.random.default_rng(args.seed)
    boot = []
    for _ in range(args.n_boot):
        s = en.iloc[rng.integers(0, len(en), len(en))]
        if set(weights.index) <= set(s.kb):
            boot.append(reweighted(s, weights))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(
        f"EN reweighted to the TR sense-count distribution: rho={reweighted(en, weights):.3f} "
        f"[{lo:.3f}, {hi:.3f}] (TR {tr[COL].mean():.3f})"
    )


if __name__ == "__main__":
    main()
