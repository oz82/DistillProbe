"""Reproduce the manuscript's descriptive typology-gradient permutation test
(cross-language replication, syntactic role): is the mean Teacher-Student gap of
the three fusional languages larger than that of the two agglutinative languages
by more than chance, given only five language-level gaps?

Exact permutation: over all ways to label k of the n languages "fusional" (k = the
number of fusional languages present), compute the difference of group means; the
one-sided p-value is the fraction of labelings whose difference is >= the observed
one. With the five first-submission languages this reproduces p=0.10 (C(5,3)=10);
the revision adds English (C(6,4)=15). Gaps are read from the canonical and
revision syntax runs.

Usage:  python scripts/typology_permutation.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "results" / "_canonical_runs.txt"
MANIFESTS = (MANIFEST, ROOT / "results" / "_revision_runs.txt")

# Ground-truth typology labels. English is (weakly) fusional / analytic Indo-European;
# it joins the fusional group, so the revision compares 4 fusional vs 2 agglutinative.
FUSIONAL = {"PL", "IT", "EL", "EN"}
SYNTAX_CFG = {"syntax_v2": "TR", "syntax_v2_eu": "EU", "syntax_v2_pl": "PL",
              "syntax_v2_it": "IT", "syntax_v2_el": "EL", "syntax_v2_en": "EN"}


def run_dirs() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for manifest in MANIFESTS:
        if not manifest.exists():
            continue
        for line in manifest.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1].startswith("results/"):
                out[parts[0]] = ROOT / parts[1]
    return out


def gap(run_dir: Path) -> float:
    """Mean Teacher-Student LOO-accuracy gap for one language's syntax run."""
    m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    # the run stores the aggregate teacher/student mean accuracies
    import re
    s = json.dumps(m)
    t = float(re.search(r'"teacher_mean_acc"\s*:\s*([0-9.eE-]+)', s).group(1))
    st = float(re.search(r'"student_mean_acc"\s*:\s*([0-9.eE-]+)', s).group(1))
    return t - st


def main() -> int:
    if not MANIFEST.exists():
        print(f"missing {MANIFEST}; run scripts/rerun_canonical.sh first", file=sys.stderr)
        return 1
    dirs = run_dirs()
    gaps = {lang: gap(dirs[cfg]) for cfg, lang in SYNTAX_CFG.items()}
    langs = list(gaps)
    print("Language-level Teacher-Student gaps:")
    for lang in langs:
        tag = "fusional" if lang in FUSIONAL else "agglutinative"
        print(f"  {lang}: {gaps[lang]:+.4f}  ({tag})")

    def diff(fus: set[str]) -> float:
        agg = [gaps[l] for l in langs if l not in fus]
        f = [gaps[l] for l in langs if l in fus]
        return sum(f) / len(f) - sum(agg) / len(agg)

    observed = diff(FUSIONAL)
    k = len(FUSIONAL & set(langs))
    perms = [set(c) for c in combinations(langs, k)]
    ge = sum(1 for p in perms if diff(p) >= observed - 1e-12)
    p_one_sided = ge / len(perms)
    print(f"\nObserved fusional-minus-agglutinative gap difference: {observed:+.4f}")
    print(f"Exact permutation over C({len(langs)},{k})={len(perms)} labelings: "
          f"{ge}/{len(perms)} >= observed  ->  one-sided p = {p_one_sided:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
