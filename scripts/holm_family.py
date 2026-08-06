"""Reproduce the manuscript's multiple-comparison correction (Materials & Methods,
statistical methodology): the primary hypothesis-test family of N=18 tests,
Holm--Bonferroni (FWER) and Benjamini--Hochberg (FDR).

The family membership is defined here explicitly (it is the set of confirmatory
directional tests reported for the idiomaticity, syntactic-role, negation, and
polysemy probes). Each p-value is read from the canonical run artefacts listed in
`results/_canonical_runs.txt`, so this script reproduces the paper's
"17/18 remain significant, largest corrected p = 2.0e-3" from the committed runs.

Usage:  python scripts/holm_family.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from distillprobe.metrics.stats import benjamini_hochberg, holm_bonferroni  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "results" / "_canonical_runs.txt"


def run_dirs() -> dict[str, Path]:
    """Parse results/_canonical_runs.txt -> {config: run_dir}."""
    out: dict[str, Path] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].startswith("results/"):
            out[parts[0]] = ROOT / parts[1]
    return out


def _metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"]


def build_family(dirs: dict[str, Path]) -> list[tuple[str, float]]:
    fam: list[tuple[str, float]] = []

    # Idiomaticity: per language, the two sequential MWU directional tests.
    idiom = {"idiom_v2": "TR", "idiom_v2_eu": "EU", "idiom_v2_pl": "PL",
             "idiom_v2_it": "IT", "idiom_v2_el": "EL"}
    for cfg, lang in idiom.items():
        comp = _metrics(dirs[cfg])["comparisons"]
        for key, short in (("VID_gt_LVC.full", "VID>LVC"), ("LVC.full_gt_LITERAL", "LVC>LIT")):
            hit = next(v for k, v in comp.items() if k.endswith(key))
            fam.append((f"idiom/{lang}/{short}", float(hit["p"])))

    # Syntactic role: per language, teacher>student paired Wilcoxon.
    syntax = {"syntax_v2": "TR", "syntax_v2_eu": "EU", "syntax_v2_pl": "PL",
              "syntax_v2_it": "IT", "syntax_v2_el": "EL"}
    for cfg, lang in syntax.items():
        m = _metrics(dirs[cfg])
        fam.append((f"syntax/{lang}/T>S", float(m["wilcoxon_one_sided_greater_p"])))

    # Negation: LaBSE>M2V paired Wilcoxon, raw and carrier modes.
    for cfg, mode in (("negation_v3_clean_raw_baselines", "raw"),
                      ("negation_v3_clean_carrier_baselines", "carrier")):
        comp = _metrics(dirs[cfg])["comparisons"]
        hit = comp["labse_gt_m2v_potion"]
        fam.append((f"negation/{mode}/LaBSE>M2V", float(hit["wilcoxon_p"])))

    # Polysemy: cross-encoder Spearman permutation test (primary, leak-free).
    ce = _metrics(dirs["polysemy_v3_primary"])["cross_encoder"]
    block = next(iter(ce.values()))
    fam.append(("polysemy/spearman-perm", float(block["spearman_perm_p"])))

    return fam


def main() -> int:
    if not MANIFEST.exists():
        print(f"missing {MANIFEST}; run scripts/rerun_canonical.sh first", file=sys.stderr)
        return 1
    dirs = run_dirs()
    fam = build_family(dirs)
    names = [n for n, _ in fam]
    pvals = [p for _, p in fam]
    holm = holm_bonferroni(pvals)
    bh = benjamini_hochberg(pvals)

    order = sorted(range(len(fam)), key=lambda i: pvals[i])
    print(f"Primary hypothesis-test family: N={len(fam)}")
    print(f"{'test':28} {'raw p':>12} {'Holm p':>12} {'BH p':>12}  sig(Holm)")
    for i in order:
        sig = "yes" if holm[i] < 0.05 else "NO"
        print(f"{names[i]:28} {pvals[i]:12.3e} {holm[i]:12.3e} {bh[i]:12.3e}  {sig}")
    n_sig = sum(1 for h in holm if h < 0.05)
    max_sig = max((h for h in holm if h < 0.05), default=float("nan"))
    print(f"\nHolm--Bonferroni: {n_sig}/{len(fam)} significant at alpha=0.05; "
          f"largest corrected p among significant = {max_sig:.2e}")
    print(f"Benjamini--Hochberg: {sum(1 for b in bh if b < 0.05)}/{len(fam)} significant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
