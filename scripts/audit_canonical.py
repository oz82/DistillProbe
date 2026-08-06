"""Faz 5 numeric audit: extract headline metrics from the canonical runs listed in
results/_canonical_runs.txt so they can be checked against the manuscript.

Usage: python scripts/audit_canonical.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "results" / "_canonical_runs.txt"


def load(run_dir: str) -> dict:
    p = ROOT / run_dir / "metrics.json"
    return json.load(open(p))["metrics"]


def main() -> None:
    runs: dict[str, str] = {}
    revs = set()
    for line in SUMMARY.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].startswith("results/"):
            runs[parts[0]] = parts[1]
            if len(parts) >= 3:
                revs.add(parts[2])
    print(f"git_rev(s) across runs: {revs}")
    print(f"configs with runs: {len(runs)}\n")

    def show(cfg, fn):
        if cfg not in runs:
            print(f"[{cfg}] MISSING")
            return
        try:
            fn(load(runs[cfg]))
        except Exception as e:  # noqa: BLE001
            print(f"[{cfg}] ERROR {e}")

    # morphology
    def morph(m):
        print("[morphology] per-category M2V vs FT consistency:")
        for r in []:
            pass
    show("morph_v2_corpus", lambda m: print("[morph] per_encoder:",
         {k: round(v, 3) for k, v in m.get("per_encoder_mean_consistency", {}).items()}))

    # idiom TR (item + declustered)
    def idiom(m):
        print("[idiom TR] means:", {k: round(v, 4) for k, v in m["per_category_mean_gap"].items()})
        for k, v in m["comparisons"].items():
            print(f"   item  {k.split(':')[-1]}: p={v['p']:.2e} δ={v['cliffs_delta']:+.3f}")
        for k, v in m.get("comparisons_declustered", {}).items():
            print(f"   lemma {k.split(':')[-1]}: p={v['p']:.2e} δ={v['cliffs_delta']:+.3f} (n_lemmas {v['n_lemmas_a']}/{v['n_lemmas_b']})")
    show("idiom_v2", idiom)

    for lang in ["eu", "pl", "it", "el"]:
        show(f"idiom_v2_{lang}", lambda m, L=lang: print(f"[idiom {L}]",
             {k: round(v, 3) for k, v in m["per_category_mean_gap"].items()},
             {k.split(':')[-1]: f"{v['p']:.1e}" for k, v in m["comparisons"].items()}))

    show("idiom_v2_me5_teacher", lambda m: print("[idiom mE5]",
         {k: round(v, 4) for k, v in m["per_category_mean_gap"].items()}))

    # polysemy
    def poly(tag):
        def f(m):
            ce = m["cross_encoder"][list(m["cross_encoder"])[0]]
            print(f"[polysemy {tag}] agreement={ce['winner_agreement_rate']:.3f} "
                  f"spearman={ce['mean_spearman']:.3f} [{ce['spearman_ci_low']:.3f},{ce['spearman_ci_high']:.3f}] "
                  f"control={ce['control_mean_spearman']:.3f} sel={ce['spearman_selectivity']:.3f} p={ce['spearman_perm_p']:.4f}")
        return f
    show("polysemy_v3_primary", poly("primary/leak-free"))
    show("polysemy_v3", poly("secondary/aug"))

    # syntax TR + 5-lang
    def syn(tag):
        def f(m):
            print(f"[syntax {tag}] teacher={m['teacher_mean_acc']:.3f} student={m['student_mean_acc']:.3f} "
                  f"diff={m['mean_acc_diff_teacher_minus_student']:+.3f} p={m['wilcoxon_one_sided_greater_p']:.2e} "
                  f"| sel student={m.get('student_selectivity',float('nan')):+.3f} teacher={m.get('teacher_selectivity',float('nan')):+.3f}")
        return f
    for cfg, tag in [("syntax_v2", "TR"), ("syntax_v2_eu", "EU"), ("syntax_v2_pl", "PL"),
                     ("syntax_v2_it", "IT"), ("syntax_v2_el", "EL")]:
        show(cfg, syn(tag))

    # xlingual
    def xling(tag):
        def f(m):
            pe = m["per_encoder"]
            for k, v in pe.items():
                print(f"[xling {tag}] {k}: P@1={v['mean_p1']:.3f} CSLS={v.get('mean_p1_csls',float('nan')):.3f} hub={v.get('mean_hubness_skew',float('nan')):.2f}")
            for k, v in m.get("retention", {}).items():
                print(f"   retention {k}: {v['mean_p1_retention']:.3f} (CSLS {v.get('mean_p1_csls_retention',float('nan')):.3f})")
        return f
    show("xlingual_v3", xling("v3"))
    show("xlingual_v3_bgem3", xling("bge-m3"))
    show("xlingual_v4_ablation", xling("ablation"))

    # negation
    def neg(tag):
        def f(m):
            pe = m["per_encoder"]
            for k, v in pe.items():
                print(f"[neg {tag}] {k}: mean_flip={v['mean_flip']:+.3f} pos_frac={v['fraction_positive_flip']:.3f} n={v['n_valid']}")
            for k, v in m.get("comparisons", {}).items():
                print(f"   {k}: p={v['wilcoxon_p']:.2e} mean_diff={v['mean_diff']:+.3f}")
        return f
    show("negation_v3_clean_raw_baselines", neg("raw"))
    show("negation_v3_clean_carrier_baselines", neg("carrier"))

    # STS
    show("sts_v1", lambda m: print("[STS]", m.get("per_encoder", m)))

    # downstream negation
    def downneg(m):
        for k, v in m["per_encoder"].items():
            print(f"[downstream-neg] {k}: non-neg={v['acc_no_negation']:.3f} "
                  f"neg={v['acc_negation']:.3f} neg_drop={v['neg_drop']:+.3f}")
        for k, v in m.get("comparisons", {}).items():
            print(f"   excess_drop {k}: {v['excess_drop']:+.3f}")
    show("downstream_negation", downneg)

    # tatoeba
    def tato(m):
        pe = m["per_encoder"]
        for k, v in pe.items():
            print(f"[tatoeba] {k}: P@1={v['mean_p1']:.3f} CSLS={v['mean_p1_csls']:.3f}")
        for k, v in m.get("retention", {}).items():
            print(f"   retention {k}: {v['mean_p1_retention']:.3f} (CSLS {v.get('mean_p1_csls_retention',float('nan')):.3f}) [{m['n_langs']} langs]")
    show("tatoeba", tato)


if __name__ == "__main__":
    main()
