"""Place the student-measured axes on one teacher-relative retention scale.

The raw axis metrics are not comparable with each other (a P@1 ratio, a Spearman
rho, a consistency, an accuracy gap, a flip rate). This script computes, per language and per student, a *common* quantity

    retention = (student - chance) / (reference - chance)

so that 1 means "the student matches the contextual reference above chance" and
0 means "no above-chance signal". Per axis:

  cross-lingual : P@1_student / P@1_reference           (chance ~ 1/n, ignored)
  syntactic role: (acc_s - 0.5) / (acc_ref - 0.5)         (LOO role classifier)
  morphology    : consistency_s / consistency_ref         (random-pair baseline ~ 0)
  polysemy      : Spearman rho(student, reference)        (reference vs itself = 1, chance = 0)
  negation      : flip-rate_s / flip-rate_ref             (positive polarity flips)

The teacher-internal idiomaticity probe is not a student measurement and is
excluded. Run identifiers are read from the canonical and revision run lists.

Usage:
    python scripts/spectrum_scale.py [--out results/spectrum_scale.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _runs() -> dict[str, str]:
    out: dict[str, str] = {}
    for f in ("results/_canonical_runs.txt", "results/_revision_runs.txt"):
        p = Path(f)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1].startswith("results/"):
                out[parts[0]] = parts[1].strip()  # later (re)runs win
    return out


def _metrics(run_dir: str) -> dict:
    payload = json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))
    return payload.get("metrics", payload)


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den <= 0:
        return None
    return float(num / den)


# (language, student key) -> configs carrying each axis. `ref` = reference encoder key.
PLAN = {
    ("tr", "m2v_potion"): dict(
        xl="xlingual_v3", syn="syntax_v2", morph="morph_tr_all_encoders",
        poly="polysemy_v4_primary", neg="negation_v3_clean_raw_baselines", ref="labse",
    ),
    ("tr", "m2v_multilingual_output"): dict(
        xl="ablation_xlingual_v5", syn="ablation_syntax_m2v_output", morph="morph_tr_all_encoders",
        poly="ablation_polysemy_m2v_output", neg="ablation_negation_raw_m2v_output", ref="labse",
    ),
    ("tr", "m2v_bgem3_pca256_sif"): dict(
        xl="ablation_xlingual_v5", syn="ablation_syntax_m2v_bgem3pca", morph="morph_tr_all_encoders",
        poly="ablation_polysemy_m2v_bgem3pca", neg="ablation_negation_raw_m2v_bgem3pca", ref="labse",
    ),
    ("tr", "st_static_mrl"): dict(
        xl="family_xlingual_ststatic", syn="family_tr_syntax_ststatic", morph="morph_tr_all_encoders",
        poly="family_tr_polysemy_ststatic", neg="family_tr_negation_raw_ststatic", ref="labse",
    ),
    ("en", "m2v_potion"): dict(
        xl=None, syn="syntax_v2_en", morph="morph_en_all_encoders",
        poly="polysemy_en_primary", neg="negation_en_clean_raw_baselines", ref="labse",
    ),
    ("en", "st_static_mrl"): dict(
        xl=None, syn="family_en_syntax_ststatic", morph="morph_en_all_encoders",
        poly="family_en_polysemy_ststatic", neg="family_en_negation_raw_ststatic", ref="labse",
    ),
    ("en", "wada_gte256_en"): dict(
        xl=None, syn="family_en_syntax_wada", morph="morph_en_all_encoders",
        poly="family_en_polysemy_wada", neg="family_en_negation_raw_wada", ref="labse",
    ),
}


def compute(runs: dict[str, str]) -> list[dict]:
    rows = []
    for (lang, student), plan in PLAN.items():
        row: dict = {"language": lang, "student": student, "reference": plan["ref"], "raw": {}}
        ref = plan["ref"]

        if plan["xl"] and plan["xl"] in runs:
            m = _metrics(runs[plan["xl"]])
            r = m.get("retention", {}).get(f"{student}_over_{ref}", {})
            row["cross_lingual"] = r.get("mean_p1_retention")
            row["raw"]["cross_lingual"] = {"student_over_ref": r.get("mean_p1_retention")}
        else:
            row["cross_lingual"] = None

        if plan["syn"] in runs:
            m = _metrics(runs[plan["syn"]])
            s, t = m.get("student_mean_acc"), m.get("teacher_mean_acc")
            row["syntactic_role"] = _ratio(s - 0.5 if s is not None else None, t - 0.5 if t else None)
            row["raw"]["syntactic_role"] = {"student_acc": s, "teacher_acc": t, "gap": m.get("mean_acc_diff_teacher_minus_student")}
        if plan["morph"] in runs:
            m = _metrics(runs[plan["morph"]]).get("per_encoder_mean_consistency", {})
            row["morphology"] = _ratio(m.get(student), m.get(ref))
            row["raw"]["morphology"] = {"student": m.get(student), "reference": m.get(ref)}
        if plan["poly"] in runs:
            m = _metrics(runs[plan["poly"]]).get("cross_encoder", {})
            d = m.get(f"{student}_vs_{ref}", {})
            row["polysemy"] = d.get("mean_spearman")
            row["raw"]["polysemy"] = {"spearman": d.get("mean_spearman"), "control": d.get("control_mean_spearman")}
        if plan["neg"] in runs:
            m = _metrics(runs[plan["neg"]]).get("per_encoder", {})
            s, t = m.get(student, {}).get("fraction_positive_flip"), m.get(ref, {}).get("fraction_positive_flip")
            row["negation"] = _ratio(s, t)
            row["raw"]["negation"] = {"student_flip_rate": s, "reference_flip_rate": t}
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/spectrum_scale.json")
    args = ap.parse_args()
    rows = compute(_runs())
    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    axes = ["cross_lingual", "syntactic_role", "morphology", "polysemy", "negation"]
    print(f"{'lang':4s} {'student':26s} " + " ".join(f"{a[:12]:>12s}" for a in axes))
    for r in rows:
        cells = [f"{r.get(a):12.3f}" if isinstance(r.get(a), float) else f"{'—':>12s}" for a in axes]
        print(f"{r['language']:4s} {r['student']:26s} " + " ".join(cells))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
