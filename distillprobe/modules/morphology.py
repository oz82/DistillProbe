from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..data import load_morphology_dataset
from ..encoders.base import Encoder
from ..reporting.plots import make_morphology_plots
from ..metrics import (
    bootstrap_ci,
    cliffs_delta,
    difference_vectors,
    paired_wilcoxon,
    pairwise_consistency,
    random_pair_baseline,
    stem_form_similarity,
)
from .base import ProbeModule, ProbeResult

log = logging.getLogger(__name__)


class MorphologyModule(ProbeModule):
    name = "morphology"

    def describe(self) -> str:
        return (
            "Morphological survival probe (Turkish suffixes).\n"
            "For each (stem, form) category and each encoder: pairwise difference-vector "
            "consistency, stem-form similarity (with bootstrap CI), and a random-pair baseline. "
            "Across encoders: paired Wilcoxon signed-rank + Cliff's delta on per-category "
            "consistency. Tokenization audit included."
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _encode_pairs(enc: Encoder, stems: list[str], forms: list[str]):
        s_vecs = enc.encode_many(stems)
        f_vecs = enc.encode_many(forms)
        return s_vecs, f_vecs

    # ------------------------------------------------------------------ run
    def run(self, encoders: dict[str, Encoder], data_path: str, config: Any) -> ProbeResult:
        ds = load_morphology_dataset(data_path)
        log.info(
            "loaded morphology dataset: language=%s version=%s categories=%d",
            ds.language,
            ds.version,
            len(ds.categories),
        )

        seed = getattr(config, "seed", 42)
        bootstrap_n = getattr(config, "bootstrap_n", 10_000)

        per_category: list[dict] = []
        token_audit: list[dict] = []
        # encoder_name -> aligned list of per-category consistencies
        consistencies: dict[str, list[float]] = {n: [] for n in encoders}

        for cat in ds.categories:
            stems = [p.stem for p in cat.pairs]
            forms = [p.form for p in cat.pairs]
            row: dict[str, Any] = {
                "id": cat.id,
                "label": cat.label,
                "n": len(cat.pairs),
            }

            for enc_name, enc in encoders.items():
                s_vecs, f_vecs = self._encode_pairs(enc, stems, forms)
                diffs = difference_vectors(s_vecs, f_vecs)
                cons = pairwise_consistency(diffs)
                sf = stem_form_similarity(s_vecs, f_vecs)
                ci = bootstrap_ci(sf, n=bootstrap_n, seed=seed)

                row[f"consistency:{enc_name}"] = float(cons)
                row[f"stem_form_sim_mean:{enc_name}"] = float(sf.mean())
                row[f"stem_form_sim_std:{enc_name}"] = float(sf.std())
                row[f"stem_form_sim_ci_low:{enc_name}"] = ci.low
                row[f"stem_form_sim_ci_high:{enc_name}"] = ci.high

                consistencies[enc_name].append(float(cons))

            per_category.append(row)

            for p in cat.pairs:
                audit = {"category": cat.id, "stem": p.stem, "form": p.form}
                for enc_name, enc in encoders.items():
                    audit[f"tokens:stem:{enc_name}"] = "|".join(enc.tokenize(p.stem))
                    audit[f"tokens:form:{enc_name}"] = "|".join(enc.tokenize(p.form))
                token_audit.append(audit)

        # Random-pair baselines (one per encoder)
        baselines: dict[str, Any] = {}
        if ds.distractor_pool:
            baseline_n_pairs = max(15, min(30, len(ds.distractor_pool) // 2))
            for enc_name, enc in encoders.items():
                bl = random_pair_baseline(
                    word_pool=list(ds.distractor_pool),
                    encode_fn=enc.encode,
                    n_pairs=baseline_n_pairs,
                    n_seeds=5,
                    base_seed=seed,
                )
                vals = list(bl.values())
                baselines[enc_name] = {
                    "per_seed": {str(k): v for k, v in bl.items()},
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "n_pairs": baseline_n_pairs,
                    "n_seeds": 5,
                }

        # Paired comparisons across encoder pairs
        names = list(encoders)
        comparisons: dict[str, Any] = {}
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                x = np.asarray(consistencies[a])
                y = np.asarray(consistencies[b])
                if x.size < 2:
                    continue
                try:
                    wres = paired_wilcoxon(x, y)
                    comparisons[f"{a}_vs_{b}"] = {
                        "wilcoxon_stat": float(wres.statistic),
                        "wilcoxon_p": float(wres.pvalue),
                        "cliffs_delta": cliffs_delta(x, y),
                        "mean_a": float(x.mean()),
                        "mean_b": float(y.mean()),
                        "n_categories": int(x.size),
                    }
                except Exception as e:  # noqa: BLE001
                    log.warning("paired test failed for %s vs %s: %s", a, b, e)

        summary = {
            "n_categories": len(ds.categories),
            "n_pairs_total": sum(len(c.pairs) for c in ds.categories),
            "encoders": [{"name": e.name, "dim": e.dim} for e in encoders.values()],
            "per_encoder_mean_consistency": {
                k: float(np.mean(v)) if v else float("nan") for k, v in consistencies.items()
            },
            "comparisons": comparisons,
            "dataset_version": ds.version,
            "language": ds.language,
            "seed": seed,
            "bootstrap_n": bootstrap_n,
        }

        return ProbeResult(
            module=self.name,
            metrics=summary,
            per_category=per_category,
            baselines=baselines,
            tokenization_audit=token_audit,
        )

    def plot(self, result_dict: dict, out_dir) -> list:
        return make_morphology_plots(result_dict, out_dir)
