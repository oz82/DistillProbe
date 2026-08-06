from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..data import load_sts_dataset
from ..encoders.base import Encoder
from .base import ProbeModule, ProbeResult

log = logging.getLogger(__name__)


class STSModule(ProbeModule):
    """Sentence Textual Similarity downstream evaluation.

    For each encoder *e* and each ``(s1, s2, gold_score)`` pair:
      cos_e = cos(e(s1), e(s2))
    Then Spearman correlation between the encoder's cosine vector and the gold
    scores. This is the canonical STS evaluation; reflects how well the encoder
    aligns with human similarity judgements on Turkish STS-Benchmark MT.

    Per-encoder Spearman + bootstrap 95% CI, plus paired comparisons.
    """

    name = "sts"

    def describe(self) -> str:
        return (
            "STS-Tr downstream probe (emrecan/stsb-mt-turkish, test split). "
            "Spearman ρ between encoder cosine similarity and gold human score. "
            "Reports per-encoder ρ, bootstrap CI, and pairwise differences."
        )

    def run(self, encoders: dict[str, Encoder], data_path: str, config: Any) -> ProbeResult:
        from scipy.stats import spearmanr

        ds = load_sts_dataset(data_path)
        log.info("loaded STS dataset: %d pairs (lang=%s)", len(ds.pairs), ds.language)
        seed = getattr(config, "seed", 42)
        bootstrap_n = getattr(config, "bootstrap_n", 10_000)

        s1 = [p.sentence1 for p in ds.pairs]
        s2 = [p.sentence2 for p in ds.pairs]
        gold = np.array([p.score for p in ds.pairs], dtype=np.float64)
        n = len(gold)

        per_encoder: dict[str, Any] = {}
        per_encoder_cos: dict[str, np.ndarray] = {}

        for en_name, enc in encoders.items():
            log.info("encoding STS pairs with %s", en_name)
            v1 = np.asarray(enc.encode_many(s1), dtype=np.float32)
            v2 = np.asarray(enc.encode_many(s2), dtype=np.float32)
            n1 = np.maximum(np.linalg.norm(v1, axis=1, keepdims=True), 1e-12)
            n2 = np.maximum(np.linalg.norm(v2, axis=1, keepdims=True), 1e-12)
            cos = ((v1 / n1) * (v2 / n2)).sum(axis=1)
            cos = cos.astype(np.float64)
            per_encoder_cos[en_name] = cos

            rho, p_val = spearmanr(cos, gold)
            rho = float(rho)
            p_val = float(p_val)

            rng = np.random.default_rng(seed)
            boots = []
            for _ in range(bootstrap_n):
                idx = rng.integers(0, n, size=n)
                rb, _ = spearmanr(cos[idx], gold[idx])
                if not np.isnan(rb):
                    boots.append(float(rb))
            barr = np.asarray(boots)
            ci_low = float(np.quantile(barr, 0.025)) if barr.size else float("nan")
            ci_high = float(np.quantile(barr, 0.975)) if barr.size else float("nan")

            per_encoder[en_name] = {
                "spearman": rho,
                "p_value": p_val,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_pairs": int(n),
            }

        # Pairwise comparisons: paired bootstrap diff in Spearman
        encoder_names = list(encoders.keys())
        comparisons: dict[str, Any] = {}
        for i, a in enumerate(encoder_names):
            for b in encoder_names[i + 1 :]:
                rng = np.random.default_rng(seed)
                diffs = []
                ca = per_encoder_cos[a]
                cb = per_encoder_cos[b]
                for _ in range(bootstrap_n):
                    idx = rng.integers(0, n, size=n)
                    ra, _ = spearmanr(ca[idx], gold[idx])
                    rb, _ = spearmanr(cb[idx], gold[idx])
                    if not (np.isnan(ra) or np.isnan(rb)):
                        diffs.append(float(ra - rb))
                darr = np.asarray(diffs)
                if darr.size:
                    # Paired bootstrap: the SAME resample indices are applied to
                    # both encoders each iteration. "mean_diff" is the bootstrap
                    # mean of rho_a - rho_b (numerically the point estimate); the
                    # CI is the percentile interval of the paired difference.
                    comparisons[f"{a}_minus_{b}"] = {
                        "mean_diff": float(darr.mean()),
                        "ci_low": float(np.quantile(darr, 0.025)),
                        "ci_high": float(np.quantile(darr, 0.975)),
                    }

        summary = {
            "n_pairs": int(n),
            "encoders": [{"name": e.name, "dim": e.dim} for e in encoders.values()],
            "per_encoder": per_encoder,
            "comparisons": comparisons,
            "dataset_version": ds.version,
            "dataset_source": ds.source,
            "language": ds.language,
            "seed": seed,
            "bootstrap_n": bootstrap_n,
        }
        return ProbeResult(
            module=self.name,
            metrics=summary,
            per_category=[],
            baselines={},
            tokenization_audit=[],
        )

    def plot(self, result_dict: dict, out_dir) -> list:
        from ..reporting.plots import make_sts_plots

        return make_sts_plots(result_dict, out_dir)
