from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..data import load_negation_dataset
from ..encoders.base import Encoder
from ..metrics import bootstrap_ci, paired_wilcoxon
from .base import ProbeModule, ProbeResult

log = logging.getLogger(__name__)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da < 1e-12 or db < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (da * db))


class NegationModule(ProbeModule):
    """Negation polarity-flip survival probe.

    For each ``(positive, negation, antonym)`` triplet (where negation is the
    Turkish "X değil" form):

      - Three cosines per encoder: ``cos(pos, neg)``, ``cos(neg, ant)``,
        ``cos(pos, ant)``.
      - **polarity_flip** = ``cos(neg, ant) − cos(neg, pos)``

    Interpretation:
      - If the encoder handles negation, ``cos(neg, ant) > cos(neg, pos)`` so
        the flip is positive: negation lands closer to the antonym than to the
        positive root.
      - Bag-of-words M2V encodes "iyi değil" as mean(iyi, değil); the result
        sits near "iyi" with a small "değil" shift, not near "kötü". Expected
        flip ≤ 0.

    Hypothesis: ``polarity_flip(LaBSE) > polarity_flip(M2V)`` per triplet
    (paired Wilcoxon, one-sided).
    """

    name = "negation"

    def describe(self) -> str:
        return (
            "Negation polarity-flip survival probe. Per (positive, 'X değil', antonym) "
            "triplet, computes polarity_flip = cos(neg, antonym) − cos(neg, positive) "
            "for each encoder. Paired Wilcoxon LaBSE > M2V."
        )

    def run(self, encoders: dict[str, Encoder], data_path: str, config: Any) -> ProbeResult:
        ds = load_negation_dataset(data_path)
        log.info(
            "loaded negation dataset: language=%s version=%s triplets=%d",
            ds.language,
            ds.version,
            len(ds.triplets),
        )

        seed = getattr(config, "seed", 42)
        bootstrap_n = getattr(config, "bootstrap_n", 10_000)
        extra = getattr(config, "extra", {}) or {}
        use_carrier = bool(extra.get("use_carrier", False))
        carrier_pos = extra.get("carrier_pos", "Bu çok {x}.")
        carrier_neg = extra.get("carrier_neg", "Bu çok {x} değil.")
        carrier_ant = extra.get("carrier_ant", "Bu çok {x}.")
        if use_carrier:
            log.info("V2 carrier mode: pos=%r, neg=%r, ant=%r", carrier_pos, carrier_neg, carrier_ant)

        per_triplet: list[dict[str, Any]] = []
        for tri in ds.triplets:
            if use_carrier:
                pos_text = carrier_pos.format(x=tri.positive)
                neg_text = carrier_neg.format(x=tri.positive)
                ant_text = carrier_ant.format(x=tri.antonym)
            else:
                pos_text = tri.positive
                neg_text = tri.negation
                ant_text = tri.antonym
            row: dict[str, Any] = {
                "positive": tri.positive,
                "negation": tri.negation,
                "antonym": tri.antonym,
                "pos_text": pos_text,
                "neg_text": neg_text,
                "ant_text": ant_text,
            }
            for en_name, enc in encoders.items():
                v_pos = enc.encode(pos_text)
                v_neg = enc.encode(neg_text)
                v_ant = enc.encode(ant_text)
                c_pn = _cos(v_pos, v_neg)
                c_na = _cos(v_neg, v_ant)
                c_pa = _cos(v_pos, v_ant)
                flip = c_na - c_pn
                row[f"cos_pos_neg:{en_name}"] = c_pn
                row[f"cos_neg_ant:{en_name}"] = c_na
                row[f"cos_pos_ant:{en_name}"] = c_pa
                row[f"polarity_flip:{en_name}"] = flip
            per_triplet.append(row)

        # Per-encoder summary
        per_encoder: dict[str, Any] = {}
        for en_name in encoders:
            flips = np.asarray(
                [r[f"polarity_flip:{en_name}"] for r in per_triplet], dtype=np.float64
            )
            valid = flips[~np.isnan(flips)]
            ci = bootstrap_ci(valid, n=bootstrap_n, seed=seed) if valid.size else None
            n_positive = int((valid > 0).sum())
            per_encoder[en_name] = {
                "mean_flip": float(valid.mean()) if valid.size else float("nan"),
                "ci_low": ci.low if ci else None,
                "ci_high": ci.high if ci else None,
                "fraction_positive_flip": float(n_positive / valid.size) if valid.size else float("nan"),
                "n_valid": int(valid.size),
            }

        # Paired test (≥2 encoders): LaBSE > M2V flip
        comparisons: dict[str, Any] = {}
        encoder_names = list(encoders.keys())
        if len(encoder_names) >= 2:
            # Pick teacher = the one whose internal name flags transformer/labse
            teacher = next(
                (n for n in encoder_names
                 if "transformer" in encoders[n].name.lower() or "labse" in encoders[n].name.lower()),
                None,
            )
            student = next((n for n in encoder_names if n != teacher), None)
            if teacher and student:
                xs = np.array(
                    [r[f"polarity_flip:{teacher}"] for r in per_triplet], dtype=np.float64
                )
                ys = np.array(
                    [r[f"polarity_flip:{student}"] for r in per_triplet], dtype=np.float64
                )
                mask = ~(np.isnan(xs) | np.isnan(ys))
                if mask.sum() >= 2:
                    diff = xs[mask] - ys[mask]
                    ci_diff = bootstrap_ci(diff, n=bootstrap_n, seed=seed)
                    try:
                        from scipy.stats import wilcoxon

                        wres = wilcoxon(xs[mask], ys[mask], alternative="greater")
                        comparisons[f"{teacher}_gt_{student}"] = {
                            "test": "wilcoxon (paired, one-sided greater)",
                            "wilcoxon_W": float(wres.statistic),
                            "wilcoxon_p": float(wres.pvalue),
                            "mean_diff": float(diff.mean()),
                            "diff_ci_low": ci_diff.low,
                            "diff_ci_high": ci_diff.high,
                            "n": int(mask.sum()),
                        }
                    except Exception as e:  # noqa: BLE001
                        log.warning("wilcoxon failed: %s", e)

        summary = {
            "n_triplets": len(per_triplet),
            "encoders": [{"name": e.name, "dim": e.dim} for e in encoders.values()],
            "per_encoder": per_encoder,
            "comparisons": comparisons,
            "use_carrier": use_carrier,
            "carrier_pos": carrier_pos if use_carrier else None,
            "carrier_neg": carrier_neg if use_carrier else None,
            "carrier_ant": carrier_ant if use_carrier else None,
            "dataset_version": ds.version,
            "language": ds.language,
            "seed": seed,
            "bootstrap_n": bootstrap_n,
        }

        return ProbeResult(
            module=self.name,
            metrics=summary,
            per_category=per_triplet,
            baselines={},
            tokenization_audit=[],
        )

    def plot(self, result_dict: dict, out_dir) -> list:
        from ..reporting.plots import make_negation_plots

        return make_negation_plots(result_dict, out_dir)
