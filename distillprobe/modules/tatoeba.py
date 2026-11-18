from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from ..encoders.base import Encoder, first_reference
from ..metrics import bootstrap_ci
from .base import ProbeModule, ProbeResult
from .xlingual import _l2, _retrieval_metrics

log = logging.getLogger(__name__)


class TatoebaModule(ProbeModule):
    """Harder cross-lingual retrieval benchmark (Tatoeba bitext mining).

    Complements the near-ceiling FLORES-200 retrieval (\\S xlingual) with the
    harder Tatoeba test sets, where the contextual reference encoder is no longer
    at P@1 ~ 1.0, so the student/teacher retention ratio is non-degenerate. Each
    language XX is evaluated as English<->XX retrieval over 1000 parallel
    sentences (mteb/tatoeba-bitext-mining); we report per-language and aggregate
    P@1 (raw + CSLS) and the student/reference retention.
    """

    name = "tatoeba"

    def describe(self) -> str:
        return (
            "Tatoeba bitext retrieval (mteb/tatoeba-bitext-mining). English<->XX P@1 "
            "(raw + CSLS) per language for each encoder, plus student/reference retention. "
            "A harder, non-ceiling complement to the FLORES-200 cross-lingual probe."
        )

    def run(self, encoders: dict[str, Encoder], data_path: str, config: Any) -> ProbeResult:
        from datasets import load_dataset

        spec = json.loads(open(data_path, encoding="utf-8").read())
        langs = spec["languages"]
        seed = getattr(config, "seed", 42)
        bootstrap_n = getattr(config, "bootstrap_n", 10_000)
        extra = getattr(config, "extra", {}) or {}
        csls_k = int(extra.get("csls_k", 10))

        # reference encoder = explicit role or contextual-name heuristic; students = the rest
        ref = first_reference(encoders)
        students = [n for n in encoders if n != ref]

        per_lang: list[dict[str, Any]] = []
        for lang in langs:
            try:
                ds = load_dataset("mteb/tatoeba-bitext-mining", f"{lang}-eng")["test"]
            except Exception as e:  # noqa: BLE001
                log.warning("skip %s: %s", lang, e)
                continue
            xx = [r["sentence1"] for r in ds]
            en = [r["sentence2"] for r in ds]
            row: dict[str, Any] = {"lang": lang, "n": len(xx)}
            for en_name, enc in encoders.items():
                vxx = _l2(np.asarray(enc.encode_many(xx), dtype=np.float32))
                ven = _l2(np.asarray(enc.encode_many(en), dtype=np.float32))
                m = _retrieval_metrics(vxx, ven, csls_k=csls_k)
                row[f"p1:{en_name}"] = m["p1"]
                row[f"p1_csls:{en_name}"] = m["p1_csls"]
                row[f"hubness_skew:{en_name}"] = m["hubness_skew"]
            per_lang.append(row)
            log.info("tatoeba %s done (n=%d)", lang, len(xx))

        # Aggregate per encoder
        per_encoder: dict[str, Any] = {}
        for en_name in encoders:
            p1 = np.array([r[f"p1:{en_name}"] for r in per_lang], dtype=np.float64)
            p1c = np.array([r[f"p1_csls:{en_name}"] for r in per_lang], dtype=np.float64)
            ci = bootstrap_ci(p1, n=bootstrap_n, seed=seed)
            per_encoder[en_name] = {
                "mean_p1": float(p1.mean()) if p1.size else float("nan"),
                "p1_ci_low": ci.low, "p1_ci_high": ci.high,
                "mean_p1_csls": float(p1c.mean()) if p1c.size else float("nan"),
                "n_langs": int(p1.size),
            }

        # Retention (student / reference), per-language then averaged
        retention: dict[str, Any] = {}
        if ref:
            for st in students:
                ratios, ratios_c = [], []
                for r in per_lang:
                    t, s = r[f"p1:{ref}"], r[f"p1:{st}"]
                    tc, sc = r[f"p1_csls:{ref}"], r[f"p1_csls:{st}"]
                    if t > 1e-9:
                        ratios.append(s / t)
                    if tc > 1e-9:
                        ratios_c.append(sc / tc)
                ra = np.asarray(ratios, dtype=np.float64)
                ci = bootstrap_ci(ra, n=bootstrap_n, seed=seed) if ra.size else None
                retention[f"{st}_over_{ref}"] = {
                    "mean_p1_retention": float(ra.mean()) if ra.size else float("nan"),
                    "ci_low": ci.low if ci else None,
                    "ci_high": ci.high if ci else None,
                    "mean_p1_csls_retention": float(np.mean(ratios_c)) if ratios_c else float("nan"),
                }

        summary = {
            "n_langs": len(per_lang),
            "languages": [r["lang"] for r in per_lang],
            "encoders": [{"name": e.name, "dim": e.dim} for e in encoders.values()],
            "reference_encoder": ref,
            "per_encoder": per_encoder,
            "retention": retention,
            "dataset_source": spec.get("source"),
            "dataset_version": spec.get("version"),
            "seed": seed,
            "bootstrap_n": bootstrap_n,
        }
        return ProbeResult(
            module=self.name, metrics=summary, per_category=per_lang,
            baselines={}, tokenization_audit=[],
        )

    def plot(self, result_dict: dict, out_dir) -> list:
        return []
