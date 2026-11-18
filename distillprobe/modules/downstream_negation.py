from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from ..encoders.base import Encoder, first_reference
from ..metrics import wilson_ci
from .base import ProbeModule, ProbeResult

log = logging.getLogger(__name__)


class DownstreamNegationModule(ProbeModule):
    """Per-axis downstream test of the negation finding (Turkish sentiment).

    Reviews are split into a NEGATION subset and a NO-NEGATION subset, with
    Positive/Negative balanced 50/50 within each subset (so the label prior is
    identical across subsets; see scripts/build_negation_sentiment.py). For each
    encoder we train a logistic-regression sentiment classifier on the encoder's
    embeddings via 5-fold stratified out-of-fold prediction, then report accuracy
    overall, on the negation subset, and on the no-negation subset.

    Key quantity: ``neg_drop = acc(no-negation) - acc(negation)``. The intrinsic
    BoW negation failure predicts a LARGER drop for the bag-of-words student than
    for the contextual reference encoder: a BoW classifier cannot recover the
    polarity flip that ``X değil``-style negation introduces.
    """

    name = "downstream_negation"

    def describe(self) -> str:
        return (
            "Downstream negation test (TRSAv1 Turkish sentiment, negation-controlled). "
            "Per encoder: 5-fold logistic-regression sentiment accuracy overall / on the "
            "negation subset / on the no-negation subset; neg_drop = acc(no-neg) - acc(neg). "
            "BoW students are predicted to drop more on negated reviews."
        )

    def run(self, encoders: dict[str, Encoder], data_path: str, config: Any) -> ProbeResult:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        data = json.loads(open(data_path, encoding="utf-8").read())
        items = data["items"]
        texts = [it["text"] for it in items]
        y = np.array([int(it["label"]) for it in items], dtype=np.int64)
        neg_mask = np.array([bool(it["has_negation"]) for it in items], dtype=bool)
        seed = getattr(config, "seed", 42)
        log.info("loaded %d sentiment items (%d negation, %d no-negation)",
                 len(items), int(neg_mask.sum()), int((~neg_mask).sum()))

        def acc(mask: np.ndarray, pred: np.ndarray) -> float:
            if mask.sum() == 0:
                return float("nan")
            return float((pred[mask] == y[mask]).mean())

        per_encoder: dict[str, Any] = {}
        for en_name, enc in encoders.items():
            log.info("encoding + classifying with %s", en_name)
            X = np.asarray(enc.encode_many(texts), dtype=np.float32)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            clf = LogisticRegression(max_iter=2000, C=1.0)
            pred = cross_val_predict(clf, X, y, cv=cv)

            acc_all = acc(np.ones_like(neg_mask), pred)
            acc_neg = acc(neg_mask, pred)
            acc_non = acc(~neg_mask, pred)
            # Wilson CIs on the subset accuracies (binomial)
            ci_neg = wilson_ci(int((pred[neg_mask] == y[neg_mask]).sum()), int(neg_mask.sum()))
            ci_non = wilson_ci(int((pred[~neg_mask] == y[~neg_mask]).sum()), int((~neg_mask).sum()))
            per_encoder[en_name] = {
                "acc_overall": acc_all,
                "acc_negation": acc_neg,
                "acc_no_negation": acc_non,
                "neg_drop": acc_non - acc_neg,
                "neg_ci": [ci_neg.low, ci_neg.high],
                "no_neg_ci": [ci_non.low, ci_non.high],
                "n_negation": int(neg_mask.sum()),
                "n_no_negation": int((~neg_mask).sum()),
            }

        # Student-vs-reference: difference in neg_drop (positive => student degrades
        # more on negation, as predicted). Reference = explicit role or contextual-name heuristic.
        comparisons: dict[str, Any] = {}
        names = list(encoders.keys())
        ref = first_reference(encoders)
        students = [n for n in names if n != ref]
        if ref:
            for st in students:
                comparisons[f"{st}_vs_{ref}"] = {
                    "neg_drop_student": per_encoder[st]["neg_drop"],
                    "neg_drop_reference": per_encoder[ref]["neg_drop"],
                    "excess_drop": per_encoder[st]["neg_drop"] - per_encoder[ref]["neg_drop"],
                }

        summary = {
            "n_items": len(items),
            "encoders": [{"name": e.name, "dim": e.dim} for e in encoders.values()],
            "per_encoder": per_encoder,
            "comparisons": comparisons,
            "reference_encoder": ref,
            "dataset_version": data.get("version"),
            "dataset_source": data.get("source"),
            "language": data.get("language", "tr"),
            "seed": seed,
        }
        return ProbeResult(
            module=self.name,
            metrics=summary,
            per_category=[],
            baselines={},
            tokenization_audit=[],
        )

    def plot(self, result_dict: dict, out_dir) -> list:
        return []
