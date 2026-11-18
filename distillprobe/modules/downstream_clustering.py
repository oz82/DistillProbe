from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..data import load_clustering_dataset
from ..encoders.base import Encoder, first_reference, is_reference_encoder
from ..metrics import bootstrap_ci
from .base import ProbeModule, ProbeResult

log = logging.getLogger(__name__)


class DownstreamClusteringModule(ProbeModule):
    """Downstream clustering test (MTEB protocol).

    Each dataset holds several clustering *sets*: texts with gold category labels.
    For every encoder and set, the texts are embedded, k-means is run with k equal
    to the number of gold categories, and the clustering is scored against the gold
    labels with V-measure (the MTEB primary clustering metric). Per-encoder score
    = mean V-measure over sets, with a bootstrap CI over sets.

    Cross-encoder quantity: ``retention = V(student) / V(reference)``, the
    task-level analogue of the intrinsic retention ratios, so the deployment claim
    "distilled static embeddings are usable for clustering" is tested directly.
    """

    name = "downstream_clustering"

    def describe(self) -> str:
        return (
            "Downstream clustering probe (MTEB-style sets). Per encoder and set: k-means with "
            "k = #gold labels, scored by V-measure (and ARI). Mean over sets with bootstrap CI; "
            "student/reference retention ratios and paired set-level differences."
        )

    def run(self, encoders: dict[str, Encoder], data_path: str, config: Any) -> ProbeResult:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, v_measure_score

        ds = load_clustering_dataset(data_path)
        seed = getattr(config, "seed", 42)
        bootstrap_n = getattr(config, "bootstrap_n", 10_000)
        extra = getattr(config, "extra", {}) or {}
        max_per_set = int(extra.get("max_sentences_per_set", 0))
        kmeans_init = int(extra.get("kmeans_n_init", 10))
        log.info(
            "loaded clustering dataset: language=%s version=%s sets=%d",
            ds.language, ds.version, len(ds.sets),
        )

        rng = np.random.default_rng(seed)
        sets = []
        for s in ds.sets:
            sents, labels = list(s.sentences), list(s.labels)
            if max_per_set and len(sents) > max_per_set:
                idx = np.sort(rng.choice(len(sents), size=max_per_set, replace=False))
                sents = [sents[i] for i in idx]
                labels = [labels[i] for i in idx]
            sets.append((sents, labels))

        per_set: list[dict[str, Any]] = []
        for i, (sents, labels) in enumerate(sets):
            per_set.append({"set_index": i, "n": len(sents), "k": len(set(labels))})

        per_encoder: dict[str, Any] = {}
        v_by_encoder: dict[str, np.ndarray] = {}
        for en_name, enc in encoders.items():
            log.info("clustering with %s", en_name)
            vs, aris = [], []
            for i, (sents, labels) in enumerate(sets):
                X = np.asarray(enc.encode_many(sents), dtype=np.float32)
                k = len(set(labels))
                km = KMeans(n_clusters=k, n_init=kmeans_init, random_state=seed)
                pred = km.fit_predict(X)
                v = float(v_measure_score(labels, pred))
                ari = float(adjusted_rand_score(labels, pred))
                vs.append(v)
                aris.append(ari)
                per_set[i][f"v_measure:{en_name}"] = v
                per_set[i][f"ari:{en_name}"] = ari
            arr = np.asarray(vs, dtype=np.float64)
            v_by_encoder[en_name] = arr
            ci = bootstrap_ci(arr, n=bootstrap_n, seed=seed)
            per_encoder[en_name] = {
                "v_measure_mean": float(arr.mean()),
                "v_measure_ci_low": ci.low,
                "v_measure_ci_high": ci.high,
                "ari_mean": float(np.mean(aris)),
                "n_sets": int(arr.size),
            }

        retention: dict[str, Any] = {}
        refs = [n for n, e in encoders.items() if is_reference_encoder(e)]
        students = [n for n in encoders if n not in refs]
        for st in students:
            for ref in refs:
                s_arr, r_arr = v_by_encoder[st], v_by_encoder[ref]
                diff = s_arr - r_arr
                ci = bootstrap_ci(diff, n=bootstrap_n, seed=seed)
                r_mean = float(r_arr.mean())
                ratio = float(s_arr.mean() / r_mean) if r_mean > 0 else float("nan")
                retention[f"{st}_over_{ref}"] = {
                    "v_measure_retention": ratio,
                    "mean_diff_student_minus_ref": float(diff.mean()),
                    "diff_ci_low": ci.low,
                    "diff_ci_high": ci.high,
                }

        summary = {
            "n_sets": len(sets),
            "encoders": [{"name": e.name, "dim": e.dim, "role": e.role} for e in encoders.values()],
            "per_encoder": per_encoder,
            "retention": retention,
            "primary_reference": first_reference(encoders),
            "dataset_version": ds.version,
            "dataset_source": ds.source,
            "language": ds.language,
            "max_sentences_per_set": max_per_set,
            "kmeans_n_init": kmeans_init,
            "seed": seed,
            "bootstrap_n": bootstrap_n,
        }
        return ProbeResult(
            module=self.name,
            metrics=summary,
            per_category=per_set,
            baselines={},
            tokenization_audit=[],
        )
