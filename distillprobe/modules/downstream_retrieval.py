from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..data import load_retrieval_dataset
from ..encoders.base import Encoder, first_reference, is_reference_encoder
from ..metrics import bootstrap_ci, mrr_at_k, ndcg_at_k, recall_at_k
from .base import ProbeModule, ProbeResult

log = logging.getLogger(__name__)


class DownstreamRetrievalModule(ProbeModule):
    """Downstream ad-hoc retrieval test (BEIR protocol, exact cosine search).

    Every judged query is embedded and ranked against the whole corpus by cosine
    similarity; nDCG@10 (BEIR/MTEB primary metric), Recall@10 and MRR@10 are
    computed per query and averaged, with bootstrap CIs over queries. No
    encoder-specific query/passage prefixes are used, so all encoders see the
    same raw text.

    Cross-encoder quantity: ``retention = nDCG@10(student) / nDCG@10(reference)``,
    the task-level analogue of the intrinsic cross-lingual retention ratio, so the
    deployment claim "distilled static embeddings are usable for retrieval" is
    tested directly.
    """

    name = "downstream_retrieval"

    def describe(self) -> str:
        return (
            "Downstream retrieval probe (BEIR-style corpus/queries/qrels). Per encoder: exact "
            "cosine ranking, nDCG@10 / Recall@10 / MRR@10 averaged over judged queries with "
            "bootstrap CIs; student/reference nDCG retention and paired per-query differences."
        )

    def run(self, encoders: dict[str, Encoder], data_path: str, config: Any) -> ProbeResult:
        ds = load_retrieval_dataset(data_path)
        seed = getattr(config, "seed", 42)
        bootstrap_n = getattr(config, "bootstrap_n", 10_000)
        extra = getattr(config, "extra", {}) or {}
        k = int(extra.get("k", 10))

        doc_ids = [d.id for d in ds.corpus]
        doc_pos = {d: i for i, d in enumerate(doc_ids)}
        doc_texts = [(d.title + " " + d.text).strip() if d.title else d.text for d in ds.corpus]
        queries = [q for q in ds.queries if q.id in ds.qrels and ds.qrels[q.id]]
        log.info(
            "loaded retrieval dataset: language=%s version=%s docs=%d judged queries=%d",
            ds.language, ds.version, len(doc_ids), len(queries),
        )

        gains = np.zeros((len(queries), len(doc_ids)), dtype=np.float64)
        for qi, q in enumerate(queries):
            for did, grade in ds.qrels[q.id].items():
                if did in doc_pos and grade > 0:
                    gains[qi, doc_pos[did]] = float(grade)

        per_encoder: dict[str, Any] = {}
        ndcg_by_encoder: dict[str, np.ndarray] = {}
        for en_name, enc in encoders.items():
            log.info("encoding corpus + queries with %s", en_name)
            D = np.asarray(enc.encode_many(doc_texts), dtype=np.float32)
            Q = np.asarray(enc.encode_many([q.text for q in queries]), dtype=np.float32)
            D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
            Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
            scores = Q @ D.T
            nd = ndcg_at_k(scores, gains, k)
            rc = recall_at_k(scores, gains, k)
            mr = mrr_at_k(scores, gains, k)
            ndcg_by_encoder[en_name] = nd
            ci = bootstrap_ci(nd, n=bootstrap_n, seed=seed)
            per_encoder[en_name] = {
                f"ndcg@{k}": float(nd.mean()),
                f"ndcg@{k}_ci_low": ci.low,
                f"ndcg@{k}_ci_high": ci.high,
                f"recall@{k}": float(rc.mean()),
                f"mrr@{k}": float(mr.mean()),
                "n_queries": int(nd.size),
            }

        retention: dict[str, Any] = {}
        refs = [n for n, e in encoders.items() if is_reference_encoder(e)]
        students = [n for n in encoders if n not in refs]
        for st in students:
            for ref in refs:
                s_arr, r_arr = ndcg_by_encoder[st], ndcg_by_encoder[ref]
                diff = s_arr - r_arr
                ci = bootstrap_ci(diff, n=bootstrap_n, seed=seed)
                r_mean = float(r_arr.mean())
                ratio = float(s_arr.mean() / r_mean) if r_mean > 0 else float("nan")
                retention[f"{st}_over_{ref}"] = {
                    f"ndcg@{k}_retention": ratio,
                    "mean_diff_student_minus_ref": float(diff.mean()),
                    "diff_ci_low": ci.low,
                    "diff_ci_high": ci.high,
                }

        summary = {
            "n_docs": len(doc_ids),
            "n_queries_judged": len(queries),
            "k": k,
            "encoders": [{"name": e.name, "dim": e.dim, "role": e.role} for e in encoders.values()],
            "per_encoder": per_encoder,
            "retention": retention,
            "primary_reference": first_reference(encoders),
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
