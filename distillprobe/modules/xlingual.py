from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..data import load_xlingual_dataset
from ..encoders.base import Encoder
from ..metrics import bootstrap_ci
from .base import ProbeModule, ProbeResult

log = logging.getLogger(__name__)


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def _csls_matrix(sim: np.ndarray, k: int = 10) -> np.ndarray:
    """Cross-domain Similarity Local Scaling (Lample et al., 2018).

    CSLS[i, j] = 2*sim[i, j] − r_S(i) − r_T(j), where r_S(i) is the mean of the
    top-k similarities of source i over all targets and r_T(j) the mean of the
    top-k similarities of target j over all sources. Penalizes hub targets and
    improves nearest-neighbour retrieval. One CSLS matrix serves both directions
    (argmax over axis 1 = source→target, over axis 0 = target→source).
    """
    n = sim.shape[0]
    k_eff = max(1, min(k, n - 1))
    rs = np.sort(sim, axis=1)[:, ::-1][:, :k_eff].mean(axis=1)  # per source row
    rt = np.sort(sim, axis=0)[::-1, :][:k_eff, :].mean(axis=0)  # per target col
    return 2.0 * sim - rs[:, None] - rt[None, :]


def _retrieval_metrics(v1: np.ndarray, v2: np.ndarray, csls_k: int = 10) -> dict[str, float]:
    """Compute symmetric retrieval metrics on parallel matched arrays.

    Both `v1` and `v2` are (N, D), assumed L2-normalized so that `v1 @ v2.T` is
    the cosine similarity matrix. Returns the average of l1→l2 and l2→l1
    metrics so each pair counts once. Also reports CSLS-based P@1 (hubness-
    mitigated) and a hubness-skewness diagnostic.
    """
    sim = v1 @ v2.T
    n = sim.shape[0]
    diag = np.arange(n)

    # P@1 (top-1 retrieval accuracy)
    pred_a = sim.argmax(axis=1)
    pred_b = sim.argmax(axis=0)
    p1_a = float((pred_a == diag).mean())
    p1_b = float((pred_b == diag).mean())

    # MRR: rank of the diagonal entry
    # rank in row i = position of column i in descending order of sim[i, :]
    order_a = (-sim).argsort(axis=1).argsort(axis=1)  # rank index
    diag_rank_a = order_a[diag, diag] + 1
    mrr_a = float((1.0 / diag_rank_a).mean())
    order_b = (-sim.T).argsort(axis=1).argsort(axis=1)
    diag_rank_b = order_b[diag, diag] + 1
    mrr_b = float((1.0 / diag_rank_b).mean())

    diag_cos = float(sim[diag, diag].mean())
    off_diag = sim.copy()
    np.fill_diagonal(off_diag, np.nan)
    off_diag_cos = float(np.nanmean(off_diag))

    # CSLS-based P@1 (mitigates hubness)
    csls = _csls_matrix(sim, k=csls_k)
    p1_csls_a = float((csls.argmax(axis=1) == diag).mean())
    p1_csls_b = float((csls.argmax(axis=0) == diag).mean())

    # Hubness diagnostic: skewness of the N1 (top-1 in-degree) distribution.
    # Each target's N1 = how many sources retrieve it as top-1; a few hubs that
    # dominate retrieval produce high positive skew. Averaged over both directions.
    # A perfectly-aligned set gives N1 ≡ 1 (zero variance) -> skew defined as 0.
    def _safe_skew(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        if x.size < 2 or float(np.std(x)) < 1e-12:
            return 0.0
        from scipy.stats import skew

        return float(skew(x))

    n1_a = np.bincount(pred_a, minlength=n)
    n1_b = np.bincount(pred_b, minlength=n)
    hub_skew = (_safe_skew(n1_a) + _safe_skew(n1_b)) / 2.0

    return {
        "p1": (p1_a + p1_b) / 2.0,
        "p1_l1_to_l2": p1_a,
        "p1_l2_to_l1": p1_b,
        "p1_csls": (p1_csls_a + p1_csls_b) / 2.0,
        "mrr": (mrr_a + mrr_b) / 2.0,
        "diag_cos_mean": diag_cos,
        "off_diag_cos_mean": off_diag_cos,
        "margin": diag_cos - off_diag_cos,
        "hubness_skew": hub_skew,
    }


class XLingualModule(ProbeModule):
    """Cross-lingual semantic alignment probe.

    Uses parallel sentences (FLORES-200 dev) to compare encoders by their
    ability to retrieve the correct translation across language pairs.

    For each encoder *e* and each unordered language pair (L1, L2):
      - encode all sentences in both languages with *e*, L2-normalize
      - similarity matrix S[i, j] = cos(e(L1_i), e(L2_j))
      - **P@1**  = mean over rows of (argmax_j S == i), averaged with the
                   transposed direction
      - **MRR**  = mean over rows of 1 / rank(S[i, i] in row i) (averaged
                   with the transposed direction)
      - **margin** = mean(S[i, i]) − mean(S[i, j ≠ i])  (diagonal vs off-diagonal)

    Aggregate per encoder = mean across all configured language pairs.

    Hypothesis: distilled M2V retains a portion of LaBSE's cross-lingual
    alignment. The retention ratio P@1(M2V) / P@1(LaBSE) per pair quantifies
    how much survives.
    """

    name = "xlingual"

    def describe(self) -> str:
        return (
            "Cross-lingual semantic alignment probe (FLORES-200 dev parallel sentences). "
            "For each language pair, computes P@1 retrieval, MRR, and diagonal-vs-off-diagonal "
            "cosine margin per encoder. Reports per-pair and aggregate metrics, plus per-pair "
            "retention ratio (student / teacher)."
        )

    def run(self, encoders: dict[str, Encoder], data_path: str, config: Any) -> ProbeResult:
        ds = load_xlingual_dataset(data_path)
        log.info(
            "loaded xlingual dataset: version=%s langs=%s pairs=%d",
            ds.version,
            ds.languages,
            len(ds.pairs),
        )

        seed = getattr(config, "seed", 42)
        bootstrap_n = getattr(config, "bootstrap_n", 10_000)
        extra = getattr(config, "extra", {}) or {}
        csls_k = int(extra.get("csls_k", 10))

        # Encode each language once per encoder (heaviest step)
        vecs: dict[str, dict[str, np.ndarray]] = {}
        for en_name, enc in encoders.items():
            log.info("encoding with %s", en_name)
            vecs[en_name] = {}
            for lang in ds.languages:
                texts = [p.text_by_lang[lang] for p in ds.pairs]
                v = enc.encode_many(texts)
                vecs[en_name][lang] = _l2(np.asarray(v, dtype=np.float32))

        # Per-pair retrieval
        per_pair: list[dict[str, Any]] = []
        for i, l1 in enumerate(ds.languages):
            for l2 in ds.languages[i + 1 :]:
                row: dict[str, Any] = {"l1": l1, "l2": l2}
                for en_name in encoders:
                    metrics = _retrieval_metrics(
                        vecs[en_name][l1], vecs[en_name][l2], csls_k=csls_k
                    )
                    for k, v in metrics.items():
                        row[f"{k}:{en_name}"] = v
                per_pair.append(row)

        # Aggregate per encoder
        per_encoder: dict[str, Any] = {}
        for en_name in encoders:
            p1_arr = np.array([r[f"p1:{en_name}"] for r in per_pair], dtype=np.float64)
            p1c_arr = np.array([r[f"p1_csls:{en_name}"] for r in per_pair], dtype=np.float64)
            mrr_arr = np.array([r[f"mrr:{en_name}"] for r in per_pair], dtype=np.float64)
            margin_arr = np.array([r[f"margin:{en_name}"] for r in per_pair], dtype=np.float64)
            hub_arr = np.array([r[f"hubness_skew:{en_name}"] for r in per_pair], dtype=np.float64)
            ci_p1 = bootstrap_ci(p1_arr, n=bootstrap_n, seed=seed)
            ci_p1c = bootstrap_ci(p1c_arr, n=bootstrap_n, seed=seed)
            ci_mrr = bootstrap_ci(mrr_arr, n=bootstrap_n, seed=seed)
            ci_mgn = bootstrap_ci(margin_arr, n=bootstrap_n, seed=seed)
            per_encoder[en_name] = {
                "mean_p1": float(p1_arr.mean()),
                "p1_ci_low": ci_p1.low, "p1_ci_high": ci_p1.high,
                "mean_p1_csls": float(p1c_arr.mean()),
                "p1_csls_ci_low": ci_p1c.low, "p1_csls_ci_high": ci_p1c.high,
                "mean_mrr": float(mrr_arr.mean()),
                "mrr_ci_low": ci_mrr.low, "mrr_ci_high": ci_mrr.high,
                "mean_margin": float(margin_arr.mean()),
                "margin_ci_low": ci_mgn.low, "margin_ci_high": ci_mgn.high,
                "mean_hubness_skew": float(hub_arr.mean()),
                "n_pairs": int(p1_arr.size),
            }

        # Cross-encoder retention (student/teacher) for all (student, teacher) pairs.
        # Teachers = encoders whose underlying name flags a contextual model
        # (transformer:, labse, or e5). Students = the rest (typically model2vec:).
        encoder_names = list(encoders.keys())
        retention: dict[str, Any] = {}
        if len(encoder_names) >= 2:
            def is_teacher(n: str) -> bool:
                lname = encoders[n].name.lower()
                return ("transformer" in lname) or ("labse" in lname) or ("e5" in lname)

            teachers = [n for n in encoder_names if is_teacher(n)]
            students = [n for n in encoder_names if not is_teacher(n)]
            for student in students:
                for teacher in teachers:
                    ratios = []
                    ratios_csls = []
                    for r in per_pair:
                        t = r[f"p1:{teacher}"]
                        s = r[f"p1:{student}"]
                        if t > 1e-9:
                            r[f"p1_retention:{student}_over_{teacher}"] = s / t
                            ratios.append(s / t)
                        tc = r[f"p1_csls:{teacher}"]
                        sc = r[f"p1_csls:{student}"]
                        if tc > 1e-9:
                            ratios_csls.append(sc / tc)
                    if not ratios:
                        continue
                    ratios_arr = np.asarray(ratios, dtype=np.float64)
                    ci_ret = bootstrap_ci(ratios_arr, n=bootstrap_n, seed=seed)
                    csls_arr = np.asarray(ratios_csls, dtype=np.float64)
                    retention[f"{student}_over_{teacher}"] = {
                        "mean_p1_retention": float(ratios_arr.mean()),
                        "ci_low": ci_ret.low,
                        "ci_high": ci_ret.high,
                        "mean_p1_csls_retention": float(csls_arr.mean()) if csls_arr.size else float("nan"),
                        "n_pairs": int(ratios_arr.size),
                    }

        summary = {
            "n_pairs": len(per_pair),
            "n_sentences": len(ds.pairs),
            "languages": ds.languages,
            "encoders": [{"name": e.name, "dim": e.dim} for e in encoders.values()],
            "per_encoder": per_encoder,
            "retention": retention,
            "dataset_version": ds.version,
            "dataset_source": ds.source,
            "seed": seed,
            "bootstrap_n": bootstrap_n,
        }

        return ProbeResult(
            module=self.name,
            metrics=summary,
            per_category=per_pair,
            baselines={},
            tokenization_audit=[],
        )

    def plot(self, result_dict: dict, out_dir) -> list:
        from ..reporting.plots import make_xlingual_plots

        return make_xlingual_plots(result_dict, out_dir)
