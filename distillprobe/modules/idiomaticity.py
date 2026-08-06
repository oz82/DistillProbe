from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..data import load_idiomaticity_dataset
from ..encoders.base import Encoder
from ..metrics import bootstrap_ci, cliffs_delta, mannwhitneyu
from .base import ProbeModule, ProbeResult

log = logging.getLogger(__name__)

# Ordered idiomaticity spectrum: leftmost = most idiomatic, rightmost = least.
# We expect gap to decrease left → right: gap(VID) > gap(LVC.full) > gap(LITERAL).
ORDERED_CATEGORIES = ["VID", "LVC.full", "LITERAL"]


class IdiomaticityModule(ProbeModule):
    """Compositionality gap probe via a contextual teacher.

    For each (phrase, tokens) pair and each encoder e:

        gap_e = 1 − cos( e(phrase),  mean_i e(token_i) )

    A bag-of-words student (Model2Vec) collapses these by construction → gap_e ≈ 0.
    A contextual teacher (LaBSE) produces a non-zero gap whose magnitude reflects
    how non-compositional the phrase is.

    Hypothesis (PARSEME ordering):
        gap_teacher(VID) > gap_teacher(LVC.full) > gap_teacher(LITERAL)

    Tests run per encoder. Pairwise Mann–Whitney U (one-sided) + Cliff's delta on
    adjacent categories of the spectrum, plus per-category bootstrap CI.
    """

    name = "idiomaticity"

    def describe(self) -> str:
        return (
            "Idiomaticity gap probe (PARSEME Turkish 1.2). "
            "For each phrase: gap = 1 − cos(encoder(phrase), mean(encoder(token_i))). "
            "Per-category bootstrap CI; ordered Mann–Whitney across the "
            "VID > LVC.full > LITERAL spectrum (one-sided). See migration.md §9.1."
        )

    # ----------------------------------------------------------- helpers
    @staticmethod
    def _gap(enc: Encoder, phrase: str, tokens: list[str], cache: dict) -> float:
        key_phrase = (enc.name, "P", phrase)
        v_phrase = cache.get(key_phrase)
        if v_phrase is None:
            v_phrase = enc.encode(phrase)
            cache[key_phrase] = v_phrase

        token_vecs: list[np.ndarray] = []
        for tok in tokens:
            key_tok = (enc.name, "T", tok)
            v = cache.get(key_tok)
            if v is None:
                v = enc.encode(tok)
                cache[key_tok] = v
            token_vecs.append(v)
        if not token_vecs:
            return float("nan")
        v_compos = np.mean(np.stack(token_vecs, axis=0), axis=0)
        # Cosine handles re-normalization via norms.
        denom = float(np.linalg.norm(v_phrase) * np.linalg.norm(v_compos))
        if denom < 1e-12:
            return float("nan")
        cos = float(np.dot(v_phrase, v_compos) / denom)
        return 1.0 - cos

    # ----------------------------------------------------------- run
    def run(self, encoders: dict[str, Encoder], data_path: str, config: Any) -> ProbeResult:
        ds = load_idiomaticity_dataset(data_path)
        log.info(
            "loaded idiomaticity dataset: language=%s version=%s items=%d source=%s",
            ds.language,
            ds.version,
            len(ds.items),
            ds.source,
        )

        seed = getattr(config, "seed", 42)
        bootstrap_n = getattr(config, "bootstrap_n", 10_000)

        # Group items by category (keep occurrence multiplicity)
        cats: dict[str, list] = {}
        for it in ds.items:
            cats.setdefault(it.category, []).append(it)

        present_categories = [c for c in ORDERED_CATEGORIES if c in cats]
        unknown = sorted(set(cats) - set(ORDERED_CATEGORIES))
        if unknown:
            log.info("note: skipping unknown categories %s", unknown)

        # Per-encoder, per-category gap arrays
        cache: dict = {}
        gaps_table: dict[str, dict[str, np.ndarray]] = {n: {} for n in encoders}

        for cat in present_categories:
            items = cats[cat]
            log.info("computing gaps for category=%s (n=%d)", cat, len(items))
            for enc_name, enc in encoders.items():
                vals: list[float] = []
                for it in items:
                    tokens = it.phrase.split()
                    g = self._gap(enc, it.phrase, tokens, cache)
                    if not np.isnan(g):
                        vals.append(g)
                gaps_table[enc_name][cat] = np.asarray(vals, dtype=np.float64)

        # Per-category aggregation
        per_category: list[dict] = []
        for cat in present_categories:
            row: dict[str, Any] = {"category": cat, "n": len(cats[cat])}
            for enc_name in encoders:
                arr = gaps_table[enc_name][cat]
                ci = bootstrap_ci(arr, n=bootstrap_n, seed=seed)
                row[f"gap_mean:{enc_name}"] = float(arr.mean()) if arr.size else float("nan")
                row[f"gap_std:{enc_name}"] = float(arr.std()) if arr.size else float("nan")
                row[f"gap_ci_low:{enc_name}"] = ci.low
                row[f"gap_ci_high:{enc_name}"] = ci.high
                row[f"gap_n_valid:{enc_name}"] = int(arr.size)
            per_category.append(row)

        # Pairwise ordered comparisons across adjacent categories on the spectrum
        comparisons: dict[str, Any] = {}
        for enc_name in encoders:
            for a, b in zip(present_categories, present_categories[1:]):
                x = gaps_table[enc_name][a]
                y = gaps_table[enc_name][b]
                if x.size < 2 or y.size < 2:
                    continue
                try:
                    res = mannwhitneyu(x, y, alternative="greater")
                    comparisons[f"{enc_name}:{a}_gt_{b}"] = {
                        "test": "mannwhitneyu",
                        "alternative": "greater",
                        "U": float(res.statistic),
                        "p": float(res.pvalue),
                        "cliffs_delta": cliffs_delta(x, y),
                        "mean_a": float(x.mean()),
                        "mean_b": float(y.mean()),
                        "n_a": int(x.size),
                        "n_b": int(y.size),
                    }
                except Exception as e:  # noqa: BLE001
                    log.warning("mannwhitney failed for %s %s vs %s: %s", enc_name, a, b, e)

        summary = {
            "n_items": len(ds.items),
            "categories": present_categories,
            "encoders": [{"name": e.name, "dim": e.dim} for e in encoders.values()],
            "per_encoder_mean_gap": {
                enc_name: {
                    cat: (
                        float(gaps_table[enc_name][cat].mean())
                        if gaps_table[enc_name][cat].size
                        else float("nan")
                    )
                    for cat in present_categories
                }
                for enc_name in encoders
            },
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
            per_category=per_category,
            baselines={},
            tokenization_audit=[],
        )

    def plot(self, result_dict: dict, out_dir) -> list:
        from ..reporting.plots import make_idiomaticity_plots

        return make_idiomaticity_plots(result_dict, out_dir)


# ---------------------------------------------------------------- V2 (subword-level)


class IdiomaticityModuleV2(ProbeModule):
    """V2 of the idiomaticity probe — subword-level access via HuggingFace transformers.

    Avoids V1's "single-token sentence" pooling artefact by:
      - building `v_joint` directly from the teacher's last hidden states at the
        phrase's subword positions WITHIN the original sentence (so the phrase
        gets attention-rich, contextualised representations);
      - building `v_compos` from the teacher's INPUT embedding layer (a static,
        pre-attention BoW lookup) at the same subword positions.

    Both vectors live in the teacher's hidden dim (e.g. 768 for LaBSE), so cosine
    is well-defined. The gap measures how much attention/context adds on top of
    a pure bag-of-subword-embeddings baseline:

        gap = 1 − cos( v_joint, v_compos )

    Hypothesis: gap(VID) > gap(LVC.full) > gap(LITERAL).
    """

    name = "idiomaticity_v2"

    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._device = None
        self._teacher_id: str | None = None
        self._teacher_label: str = "teacher"

    def describe(self) -> str:
        return (
            "Idiomaticity V2 (subword-level). Joint = mean teacher hidden state at phrase subword "
            "positions in the original sentence; compositional baseline = mean teacher input "
            "embeddings (pre-attention) at the same positions. Both 768-d (LaBSE). "
            "Cosine gap; per-category bootstrap CI; sequential Mann–Whitney U on the "
            "VID > LVC.full > LITERAL spectrum."
        )

    # ----------------------------------------------------------- internals
    def _ensure_model(self, model_id: str) -> None:
        if self._model is not None and self._teacher_id == model_id:
            return
        from transformers import AutoModel, AutoTokenizer
        import torch

        log.info("loading teacher (HF): %s", model_id)
        self._teacher_id = model_id
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self._model = AutoModel.from_pretrained(model_id).eval()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(self._device)
        log.info("teacher loaded on %s", self._device)
        # Build a clean encoder label for keys/plots.
        short = model_id.split("/")[-1].lower()
        self._teacher_label = f"{short}_v2_subword"

    def _phrase_vectors(
        self, sentence_tokens: list[str], phrase_idx_set: set[int], max_length: int
    ):
        import torch

        encoded = self._tokenizer(
            sentence_tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        word_ids = encoded.word_ids()
        positions = [
            i for i, w in enumerate(word_ids) if w is not None and w in phrase_idx_set
        ]
        if not positions:
            return None, None
        inputs = {k: v.to(self._device) for k, v in encoded.items()}
        with torch.no_grad():
            out = self._model(**inputs)
            h = out.last_hidden_state[0]
            ids = inputs["input_ids"][0]
            emb = self._model.get_input_embeddings()(ids)
        v_joint = h[positions].mean(dim=0).detach().cpu().numpy().astype(np.float32)
        v_compos = emb[positions].mean(dim=0).detach().cpu().numpy().astype(np.float32)
        return v_joint, v_compos

    # ----------------------------------------------------------- run
    def run(self, encoders: dict[str, Encoder], data_path: str, config: Any) -> ProbeResult:
        ds = load_idiomaticity_dataset(data_path)
        log.info(
            "loaded idiomaticity dataset: language=%s version=%s items=%d source=%s",
            ds.language,
            ds.version,
            len(ds.items),
            ds.source,
        )

        seed = getattr(config, "seed", 42)
        bootstrap_n = getattr(config, "bootstrap_n", 10_000)
        extra = getattr(config, "extra", {}) or {}
        teacher_id = extra.get("teacher_model_id", "sentence-transformers/LaBSE")
        max_length = int(extra.get("teacher_max_length", 256))
        self._ensure_model(teacher_id)
        label = self._teacher_label

        cats: dict[str, list] = {}
        for it in ds.items:
            cats.setdefault(it.category, []).append(it)
        present = [c for c in ORDERED_CATEGORIES if c in cats]
        unknown = sorted(set(cats) - set(ORDERED_CATEGORIES))
        if unknown:
            log.info("note: skipping unknown categories %s", unknown)

        from collections import defaultdict

        gaps: dict[str, list[float]] = {c: [] for c in present}
        gaps_by_lemma: dict[str, dict[str, list[float]]] = {
            c: defaultdict(list) for c in present
        }
        n_skipped_no_alignment = 0
        for cat in present:
            log.info("V2 gap for category=%s (n=%d)", cat, len(cats[cat]))
            for it in cats[cat]:
                if not it.sentence_tokens or not it.phrase_token_indices:
                    n_skipped_no_alignment += 1
                    continue
                v_joint, v_compos = self._phrase_vectors(
                    it.sentence_tokens, set(it.phrase_token_indices), max_length
                )
                if v_joint is None:
                    n_skipped_no_alignment += 1
                    continue
                denom = float(np.linalg.norm(v_joint) * np.linalg.norm(v_compos))
                if denom < 1e-12:
                    continue
                cos = float(np.dot(v_joint, v_compos) / denom)
                gaps[cat].append(1.0 - cos)
                gaps_by_lemma[cat][it.lemma_phrase].append(1.0 - cos)
        if n_skipped_no_alignment:
            log.warning(
                "skipped %d items lacking subword alignment (V1 dataset?)",
                n_skipped_no_alignment,
            )

        per_category: list[dict] = []
        for cat in present:
            arr = np.asarray(gaps[cat], dtype=np.float64)
            ci = bootstrap_ci(arr, n=bootstrap_n, seed=seed)
            per_category.append(
                {
                    "category": cat,
                    "n": int(arr.size),
                    f"gap_mean:{label}": float(arr.mean()) if arr.size else float("nan"),
                    f"gap_std:{label}": float(arr.std()) if arr.size else float("nan"),
                    f"gap_ci_low:{label}": ci.low,
                    f"gap_ci_high:{label}": ci.high,
                    f"gap_n_valid:{label}": int(arr.size),
                }
            )

        comparisons: dict[str, Any] = {}
        for a, b in zip(present, present[1:]):
            x = np.asarray(gaps[a])
            y = np.asarray(gaps[b])
            if x.size < 2 or y.size < 2:
                continue
            try:
                res = mannwhitneyu(x, y, alternative="greater")
                comparisons[f"{label}:{a}_gt_{b}"] = {
                    "test": "mannwhitneyu",
                    "alternative": "greater",
                    "U": float(res.statistic),
                    "p": float(res.pvalue),
                    "cliffs_delta": cliffs_delta(x, y),
                    "mean_a": float(x.mean()),
                    "mean_b": float(y.mean()),
                    "n_a": int(x.size),
                    "n_b": int(y.size),
                }
            except Exception as e:  # noqa: BLE001
                log.warning("mannwhitney failed: %s vs %s: %s", a, b, e)

        # De-clustered (lemma-level) test: aggregate per-MWE-lemma mean gap so each
        # idiom counts once, removing the pseudoreplication in the item-level
        # Mann-Whitney (an idiom recurring across many sentences would otherwise be
        # counted many times as if independent).
        lemma_gaps: dict[str, np.ndarray] = {
            cat: np.asarray(
                [float(np.mean(v)) for v in gaps_by_lemma[cat].values() if v],
                dtype=np.float64,
            )
            for cat in present
        }
        comparisons_declustered: dict[str, Any] = {}
        for a, b in zip(present, present[1:]):
            x = lemma_gaps[a]
            y = lemma_gaps[b]
            if x.size < 2 or y.size < 2:
                continue
            try:
                res = mannwhitneyu(x, y, alternative="greater")
                comparisons_declustered[f"{label}:{a}_gt_{b}"] = {
                    "test": "mannwhitneyu (lemma-declustered)",
                    "alternative": "greater",
                    "U": float(res.statistic),
                    "p": float(res.pvalue),
                    "cliffs_delta": cliffs_delta(x, y),
                    "mean_a": float(x.mean()),
                    "mean_b": float(y.mean()),
                    "n_lemmas_a": int(x.size),
                    "n_lemmas_b": int(y.size),
                }
            except Exception as e:  # noqa: BLE001
                log.warning("declustered mannwhitney failed: %s vs %s: %s", a, b, e)

        summary = {
            "n_items": len(ds.items),
            "n_skipped_no_alignment": n_skipped_no_alignment,
            "categories": present,
            "teacher_model_id": teacher_id,
            "teacher_label": label,
            "n_lemmas": {cat: int(len(gaps_by_lemma[cat])) for cat in present},
            "per_category_mean_gap": {
                cat: float(np.asarray(gaps[cat]).mean()) if gaps[cat] else float("nan")
                for cat in present
            },
            "comparisons": comparisons,
            "comparisons_declustered": comparisons_declustered,
            "dataset_version": ds.version,
            "dataset_source": ds.source,
            "language": ds.language,
            "seed": seed,
            "bootstrap_n": bootstrap_n,
            "max_length": max_length,
        }

        return ProbeResult(
            module=self.name,
            metrics=summary,
            per_category=per_category,
            baselines={},
            tokenization_audit=[],
        )

    def plot(self, result_dict: dict, out_dir) -> list:
        from ..reporting.plots import make_idiomaticity_plots

        return make_idiomaticity_plots(result_dict, out_dir)
