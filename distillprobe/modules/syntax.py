from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from ..data import load_syntax_dataset
from ..encoders.base import Encoder, first_student
from ..metrics import bootstrap_ci, paired_wilcoxon
from .base import ProbeModule, ProbeResult

log = logging.getLogger(__name__)


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da < 1e-12 or db < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (da * db))


class SyntaxModule(ProbeModule):
    """Syntactic role survival probe.

    For each frequent noun lemma X that occurs in PARSEME-tr (UD-Turkish-IMST
    layer) as both `nsubj` and `obj` ≥10 times:

      - **Teacher (LaBSE) prototypes** — subword-level pooling at X's position
        inside the original sentence. Provides a contextualised, position-aware
        representation:
          ``subj_proto_L(X) = mean over subj contexts of LaBSE.last_hidden_state at X``
          ``obj_proto_L(X)  = mean over  obj contexts of LaBSE.last_hidden_state at X``

      - **Student (M2V) prototypes** — sentence-level bag-of-words. Position-
        independent by construction:
          ``subj_proto_M(X) = mean over subj contexts of M2V.encode(sentence)``
          ``obj_proto_M(X)  = mean over  obj contexts of M2V.encode(sentence)``

    Per-word **role separation**:
      ``role_sep = 1 − cos(subj_proto, obj_proto)``

    Hypothesis: ``role_sep_LaBSE > role_sep_M2V``. LaBSE captures positional /
    syntactic information *plus* collocational information; M2V's BoW captures
    only collocational. The signed difference per word ``role_sep_L − role_sep_M``
    is an estimate of pure syntactic information lost in distillation; tested
    with paired Wilcoxon.
    """

    name = "syntax"

    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._device = None
        self._teacher_id: str | None = None
        self._teacher_label: str = "teacher"

    def describe(self) -> str:
        return (
            "Syntactic role survival probe (PARSEME 1.2 / UD-Turkish-IMST). For each lemma "
            "appearing in both nsubj and obj, computes role_sep = 1 − cos(subj_proto, "
            "obj_proto) for the teacher (LaBSE subword-pooled at target position in context) "
            "and the student (M2V sentence BoW). Paired Wilcoxon on per-word role_sep diff."
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
        short = model_id.split("/")[-1].lower()
        self._teacher_label = f"{short}_targetpool"

    def _target_vec_in_context(
        self,
        sentence_tokens: list[str],
        target_idx: int,
        max_length: int,
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
        positions = [i for i, w in enumerate(word_ids) if w == target_idx]
        if not positions:
            return None
        inputs = {k: v.to(self._device) for k, v in encoded.items()}
        with torch.no_grad():
            out = self._model(**inputs)
        h = out.last_hidden_state[0]
        return h[positions].mean(dim=0).detach().cpu().numpy().astype(np.float32)

    # ----------------------------------------------------------- run
    def run(self, encoders: dict[str, Encoder], data_path: str, config: Any) -> ProbeResult:
        ds = load_syntax_dataset(data_path)
        log.info(
            "loaded syntax dataset: language=%s version=%s items=%d roles=%s",
            ds.language,
            ds.version,
            len(ds.items),
            ds.target_roles,
        )

        seed = getattr(config, "seed", 42)
        bootstrap_n = getattr(config, "bootstrap_n", 10_000)
        extra = getattr(config, "extra", {}) or {}
        teacher_id = extra.get("teacher_model_id", "sentence-transformers/LaBSE")
        max_length = int(extra.get("teacher_max_length", 256))
        self._ensure_model(teacher_id)

        teacher_label = self._teacher_label
        student_name = first_student(encoders)
        student: Encoder | None = encoders.get(student_name) if student_name else None

        if len(ds.target_roles) != 2:
            raise ValueError(
                f"this module supports exactly 2 target_roles; got {ds.target_roles}"
            )
        role_a, role_b = ds.target_roles

        by_word_role: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        for it in ds.items:
            by_word_role[it.word][it.role].append(it)

        per_word: list[dict[str, Any]] = []

        for word, role_dict in by_word_role.items():
            if not (role_dict.get(role_a) and role_dict.get(role_b)):
                continue
            row: dict[str, Any] = {"word": word}
            for role in (role_a, role_b):
                row[f"n:{role}"] = len(role_dict[role])

            # Teacher prototypes (subword-pool at target)
            teacher_protos: dict[str, np.ndarray] = {}
            for role in (role_a, role_b):
                vecs = []
                for it in role_dict[role]:
                    v = self._target_vec_in_context(
                        it.sentence_tokens, it.target_token_index, max_length
                    )
                    if v is not None:
                        vecs.append(v)
                if vecs:
                    teacher_protos[role] = np.mean(np.stack(vecs, axis=0), axis=0)

            if role_a in teacher_protos and role_b in teacher_protos:
                row[f"role_sep:{teacher_label}"] = 1.0 - _safe_cosine(
                    teacher_protos[role_a], teacher_protos[role_b]
                )

            # Student prototypes (sentence BoW)
            if student is not None:
                student_protos: dict[str, np.ndarray] = {}
                for role in (role_a, role_b):
                    sents = [it.sentence_text for it in role_dict[role]]
                    if not sents:
                        continue
                    vecs = student.encode_many(sents)
                    student_protos[role] = vecs.mean(axis=0)
                if role_a in student_protos and role_b in student_protos:
                    row[f"role_sep:{student_name}"] = 1.0 - _safe_cosine(
                        student_protos[role_a], student_protos[role_b]
                    )

            per_word.append(row)

        # Aggregate
        teacher_key = f"role_sep:{teacher_label}"
        student_key = f"role_sep:{student_name}" if student_name else None

        teacher_seps = np.array(
            [r[teacher_key] for r in per_word if teacher_key in r and not np.isnan(r[teacher_key])],
            dtype=np.float64,
        )
        student_seps = np.array(
            [r[student_key] for r in per_word if student_key and student_key in r and not np.isnan(r[student_key])],
            dtype=np.float64,
        ) if student_name else np.array([], dtype=np.float64)

        ci_t = bootstrap_ci(teacher_seps, n=bootstrap_n, seed=seed) if teacher_seps.size else None
        ci_s = bootstrap_ci(student_seps, n=bootstrap_n, seed=seed) if student_seps.size else None

        # Paired diff (LaBSE − M2V) per word
        diffs: list[float] = []
        if student_name:
            for r in per_word:
                if teacher_key in r and student_key in r:
                    a = r[teacher_key]
                    b = r[student_key]
                    if not (np.isnan(a) or np.isnan(b)):
                        diffs.append(a - b)
        diffs_arr = np.asarray(diffs, dtype=np.float64)
        ci_diff = bootstrap_ci(diffs_arr, n=bootstrap_n, seed=seed) if diffs_arr.size else None

        # Paired Wilcoxon (one-sided: teacher > student)
        wil_p = float("nan")
        wil_stat = float("nan")
        if diffs_arr.size >= 2:
            try:
                from scipy.stats import wilcoxon

                res = wilcoxon(diffs_arr, alternative="greater")
                wil_p = float(res.pvalue)
                wil_stat = float(res.statistic)
            except Exception as e:  # noqa: BLE001
                log.warning("wilcoxon failed: %s", e)

        summary: dict[str, Any] = {
            "n_words": len(per_word),
            "target_roles": [role_a, role_b],
            "teacher_model_id": teacher_id,
            "teacher_label": teacher_label,
            "student_name": student_name,
            "encoders": [{"name": e.name, "dim": e.dim} for e in encoders.values()],
            "mean_role_sep_teacher": float(teacher_seps.mean()) if teacher_seps.size else float("nan"),
            "teacher_ci_low": ci_t.low if ci_t else None,
            "teacher_ci_high": ci_t.high if ci_t else None,
            "mean_role_sep_student": float(student_seps.mean()) if student_seps.size else float("nan"),
            "student_ci_low": ci_s.low if ci_s else None,
            "student_ci_high": ci_s.high if ci_s else None,
            "mean_diff_teacher_minus_student": float(diffs_arr.mean()) if diffs_arr.size else float("nan"),
            "diff_ci_low": ci_diff.low if ci_diff else None,
            "diff_ci_high": ci_diff.high if ci_diff else None,
            "wilcoxon_one_sided_greater_p": wil_p,
            "wilcoxon_statistic": wil_stat,
            "n_paired": int(diffs_arr.size),
            "dataset_version": ds.version,
            "dataset_source": ds.source,
            "language": ds.language,
            "seed": seed,
            "bootstrap_n": bootstrap_n,
        }

        return ProbeResult(
            module=self.name,
            metrics=summary,
            per_category=per_word,
            baselines={},
            tokenization_audit=[],
        )

    def plot(self, result_dict: dict, out_dir) -> list:
        from ..reporting.plots import make_syntax_plots

        return make_syntax_plots(result_dict, out_dir)


# ---------------------------------------------------------------- V2


class SyntaxModuleV2(ProbeModule):
    """V2: per-word leave-one-out nearest-centroid classification of subj vs obj.

    Same task framed identically for both encoders: given an occurrence's
    encoded features, predict its UD role from the (excluding-itself) class
    centroids. Both encoders use the *same* protocol (LOO + nearest centroid),
    so accuracies are directly comparable.

      - **Teacher features**: LaBSE last_hidden_state pooled at the target word
        position in the original sentence (positional + collocational signal).
      - **Student features**: M2V sentence-level bag-of-words encoding
        (positional information cannot be present; collocational only).

    LaBSE accuracy − M2V accuracy is an estimate of the positional information
    content of the syntactic role distinction, in classifier-discriminability
    units. Random binary chance = 0.5.

    Hypothesis: ``acc(LaBSE) > acc(M2V)`` per word (paired Wilcoxon, one-sided).
    """

    name = "syntax_v2"

    def __init__(self):
        self._tokenizer = None
        self._model = None
        self._device = None
        self._teacher_id: str | None = None
        self._teacher_label: str = "teacher"

    def describe(self) -> str:
        return (
            "V2 syntax probe — per-word leave-one-out nearest-centroid classification "
            "of nsubj vs obj. Same protocol applied to both LaBSE target-pool features "
            "and M2V sentence-BoW features; accuracy difference quantifies positional "
            "information loss in distillation."
        )

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
        short = model_id.split("/")[-1].lower()
        self._teacher_label = f"{short}_targetpool"

    def _target_vec(self, sentence_tokens: list[str], target_idx: int, max_length: int):
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
        positions = [i for i, w in enumerate(word_ids) if w == target_idx]
        if not positions:
            return None
        inputs = {k: v.to(self._device) for k, v in encoded.items()}
        with torch.no_grad():
            out = self._model(**inputs)
        h = out.last_hidden_state[0]
        return h[positions].mean(dim=0).detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _loo_nearest_centroid_accuracy(features: np.ndarray, labels: np.ndarray) -> float:
        n = features.shape[0]
        if n < 4:
            return float("nan")
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        f = features / norms
        correct = 0
        valid = 0
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            tlabels = labels[mask]
            if not tlabels.any() or tlabels.all():
                continue
            c0 = f[mask & (labels == 0)].mean(axis=0)
            c1 = f[mask & (labels == 1)].mean(axis=0)
            c0 /= max(float(np.linalg.norm(c0)), 1e-12)
            c1 /= max(float(np.linalg.norm(c1)), 1e-12)
            sim0 = float(np.dot(f[i], c0))
            sim1 = float(np.dot(f[i], c1))
            pred = 0 if sim0 > sim1 else 1
            if pred == int(labels[i]):
                correct += 1
            valid += 1
        return correct / valid if valid else float("nan")

    def run(self, encoders: dict[str, Encoder], data_path: str, config: Any) -> ProbeResult:
        from collections import defaultdict

        ds = load_syntax_dataset(data_path)
        seed = getattr(config, "seed", 42)
        bootstrap_n = getattr(config, "bootstrap_n", 10_000)
        extra = getattr(config, "extra", {}) or {}
        teacher_id = extra.get("teacher_model_id", "sentence-transformers/LaBSE")
        max_length = int(extra.get("teacher_max_length", 256))
        ctrl_shuffles = int(extra.get("control_shuffles", 5))
        ctrl_rng = np.random.default_rng(seed)
        self._ensure_model(teacher_id)

        student_name = first_student(encoders)
        student = encoders.get(student_name) if student_name else None

        if len(ds.target_roles) != 2:
            raise ValueError("V2 supports exactly 2 target_roles")
        role_a, role_b = ds.target_roles  # original order, e.g. nsubj, obj

        by_word = defaultdict(list)
        for it in ds.items:
            by_word[it.word].append(it)

        teacher_label = self._teacher_label
        student_label = (student_name + "_bow") if student_name else None

        per_word: list[dict[str, Any]] = []
        n_align_dropped = 0
        for word, items in by_word.items():
            # Teacher target-pool features. Drop occurrences whose target token
            # fails subword alignment (e.g. truncated past max_length) rather than
            # zero-filling them: a zero vector deterministically biases the
            # nearest-centroid prediction and dilutes the class centroids, and it
            # would leave the teacher and student compared on different occurrence
            # sets. Filtering keeps both encoders on the SAME aligned occurrences.
            t_vecs = []
            kept = []
            for it in items:
                v = self._target_vec(it.sentence_tokens, it.target_token_index, max_length)
                if v is None:
                    n_align_dropped += 1
                    continue
                t_vecs.append(v)
                kept.append(it)
            items = kept
            n = len(items)
            labels = np.array(
                [0 if it.role == role_a else 1 for it in items], dtype=np.int64
            )
            if labels.sum() == 0 or labels.sum() == n:
                continue

            row: dict[str, Any] = {
                "word": word,
                "n": n,
                f"n:{role_a}": int((labels == 0).sum()),
                f"n:{role_b}": int((labels == 1).sum()),
            }

            t_features = np.stack(t_vecs, axis=0)
            row[f"loo_acc:{teacher_label}"] = float(
                self._loo_nearest_centroid_accuracy(t_features, labels)
            )

            # Control task (Hewitt & Liang, 2019): rerun the same parameter-free
            # LOO classifier on label-permuted (class-count-preserving) labels.
            # Near-chance control accuracy means the real accuracy reflects the
            # representation, not probe capacity. The same permutations are
            # applied to both encoders for a fair comparison.
            idx_perms = [ctrl_rng.permutation(n) for _ in range(ctrl_shuffles)]

            def _control_acc(features: np.ndarray) -> float:
                accs = [
                    self._loo_nearest_centroid_accuracy(features, labels[p])
                    for p in idx_perms
                ]
                accs = [a for a in accs if not np.isnan(a)]
                return float(np.mean(accs)) if accs else float("nan")

            row[f"loo_ctrl:{teacher_label}"] = _control_acc(t_features)

            # Student sentence-BoW features
            if student is not None:
                sents = [it.sentence_text for it in items]
                s_features = np.asarray(student.encode_many(sents), dtype=np.float32)
                row[f"loo_acc:{student_label}"] = float(
                    self._loo_nearest_centroid_accuracy(s_features, labels)
                )
                row[f"loo_ctrl:{student_label}"] = _control_acc(s_features)

            per_word.append(row)

        if n_align_dropped:
            log.warning(
                "syntax V2: dropped %d occurrence(s) with unaligned/truncated target "
                "token from both encoders", n_align_dropped
            )

        # Aggregate
        teacher_accs = np.asarray(
            [r[f"loo_acc:{teacher_label}"] for r in per_word], dtype=np.float64
        )
        student_accs = (
            np.asarray(
                [r.get(f"loo_acc:{student_label}", np.nan) for r in per_word],
                dtype=np.float64,
            )
            if student_label
            else np.array([], dtype=np.float64)
        )

        # Control-task accuracies and selectivity (real − control).
        teacher_ctrl = np.asarray(
            [r.get(f"loo_ctrl:{teacher_label}", np.nan) for r in per_word], dtype=np.float64
        )
        student_ctrl = (
            np.asarray(
                [r.get(f"loo_ctrl:{student_label}", np.nan) for r in per_word], dtype=np.float64
            )
            if student_label
            else np.array([], dtype=np.float64)
        )
        teacher_ctrl_mean = float(np.nanmean(teacher_ctrl)) if teacher_ctrl.size and not np.all(np.isnan(teacher_ctrl)) else float("nan")
        student_ctrl_mean = float(np.nanmean(student_ctrl)) if student_ctrl.size and not np.all(np.isnan(student_ctrl)) else float("nan")
        teacher_selectivity = (
            float(np.nanmean(teacher_accs)) - teacher_ctrl_mean if teacher_accs.size else float("nan")
        )
        student_selectivity = (
            float(np.nanmean(student_accs)) - student_ctrl_mean
            if student_accs.size and not np.all(np.isnan(student_accs))
            else float("nan")
        )

        ci_t = bootstrap_ci(teacher_accs, n=bootstrap_n, seed=seed)
        ci_s = bootstrap_ci(student_accs[~np.isnan(student_accs)], n=bootstrap_n, seed=seed) if student_accs.size else None

        wil_p = float("nan")
        wil_stat = float("nan")
        diff_arr = np.array([], dtype=np.float64)
        ci_diff = None
        if student_accs.size:
            mask = ~np.isnan(teacher_accs) & ~np.isnan(student_accs)
            diff_arr = teacher_accs[mask] - student_accs[mask]
            ci_diff = bootstrap_ci(diff_arr, n=bootstrap_n, seed=seed)
            if diff_arr.size >= 2:
                try:
                    from scipy.stats import wilcoxon

                    res = wilcoxon(diff_arr, alternative="greater")
                    wil_p = float(res.pvalue)
                    wil_stat = float(res.statistic)
                except Exception as e:  # noqa: BLE001
                    log.warning("wilcoxon failed: %s", e)

        summary = {
            "n_words": len(per_word),
            "target_roles": [role_a, role_b],
            "teacher_model_id": teacher_id,
            "teacher_label": teacher_label,
            "student_label": student_label,
            "encoders": [{"name": e.name, "dim": e.dim} for e in encoders.values()],
            "teacher_mean_acc": float(teacher_accs.mean()) if teacher_accs.size else float("nan"),
            "teacher_ci_low": ci_t.low,
            "teacher_ci_high": ci_t.high,
            "student_mean_acc": float(student_accs[~np.isnan(student_accs)].mean()) if student_accs.size else float("nan"),
            "student_ci_low": ci_s.low if ci_s else None,
            "student_ci_high": ci_s.high if ci_s else None,
            "mean_acc_diff_teacher_minus_student": float(diff_arr.mean()) if diff_arr.size else float("nan"),
            "diff_ci_low": ci_diff.low if ci_diff else None,
            "diff_ci_high": ci_diff.high if ci_diff else None,
            "wilcoxon_one_sided_greater_p": wil_p,
            "wilcoxon_statistic": wil_stat,
            "n_paired": int(diff_arr.size),
            "random_baseline_acc": 0.5,
            "teacher_control_acc": teacher_ctrl_mean,
            "student_control_acc": student_ctrl_mean,
            "teacher_selectivity": teacher_selectivity,
            "student_selectivity": student_selectivity,
            "control_shuffles": ctrl_shuffles,
            "dataset_version": ds.version,
            "language": ds.language,
            "seed": seed,
            "bootstrap_n": bootstrap_n,
        }

        return ProbeResult(
            module=self.name,
            metrics=summary,
            per_category=per_word,
            baselines={},
            tokenization_audit=[],
        )

    def plot(self, result_dict: dict, out_dir) -> list:
        from ..reporting.plots import make_syntax_v2_plots

        return make_syntax_v2_plots(result_dict, out_dir)
