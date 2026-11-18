from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..data import load_polysemy_dataset
from ..encoders.base import Encoder
from ..metrics import bootstrap_ci
from .base import ProbeModule, ProbeResult

log = logging.getLogger(__name__)


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da < 1e-12 or db < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (da * db))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


class PolysemyModule(ProbeModule):
    """Sense-ranking cross-encoder agreement probe.

    NB: the headline cross-encoder metric is a *symmetric agreement* between two
    encoders' sense rankings, not a directional "retention" of the teacher by the
    student; it is reported and labelled as agreement accordingly.

    For each polysemic word *w* and each encoder *e*:

      - per-sense prototype:  ``proto_e_i = mean( e(def_i), e(example_i) )``
      - word vector:          ``v_e_w     = e(w)``
      - sense similarities:   ``sim_e_i   = cos( v_e_w , proto_e_i )``

    Per-encoder summary statistics (aggregated across words):

      * top1_score     — max similarity to any sense
      * margin         — top1 minus runner-up similarity
      * entropy        — Shannon entropy of softmax over sense similarities
                         (lower = more peaked = encoder commits to one sense)
      * first_sense_bias — fraction of words whose top1 is the first TDK sense
                           (TDK orders senses roughly by salience/frequency)

    Cross-encoder summary (when ≥2 encoders are configured, e.g. LaBSE + M2V):

      * winner_agreement_rate — fraction of words where both encoders pick the
                                same top sense
      * mean_spearman         — average Spearman ρ between the two encoders'
                                sense-similarity rankings (per-word, then mean)

    Hypothesis: distillation-induced bag-of-words M2V should retain non-trivial
    sense ordering compared to LaBSE → mean_spearman > 0 and
    winner_agreement_rate > 1/avg(n_senses) (random baseline ≈ 0.16 for our
    dataset's avg ~6 senses).
    """

    name = "polysemy"

    def describe(self) -> str:
        return (
            "Polysemy retention probe. Per word and per encoder, computes per-sense "
            "prototype = mean(encoder(definition), encoder(example)), then ranks senses by "
            "cosine to the static word vector. Reports per-encoder concentration "
            "(top1, margin, entropy) and cross-encoder agreement (winner match, Spearman)."
        )

    # ----------------------------------------------------------- run
    def run(self, encoders: dict[str, Encoder], data_path: str, config: Any) -> ProbeResult:
        ds = load_polysemy_dataset(data_path)
        log.info(
            "loaded polysemy dataset: language=%s version=%s words=%d",
            ds.language,
            ds.version,
            len(ds.words),
        )

        seed = getattr(config, "seed", 42)
        bootstrap_n = getattr(config, "bootstrap_n", 10_000)

        # Per-encoder text cache (LaBSE is the bottleneck; many definitions repeat structure).
        cache: dict[tuple[str, str], np.ndarray] = {}

        def enc_cached(enc: Encoder, text: str) -> np.ndarray:
            key = (enc.name, text)
            v = cache.get(key)
            if v is None:
                v = np.asarray(enc.encode(text))
                cache[key] = v
            return v

        encoder_names = list(encoders.keys())
        per_word: list[dict[str, Any]] = []

        for w in ds.words:
            if len(w.senses) < 2:
                continue
            row: dict[str, Any] = {"word": w.word, "n_senses": len(w.senses)}

            for en_name, enc in encoders.items():
                v_word = enc_cached(enc, w.word)
                sims: list[float] = []
                for s in w.senses:
                    texts: list[str] = []
                    if s.definition:
                        texts.append(s.definition)
                    texts.extend(s.examples)
                    if not texts:
                        sims.append(float("nan"))
                        continue
                    vecs = np.stack([enc_cached(enc, t) for t in texts], axis=0)
                    proto = vecs.mean(axis=0)
                    sims.append(_safe_cosine(v_word, proto))

                arr = np.asarray(sims, dtype=np.float64)
                valid = ~np.isnan(arr)
                if valid.sum() < 2:
                    # Drop word if too few sense comparisons valid for this encoder
                    row[f"top1_idx:{en_name}"] = -1
                    row[f"top1_score:{en_name}"] = float("nan")
                    row[f"margin:{en_name}"] = float("nan")
                    row[f"entropy:{en_name}"] = float("nan")
                    row[f"sense_sims:{en_name}"] = arr.tolist()
                    continue

                arr_valid = arr[valid]
                idx_map = np.where(valid)[0]
                top1_local = int(np.argmax(arr_valid))
                top1_global = int(idx_map[top1_local])
                ordered = np.sort(arr_valid)[::-1]
                margin = float(ordered[0] - ordered[1])
                p = _softmax(arr_valid)
                ent = _entropy(p)

                row[f"top1_idx:{en_name}"] = top1_global
                row[f"top1_score:{en_name}"] = float(arr_valid[top1_local])
                row[f"margin:{en_name}"] = margin
                row[f"entropy:{en_name}"] = ent
                row[f"sense_sims:{en_name}"] = arr.tolist()
                row[f"first_sense_bias:{en_name}"] = bool(top1_global == 0)

            # Cross-encoder agreement when ≥2 encoders
            if len(encoder_names) >= 2:
                from scipy.stats import spearmanr

                a, b = encoder_names[0], encoder_names[1]
                ia = row.get(f"top1_idx:{a}", -1)
                ib = row.get(f"top1_idx:{b}", -1)
                if ia >= 0 and ib >= 0:
                    row[f"agree_winner:{a}_{b}"] = bool(ia == ib)
                xa = np.asarray(row.get(f"sense_sims:{a}", []), dtype=np.float64)
                xb = np.asarray(row.get(f"sense_sims:{b}", []), dtype=np.float64)
                mask = ~(np.isnan(xa) | np.isnan(xb))
                if mask.sum() >= 2:
                    rho, _ = spearmanr(xa[mask], xb[mask])
                    if not np.isnan(rho):
                        row[f"spearman:{a}_{b}"] = float(rho)

            per_word.append(row)

        # Aggregate per-encoder summary
        summary: dict[str, Any] = {}
        for en_name in encoder_names:
            top1s = np.asarray(
                [r[f"top1_score:{en_name}"] for r in per_word], dtype=np.float64
            )
            margins = np.asarray(
                [r[f"margin:{en_name}"] for r in per_word], dtype=np.float64
            )
            ents = np.asarray(
                [r[f"entropy:{en_name}"] for r in per_word], dtype=np.float64
            )
            biases = np.asarray(
                [int(r.get(f"first_sense_bias:{en_name}", False)) for r in per_word],
                dtype=np.float64,
            )
            t = top1s[~np.isnan(top1s)]
            m = margins[~np.isnan(margins)]
            e = ents[~np.isnan(ents)]
            ci_top1 = bootstrap_ci(t, n=bootstrap_n, seed=seed) if t.size else None
            ci_margin = bootstrap_ci(m, n=bootstrap_n, seed=seed) if m.size else None
            ci_ent = bootstrap_ci(e, n=bootstrap_n, seed=seed) if e.size else None
            summary[en_name] = {
                "mean_top1_score": float(t.mean()) if t.size else float("nan"),
                "top1_ci_low": ci_top1.low if ci_top1 else None,
                "top1_ci_high": ci_top1.high if ci_top1 else None,
                "mean_margin": float(m.mean()) if m.size else float("nan"),
                "margin_ci_low": ci_margin.low if ci_margin else None,
                "margin_ci_high": ci_margin.high if ci_margin else None,
                "mean_entropy": float(e.mean()) if e.size else float("nan"),
                "entropy_ci_low": ci_ent.low if ci_ent else None,
                "entropy_ci_high": ci_ent.high if ci_ent else None,
                "first_sense_bias_rate": float(biases.mean()) if biases.size else float("nan"),
                "n_valid": int(t.size),
            }

        # Cross-encoder summary
        cross: dict[str, Any] = {}
        if len(encoder_names) >= 2:
            a, b = encoder_names[0], encoder_names[1]
            agrees = [int(r[f"agree_winner:{a}_{b}"]) for r in per_word if f"agree_winner:{a}_{b}" in r]
            spears = [r[f"spearman:{a}_{b}"] for r in per_word if f"spearman:{a}_{b}" in r]
            avg_n_senses = float(np.mean([r["n_senses"] for r in per_word])) if per_word else float("nan")
            agree_rate = float(np.mean(agrees)) if agrees else float("nan")
            spear_arr = np.asarray(spears, dtype=np.float64)
            spear_ci = bootstrap_ci(spear_arr, n=bootstrap_n, seed=seed) if spear_arr.size else None
            # Permutation control (analogue of a Hewitt & Liang control task for a
            # ranking probe): break the *sense correspondence* between the two
            # encoders by shuffling encoder B's sense order per word, then recompute
            # the dataset-level winner-agreement rate and mean Spearman under this
            # null. Selectivity = observed − control; the permutation p-value is
            # P(null statistic ≥ observed). This shows the agreement reflects shared
            # sense structure, not the marginal shape of the similarity vectors.
            from scipy.stats import spearmanr

            pairs: list[tuple[np.ndarray, np.ndarray]] = []
            for r in per_word:
                xa = np.asarray(r.get(f"sense_sims:{a}", []), dtype=np.float64)
                xb = np.asarray(r.get(f"sense_sims:{b}", []), dtype=np.float64)
                m = ~(np.isnan(xa) | np.isnan(xb))
                if m.sum() >= 2:
                    pairs.append((xa[m], xb[m]))

            n_perm = int(getattr(config, "control_perms", 1000))
            ctrl_rng = np.random.default_rng(seed)

            def _dataset_stats(shuffle: bool) -> tuple[float, float]:
                hits = 0
                tot = 0
                rhos: list[float] = []
                for xa_v, xb_v in pairs:
                    if shuffle:
                        xb_use = xb_v[ctrl_rng.permutation(len(xb_v))]
                    else:
                        xb_use = xb_v
                    tot += 1
                    if int(np.argmax(xa_v)) == int(np.argmax(xb_use)):
                        hits += 1
                    rho, _ = spearmanr(xa_v, xb_use)
                    if not np.isnan(rho):
                        rhos.append(rho)
                ag = hits / tot if tot else float("nan")
                rh = float(np.mean(rhos)) if rhos else float("nan")
                return ag, rh

            obs_agree, obs_rho = _dataset_stats(shuffle=False)
            null_agree = np.empty(n_perm, dtype=np.float64)
            null_rho = np.empty(n_perm, dtype=np.float64)
            for i in range(n_perm):
                null_agree[i], null_rho[i] = _dataset_stats(shuffle=True)
            ctrl_agree_mean = float(np.nanmean(null_agree))
            ctrl_rho_mean = float(np.nanmean(null_rho))
            p_agree = float((np.sum(null_agree >= obs_agree) + 1) / (n_perm + 1))
            p_rho = float((np.sum(null_rho >= obs_rho) + 1) / (n_perm + 1))

            cross[f"{a}_vs_{b}"] = {
                "winner_agreement_rate": agree_rate,
                "random_agreement_baseline": (1.0 / avg_n_senses) if avg_n_senses else float("nan"),
                "mean_spearman": float(spear_arr.mean()) if spear_arr.size else float("nan"),
                "spearman_ci_low": spear_ci.low if spear_ci else None,
                "spearman_ci_high": spear_ci.high if spear_ci else None,
                "n_words": len(per_word),
                "avg_n_senses": avg_n_senses,
                # control-task / selectivity
                "control_winner_agreement": ctrl_agree_mean,
                "winner_agreement_selectivity": obs_agree - ctrl_agree_mean,
                "winner_agreement_perm_p": p_agree,
                "control_mean_spearman": ctrl_rho_mean,
                "spearman_selectivity": obs_rho - ctrl_rho_mean,
                "spearman_perm_p": p_rho,
                "control_perms": n_perm,
                "n_control_pairs": len(pairs),
            }

        metrics = {
            "n_words": len(per_word),
            "encoders": [{"name": e.name, "dim": e.dim} for e in encoders.values()],
            "per_encoder": summary,
            "cross_encoder": cross,
            "dataset_version": ds.version,
            "dataset_source": ds.source,
            "language": ds.language,
            "seed": seed,
            "bootstrap_n": bootstrap_n,
        }

        return ProbeResult(
            module=self.name,
            metrics=metrics,
            per_category=per_word,
            baselines={},
            tokenization_audit=[],
        )

    def plot(self, result_dict: dict, out_dir) -> list:
        from ..reporting.plots import make_polysemy_plots

        return make_polysemy_plots(result_dict, out_dir)
