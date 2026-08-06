"""Plot helpers for module results. Uses matplotlib with the Agg backend."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _figsize(n_cats: int, height: float = 4.6) -> tuple[float, float]:
    return (max(8.0, n_cats * 0.95), height)


def plot_consistency_bars(
    per_category: list[dict],
    encoder_names: list[str],
    baselines: dict | None,
    out_path: Path,
    title: str = "Per-category difference-vector consistency",
) -> None:
    labels = [r.get("label", r["id"]) for r in per_category]
    n_cats = len(labels)
    n_enc = len(encoder_names)
    x = np.arange(n_cats)
    width = 0.8 / n_enc

    fig, ax = plt.subplots(figsize=_figsize(n_cats))
    for i, en in enumerate(encoder_names):
        vals = [r[f"consistency:{en}"] for r in per_category]
        offset = (i - (n_enc - 1) / 2) * width
        ax.bar(x + offset, vals, width=width, label=en, edgecolor="black", linewidth=0.4)

    if baselines:
        means = [b.get("mean", 0.0) for b in baselines.values()]
        if means:
            avg = float(np.mean(means))
            ax.axhline(
                y=avg,
                linestyle="--",
                color="gray",
                alpha=0.85,
                linewidth=1.0,
                label=f"random baseline ≈ {avg:+.4f}",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("mean pairwise cosine of difference vectors")
    ax.set_title(title)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_stem_form_similarity_bars(
    per_category: list[dict],
    encoder_names: list[str],
    out_path: Path,
    title: str = "Stem–form cosine similarity (bootstrap 95% CI)",
) -> None:
    labels = [r.get("label", r["id"]) for r in per_category]
    n_cats = len(labels)
    n_enc = len(encoder_names)
    x = np.arange(n_cats)
    width = 0.8 / n_enc

    fig, ax = plt.subplots(figsize=_figsize(n_cats))
    for i, en in enumerate(encoder_names):
        means = np.array([r[f"stem_form_sim_mean:{en}"] for r in per_category])
        lows = np.array([r[f"stem_form_sim_ci_low:{en}"] for r in per_category])
        highs = np.array([r[f"stem_form_sim_ci_high:{en}"] for r in per_category])
        err = np.vstack([means - lows, highs - means])
        offset = (i - (n_enc - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            label=en,
            yerr=err,
            capsize=3,
            ecolor="black",
            edgecolor="black",
            linewidth=0.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("cosine(stem, form)")
    ax.set_title(title)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_delta_bars(
    per_category: list[dict],
    encoder_a: str,
    encoder_b: str,
    out_path: Path,
    title: str | None = None,
) -> None:
    labels = [r.get("label", r["id"]) for r in per_category]
    deltas = np.array(
        [r[f"consistency:{encoder_a}"] - r[f"consistency:{encoder_b}"] for r in per_category]
    )
    order = np.argsort(deltas)
    labels_sorted = [labels[i] for i in order]
    deltas_sorted = deltas[order]
    colors = ["#3a86ff" if d >= 0 else "#ff6b6b" for d in deltas_sorted]

    fig, ax = plt.subplots(figsize=_figsize(len(labels)))
    ax.bar(np.arange(len(labels)), deltas_sorted, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels_sorted, rotation=30, ha="right")
    ax.set_ylabel(f"consistency({encoder_a}) − consistency({encoder_b})")
    if title is None:
        title = f"Per-category consistency delta: {encoder_a} − {encoder_b}"
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def discover_encoder_names(per_category: list[dict]) -> list[str]:
    """Read encoder names off a per_category row in insertion order.

    Insertion order in `per_category` rows reflects the encoder ordering from the
    YAML config (which is meaningful: typically the focal model first, then the
    baseline). Sorting alphabetically would invert the natural reading direction
    of the delta plot.
    """
    if not per_category:
        return []
    seen: set[str] = set()
    names: list[str] = []
    for k in per_category[0]:
        if k.startswith("consistency:"):
            n = k.split(":", 1)[1]
            if n not in seen:
                seen.add(n)
                names.append(n)
    return names


def plot_gap_bars(
    per_category: list[dict],
    encoder_names: list[str],
    out_path: Path,
    title: str = "Per-category compositionality gap (bootstrap 95% CI)",
    tight_y: bool = False,
) -> None:
    """Bar chart of per-category gap mean with bootstrap CI error bars.

    `tight_y=True` zooms the y-axis to the data range (helpful when bar values are
    far from 0 — e.g. V2 subword-level gaps in [0.90, 0.93]). Default is the
    standard 0-anchored scale.
    """
    cats = [r["category"] for r in per_category]
    n_cats = len(cats)
    n_enc = len(encoder_names)
    x = np.arange(n_cats)
    width = 0.8 / n_enc

    fig, ax = plt.subplots(figsize=_figsize(n_cats))
    all_lows: list[float] = []
    all_highs: list[float] = []
    for i, en in enumerate(encoder_names):
        means = np.array([r[f"gap_mean:{en}"] for r in per_category])
        lows = np.array([r[f"gap_ci_low:{en}"] for r in per_category])
        highs = np.array([r[f"gap_ci_high:{en}"] for r in per_category])
        all_lows.extend(lows.tolist())
        all_highs.extend(highs.tolist())
        err = np.vstack([means - lows, highs - means])
        offset = (i - (n_enc - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            label=en,
            yerr=err,
            capsize=3,
            ecolor="black",
            edgecolor="black",
            linewidth=0.4,
        )

    if tight_y and all_lows:
        lo = float(min(all_lows))
        hi = float(max(all_highs))
        rng = max(hi - lo, 1e-6)
        pad = max(0.15 * rng, 0.002)
        ax.set_ylim(lo - pad, hi + pad)
        title = title + " — zoomed"
    else:
        ax.axhline(0, color="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=0)
    ax.set_ylabel("gap = 1 − cos(encoder(phrase), mean(encoder(token_i)))")
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def discover_gap_encoder_names(per_category: list[dict]) -> list[str]:
    if not per_category:
        return []
    seen: set[str] = set()
    names: list[str] = []
    for k in per_category[0]:
        if k.startswith("gap_mean:"):
            n = k.split(":", 1)[1]
            if n not in seen:
                seen.add(n)
                names.append(n)
    return names


def make_idiomaticity_plots(result_dict: dict, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_category = result_dict.get("per_category", [])
    if not per_category:
        return []
    encoder_names = discover_gap_encoder_names(per_category)
    if not encoder_names:
        return []

    # Heuristic: zoom y-axis when every bar is far from 0 (typical of V2 subword-
    # level gaps where the absolute value sits near 0.92 and category differences
    # are ~0.005 — these would be invisible on a 0-anchored axis).
    all_means = [r[f"gap_mean:{en}"] for en in encoder_names for r in per_category]
    tight = bool(all_means) and min(all_means) > 0.3

    paths: list[Path] = []
    p = out_dir / "gap_per_category.png"
    plot_gap_bars(per_category, encoder_names, p, tight_y=tight)
    paths.append(p)
    return paths


def make_polysemy_plots(result_dict: dict, out_dir: Path) -> list[Path]:
    """Plots for the polysemy probe.

    Produces:
      - `summary_metrics.png` — per-encoder mean top1, margin, entropy with bootstrap CI
      - `cross_encoder_agreement.png` — winner-agreement rate vs random baseline,
        plus mean Spearman ρ
      - `per_word_top1.png` — per-word top1 score per encoder
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = result_dict.get("metrics", {})
    per_encoder = metrics.get("per_encoder", {})
    cross = metrics.get("cross_encoder", {})
    per_word = result_dict.get("per_category", [])
    if not per_encoder:
        return []

    encoder_names = list(per_encoder.keys())
    paths: list[Path] = []

    # 1) Summary metrics per encoder
    metric_keys = [
        ("mean_top1_score", "top1_ci_low", "top1_ci_high", "top1 (mean)"),
        ("mean_margin", "margin_ci_low", "margin_ci_high", "margin (top1−top2)"),
        ("mean_entropy", "entropy_ci_low", "entropy_ci_high", "entropy"),
    ]
    fig, axes = plt.subplots(1, len(metric_keys), figsize=(3.6 * len(metric_keys), 4))
    if len(metric_keys) == 1:
        axes = [axes]
    for ax, (mkey, lo_key, hi_key, label) in zip(axes, metric_keys):
        means = np.array([per_encoder[en][mkey] for en in encoder_names])
        lows = np.array([per_encoder[en].get(lo_key) or per_encoder[en][mkey] for en in encoder_names])
        highs = np.array([per_encoder[en].get(hi_key) or per_encoder[en][mkey] for en in encoder_names])
        err = np.vstack([means - lows, highs - means])
        x = np.arange(len(encoder_names))
        ax.bar(
            x, means, yerr=err, capsize=4, ecolor="black",
            color=["#3a86ff", "#ff8c42", "#8ac926", "#bb6bd9"][: len(encoder_names)],
            edgecolor="black", linewidth=0.4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(encoder_names, rotation=20, ha="right")
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Polysemy probe — per-encoder summary (95% bootstrap CI)")
    plt.tight_layout()
    p1 = out_dir / "summary_metrics.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    paths.append(p1)

    # 2) Cross-encoder agreement
    if cross:
        cross_key = next(iter(cross))
        cdata = cross[cross_key]
        fig, ax = plt.subplots(figsize=(7.5, 4))
        labels = ["winner agreement", "random baseline", "mean Spearman ρ"]
        values = [
            cdata["winner_agreement_rate"],
            cdata["random_agreement_baseline"],
            cdata["mean_spearman"],
        ]
        colors = ["#3a86ff", "#999999", "#8ac926"]
        # Spearman CI as error bar if present
        spearman_err = None
        if cdata.get("spearman_ci_low") is not None and cdata.get("spearman_ci_high") is not None:
            lo = cdata["spearman_ci_low"]
            hi = cdata["spearman_ci_high"]
            spearman_err = [[values[2] - lo], [hi - values[2]]]
        x = np.arange(3)
        bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.4)
        if spearman_err is not None:
            ax.errorbar(
                [2], [values[2]], yerr=spearman_err, fmt="none", ecolor="black", capsize=5
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylim(min(0, min(values) - 0.05), max(values) + 0.1)
        ax.set_title(f"Polysemy: {cross_key} agreement (n={cdata['n_words']} words)")
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10,
            )
        plt.tight_layout()
        p2 = out_dir / "cross_encoder_agreement.png"
        fig.savefig(p2, dpi=130)
        plt.close(fig)
        paths.append(p2)

    # 3) Per-word top1 scores
    if per_word:
        words = [r["word"] for r in per_word]
        n_w = len(words)
        n_e = len(encoder_names)
        x = np.arange(n_w)
        width = 0.8 / n_e
        fig, ax = plt.subplots(figsize=(max(8, n_w * 0.65), 4.5))
        for i, en in enumerate(encoder_names):
            scores = [r.get(f"top1_score:{en}", float("nan")) for r in per_word]
            offset = (i - (n_e - 1) / 2) * width
            ax.bar(
                x + offset, scores, width=width, label=en,
                edgecolor="black", linewidth=0.3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(words, rotation=30, ha="right")
        ax.set_ylabel("top1 cos(word, sense_prototype)")
        ax.set_title("Polysemy probe — per-word top1 score")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        p3 = out_dir / "per_word_top1.png"
        fig.savefig(p3, dpi=130)
        plt.close(fig)
        paths.append(p3)

    return paths


def make_syntax_plots(result_dict: dict, out_dir: Path) -> list[Path]:
    """Plots for the syntax probe.

    Produces:
      - `role_sep_summary.png`  — mean role_sep for teacher vs student with bootstrap CI,
        plus the paired difference and Wilcoxon p value as a caption.
      - `per_word_role_sep.png` — per-word teacher vs student role separation
        (sorted by paired diff descending).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = result_dict.get("metrics", {})
    per_word = result_dict.get("per_category", [])
    if not metrics or not per_word:
        return []

    teacher_label = metrics.get("teacher_label", "teacher")
    student_name = metrics.get("student_name") or "student"
    paths: list[Path] = []

    # ── 1) Summary
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    means = [
        metrics.get("mean_role_sep_teacher", float("nan")),
        metrics.get("mean_role_sep_student", float("nan")),
        metrics.get("mean_diff_teacher_minus_student", float("nan")),
    ]
    lows = [
        metrics.get("teacher_ci_low") or means[0],
        metrics.get("student_ci_low") or means[1],
        metrics.get("diff_ci_low") or means[2],
    ]
    highs = [
        metrics.get("teacher_ci_high") or means[0],
        metrics.get("student_ci_high") or means[1],
        metrics.get("diff_ci_high") or means[2],
    ]
    err = np.vstack([np.array(means) - np.array(lows), np.array(highs) - np.array(means)])
    labels = [f"teacher ({teacher_label})", f"student ({student_name})", "diff (teacher − student)"]
    colors = ["#3a86ff", "#ff8c42", "#8ac926"]
    x = np.arange(3)
    bars = ax.bar(x, means, yerr=err, capsize=5, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("role_sep = 1 − cos(subj_proto, obj_proto)")
    p = metrics.get("wilcoxon_one_sided_greater_p")
    n = metrics.get("n_paired", "?")
    title_extra = ""
    if p is not None and not np.isnan(p):
        title_extra = f"   (paired Wilcoxon teacher>student: p={p:.2e}, n={n})"
    ax.set_title(f"Syntactic role separation — bootstrap 95% CI{title_extra}")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
            ha="center", va="bottom", fontsize=10,
        )
    plt.tight_layout()
    p1 = out_dir / "role_sep_summary.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    paths.append(p1)

    # ── 2) Per-word
    teacher_key = f"role_sep:{teacher_label}"
    student_key = f"role_sep:{student_name}" if student_name else None
    rows = [
        r for r in per_word
        if teacher_key in r and (student_key is None or student_key in r)
    ]
    if rows:
        diffs = [
            r[teacher_key] - (r.get(student_key, 0.0) if student_key else 0.0)
            for r in rows
        ]
        order = np.argsort(-np.array(diffs))
        rows = [rows[i] for i in order]
        words = [r["word"] for r in rows]
        n_w = len(words)
        x = np.arange(n_w)
        width = 0.4
        teacher_vals = [r[teacher_key] for r in rows]
        student_vals = [r.get(student_key, 0.0) for r in rows] if student_key else [0.0] * n_w
        fig, ax = plt.subplots(figsize=(max(10, n_w * 0.18), 4.6))
        ax.bar(x - width / 2, teacher_vals, width=width, label=f"teacher ({teacher_label})",
               color="#3a86ff", edgecolor="black", linewidth=0.3)
        if student_key:
            ax.bar(x + width / 2, student_vals, width=width, label=f"student ({student_name})",
                   color="#ff8c42", edgecolor="black", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(words, rotation=80, ha="right", fontsize=7)
        ax.set_ylabel("role_sep = 1 − cos(subj_proto, obj_proto)")
        ax.set_title("Per-word role separation (sorted by teacher − student diff)")
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        p2 = out_dir / "per_word_role_sep.png"
        fig.savefig(p2, dpi=130)
        plt.close(fig)
        paths.append(p2)

    return paths


def make_xlingual_plots(result_dict: dict, out_dir: Path) -> list[Path]:
    """Plots for the cross-lingual alignment probe.

    Produces:
      - `p1_per_pair.png`     — P@1 per language pair, encoder-grouped bars
      - `summary_metrics.png` — per-encoder mean P@1 / MRR / margin (CI)
      - `retention_per_pair.png` — student/teacher P@1 retention per pair
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = result_dict.get("metrics", {})
    per_encoder = metrics.get("per_encoder", {})
    per_pair = result_dict.get("per_category", [])
    retention = metrics.get("retention", {})
    encoder_names = list(per_encoder.keys())
    if not encoder_names or not per_pair:
        return []

    paths: list[Path] = []

    # ── 1) P@1 per language pair
    pair_labels = [f"{r['l1']}-{r['l2']}" for r in per_pair]
    n_p = len(pair_labels)
    n_e = len(encoder_names)
    x = np.arange(n_p)
    width = 0.8 / n_e
    fig, ax = plt.subplots(figsize=(max(10, n_p * 0.5), 4.6))
    for i, en in enumerate(encoder_names):
        vals = [r[f"p1:{en}"] for r in per_pair]
        offset = (i - (n_e - 1) / 2) * width
        ax.bar(
            x + offset, vals, width=width, label=en,
            edgecolor="black", linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("P@1 (mean of l1→l2 and l2→l1)")
    ax.set_title("Cross-lingual retrieval P@1 per language pair")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p1 = out_dir / "p1_per_pair.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    paths.append(p1)

    # ── 2) Summary metrics
    metric_keys = [
        ("mean_p1", "p1_ci_low", "p1_ci_high", "mean P@1"),
        ("mean_mrr", "mrr_ci_low", "mrr_ci_high", "mean MRR"),
        ("mean_margin", "margin_ci_low", "margin_ci_high", "diag − off-diag cos"),
    ]
    fig, axes = plt.subplots(1, len(metric_keys), figsize=(3.7 * len(metric_keys), 4))
    for ax, (mkey, lo_key, hi_key, label) in zip(axes, metric_keys):
        means = np.array([per_encoder[en][mkey] for en in encoder_names])
        lows = np.array([per_encoder[en].get(lo_key) or per_encoder[en][mkey] for en in encoder_names])
        highs = np.array([per_encoder[en].get(hi_key) or per_encoder[en][mkey] for en in encoder_names])
        err = np.vstack([means - lows, highs - means])
        x = np.arange(len(encoder_names))
        ax.bar(
            x, means, yerr=err, capsize=5,
            color=["#3a86ff", "#ff8c42", "#8ac926", "#bb6bd9"][: len(encoder_names)],
            edgecolor="black", linewidth=0.4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(encoder_names, rotation=20, ha="right")
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Cross-lingual probe — per-encoder summary (95% bootstrap CI)")
    plt.tight_layout()
    p2 = out_dir / "summary_metrics.png"
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    paths.append(p2)

    # ── 3) Retention per pair
    if retention:
        ret_key = next(iter(retention))
        ratios = [
            r.get(f"p1_retention:{ret_key}", float("nan"))
            for r in per_pair
        ]
        valid = [(lab, ratio) for lab, ratio in zip(pair_labels, ratios) if not np.isnan(ratio)]
        if valid:
            order = np.argsort([r for _, r in valid])  # ascending: worst retention first
            valid_sorted = [valid[i] for i in order]
            labs = [v[0] for v in valid_sorted]
            vals = [v[1] for v in valid_sorted]
            x = np.arange(len(labs))
            fig, ax = plt.subplots(figsize=(max(10, len(labs) * 0.5), 4.6))
            ax.bar(x, vals, color="#bb6bd9", edgecolor="black", linewidth=0.4)
            ax.axhline(1.0, color="black", linewidth=0.6, linestyle="--", alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(f"P@1({ret_key.split('_over_')[0]}) / P@1({ret_key.split('_over_')[1]})")
            mean = retention[ret_key]["mean_p1_retention"]
            lo = retention[ret_key]["ci_low"]
            hi = retention[ret_key]["ci_high"]
            ax.set_title(
                f"Cross-lingual P@1 retention — student / teacher "
                f"(mean = {mean:.3f} [{lo:.3f}, {hi:.3f}])"
            )
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            p3 = out_dir / "retention_per_pair.png"
            fig.savefig(p3, dpi=130)
            plt.close(fig)
            paths.append(p3)

    return paths


def make_negation_plots(result_dict: dict, out_dir: Path) -> list[Path]:
    """Plots for the negation polarity-flip probe."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = result_dict.get("metrics", {})
    per_triplet = result_dict.get("per_category", [])
    if not metrics or not per_triplet:
        return []
    per_encoder = metrics.get("per_encoder", {})
    if not per_encoder:
        return []
    encoder_names = list(per_encoder.keys())
    paths: list[Path] = []

    # ── 1) Per-triplet polarity flip
    # Sort triplets by first encoder's flip ascending (so plot reads "worst → best")
    primary = encoder_names[0]
    rows = sorted(per_triplet, key=lambda r: r.get(f"polarity_flip:{primary}", 0.0))
    n = len(rows)
    n_e = len(encoder_names)
    x = np.arange(n)
    width = 0.8 / n_e
    fig, ax = plt.subplots(figsize=(max(10, n * 0.45), 4.6))
    for i, en in enumerate(encoder_names):
        vals = [r.get(f"polarity_flip:{en}", float("nan")) for r in rows]
        offset = (i - (n_e - 1) / 2) * width
        ax.bar(
            x + offset, vals, width=width, label=en,
            edgecolor="black", linewidth=0.4,
        )
    labels = [f"{r['positive']} | {r['antonym']}" for r in rows]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("polarity_flip = cos(neg, antonym) − cos(neg, positive)")
    ax.set_title("Per-triplet polarity flip (sorted by primary encoder)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p1 = out_dir / "polarity_flip_per_triplet.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    paths.append(p1)

    # ── 2) Summary
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    means = np.array([per_encoder[en]["mean_flip"] for en in encoder_names])
    lows = np.array([per_encoder[en].get("ci_low") or per_encoder[en]["mean_flip"] for en in encoder_names])
    highs = np.array([per_encoder[en].get("ci_high") or per_encoder[en]["mean_flip"] for en in encoder_names])
    err = np.vstack([means - lows, highs - means])
    bars = ax.bar(
        np.arange(len(encoder_names)), means, yerr=err, capsize=5,
        color=["#3a86ff", "#ff8c42", "#8ac926", "#bb6bd9"][: len(encoder_names)],
        edgecolor="black", linewidth=0.4,
    )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(np.arange(len(encoder_names)))
    ax.set_xticklabels(encoder_names)
    ax.set_ylabel("mean polarity_flip (95% bootstrap CI)")
    title_extra = ""
    cmps = metrics.get("comparisons", {})
    if cmps:
        cmp_key, cmp_val = next(iter(cmps.items()))
        title_extra = (
            f"\npaired Wilcoxon ({cmp_key.replace('_gt_', ' > ')}): "
            f"p = {cmp_val['wilcoxon_p']:.2e}, "
            f"diff = {cmp_val['mean_diff']:.3f} [{cmp_val['diff_ci_low']:.3f}, {cmp_val['diff_ci_high']:.3f}]"
        )
    ax.set_title("Negation polarity-flip — per-encoder summary" + title_extra)
    for bar, v, en in zip(bars, means, encoder_names):
        frac = per_encoder[en].get("fraction_positive_flip", float("nan"))
        ax.text(
            bar.get_x() + bar.get_width() / 2, v,
            f"{v:.3f}\n{frac*100:.0f}% > 0",
            ha="center", va="bottom" if v >= 0 else "top",
            fontsize=10,
        )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p2 = out_dir / "summary.png"
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    paths.append(p2)

    return paths


def make_syntax_v2_plots(result_dict: dict, out_dir: Path) -> list[Path]:
    """Plots for the V2 syntax classifier probe."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = result_dict.get("metrics", {})
    per_word = result_dict.get("per_category", [])
    if not metrics or not per_word:
        return []
    teacher_label = metrics.get("teacher_label", "teacher")
    student_label = metrics.get("student_label") or "student"
    paths: list[Path] = []

    # Summary
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    means = [
        metrics.get("teacher_mean_acc", float("nan")),
        metrics.get("student_mean_acc", float("nan")),
        metrics.get("mean_acc_diff_teacher_minus_student", float("nan")),
    ]
    lows = [
        metrics.get("teacher_ci_low") or means[0],
        metrics.get("student_ci_low") or means[1],
        metrics.get("diff_ci_low") or means[2],
    ]
    highs = [
        metrics.get("teacher_ci_high") or means[0],
        metrics.get("student_ci_high") or means[1],
        metrics.get("diff_ci_high") or means[2],
    ]
    err = np.vstack([np.array(means) - np.array(lows), np.array(highs) - np.array(means)])
    labels = [f"teacher\n({teacher_label})", f"student\n({student_label})", "diff (T − S)"]
    colors = ["#3a86ff", "#ff8c42", "#8ac926"]
    x = np.arange(3)
    bars = ax.bar(x, means, yerr=err, capsize=5, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(0.5, color="gray", linewidth=0.6, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("LOO nearest-centroid accuracy (random = 0.5)")
    p = metrics.get("wilcoxon_one_sided_greater_p")
    n = metrics.get("n_paired", "?")
    title_extra = ""
    if p is not None and not np.isnan(p):
        title_extra = f"\npaired Wilcoxon T>S: p={p:.2e}, n={n}"
    ax.set_title("Syntax V2 — per-word LOO nearest-centroid accuracy" + title_extra)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}",
            ha="center", va="bottom" if v >= 0 else "top", fontsize=10,
        )
    plt.tight_layout()
    p1 = out_dir / "summary.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    paths.append(p1)

    # Per-word
    rows = sorted(
        per_word,
        key=lambda r: -((r.get(f"loo_acc:{teacher_label}", 0) - r.get(f"loo_acc:{student_label}", 0))),
    )
    n_w = len(rows)
    x = np.arange(n_w)
    width = 0.4
    t_v = [r.get(f"loo_acc:{teacher_label}", float("nan")) for r in rows]
    s_v = [r.get(f"loo_acc:{student_label}", float("nan")) for r in rows]
    fig, ax = plt.subplots(figsize=(max(10, n_w * 0.18), 4.6))
    ax.bar(x - width / 2, t_v, width=width, label=teacher_label, color="#3a86ff", edgecolor="black", linewidth=0.3)
    ax.bar(x + width / 2, s_v, width=width, label=student_label, color="#ff8c42", edgecolor="black", linewidth=0.3)
    ax.axhline(0.5, color="gray", linewidth=0.6, linestyle="--", alpha=0.6, label="random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([r["word"] for r in rows], rotation=80, ha="right", fontsize=7)
    ax.set_ylabel("LOO accuracy")
    ax.set_title("Per-word LOO accuracy (sorted by teacher − student)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p2 = out_dir / "per_word_acc.png"
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    paths.append(p2)

    return paths


def make_sts_plots(result_dict: dict, out_dir: Path) -> list[Path]:
    """Plot per-encoder Spearman ρ on STS, with bootstrap CI."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = result_dict.get("metrics", {})
    per_encoder = metrics.get("per_encoder", {})
    if not per_encoder:
        return []
    encs = list(per_encoder.keys())
    means = np.array([per_encoder[e]["spearman"] for e in encs])
    lows = np.array([per_encoder[e]["ci_low"] for e in encs])
    highs = np.array([per_encoder[e]["ci_high"] for e in encs])
    err = np.vstack([means - lows, highs - means])

    fig, ax = plt.subplots(figsize=(max(7.5, len(encs) * 1.5), 4.5))
    x = np.arange(len(encs))
    bars = ax.bar(
        x, means, yerr=err, capsize=5,
        color=["#3a86ff", "#ff8c42", "#8ac926", "#bb6bd9", "#ff6b6b"][: len(encs)],
        edgecolor="black", linewidth=0.4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(encs, rotation=15, ha="right")
    ax.set_ylabel("Spearman ρ (cosine vs gold)")
    ax.set_ylim(0, 1)
    ax.set_title(f"STS-Tr — Spearman correlation per encoder (n={metrics.get('n_pairs', '?')})")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    p1 = out_dir / "spearman.png"
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    return [p1]


def make_morphology_plots(result_dict: dict, out_dir: Path) -> list[Path]:
    """Generate the standard morphology plot set into out_dir. Returns written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_category = result_dict.get("per_category", [])
    if not per_category:
        return []
    encoder_names = discover_encoder_names(per_category)
    baselines = result_dict.get("baselines") or {}

    paths: list[Path] = []
    p1 = out_dir / "consistency_per_category.png"
    plot_consistency_bars(per_category, encoder_names, baselines, p1)
    paths.append(p1)

    p2 = out_dir / "stem_form_similarity_per_category.png"
    plot_stem_form_similarity_bars(per_category, encoder_names, p2)
    paths.append(p2)

    if len(encoder_names) >= 2:
        a, b = encoder_names[0], encoder_names[1]
        p3 = out_dir / f"delta_consistency_{a}_minus_{b}.png"
        plot_delta_bars(per_category, a, b, p3)
        paths.append(p3)

    return paths
