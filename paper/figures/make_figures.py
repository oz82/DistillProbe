"""Generate paper figures from DistillProbe results.

Figures (each saved as PDF for LaTeX, PNG for previews, TIFF for the IEEE portal):
  1. loss_spectrum          — teacher-relative retention of the five student axes,
                              Turkish vs English (potion), with the positional gap inset
  2. typology_gradient      — syntax Teacher-Student gap across six languages
  3. ablation_heatmap       — teacher-relative retention, students x axes (TR and EN)
  4. downstream_retention   — task-level retention (STS, retrieval, clustering) per
                              student against both contextual references

All values are read live from the canonical / revision run manifests, so the
figures never go stale relative to the numbers in the text.

Usage:
    python paper/figures/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# IEEE rejects Type 3 fonts in submitted graphics; 42 embeds TrueType instead.
# Figures are authored at the final printed size (IEEE two-column text width,
# 7.16 in) and included at \textwidth, so on-figure type stays at 8-9 pt.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
matplotlib.rcParams["font.size"] = 8

TEXT_WIDTH_IN = 7.16
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from spectrum_scale import compute as spectrum_compute  # noqa: E402

AXES = ["cross_lingual", "syntactic_role", "morphology", "polysemy", "negation"]
AXIS_LABEL = {
    "cross_lingual": "Cross-lingual\nalignment",
    "syntactic_role": "Syntactic\nrole",
    "morphology": "Morphology",
    "polysemy": "Polysemy\nranking",
    "negation": "Negation\npolarity flip",
}
STUDENT_LABEL = {
    "m2v_potion": "potion (bge-m3 + Tokenlearn)",
    "m2v_multilingual_output": "M2V_output (LaBSE + PCA/Zipf)",
    "m2v_bgem3_pca256_sif": "bge-m3 + PCA/SIF (ours)",
    "st_static_mrl": "static-similarity-mrl (no teacher)",
    "wada_gte256_en": "Wada et al. SWE (GTE, EN)",
}
BLUE, ORANGE, RED, GREY = "#3a86ff", "#ff8c42", "#ff4d4d", "#9e9e9e"


def _run_dirs() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for name in ("_canonical_runs.txt", "_revision_runs.txt"):
        p = ROOT / "results" / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1].startswith("results/"):
                out[parts[0]] = ROOT / parts[1].strip()
    return out


def _metrics(run_dir: Path) -> dict:
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return payload.get("metrics", payload)


def _save(fig, out_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=200)
    try:
        fig.savefig(out_dir / f"{stem}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    except Exception as e:  # noqa: BLE001 - TIFF export is optional
        print(f"  (tiff skipped: {e})")
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png")


# ---------------------------------------------------------------- Fig 1
def make_loss_spectrum(out_dir: Path, rows: list[dict], dirs: dict[str, Path]) -> None:
    tr = next(r for r in rows if r["language"] == "tr" and r["student"] == "m2v_potion")
    en = next(r for r in rows if r["language"] == "en" and r["student"] == "m2v_potion")
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.2))
    x = np.arange(len(AXES))
    w = 0.36
    tr_vals = [tr.get(a) if tr.get(a) is not None else np.nan for a in AXES]
    en_vals = [en.get(a) if en.get(a) is not None else np.nan for a in AXES]
    b1 = ax.bar(x - w / 2, tr_vals, w, color=BLUE, edgecolor="black", linewidth=0.5, label="Turkish")
    b2 = ax.bar(x + w / 2, en_vals, w, color=ORANGE, edgecolor="black", linewidth=0.5, label="English")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, 0.02, "shared\n17-lang\nrun", ha="center",
                        va="bottom", fontsize=6.5, color="#555")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center",
                        va="bottom", fontsize=7.5)
    ax.axhline(1.0, color=GREY, lw=0.8, ls="--")
    ax.text(1.5, 1.02, "reference encoder (= 1.0)", fontsize=7, color="#555", ha="center")
    ax.set_ylim(0, 1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([AXIS_LABEL[a] for a in AXES], fontsize=8)
    ax.set_ylabel("Teacher-relative retention\n(student − chance) / (reference − chance)", fontsize=8)
    ax.set_title("Loss spectrum on a common scale: potion-multilingual-128M vs. LaBSE reference", fontsize=9)
    ax.legend(loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(0.0, 0.93))
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Inset: the positional component (teacher-student LOO gap) that the scale hides.
    ins = ax.inset_axes([0.80, 0.42, 0.17, 0.40])
    gaps = [tr["raw"]["syntactic_role"]["gap"], en["raw"]["syntactic_role"]["gap"]]
    ins.bar([0, 1], gaps, color=[BLUE, ORANGE], edgecolor="black", linewidth=0.5, width=0.6)
    ins.set_xticks([0, 1])
    ins.set_xticklabels(["TR", "EN"], fontsize=6.5)
    ins.set_ylim(0, 0.14)
    ins.set_title("positional gap\n(teacher − student acc.)", fontsize=6.5)
    ins.tick_params(axis="y", labelsize=6)
    for i, g in enumerate(gaps):
        ins.text(i, g + 0.004, f"+{g:.3f}", ha="center", va="bottom", fontsize=6)
    _save(fig, out_dir, "loss_spectrum")


# ---------------------------------------------------------------- Fig 2
_SYNTAX_CFG = {"syntax_v2": "TR", "syntax_v2_eu": "EU", "syntax_v2_pl": "PL",
               "syntax_v2_it": "IT", "syntax_v2_el": "EL", "syntax_v2_en": "EN"}
_TYPOLOGY = {"TR": "agglutinative", "EU": "agglutinative", "PL": "fusional", "IT": "fusional",
             "EL": "fusional", "EN": "fusional"}


def make_typology_gradient(out_dir: Path, dirs: dict[str, Path]) -> None:
    gaps, los, his = {}, {}, {}
    for cfg, lang in _SYNTAX_CFG.items():
        if cfg not in dirs:
            continue
        m = _metrics(dirs[cfg])
        gaps[lang] = float(m["mean_acc_diff_teacher_minus_student"])
        los[lang] = float(m.get("diff_ci_low", np.nan))
        his[lang] = float(m.get("diff_ci_high", np.nan))
    order = sorted(gaps, key=lambda lang: (_TYPOLOGY[lang] != "agglutinative", gaps[lang]))
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.9))
    x = np.arange(len(order))
    colors = [BLUE if _TYPOLOGY[lang] == "agglutinative" else ORANGE for lang in order]
    vals = [gaps[lang] for lang in order]
    err = np.array([[vals[i] - los[lang], his[lang] - vals[i]] for i, lang in enumerate(order)]).T
    ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5, width=0.6,
           yerr=err, capsize=3, error_kw={"lw": 0.8})
    for i, lang in enumerate(order):
        ax.text(i, his[lang] + 0.004, f"+{vals[i]:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=8)
    ax.set_ylabel("Teacher − student LOO accuracy gap", fontsize=8)
    ax.set_ylim(0, max(his.values()) + 0.03)
    ax.set_title("Syntactic-role positional loss across six languages (descriptive gradient)", fontsize=9)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color=BLUE, label="agglutinative (TR, EU)"),
                       Patch(color=ORANGE, label="fusional / analytic Indo-European (PL, IT, EL, EN)")],
              fontsize=7.5, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, out_dir, "typology_gradient")


# ---------------------------------------------------------------- Fig 3
def make_ablation_heatmap(out_dir: Path, rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.9), gridspec_kw={"width_ratios": [5, 4]})
    for ax, lang, title in zip(axes, ("tr", "en"), ("Turkish", "English"), strict=True):
        sub = [r for r in rows if r["language"] == lang]
        axes_here = [a for a in AXES if any(r.get(a) is not None for r in sub)]
        mat = np.array([[r.get(a) if r.get(a) is not None else np.nan for a in axes_here] for r in sub])
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1.05, aspect="auto")
        ax.set_xticks(range(len(axes_here)))
        ax.set_xticklabels([AXIS_LABEL[a].replace("\n", " ") for a in axes_here], fontsize=7, rotation=25, ha="right")
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([STUDENT_LABEL.get(r["student"], r["student"]) for r in sub], fontsize=7)
        if lang == "en":
            ax.yaxis.tick_right()
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if v > 0.6 else "black")
        ax.set_title(f"{title}: teacher-relative retention (vs. LaBSE)", fontsize=8.5)
    fig.subplots_adjust(wspace=0.15)
    # Place the colour bar to the right of the English panel's right-hand row labels,
    # so it cannot overlap them (fig.colorbar(ax=...) only clears the axes frame).
    fig.canvas.draw()
    label_right = axes[1].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).x1
    pos = axes[1].get_position()
    cax = fig.add_axes((label_right + 0.015, pos.y0, 0.012, pos.height))
    fig.colorbar(im, cax=cax)
    cax.tick_params(labelsize=7)
    fig.savefig(out_dir / "ablation_heatmap.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "ablation_heatmap.png", dpi=200, bbox_inches="tight")
    try:
        fig.savefig(out_dir / "ablation_heatmap.tiff", dpi=600, bbox_inches="tight",
                    pil_kwargs={"compression": "tiff_lzw"})
    except Exception as e:  # noqa: BLE001
        print(f"  (tiff skipped: {e})")
    plt.close(fig)
    print("  wrote ablation_heatmap.pdf/.png")


# ---------------------------------------------------------------- Fig 4
_DOWNSTREAM = [
    ("sts_tr_all_encoders", "sts", "STS-Tr (ρ)", "tr"),
    ("sts_en_all_encoders", "sts", "STS-B EN (ρ)", "en"),
    ("downstream_retrieval_tr_tquad", "retrieval", "TQuAD (nDCG@10)", "tr"),
    ("downstream_retrieval_tr_scifact", "retrieval", "SciFact-TR (nDCG@10)", "tr"),
    ("downstream_retrieval_en_scifact", "retrieval", "SciFact EN (nDCG@10)", "en"),
    ("downstream_clustering_tr_abstracts", "clustering", "TR abstracts (V)", "tr"),
    ("downstream_clustering_tr_columns", "clustering", "TR columns (V)", "tr"),
    ("downstream_clustering_en_20ng", "clustering", "20NG EN (V)", "en"),
]
_STUDENTS_DOWN = ["m2v_potion", "m2v_multilingual_output", "m2v_bgem3_pca256_sif", "st_static_mrl"]


def _task_scores(kind: str, m: dict) -> dict[str, float]:
    if kind == "sts":
        return {k: v["spearman"] for k, v in m["per_encoder"].items()}
    if kind == "retrieval":
        kk = f"ndcg@{m.get('k', 10)}"
        return {k: v[kk] for k, v in m["per_encoder"].items()}
    return {k: v["v_measure_mean"] for k, v in m["per_encoder"].items()}


def make_downstream_retention(out_dir: Path, dirs: dict[str, Path]) -> None:
    tasks = [(cfg, kind, label, lang) for cfg, kind, label, lang in _DOWNSTREAM if cfg in dirs]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.3), sharey=True)
    for ax, ref, ref_label in zip(axes, ("labse", "me5_base"), ("LaBSE", "multilingual-E5-base"), strict=True):
        x = np.arange(len(tasks))
        n_s = len(_STUDENTS_DOWN)
        w = 0.8 / n_s
        palette = [BLUE, "#8ecae6", ORANGE, "#6a994e"]
        for si, st in enumerate(_STUDENTS_DOWN):
            vals = []
            for cfg, kind, _, _ in tasks:
                sc = _task_scores(kind, _metrics(dirs[cfg]))
                vals.append(sc[st] / sc[ref] if st in sc and ref in sc and sc[ref] > 0 else np.nan)
            ax.bar(x - 0.4 + w * (si + 0.5), vals, w, color=palette[si], edgecolor="black",
                   linewidth=0.4, label=STUDENT_LABEL[st])
        ax.axhline(1.0, color=GREY, lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([t[2] for t in tasks], fontsize=6.5, rotation=30, ha="right")
        ax.set_title(f"Task-level retention vs. {ref_label}", fontsize=8.5)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("student score / reference score", fontsize=8)
    axes[0].set_ylim(0, 1.35)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
        fig.savefig(out_dir / f"downstream_retention.{ext}", bbox_inches="tight", **kw)
    try:
        fig.savefig(out_dir / "downstream_retention.tiff", dpi=600, bbox_inches="tight",
                    pil_kwargs={"compression": "tiff_lzw"})
    except Exception as e:  # noqa: BLE001
        print(f"  (tiff skipped: {e})")
    plt.close(fig)
    print("  wrote downstream_retention.pdf/.png")


def main() -> int:
    out_dir = HERE
    dirs = _run_dirs()
    rows = spectrum_compute({k: str(v) for k, v in dirs.items()})
    print("Fig 1: loss spectrum (common scale)")
    make_loss_spectrum(out_dir, rows, dirs)
    print("Fig 2: typology gradient")
    make_typology_gradient(out_dir, dirs)
    print("Fig 3: ablation heatmap")
    make_ablation_heatmap(out_dir, rows)
    print("Fig 4: downstream retention")
    make_downstream_retention(out_dir, dirs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
