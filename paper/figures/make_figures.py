"""Generate paper figures from DistillProbe results.

Two figures:
  1. loss_spectrum.{pdf,png}  — six axes ordered from preserved to lost
  2. typology_gradient.{pdf,png} — syntax T-S gap across 5 languages, grouped by typology

Both saved as PDF (for LaTeX inclusion) and PNG (for previews).

Usage:
    python paper/figures/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# IEEE rejects Type 3 fonts in submitted graphics; 42 embeds TrueType instead.
# Both figures are authored at the final printed size (IEEE two-column text
# width, 7.16 in) and included at \textwidth, so nothing is scaled down and the
# on-figure type stays at its nominal 8-9 pt — IEEE's stated legibility floor.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
matplotlib.rcParams["font.size"] = 8

TEXT_WIDTH_IN = 7.16  # IEEE two-column text width


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MANIFEST = ROOT / "results" / "_canonical_runs.txt"

# Fallback per-language syntax Teacher-Student gaps (used only if the canonical
# run manifest is absent; otherwise read live so the figure never goes stale).
_SYNTAX_GAP_FALLBACK = {"TR": 0.052, "EU": 0.070, "PL": 0.095, "IT": 0.091, "EL": 0.088}
_SYNTAX_CFG = {"syntax_v2": "TR", "syntax_v2_eu": "EU", "syntax_v2_pl": "PL",
               "syntax_v2_it": "IT", "syntax_v2_el": "EL"}


def _syntax_gaps() -> dict[str, float]:
    """Read the five syntax Teacher-Student gaps from the canonical runs."""
    if not MANIFEST.exists():
        return dict(_SYNTAX_GAP_FALLBACK)
    dirs = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].startswith("results/"):
            dirs[parts[0]] = ROOT / parts[1]
    gaps = {}
    for cfg, lang in _SYNTAX_CFG.items():
        try:
            m = json.loads((dirs[cfg] / "metrics.json").read_text(encoding="utf-8"))["metrics"]
            gaps[lang] = float(m["mean_acc_diff_teacher_minus_student"])
        except (KeyError, FileNotFoundError):
            gaps[lang] = _SYNTAX_GAP_FALLBACK[lang]
    return gaps

# ---------------------------------------------------------------- Fig 1
def make_loss_spectrum(out_dir: Path) -> None:
    """Five student-measured axes, ordered by retention of each axis's
    BoW-challenging component (high = preserved, low = structurally lost).
    The teacher-internal idiomaticity probe is NOT plotted (it is a property of
    the contextual encoder, not a student-retention value)."""
    # Tiers are assigned explicitly per axis (they are semantic judgements, not a
    # numeric threshold): the paper's text groups morphology (0.25, category-
    # dependent 0.15-0.36) with the PARTIAL axes, while the syntax bar plots the
    # positional COMPONENT, which is structurally lost.
    axes = [
        ("Cross-lingual\nalignment", 0.66, "P@1 retention\n0.66 (17 lang)", "preserved"),
        ("Polysemy\nranking", 0.34, "Spearman ρ\n0.34 leak-free", "partial"),
        ("Morphology\n(per category)", 0.25, "mean consist.\n0.25 (varies)", "partial"),
        ("Syntactic role\n(positional)", 0.05, "positional comp.\nlost (+0.05 gap)", "lost"),
        ("Negation\npolarity flip", 0.00, "0/98 flips", "lost"),
    ]
    TIER_COLOR = {"preserved": "#3a86ff", "partial": "#ff8c42", "lost": "#ff4d4d"}
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.1))
    labels = [a[0] for a in axes]
    values = [a[1] for a in axes]
    annot = [a[2] for a in axes]
    colors = [TIER_COLOR[a[3]] for a in axes]
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6, width=0.65)

    # Annotations on bars
    for bar, label in zip(bars, annot):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222",
        )

    # Y axis. Headroom leaves the top band free for the legend row so it never
    # overprints the spectrum labels.
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Retention proxy (axis-specific)", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.tick_params(axis="both", labelsize=8)

    # Title and grid
    ax.set_title(
        "DistillProbe loss spectrum: Model2Vec retention across five student-measured axes",
        fontsize=9,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    legend_patches = [
        Patch(facecolor="#3a86ff", edgecolor="black", label="Preserved"),
        Patch(facecolor="#ff8c42", edgecolor="black", label="Partial"),
        Patch(facecolor="#ff4d4d", edgecolor="black", label="Lost (structural)"),
    ]
    # Legend centred on the free top band, horizontal, so the two spectrum
    # labels keep the top corners to themselves (the earlier "upper right"
    # placement clipped "BoW-impossible" to "BoW-impo").
    ax.legend(handles=legend_patches, loc="upper center", ncol=3,
              framealpha=1.0, fontsize=8, borderpad=0.4, columnspacing=1.2)

    # Top annotation: BoW-natural <-> BoW-impossible spectrum
    ax.annotate(
        "BoW-natural",
        xy=(0.0, 0.99), xycoords=("data", "axes fraction"),
        ha="left", va="top", fontsize=8, fontstyle="italic", color="#3a86ff"
    )
    ax.annotate(
        "BoW-impossible",
        xy=(len(labels) - 1, 0.99), xycoords=("data", "axes fraction"),
        ha="right", va="top", fontsize=8, fontstyle="italic", color="#ff4d4d"
    )

    plt.tight_layout()
    fig.savefig(out_dir / "loss_spectrum.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "loss_spectrum.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "loss_spectrum.tiff", dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})  # kept for venues wanting TIFF
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2
def make_typology_gradient(out_dir: Path) -> None:
    """Syntax T-S gap across 5 languages, grouped by typology."""
    g = _syntax_gaps()
    rows = [
        ("TR", "Turkish\n(aggl.-acc)",      g["TR"], "agglutinative"),
        ("EU", "Basque\n(aggl.-erg)",       g["EU"], "agglutinative"),
        ("PL", "Polish\n(fus. Slavic)",     g["PL"], "fusional"),
        ("IT", "Italian\n(fus. Romance)",   g["IT"], "fusional"),
        ("EL", "Greek\n(fus. Hellenic)",    g["EL"], "fusional"),
    ]
    typology_color = {"agglutinative": "#3a86ff", "fusional": "#ff8c42"}

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.0))
    x = np.arange(len(rows))
    values = [r[2] for r in rows]
    colors = [typology_color[r[3]] for r in rows]
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6, width=0.65)

    # Group means
    agg_vals = [r[2] for r in rows if r[3] == "agglutinative"]
    fus_vals = [r[2] for r in rows if r[3] == "fusional"]
    agg_mean = float(np.mean(agg_vals))
    fus_mean = float(np.mean(fus_vals))

    # The group means are drawn as reference lines but labelled in the legend
    # rather than as floating text: the previous right-aligned labels landed on
    # top of Greek's bar annotation (fusional mean 0.091 vs Greek +0.088).
    ax.axhline(agg_mean, color="#3a86ff", linestyle="--", alpha=0.6, linewidth=1.2)
    ax.axhline(fus_mean, color="#ff8c42", linestyle="--", alpha=0.6, linewidth=1.2)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.0035,
            f"{v:+.3f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([r[1] for r in rows], fontsize=8)
    ax.set_ylabel("Teacher − Student LOO accuracy", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_ylim(0, max(values) * 1.45)
    ax.set_title(
        "Typology gradient (syntactic-role probe; descriptive, n.s. at n=5)",
        fontsize=9,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_patches = [
        Patch(facecolor="#3a86ff", edgecolor="black",
              label=f"Agglutinative (dashed mean {agg_mean:+.3f})"),
        Patch(facecolor="#ff8c42", edgecolor="black",
              label=f"Fusional, Indo-European (dashed mean {fus_mean:+.3f})"),
    ]
    ax.legend(handles=legend_patches, loc="upper center", ncol=2,
              framealpha=1.0, fontsize=8, borderpad=0.4, columnspacing=1.2)

    plt.tight_layout()
    fig.savefig(out_dir / "typology_gradient.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "typology_gradient.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "typology_gradient.tiff", dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})  # kept for venues wanting TIFF
    plt.close(fig)


def main() -> None:
    out_dir = HERE
    out_dir.mkdir(exist_ok=True)
    make_loss_spectrum(out_dir)
    make_typology_gradient(out_dir)
    print(f"figures written to {out_dir}")


if __name__ == "__main__":
    main()
