"""Build an STS probe dataset from a HuggingFace STS-Benchmark-style dataset.

Default is STS-Tr (`emrecan/stsb-mt-turkish`, STS-Benchmark machine-translated to
Turkish). The original English STS-Benchmark (`mteb/stsbenchmark-sts`) is the
`--language en` counterpart used by the English battery.

Usage:
    python scripts/build_sts_dataset.py --out data/sts/tr_stsb_test.json
    python scripts/build_sts_dataset.py --language en --hf-dataset mteb/stsbenchmark-sts \\
        --out data/sts/en_stsb_test.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from distillprobe.data.schemas import STSDataset, STSPair

DEFAULT_DATASET = {
    "tr": ("emrecan/stsb-mt-turkish", "STS-Benchmark machine-translated to Turkish"),
    "en": ("mteb/stsbenchmark-sts", "original English STS-Benchmark"),
}
SCORE_FIELDS = ("score", "similarity_score", "label")


def _score_field(columns: list[str]) -> str:
    for f in SCORE_FIELDS:
        if f in columns:
            return f
    raise ValueError(f"no score column among {columns}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--version", default="v1-2026-05-06")
    p.add_argument("--language", default="tr", choices=sorted(DEFAULT_DATASET))
    p.add_argument(
        "--hf-dataset", default=None, help="override the per-language default dataset id"
    )
    args = p.parse_args()

    import datasets

    default_id, default_note = DEFAULT_DATASET[args.language]
    dataset_id = args.hf_dataset or default_id
    ds = datasets.load_dataset(dataset_id, split=args.split)
    field = _score_field(ds.column_names)
    scores = [float(r[field]) for r in ds]
    # STS-B convention is [0, 5]; some mirrors ship [0, 1]-normalised scores.
    scale = 5.0 if max(scores) <= 1.0 else 1.0
    pairs = [
        STSPair(sentence1=r["sentence1"], sentence2=r["sentence2"], score=float(r[field]) * scale)
        for r in ds
    ]
    note = default_note if dataset_id == default_id else "custom dataset"
    if scale != 1.0:
        note += "; scores rescaled from [0,1] to [0,5]"
    out = STSDataset(
        language=args.language,
        version=args.version,
        source=f"{dataset_id} (split={args.split}); {note}",
        pairs=pairs,
    )
    p_out = Path(args.out)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    p_out.write_text(out.model_dump_json(indent=2), encoding="utf-8")
    print(
        f"wrote {p_out} ({len(pairs)} pairs, score field={field}, scale={scale})", file=sys.stderr
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
