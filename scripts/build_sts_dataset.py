"""Build STS-Tr (Turkish STS-Benchmark MT) probe dataset.

Downloads `emrecan/stsb-mt-turkish` from HuggingFace and serializes the chosen
split into the JSON form expected by the STSModule.

Usage:
    python scripts/build_sts_dataset.py --out data/sts/tr_stsb_test.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from distillprobe.data.schemas import STSDataset, STSPair


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test", choices=["train", "validation", "test"])
    p.add_argument("--version", default="v1-2026-05-06")
    args = p.parse_args()

    import datasets

    ds = datasets.load_dataset("emrecan/stsb-mt-turkish", split=args.split)
    pairs = [
        STSPair(sentence1=r["sentence1"], sentence2=r["sentence2"], score=float(r["score"]))
        for r in ds
    ]
    out = STSDataset(
        language="tr",
        version=args.version,
        source=f"emrecan/stsb-mt-turkish (split={args.split}); STS-Benchmark machine-translated to Turkish",
        pairs=pairs,
    )
    p_out = Path(args.out)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    p_out.write_text(out.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote {p_out} ({len(pairs)} pairs)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
