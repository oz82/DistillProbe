"""Build a negation-controlled English sentiment downstream dataset (Amazon polarity).

English counterpart of `build_negation_sentiment.py` (TRSAv1). Same design: binary
reviews split into a NEGATION and a NO-NEGATION subset, each balanced 50/50 on the
label so an accuracy gap between subsets reflects negation handling only.

The negation detector differs from the Turkish one by necessity: English marks
negation with free words (`not`, `n't`, `never`, ...) and has no verbal negation
suffix, so there is no analogue of the Turkish `-mA-` pattern.

Source: fancyzhx/amazon_polarity (HuggingFace), test split (400k reviews).
Output: data/sts/en_negation_sentiment.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

NEGATION = re.compile(
    r"(\b(not|no|never|none|nothing|nobody|nowhere|neither|nor|without|hardly|barely|"
    r"scarcely|cannot)\b|n't\b)",
    re.IGNORECASE,
)


def has_negation(text: str) -> bool:
    return bool(NEGATION.search(text))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sts/en_negation_sentiment.json")
    ap.add_argument("--split", default="test")
    ap.add_argument(
        "--per-cell",
        type=int,
        default=750,
        help="examples per (subset x label) cell; total = 4 x this",
    )
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--max-words", type=int, default=40)
    args = ap.parse_args()

    from datasets import load_dataset

    # Streaming: the cells fill after a few thousand reviews, so there is no need to
    # materialise the 400k-row split (or its 1.3 GB train sibling) locally.
    ds = load_dataset("fancyzhx/amazon_polarity", split=args.split, streaming=True)

    buckets: dict[tuple[bool, int], list[str]] = {
        (False, 0): [],
        (False, 1): [],
        (True, 0): [],
        (True, 1): [],
    }
    seen: set[str] = set()

    for row in ds:
        lab = int(row["label"])  # 0 = negative, 1 = positive
        text = (row["content"] or "").strip()
        nw = len(text.split())
        if nw < args.min_words or nw > args.max_words:
            continue
        key_text = text.lower()
        if key_text in seen:
            continue
        seen.add(key_text)
        neg = has_negation(text)
        cell = buckets[(neg, lab)]
        if len(cell) < args.per_cell:
            cell.append(text)
        if all(len(v) >= args.per_cell for v in buckets.values()):
            break

    items = []
    for (neg, lab), texts in buckets.items():
        for t in texts:
            items.append({"text": t, "label": lab, "has_negation": neg})

    sizes = {f"neg={k[0]},label={k[1]}": len(v) for k, v in buckets.items()}
    out = {
        "language": "en",
        "version": "negsent-en-2026-09-14",
        "source": f"fancyzhx/amazon_polarity (HuggingFace, split={args.split}); binary Pos/Neg; "
        "negation-controlled (Pos/Neg balanced 50/50 within each negation subset); "
        "negation detected by free-word negators (not, n't, never, no, ...).",
        "cell_sizes": sizes,
        "items": items,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {outp}: {len(items)} items; cells {sizes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
