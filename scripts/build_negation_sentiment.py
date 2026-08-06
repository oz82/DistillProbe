"""Build a negation-controlled Turkish sentiment downstream dataset from TRSAv1.

Goal: a per-axis downstream test of the negation finding. We split binary
(Positive/Negative) reviews into a NEGATION subset and a NO-NEGATION subset, and
within EACH subset balance Positive/Negative 50/50 so the label prior is identical
across subsets. Any accuracy gap between subsets then reflects handling of
negation, not a label-distribution confound.

Source: maydogan/TRSAv1 (HuggingFace). Output: data/sts/tr_negation_sentiment.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Turkish negation markers. Explicit lexical negators (high precision) plus a
# conservative verbal-negation suffix pattern (-mA- + common tense/person endings).
EXPLICIT = re.compile(
    r"\b(değil\w*|yok\w*|hiç\w*|asla|olmadı\w*|olmaz\w*|olmamış\w*)\b", re.IGNORECASE
)
VERBAL_NEG = re.compile(
    r"\b\w*?m[ae](dı|di|du|dü|yor|z|m|sın|sin|sun|sün|malı|meli|mış|miş|muş|müş)\b",
    re.IGNORECASE,
)


def has_negation(text: str) -> bool:
    return bool(EXPLICIT.search(text)) or bool(VERBAL_NEG.search(text))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sts/tr_negation_sentiment.json")
    ap.add_argument("--per-cell", type=int, default=750,
                    help="examples per (subset x label) cell; total = 4 x this")
    ap.add_argument("--min-words", type=int, default=3)
    ap.add_argument("--max-words", type=int, default=40)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("maydogan/TRSAv1")["train"]

    # Buckets: (has_neg, label) -> list of texts, deduplicated
    buckets: dict[tuple[bool, int], list[str]] = {
        (False, 0): [], (False, 1): [], (True, 0): [], (True, 1): []
    }
    seen: set[str] = set()
    label_map = {"Positive": 1, "Negative": 0}

    for row in ds:
        score = row["score"]
        if score not in label_map:
            continue  # drop Neutral
        text = (row["review"] or "").strip()
        nw = len(text.split())
        if nw < args.min_words or nw > args.max_words:
            continue
        key_text = text.lower()
        if key_text in seen:
            continue
        seen.add(key_text)
        lab = label_map[score]
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
        "language": "tr",
        "version": "negsent-2026-06-20",
        "source": "maydogan/TRSAv1 (HuggingFace); binary Pos/Neg; negation-controlled "
                  "(Pos/Neg balanced 50/50 within each negation subset).",
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
