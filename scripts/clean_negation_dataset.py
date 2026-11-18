"""Clean the v2 negation triplet dataset down to genuine lexical antonyms.

Motivation: the v2 KeNet-derived set is contaminated by
morphological-negation pairs — same-stem ``-lI``/``-sIz`` privatives such as
``bastonlu``/``bastonsuz`` ("with cane"/"without cane"). For these the "antonym"
is itself a morphological negation that shares the stem token, so the headline
``0/N`` polarity-flip result is computed over non-antonyms and the
astronomical p-values are inflated. We keep only genuine *lexical* antonyms.

Filter (a triplet is DISCARDED if any holds):
  1. positive and antonym are a same-stem ``-lI``/``-sIz`` privative pair;
  2. either member is a ``-sIz`` privative derivation (morphological negation);
  3. positive and antonym share a long common stem (other morphological paradigms).

Hand-curated v1 triplets (genuine gradable antonyms: iyi/kötü, büyük/küçük…)
are always preserved unless they trip a filter (none do).

Output: ``data/negation/tr_negation.v3_clean.json`` + a discard log.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LI = ("lı", "li", "lu", "lü")
SIZ = ("sız", "siz", "suz", "süz")


def stem_li(w: str) -> str | None:
    for s in LI:
        if w.endswith(s) and len(w) > len(s) + 1:
            return w[: -len(s)]
    return None


def stem_siz(w: str) -> str | None:
    for s in SIZ:
        if w.endswith(s) and len(w) > len(s) + 1:
            return w[: -len(s)]
    return None


def is_siz(w: str) -> bool:
    return any(w.endswith(s) for s in SIZ)


def common_prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def reason_to_discard(pos: str, ant: str) -> str | None:
    p, a = pos.lower().strip(), ant.lower().strip()
    # 1. same-stem -lI / -sIz privative pair (either direction)
    sp_li, sp_siz = stem_li(p), stem_siz(p)
    sa_li, sa_siz = stem_li(a), stem_siz(a)
    if sp_li and sa_siz and sp_li == sa_siz:
        return "same-stem -lI/-sIz privative"
    if sp_siz and sa_li and sp_siz == sa_li:
        return "same-stem -lI/-sIz privative"
    # 2. either member is a -sIz privative (morphological negation)
    if is_siz(p) or is_siz(a):
        return "-sIz privative member"
    # 3. share a long common stem (>=4 chars and >=60% of shorter word) -> morphological derivation
    cpl = common_prefix_len(p, a)
    if cpl >= 4 and cpl >= 0.6 * min(len(p), len(a)):
        return "shared stem (morphological derivation)"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/negation/tr_negation.v2.json")
    ap.add_argument("--out", default="data/negation/tr_negation.v3_clean.json")
    ap.add_argument("--log", default="data/negation/tr_negation.v3_clean.discarded.json")
    ap.add_argument("--version", default="v3-clean-2026-06-19")
    args = ap.parse_args()

    d = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    kept, discarded = [], []
    for t in d["triplets"]:
        r = reason_to_discard(t["positive"], t["antonym"])
        if r is None:
            kept.append(t)
        else:
            discarded.append({**t, "discard_reason": r})

    out = {
        "language": d["language"],
        "version": args.version,
        "source": (
            f"Cleaned from {d.get('version', 'v2')}: removed morphological-negation "
            f"contaminants (same-stem -lI/-sIz privatives, -sIz members, shared-stem "
            f"derivations). Kept {len(kept)}/{len(d['triplets'])} genuine lexical antonyms."
        ),
        "triplets": kept,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.log).write_text(
        json.dumps({"n_discarded": len(discarded), "discarded": discarded}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    from collections import Counter
    reasons = Counter(x["discard_reason"] for x in discarded)
    print(f"input triplets : {len(d['triplets'])}")
    print(f"kept (clean)   : {len(kept)}")
    print(f"discarded      : {len(discarded)}")
    for r, c in reasons.most_common():
        print(f"  - {r}: {c}")
    print(f"wrote {args.out}")
    print(f"wrote {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
