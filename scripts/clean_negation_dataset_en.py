"""Clean the English v2 negation triplets down to genuine lexical antonyms.

English counterpart of `clean_negation_dataset.py`. The Turkish contaminants were
same-stem ``-lI``/``-sIz`` privative pairs; in English the morphological-negation
contaminants are prefixal (``happy``/``unhappy``, ``moral``/``immoral``) and the
``-ful``/``-less`` pair (``careful``/``careless``). For these the "antonym" shares
the stem token with the positive word, so a bag-of-words encoder is not being
tested on lexical antonymy at all.

Filter (a triplet is DISCARDED if any holds):
  1. antonym == negative prefix + positive (or vice versa): un-, in-, im-, il-,
     ir-, non-, dis-, a- (the a- prefix only on stems of >= 4 letters);
  2. either member is a productive morphological negation on its own: ``un-``/``non-``
     + a WordNet adjective, or ``-less`` on a WordNet noun/verb stem;
  3. positive and antonym share a long common stem (>= 4 chars and >= 60% of the
     shorter word) — same heuristic as the Turkish cleaner;
  4. positive and antonym share a long common *suffix* (>= 5 chars and >= 60%),
     i.e. a prefix-swap derivation such as ``ipsilateral``/``contralateral``.

Output: ``data/negation/en_negation.v3_clean.json`` + a discard log.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

NEG_PREFIXES = ("un", "in", "im", "il", "ir", "non", "dis", "a")
PRODUCTIVE_PREFIXES = ("un", "non")


def common_prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def _is_prefixed_negation_of(longer: str, shorter: str) -> bool:
    for pre in NEG_PREFIXES:
        if longer == pre + shorter:
            return pre != "a" or len(shorter) >= 4
    return False


def _wn_has(word: str, pos: str) -> bool:
    from nltk.corpus import wordnet as wn

    return bool(wn.synsets(word, pos))


def is_morphological_negation(w: str) -> bool:
    for pre in PRODUCTIVE_PREFIXES:
        if w.startswith(pre) and len(w) - len(pre) >= 3 and _wn_has(w[len(pre) :], "a"):
            return True
    if w.endswith("less") and len(w) > 6:
        stem = w[:-4]
        return _wn_has(stem, "n") or _wn_has(stem, "v")
    return False


def reason_to_discard(pos: str, ant: str) -> str | None:
    p, a = pos.lower().strip(), ant.lower().strip()
    if _is_prefixed_negation_of(p, a) or _is_prefixed_negation_of(a, p):
        return "same-stem negative-prefix pair"
    if is_morphological_negation(p) or is_morphological_negation(a):
        return "morphological-negation member (un-/non- or -less)"
    cpl = common_prefix_len(p, a)
    if cpl >= 4 and cpl >= 0.6 * min(len(p), len(a)):
        return "shared stem (morphological derivation)"
    # English derives antonyms by prefix swap too (ipsilateral/contralateral,
    # endogamous/exogamous), which leaves a shared *suffix* stem instead.
    csl = common_prefix_len(p[::-1], a[::-1])
    if csl >= 5 and csl >= 0.6 * min(len(p), len(a)):
        return "shared stem (prefix-swap derivation)"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/negation/en_negation.v2.json")
    ap.add_argument("--out", default="data/negation/en_negation.v3_clean.json")
    ap.add_argument("--log", default="data/negation/en_negation.v3_clean.discarded.json")
    ap.add_argument("--version", default="v3-clean-en-2026-09-14")
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
            f"contaminants (negative-prefix same-stem pairs, un-/non-/-less members, "
            f"shared-stem derivations). Kept {len(kept)}/{len(d['triplets'])} genuine lexical antonyms."
        ),
        "triplets": kept,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.log).write_text(
        json.dumps(
            {"n_discarded": len(discarded), "discarded": discarded}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

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
