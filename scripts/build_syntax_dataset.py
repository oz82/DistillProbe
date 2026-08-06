"""Build a syntactic role probe dataset from PARSEME 1.2 Turkish CUPT files.

PARSEME-tr's morpho-syntactic layer comes from UD-Turkish-IMST, so the DEPREL
column carries Universal Dependencies labels (`nsubj`, `obj`, …). For each
target lemma that appears as **both** subject and object frequently enough,
emit one record per occurrence with the surrounding sentence and the target
token index (0-based, into the non-range token list).

Usage:
    python scripts/build_syntax_dataset.py \\
        --parseme data/idiomaticity/parseme_tr/{train,dev,test}.cupt \\
        --out data/syntax/tr_syntax.v1.json \\
        --target-roles nsubj obj \\
        --target-pos NOUN PROPN \\
        --min-per-role 10 --max-per-role 60
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from distillprobe.data.parseme import iter_many_cupt
from distillprobe.data.schemas import SyntaxDataset, SyntaxItem


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--parseme", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target-roles", nargs="+", default=["nsubj", "obj"])
    p.add_argument("--target-pos", nargs="+", default=["NOUN", "PROPN"])
    p.add_argument("--min-per-role", type=int, default=10)
    p.add_argument("--max-per-role", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--version", default="v1-2026-05-05")
    p.add_argument("--language", default="tr", help="ISO code for the dataset's language field")
    p.add_argument("--source", default=None, help="override the provenance string")
    args = p.parse_args()

    target_roles = set(args.target_roles)
    target_pos = set(args.target_pos)
    rng = random.Random(args.seed)

    paths = [Path(s) for s in args.parseme]
    for path in paths:
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 2

    # Pass 1: index — lemma → role → list[(sentence_tokens, target_word_idx, sentence_text)]
    by_lemma_role: dict[str, dict[str, list[tuple[list[str], int, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    n_sents = 0
    n_hits = 0
    for sent in iter_many_cupt(paths):
        n_sents += 1
        if not sent.text:
            continue
        # Build word-level (non-range) token list and an index map
        sentence_tokens: list[str] = []
        full_to_word: dict[int, int] = {}
        for i, t in enumerate(sent.tokens):
            if t.is_range:
                continue
            full_to_word[i] = len(sentence_tokens)
            sentence_tokens.append(t.form)

        for full_idx, t in enumerate(sent.tokens):
            if t.is_range:
                continue
            if t.upos not in target_pos:
                continue
            if t.deprel not in target_roles:
                continue
            lemma = (t.lemma or "").lower()
            if not lemma or lemma == "_":
                continue
            word_idx = full_to_word[full_idx]
            by_lemma_role[lemma][t.deprel].append((sentence_tokens, word_idx, sent.text))
            n_hits += 1

    print(f"scanned {n_sents} sentences, found {n_hits} role-token hits", file=sys.stderr)

    # Filter: keep lemmas with ≥min_per_role in EACH target role
    qualifying: list[str] = []
    for lemma, roles in by_lemma_role.items():
        if all(len(roles.get(r, [])) >= args.min_per_role for r in args.target_roles):
            qualifying.append(lemma)
    print(f"qualifying lemmas: {len(qualifying)}", file=sys.stderr)

    # Subsample per (lemma, role) to max_per_role
    items: list[SyntaxItem] = []
    distribution = Counter()
    for lemma in qualifying:
        for role in args.target_roles:
            occ = by_lemma_role[lemma][role]
            if len(occ) > args.max_per_role:
                occ = rng.sample(occ, args.max_per_role)
            for sentence_tokens, target_idx, sent_text in occ:
                items.append(
                    SyntaxItem(
                        word=lemma,
                        role=role,
                        sentence_text=sent_text,
                        sentence_tokens=sentence_tokens,
                        target_token_index=target_idx,
                    )
                )
                distribution[(lemma, role)] += 1

    print(f"wrote {len(items)} items across {len(qualifying)} lemmas", file=sys.stderr)
    # Show top lemmas by combined occurrence
    top = sorted(
        qualifying,
        key=lambda l: -sum(distribution[(l, r)] for r in args.target_roles),
    )[:20]
    print("=== top 20 lemmas (by total kept occurrences) ===", file=sys.stderr)
    for lemma in top:
        per_role = " ".join(f"{r}={distribution[(lemma, r)]}" for r in args.target_roles)
        print(f"  {lemma:14s} {per_role}", file=sys.stderr)

    ds = SyntaxDataset(
        language=args.language,
        version=args.version,
        source=args.source or (
            f"PARSEME 1.2 ({args.language}) UD DEPREL annotation; "
            f"target_roles={args.target_roles}, target_pos={args.target_pos}, "
            f"min_per_role={args.min_per_role}, max_per_role={args.max_per_role}"
        ),
        target_roles=list(args.target_roles),
        items=items,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ds.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
