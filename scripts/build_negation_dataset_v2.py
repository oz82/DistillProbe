"""Build extended negation triplet dataset from KeNet adjective ANTONYM relations.

Parses `data/kenet/turkish_wordnet.xml` (StarlangSoftware/TurkishWordNet),
finds all (synset_a, synset_b) pairs where both are adjectives (POS=a) and
linked by an ANTONYM relation, then generates (positive, "positive değil",
antonym) triplets — one per ordered pair, deduplicated.

Output: combines KeNet-derived triplets with the v1 hand-curated list (25)
into a single larger v2 dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from distillprobe.data.schemas import NegationDataset, NegationTriplet


def parse_kenet(path: Path) -> dict:
    """Return dict[synset_id] -> {pos, literals, antonyms (list of synset_ids), def}."""
    tree = ET.parse(path)
    root = tree.getroot()
    out = {}
    for syn in root.findall("SYNSET"):
        sid = (syn.findtext("ID") or "").strip()
        if not sid:
            continue
        pos = (syn.findtext("POS") or "").strip()
        defn = (syn.findtext("DEF") or "").strip()
        # literals
        literals = []
        synonym = syn.find("SYNONYM")
        if synonym is not None:
            for lit in synonym.findall("LITERAL"):
                txt = (lit.text or "").strip()
                if txt:
                    literals.append(txt)
        # antonyms
        antonyms = []
        for sr in syn.findall("SR"):
            rel_type = (sr.findtext("TYPE") or "").strip().upper()
            if rel_type != "ANTONYM":
                continue
            # target id is the text content of <SR> before <TYPE>
            target = (sr.text or "").strip()
            if target:
                antonyms.append(target)
        out[sid] = {"pos": pos, "literals": literals, "antonyms": antonyms, "def": defn}
    return out


def is_clean_word(s: str) -> bool:
    """Single-word literal, alphabetic Turkish characters only, length ≥ 2."""
    if not s or " " in s or "(" in s or "-" in s or "/" in s:
        return False
    if len(s) < 2:
        return False
    return all(c.isalpha() for c in s)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kenet", required=True)
    p.add_argument("--seed-v1", required=True, help="path to v1 hand-curated dataset")
    p.add_argument("--out", required=True)
    p.add_argument("--max-kenet-triplets", type=int, default=200)
    p.add_argument("--version", default="v2-kenet-2026-05-06")
    args = p.parse_args()

    print("parsing KeNet…", file=sys.stderr)
    syns = parse_kenet(Path(args.kenet))
    print(f"  {len(syns)} synsets", file=sys.stderr)

    # Extract adjective ANTONYM pairs
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str]] = []
    for sid, s in syns.items():
        if s["pos"] != "a":
            continue
        src_words = [w for w in s["literals"] if is_clean_word(w)]
        if not src_words:
            continue
        for ant_id in s["antonyms"]:
            tgt = syns.get(ant_id)
            if tgt is None or tgt["pos"] != "a":
                continue
            tgt_words = [w for w in tgt["literals"] if is_clean_word(w)]
            if not tgt_words:
                continue
            # one pair per (src, tgt) with first words; dedupe symmetrically
            for src in src_words[:2]:
                for tgt_w in tgt_words[:2]:
                    key = tuple(sorted([src.lower(), tgt_w.lower()]))
                    if key in seen or src.lower() == tgt_w.lower():
                        continue
                    seen.add(key)
                    pairs.append((src.lower(), tgt_w.lower()))

    print(f"  KeNet adjective ANTONYM pairs (deduped): {len(pairs)}", file=sys.stderr)
    if len(pairs) > args.max_kenet_triplets:
        import random

        rng = random.Random(42)
        pairs = rng.sample(pairs, args.max_kenet_triplets)
        print(f"  subsampled to {len(pairs)}", file=sys.stderr)

    # Combine with v1 hand-curated triplets
    v1 = json.loads(Path(args.seed_v1).read_text(encoding="utf-8"))
    v1_keys = {(t["positive"].lower(), t["antonym"].lower()) for t in v1["triplets"]}
    v1_keys |= {(t["antonym"].lower(), t["positive"].lower()) for t in v1["triplets"]}

    triplets: list[NegationTriplet] = []
    # Add v1 triplets first (preserve hand-curated)
    for t in v1["triplets"]:
        triplets.append(NegationTriplet(**t))
    # Add KeNet triplets, skipping duplicates with v1
    n_added = 0
    for src, tgt in pairs:
        if (src, tgt) in v1_keys or (tgt, src) in v1_keys:
            continue
        triplets.append(
            NegationTriplet(
                positive=src,
                negation=f"{src} değil",
                antonym=tgt,
                note="from KeNet adjective ANTONYM",
            )
        )
        n_added += 1

    print(f"  triplets total: {len(triplets)} (v1: {len(v1['triplets'])}, +KeNet: {n_added})", file=sys.stderr)

    out_ds = NegationDataset(
        language="tr",
        version=args.version,
        source=(
            f"v1 hand-curated ({len(v1['triplets'])} triplets) + KeNet adjective ANTONYM "
            f"({n_added} new triplets, deduped vs v1)"
        ),
        triplets=triplets,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(out_ds.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
