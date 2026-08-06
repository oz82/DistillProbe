"""Extract polysemic single-word literals from KeNet for polysemy probe candidates.

Outputs a list of literals that appear in ≥ min-senses distinct synsets,
prioritizing nouns. Used as input to build_polysemy_dataset.py --candidates.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kenet", required=True)
    p.add_argument("--min-senses", type=int, default=3)
    p.add_argument("--max-candidates", type=int, default=60)
    p.add_argument("--out")
    args = p.parse_args()

    tree = ET.parse(args.kenet)
    root = tree.getroot()
    lit_to_syn: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for syn in root.findall("SYNSET"):
        sid = (syn.findtext("ID") or "").strip()
        pos = (syn.findtext("POS") or "").strip()
        synonym = syn.find("SYNONYM")
        if synonym is None:
            continue
        for lit in synonym.findall("LITERAL"):
            txt = (lit.text or "").strip()
            if (
                txt and " " not in txt and "(" not in txt and "-" not in txt
                and len(txt) >= 2 and txt.isalpha()
            ):
                lit_to_syn[txt].append((sid, pos))

    polysemic: list[tuple[str, int]] = []
    for lit, syns in lit_to_syn.items():
        if len(syns) < args.min_senses:
            continue
        poses = [s[1] for s in syns]
        # majority noun
        if poses.count("n") >= len(poses) // 2:
            polysemic.append((lit, len(syns)))

    polysemic.sort(key=lambda x: -x[1])
    polysemic = polysemic[: args.max_candidates]

    print(f"polysemic literals (top {len(polysemic)}):", file=sys.stderr)
    for lit, n in polysemic[:20]:
        print(f"  {lit:18s} synsets={n}", file=sys.stderr)
    if len(polysemic) > 20:
        print(f"  …", file=sys.stderr)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            "\n".join(lit for lit, _ in polysemic) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.out}", file=sys.stderr)

    print(" ".join(lit for lit, _ in polysemic))
    return 0


if __name__ == "__main__":
    sys.exit(main())
