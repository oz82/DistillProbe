"""Augment a polysemy v1 dataset with sense-assigned PARSEME Turkish sentences.

For each polysemic word, scan the PARSEME 1.2 Turkish CUPT corpus for sentences
where the word appears (lemma match). Encode candidates with LaBSE; for each
TDK sense, build a seed prototype = mean(LaBSE(definition) + LaBSE(existing
examples)). Each candidate is assigned to its argmax sense; the top-k highest-
cosine candidates per sense (subject to `--min-cos`) are appended to the
sense's example list. Output: v2 JSON.

Usage:
    python scripts/augment_polysemy_with_parseme.py \\
        --in  data/polysemy/tr_polysemy.v1.json \\
        --out data/polysemy/tr_polysemy.v2.json \\
        --parseme data/idiomaticity/parseme_tr/train.cupt \\
                  data/idiomaticity/parseme_tr/dev.cupt \\
                  data/idiomaticity/parseme_tr/test.cupt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from distillprobe.data.parseme import iter_many_cupt
from distillprobe.data.schemas import PolysemyDataset


def index_parseme_by_lemma(paths: list[Path], targets: list[str]) -> dict[str, list[str]]:
    """For each target lemma (lower-cased), collect PARSEME sentence texts whose
    tokens include that lemma."""
    target_set = {t.lower() for t in targets}
    index: dict[str, list[str]] = {t: [] for t in target_set}
    n_sents = 0
    for sent in iter_many_cupt(paths):
        n_sents += 1
        if not sent.text:
            continue
        seen_here: set[str] = set()
        for tok in sent.tokens:
            if tok.is_range:
                continue
            lem = (tok.lemma or "").lower()
            if lem in target_set and lem not in seen_here:
                index[lem].append(sent.text)
                seen_here.add(lem)
    print(f"scanned {n_sents} PARSEME sentences", file=sys.stderr)
    return index


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return v / n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--parseme", nargs="+", required=True)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--min-cos", type=float, default=0.30)
    p.add_argument("--max-candidates", type=int, default=200,
                   help="cap candidate PARSEME sentences scanned per word")
    p.add_argument("--teacher", default="sentence-transformers/LaBSE")
    p.add_argument("--version", default="v2-2026-05-05")
    args = p.parse_args()

    raw = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    ds = PolysemyDataset(**raw)
    target_lemmas = [w.word for w in ds.words]
    print(f"target lemmas ({len(target_lemmas)}): {target_lemmas}", file=sys.stderr)

    print("indexing PARSEME by lemma...", file=sys.stderr)
    index = index_parseme_by_lemma([Path(s) for s in args.parseme], target_lemmas)
    for w_lemma, sents in sorted(index.items(), key=lambda x: -len(x[1])):
        print(f"  {w_lemma:10s} {len(sents):5d} sentences", file=sys.stderr)

    # Lazy-load LaBSE only after indexing (so we get the report fast)
    print(f"loading teacher: {args.teacher}", file=sys.stderr)
    from distillprobe.encoders.transformer_encoder import TransformerEncoder

    enc = TransformerEncoder(args.teacher)

    n_added_total = 0
    n_assigned_skipped_low_cos = 0

    for word in ds.words:
        candidates = index.get(word.word.lower(), [])
        if not candidates:
            print(f"  [{word.word}] no PARSEME candidates — leaving unchanged", file=sys.stderr)
            continue
        if len(candidates) > args.max_candidates:
            candidates = candidates[: args.max_candidates]

        cand_vecs = enc.encode_many(candidates)
        cand_norm = _normalize(cand_vecs)

        # Per-sense seed prototype from definition + existing examples
        seed_protos: list[np.ndarray | None] = []
        for s in word.senses:
            texts: list[str] = []
            if s.definition:
                texts.append(s.definition)
            texts.extend(s.examples)
            if not texts:
                seed_protos.append(None)
                continue
            v = enc.encode_many(texts).mean(axis=0)
            seed_protos.append(_normalize(v.reshape(1, -1))[0])

        valid_indices = [i for i, sp in enumerate(seed_protos) if sp is not None]
        if not valid_indices:
            continue
        seeds_arr = np.stack([seed_protos[i] for i in valid_indices], axis=0)
        sim_matrix = cand_norm @ seeds_arr.T  # (Nc, Nv)
        argmax_local = sim_matrix.argmax(axis=1)  # values in [0, Nv)

        per_sense_added = [0] * len(word.senses)
        for local_idx, global_idx in enumerate(valid_indices):
            mask = argmax_local == local_idx
            if not mask.any():
                continue
            cand_idx_arr = np.where(mask)[0]
            cand_sims = sim_matrix[cand_idx_arr, local_idx]
            order = np.argsort(-cand_sims)
            taken = 0
            for o in order:
                if taken >= args.top_k:
                    break
                cos = float(cand_sims[o])
                if cos < args.min_cos:
                    n_assigned_skipped_low_cos += 1
                    continue
                cidx = int(cand_idx_arr[o])
                word.senses[global_idx].examples.append(candidates[cidx])
                taken += 1
                n_added_total += 1
            per_sense_added[global_idx] = taken

        print(
            f"  [{word.word}] candidates={len(candidates):3d}, "
            f"per-sense added: {per_sense_added}, "
            f"per-sense total now: {[len(s.examples) for s in word.senses]}",
            file=sys.stderr,
        )

    print("", file=sys.stderr)
    print(f"total examples added: {n_added_total}", file=sys.stderr)
    print(f"skipped (cos < {args.min_cos}): {n_assigned_skipped_low_cos}", file=sys.stderr)

    ds = ds.model_copy(
        update={
            "version": args.version,
            "source": ds.source + "; augmented with PARSEME 1.2 TR via LaBSE argmax-assignment",
        }
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ds.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
