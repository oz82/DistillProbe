"""Serialise MTEB-style clustering and BEIR-style retrieval datasets from the Hub.

Both layouts are the ones MTEB / TR-MTEB publish, so a single script covers the
Turkish originals (`trmteb/*`) and the English counterparts (`mteb/*`).

Usage:
    python scripts/build_downstream_dataset.py clustering \\
        --hf-dataset trmteb/ts_abstract_corpus --language tr \\
        --out data/downstream/tr_clustering_abstracts.json

    python scripts/build_downstream_dataset.py retrieval \\
        --hf-dataset trmteb/tquad --language tr --qrels-split test \\
        --out data/downstream/tr_retrieval_tquad.json
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

from distillprobe.data.schemas import (
    ClusteringDataset,
    ClusteringSet,
    RetrievalDataset,
    RetrievalDoc,
    RetrievalQuery,
)


def _as_list(value) -> list:
    # Some Hub mirrors store the per-set lists as their string repr.
    if isinstance(value, str):
        return list(ast.literal_eval(value))
    return list(value)


def build_clustering(args) -> ClusteringDataset:
    import random

    import datasets

    ds = datasets.load_dataset(args.hf_dataset, split=args.split)
    sets: list[ClusteringSet] = []
    for row in ds:
        sents = [str(s) for s in _as_list(row["sentences"])]
        labels = [str(x) for x in _as_list(row["labels"])]
        if len(sents) != len(labels):
            raise ValueError(f"set with {len(sents)} sentences but {len(labels)} labels")
        sets.append(ClusteringSet(sentences=sents, labels=labels))

    note = "MTEB clustering layout"
    if all(len(set(s.labels)) == 1 for s in sets):
        # TR-MTEB stores one *class* per row rather than one clustering set per row.
        # Pool the classes and draw seeded stratified-free random subsets, which is
        # how MTEB obtains several sets (and a CI) from a single labelled corpus.
        pool = [(t, lab) for s in sets for t, lab in zip(s.sentences, s.labels, strict=True)]
        rng = random.Random(args.seed)
        size = min(args.set_size, len(pool)) if args.set_size else len(pool)
        sets = []
        for _ in range(args.n_sets):
            sample = rng.sample(pool, size)
            sets.append(
                ClusteringSet(sentences=[t for t, _ in sample], labels=[lab for _, lab in sample])
            )
        note = (
            f"rows are classes; pooled {len(pool)} texts / {len({lab for _, lab in pool})} classes "
            f"into {args.n_sets} seeded random sets of {size} (seed={args.seed})"
        )
    sizes = [len(s.sentences) for s in sets]
    ks = [len(set(s.labels)) for s in sets]
    print(
        f"{len(sets)} sets; sizes {min(sizes)}–{max(sizes)}; k {min(ks)}–{max(ks)}; {note}",
        file=sys.stderr,
    )
    return ClusteringDataset(
        language=args.language,
        version=args.version,
        source=f"{args.hf_dataset} (split={args.split}); {note}",
        sets=sets,
    )


def build_retrieval(args) -> RetrievalDataset:
    import datasets

    corpus = datasets.load_dataset(args.hf_dataset, "corpus", split="corpus")
    queries = datasets.load_dataset(args.hf_dataset, "queries", split="queries")
    qrels = datasets.load_dataset(args.hf_dataset, args.qrels_config, split=args.qrels_split)

    docs = [
        RetrievalDoc(id=str(r["_id"]), title=str(r.get("title") or ""), text=str(r["text"]))
        for r in corpus
    ]
    qs = [RetrievalQuery(id=str(r["_id"]), text=str(r["text"])) for r in queries]
    rel: dict[str, dict[str, float]] = {}
    for r in qrels:
        rel.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = float(r["score"])
    judged = sum(1 for q in qs if q.id in rel)
    print(
        f"{len(docs)} docs, {len(qs)} queries ({judged} judged), {len(rel)} qrel queries",
        file=sys.stderr,
    )
    return RetrievalDataset(
        language=args.language,
        version=args.version,
        source=f"{args.hf_dataset} (qrels={args.qrels_config}/{args.qrels_split}); BEIR layout",
        corpus=docs,
        queries=qs,
        qrels=rel,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="kind", required=True)
    for kind in ("clustering", "retrieval"):
        sp = sub.add_parser(kind)
        sp.add_argument("--hf-dataset", required=True)
        sp.add_argument("--language", required=True)
        sp.add_argument("--out", required=True)
        sp.add_argument("--version", default="v1-2026-09-14")
        if kind == "clustering":
            sp.add_argument("--split", default="test")
            sp.add_argument("--n-sets", type=int, default=10,
                            help="sets drawn when rows are classes (TR-MTEB layout)")
            sp.add_argument("--set-size", type=int, default=2000,
                            help="texts per drawn set (0 = whole pool)")
            sp.add_argument("--seed", type=int, default=42)
        else:
            sp.add_argument("--qrels-config", default="default")
            sp.add_argument("--qrels-split", default="test")
    args = p.parse_args()

    ds = build_clustering(args) if args.kind == "clustering" else build_retrieval(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ds.model_dump_json(indent=None), encoding="utf-8")
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
