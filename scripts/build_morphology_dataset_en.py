"""Build the English morphology probe dataset from UD English-EWT + UniMorph.

English counterpart of `build_morphology_dataset_v2.py`. The Turkish set takes
corpus-attested (lemma, form) pairs from the UD layer of PARSEME-tr, keeps one
inflectional category per token via FEATS, and validates that the form is a
*regular* suffixation of the lemma (vowel harmony + consonant assimilation).
Here the same protocol runs on UD English-EWT: FEATS selects the category, an
orthographic validator keeps regular suffixations only (``cat/cats``,
``stop/stopped``, ``try/tries``) and rejects suppletive or irregular forms
(``go/went``, ``child/children``), and every kept pair must also occur in UniMorph
`eng` with the matching tag, which filters treebank tagging errors.

English has far fewer productive inflections than Turkish; the six categories
below are what the language offers, so this axis is *expected* to look different
(near-ceiling suffix regularity) — that contrast is the point of the replication.

Usage:
    python scripts/build_morphology_dataset_en.py \\
        --conllu data/ud/UD_English-EWT/en_ewt-ud-{train,dev,test}.conllu \\
        --unimorph data/morphology/unimorph_eng/eng \\
        --out data/morphology/en_categories.v1.json
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import OrderedDict
from pathlib import Path

from distillprobe.data.parseme import iter_many_cupt
from distillprobe.data.schemas import MorphologyCategory, MorphologyDataset, StemFormPair

VOWELS = "aeiou"
SIBILANT_ENDINGS = ("s", "x", "z", "ch", "sh")


def _doubled(lemma: str) -> str:
    return lemma + lemma[-1]


def _y_to_i(lemma: str) -> str | None:
    if len(lemma) > 2 and lemma.endswith("y") and lemma[-2] not in VOWELS:
        return lemma[:-1] + "i"
    return None


def v_s_suffix(lemma: str, form: str) -> bool:
    """Regular -s / -es / -ies (noun plural and 3sg present share this rule)."""
    if form == lemma + "s":
        return True
    if form == lemma + "es" and (lemma.endswith(SIBILANT_ENDINGS) or lemma.endswith("o")):
        return True
    yi = _y_to_i(lemma)
    return yi is not None and form == yi + "es"


def v_ed(lemma: str, form: str) -> bool:
    if form == lemma + "ed":
        return True
    if lemma.endswith("e") and form == lemma + "d":
        return True
    if form == _doubled(lemma) + "ed":
        return True
    yi = _y_to_i(lemma)
    return yi is not None and form == yi + "ed"


def v_ing(lemma: str, form: str) -> bool:
    if form == lemma + "ing":
        return True
    if lemma.endswith("e") and not lemma.endswith("ee") and form == lemma[:-1] + "ing":
        return True
    return form == _doubled(lemma) + "ing"


def _v_degree(lemma: str, form: str, suffix: str) -> bool:
    if form == lemma + suffix:
        return True
    if lemma.endswith("e") and form == lemma + suffix[1:]:
        return True
    if form == _doubled(lemma) + suffix:
        return True
    yi = _y_to_i(lemma)
    return yi is not None and form == yi + suffix


def v_er(lemma: str, form: str) -> bool:
    return _v_degree(lemma, form, "er")


def v_est(lemma: str, form: str) -> bool:
    return _v_degree(lemma, form, "est")


# Each category: FEATS predicate on a UPOS, orthographic validator, UniMorph tag.
CATEGORIES = [
    {
        "id": "plural_s",
        "label": "noun plural -s",
        "upos": {"NOUN"},
        "feats": {"Number": "Plur"},
        "validator": v_s_suffix,
        "unimorph": "N;PL",
        "description": "Regular noun plural -s/-es/-ies (Number=Plur), UD English-EWT.",
    },
    {
        "id": "past_ed",
        "label": "verb past -ed",
        "upos": {"VERB"},
        "feats": {"Tense": "Past", "VerbForm": "Fin"},
        "validator": v_ed,
        "unimorph": "V;PST",
        "description": "Regular past tense -ed/-d/-ied with consonant doubling (Tense=Past, VerbForm=Fin).",
    },
    {
        "id": "participle_ing",
        "label": "verb -ing",
        "upos": {"VERB"},
        "feats_any": ({"VerbForm": "Ger"}, {"VerbForm": "Part", "Tense": "Pres"}),
        "validator": v_ing,
        "unimorph": "V;V.PTCP;PRS",
        "description": "Present participle / gerund -ing with e-deletion and doubling (VerbForm=Ger or Part+Tense=Pres).",
    },
    {
        "id": "third_singular_s",
        "label": "verb 3sg present -s",
        "upos": {"VERB"},
        "feats": {"Tense": "Pres", "Person": "3", "Number": "Sing"},
        "validator": v_s_suffix,
        "unimorph": "V;PRS;3;SG",
        "description": "Third-person singular present -s/-es/-ies (Tense=Pres, Person=3, Number=Sing).",
    },
    {
        "id": "comparative_er",
        "label": "adjective comparative -er",
        "upos": {"ADJ", "ADV"},
        "feats": {"Degree": "Cmp"},
        "validator": v_er,
        "unimorph": "ADJ;CMPR",
        "description": "Synthetic comparative -er/-ier (Degree=Cmp); periphrastic 'more X' is not a form.",
    },
    {
        "id": "superlative_est",
        "label": "adjective superlative -est",
        "upos": {"ADJ", "ADV"},
        "feats": {"Degree": "Sup"},
        "validator": v_est,
        "unimorph": "ADJ;SPRL",
        "description": "Synthetic superlative -est/-iest (Degree=Sup).",
    },
]


def feats_match(feats: dict[str, str], required: dict[str, str]) -> bool:
    return all(feats.get(k) == v for k, v in required.items())


def category_matches(cat: dict, upos: str, feats: dict[str, str]) -> bool:
    if upos not in cat["upos"]:
        return False
    if "feats_any" in cat:
        return any(feats_match(feats, req) for req in cat["feats_any"])
    return feats_match(feats, cat["feats"])


def load_unimorph(path: Path) -> set[tuple[str, str, str]]:
    """Return {(lemma, form, tag)} from a UniMorph TSV (lemma \\t form \\t tag)."""
    out: set[tuple[str, str, str]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 3:
                continue
            out.add((cols[0].lower(), cols[1].lower(), cols[2]))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--conllu", nargs="+", required=True, help="UD CoNLL-U files")
    p.add_argument("--unimorph", required=True, help="UniMorph eng TSV")
    p.add_argument("--out", required=True)
    p.add_argument("--max-per-category", type=int, default=200)
    p.add_argument("--min-stem-len", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--version", default="v1-en-corpus-2026-09-14")
    args = p.parse_args()

    unimorph = load_unimorph(Path(args.unimorph))
    print(f"UniMorph eng: {len(unimorph)} (lemma, form, tag) triples", file=sys.stderr)

    paths = [Path(s) for s in args.conllu]
    by_cat: dict[str, OrderedDict[tuple[str, str], None]] = {
        c["id"]: OrderedDict() for c in CATEGORIES
    }
    rejected = {"validator": 0, "unimorph": 0}
    n_sents = 0
    for sent in iter_many_cupt(paths):
        n_sents += 1
        for tok in sent.tokens:
            if tok.is_range or not tok.lemma or tok.lemma == "_":
                continue
            lemma = tok.lemma.lower()
            form = tok.form.lower()
            if not lemma.isalpha() or not form.isalpha() or lemma == form:
                continue
            if len(lemma) < args.min_stem_len:
                continue
            feats = tok.feats_dict()
            if not feats:
                continue
            for cat in CATEGORIES:
                if not category_matches(cat, tok.upos, feats):
                    continue
                if not cat["validator"](lemma, form):
                    rejected["validator"] += 1
                    break
                if (lemma, form, cat["unimorph"]) not in unimorph:
                    rejected["unimorph"] += 1
                    break
                by_cat[cat["id"]][(lemma, form)] = None
                break  # one category per token

    print(
        f"scanned {n_sents} sentences; rejected by validator={rejected['validator']}, "
        f"by UniMorph cross-check={rejected['unimorph']}",
        file=sys.stderr,
    )

    distractor_pool: list[str] = []
    seen: set[str] = set()
    for sent in iter_many_cupt(paths):
        for tok in sent.tokens:
            if tok.is_range or tok.upos != "NOUN":
                continue
            lem = (tok.lemma or "").lower()
            if lem and lem != "_" and len(lem) >= 3 and lem.isalpha() and lem not in seen:
                seen.add(lem)
                distractor_pool.append(lem)
                if len(distractor_pool) >= 60:
                    break
        if len(distractor_pool) >= 60:
            break

    rng = random.Random(args.seed)
    categories: list[MorphologyCategory] = []
    for cat in CATEGORIES:
        items = list(by_cat[cat["id"]].keys())
        if len(items) > args.max_per_category:
            items = rng.sample(items, args.max_per_category)
        print(f"  {cat['id']:20s} {len(items):4d} pairs", file=sys.stderr)
        categories.append(
            MorphologyCategory(
                id=cat["id"],
                label=cat["label"],
                description=cat["description"],
                pairs=[StemFormPair(stem=lemma, form=form) for lemma, form in items],
            )
        )

    ds = MorphologyDataset(
        language="en",
        version=args.version,
        categories=categories,
        distractor_pool=distractor_pool,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ds.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote {out}; total pairs: {sum(len(c.pairs) for c in categories)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
