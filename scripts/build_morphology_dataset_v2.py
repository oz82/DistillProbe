"""Build extended morphology dataset from PARSEME-tr UD layer.

Iterates PARSEME-tr CUPT files, extracts (lemma, form) pairs by Universal
Dependencies morphological features (Number=Plur for plural_lAr, Case=Loc for
locative_DA, ...). Filters strictly by Turkish vowel harmony + suffix pattern
to throw out broken-lemma pairs.

Output: corpus-derived dataset with same schema as v1; intended to be larger
(~10× v1) but inherits PARSEME lemma quality concerns. Used for replication
under expanded N.

Stem validity (v3): the PARSEME-tr lemma column is machine-produced and sometimes
holds a truncated stem (`çapk` for `çapkın`, `pua` for `puan`) or an already
inflected word (`bireyler`, `gözlerimiz`). With `--lexicon <KeNet XML>` a stem is
kept only if it is attested as a bare word form somewhere in the corpus OR is a
single-word KeNet literal; everything else is dropped before sampling.
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from collections import OrderedDict, defaultdict
from pathlib import Path


def normalize_tr(s: str) -> str:
    """NFC-normalize and drop the redundant U+0307 combining dot that Unicode
    lowercasing of Turkish 'İ' produces (e.g. 'i̇şsiz' -> 'işsiz'). Keeps
    legitimate words while letting the isalpha() filter reject true garbage."""
    s = unicodedata.normalize("NFC", s)
    return s.replace("̇", "")

from distillprobe.data.parseme import iter_many_cupt
from distillprobe.data.schemas import (
    MorphologyCategory,
    MorphologyDataset,
    StemFormPair,
)

VOWELS = "aeıioöuü"
BACK_UNROUNDED = "aı"
BACK_ROUNDED = "ou"
BACK_VOWELS = BACK_UNROUNDED + BACK_ROUNDED
FRONT_UNROUNDED = "ei"
FRONT_ROUNDED = "öü"
FRONT_VOWELS = FRONT_UNROUNDED + FRONT_ROUNDED
VOICELESS = "pçtkfsşh"


def last_vowel(word: str) -> str | None:
    for c in reversed(word.lower()):
        if c in VOWELS:
            return c
    return None


def _starts_match(lemma: str, form: str) -> bool:
    return form.lower().startswith(lemma.lower())


# ---------------------------------------------------------------- validators


def v_plural(lemma: str, form: str) -> bool:
    """form == lemma + lar/ler with vowel harmony."""
    lemma_l = lemma.lower()
    form_l = form.lower()
    if not form_l.startswith(lemma_l):
        return False
    suf = form_l[len(lemma_l) :]
    if suf not in ("lar", "ler"):
        return False
    lv = last_vowel(lemma_l)
    if lv is None:
        return False
    return (suf == "lar" and lv in BACK_VOWELS) or (suf == "ler" and lv in FRONT_VOWELS)


def _validate_case(lemma: str, form: str, allowed_suffixes: set[str]) -> bool:
    """Generic helper: form == lemma + one_of(allowed_suffixes), with vowel-harmony
    and consonant-assimilation consistency."""
    lemma_l = lemma.lower()
    form_l = form.lower()
    if not form_l.startswith(lemma_l):
        return False
    suf = form_l[len(lemma_l) :]
    if suf not in allowed_suffixes:
        return False
    lv = last_vowel(lemma_l)
    if lv is None:
        return False
    last_char = lemma_l[-1] if lemma_l else ""
    expected_d = "t" if last_char in VOICELESS else "d"
    # Suffixes here begin with d or t (locative/ablative)
    if suf and suf[0] in "dt" and suf[0] != expected_d:
        return False
    # Vowel: the *first* vowel of suffix must match harmony
    suf_v = next((c for c in suf if c in VOWELS), None)
    if suf_v is None:
        return False
    return (suf_v in BACK_UNROUNDED and lv in BACK_UNROUNDED + BACK_ROUNDED) or (
        suf_v in FRONT_UNROUNDED and lv in FRONT_UNROUNDED + FRONT_ROUNDED
    )


def v_locative(lemma: str, form: str) -> bool:
    return _validate_case(lemma, form, {"da", "de", "ta", "te"})


def v_ablative(lemma: str, form: str) -> bool:
    return _validate_case(lemma, form, {"dan", "den", "tan", "ten"})


def v_dative(lemma: str, form: str) -> bool:
    """form == lemma + a/e (consonant-final) or + ya/ye (vowel-final + buffer y)."""
    lemma_l = lemma.lower()
    form_l = form.lower()
    if not form_l.startswith(lemma_l):
        return False
    suf = form_l[len(lemma_l) :]
    if suf not in ("a", "e", "ya", "ye"):
        return False
    lv = last_vowel(lemma_l)
    if lv is None:
        return False
    last_char = lemma_l[-1] if lemma_l else ""
    is_vowel_final = last_char in VOWELS
    suf_v = suf[-1]
    if is_vowel_final:
        if suf in ("a", "e"):  # missing buffer y
            return False
    else:
        if suf in ("ya", "ye"):  # spurious buffer
            return False
    return (suf_v == "a" and lv in BACK_VOWELS) or (suf_v == "e" and lv in FRONT_VOWELS)


def v_genitive(lemma: str, form: str) -> bool:
    """form == lemma + ın/in/un/ün or nın/nin/nun/nün."""
    lemma_l = lemma.lower()
    form_l = form.lower()
    if not form_l.startswith(lemma_l):
        return False
    suf = form_l[len(lemma_l) :]
    if suf not in ("ın", "in", "un", "ün", "nın", "nin", "nun", "nün"):
        return False
    lv = last_vowel(lemma_l)
    if lv is None:
        return False
    last_char = lemma_l[-1] if lemma_l else ""
    is_vowel_final = last_char in VOWELS
    if is_vowel_final and not suf.startswith("n"):
        return False
    if (not is_vowel_final) and suf.startswith("n"):
        return False
    suf_v = suf[-2] if suf.startswith("n") else suf[0]
    # Vowel harmony with rounding
    if suf_v == "ı":
        return lv in BACK_UNROUNDED
    if suf_v == "i":
        return lv in FRONT_UNROUNDED
    if suf_v == "u":
        return lv in BACK_ROUNDED
    if suf_v == "ü":
        return lv in FRONT_ROUNDED
    return False


# ---------------------------------------------------------------- categories

# Each category: (id, label, feats_predicate, suffix_validator)
CATEGORIES = [
    {
        "id": "plural_lAr",
        "label": "plural -lAr",
        "feats": {"Number": "Plur"},
        "validator": v_plural,
        "description": "Çoğul eki -lar/-ler, PARSEME-tr UD katmanından (Number=Plur).",
    },
    {
        "id": "locative_DA",
        "label": "locative -DA",
        "feats": {"Case": "Loc"},
        "validator": v_locative,
        "description": "Bulunma hâli -da/-de/-ta/-te (Case=Loc).",
    },
    {
        "id": "ablative_DAn",
        "label": "ablative -DAn",
        "feats": {"Case": "Abl"},
        "validator": v_ablative,
        "description": "Ayrılma hâli -dan/-den/-tan/-ten (Case=Abl).",
    },
    {
        "id": "dative_A",
        "label": "dative -A",
        "feats": {"Case": "Dat"},
        "validator": v_dative,
        "description": "Yönelme hâli -a/-e/-ya/-ye (Case=Dat).",
    },
    {
        "id": "genitive_In",
        "label": "genitive -In",
        "feats": {"Case": "Gen"},
        "validator": v_genitive,
        "description": "Tamlayan hâli -ın/-in/-un/-ün (Case=Gen).",
    },
]


def feats_match(feats: dict[str, str], required: dict[str, str]) -> bool:
    return all(feats.get(k) == v for k, v in required.items())


def load_kenet_literals(path: Path) -> set[str]:
    """Single-word, alphabetic KeNet literals (normalized), used as a stem lexicon."""
    import xml.etree.ElementTree as ET

    out: set[str] = set()
    for syn in ET.parse(path).getroot().findall("SYNSET"):
        synonym = syn.find("SYNONYM")
        if synonym is None:
            continue
        for lit in synonym.findall("LITERAL"):
            w = normalize_tr((lit.text or "").strip().lower())
            if w.isalpha():
                out.add(w)
    return out


def bare_attested_forms(paths: list[Path]) -> set[str]:
    """Words that occur in the corpus with surface form identical to their lemma."""
    out: set[str] = set()
    for sent in iter_many_cupt(paths):
        for tok in sent.tokens:
            if tok.is_range:
                continue
            form = normalize_tr(tok.form.lower())
            lemma = normalize_tr((tok.lemma or "").lower())
            if form == lemma and form.isalpha():
                out.add(form)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--parseme", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-per-category", type=int, default=200)
    p.add_argument("--min-stem-len", type=int, default=2)
    p.add_argument("--version", default="v2-corpus-2026-05-06")
    p.add_argument(
        "--lexicon",
        default=None,
        help="KeNet XML; when given, a stem must be a bare-attested corpus word or a KeNet "
        "literal (drops truncated and already-inflected lemmatizer outputs)",
    )
    args = p.parse_args()

    paths = [Path(s) for s in args.parseme]
    valid_stem: set[str] | None = None
    if args.lexicon:
        lexicon = load_kenet_literals(Path(args.lexicon))
        bare = bare_attested_forms(paths)
        valid_stem = lexicon | bare
        print(
            f"stem lexicon: {len(lexicon)} KeNet literals + {len(bare)} bare-attested forms",
            file=sys.stderr,
        )
    n_rejected_stem = 0

    by_cat: dict[str, "OrderedDict[tuple[str, str], None]"] = {
        c["id"]: OrderedDict() for c in CATEGORIES
    }
    n_sents = 0
    for sent in iter_many_cupt(paths):
        n_sents += 1
        for tok in sent.tokens:
            if tok.is_range or tok.upos != "NOUN":
                continue
            if not tok.lemma or tok.lemma == "_":
                continue
            # Lower-case BEFORE normalizing: Python lowercases Turkish 'İ' (U+0130)
            # to 'i' + U+0307, which normalize_tr then collapses to a clean 'i'.
            lemma = normalize_tr(tok.lemma.lower())
            form = normalize_tr(tok.form.lower())
            # Reject inflected/garbage stems: apostrophe-suffixed proper nouns
            # (fadime', vakfı'n), slash/digit tokens, and possessive-marked stems
            # all fail isalpha() after normalization; dotted-i artefacts are fixed.
            if not lemma.isalpha() or not form.isalpha():
                continue
            if len(lemma) < args.min_stem_len:
                continue
            feats = tok.feats_dict()
            if not feats:
                continue
            for cat in CATEGORIES:
                if not feats_match(feats, cat["feats"]):
                    continue
                if not cat["validator"](lemma, form):
                    continue
                if valid_stem is not None and lemma not in valid_stem:
                    n_rejected_stem += 1
                    break
                key = (lemma, form)
                by_cat[cat["id"]][key] = None
                break  # one category per token

    print(f"scanned {n_sents} sentences", file=sys.stderr)
    if valid_stem is not None:
        print(f"rejected by stem-validity filter: {n_rejected_stem} tokens", file=sys.stderr)

    distractor_pool: list[str] = []
    seen_distractors: set[str] = set()
    for sent in iter_many_cupt([Path(s) for s in args.parseme]):
        for tok in sent.tokens:
            if tok.is_range or tok.upos != "NOUN":
                continue
            lem = normalize_tr((tok.lemma or "").lower())
            if (
                lem
                and lem != "_"
                and len(lem) >= 3
                and lem not in seen_distractors
                and lem.isalpha()
            ):
                seen_distractors.add(lem)
                distractor_pool.append(lem)
                if len(distractor_pool) >= 60:
                    break
        if len(distractor_pool) >= 60:
            break

    categories: list[MorphologyCategory] = []
    for cat in CATEGORIES:
        pairs_dict = by_cat[cat["id"]]
        items = list(pairs_dict.keys())
        if len(items) > args.max_per_category:
            import random

            rng = random.Random(42)
            items = rng.sample(items, args.max_per_category)
        pairs = [StemFormPair(stem=lemma, form=form) for lemma, form in items]
        print(f"  {cat['id']:20s} {len(pairs):4d} pairs", file=sys.stderr)
        categories.append(
            MorphologyCategory(
                id=cat["id"],
                label=cat["label"],
                description=cat.get("description"),
                pairs=pairs,
            )
        )

    ds = MorphologyDataset(
        language="tr",
        version=args.version,
        categories=categories,
        distractor_pool=distractor_pool,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ds.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote {out}", file=sys.stderr)
    print(f"total pairs: {sum(len(c.pairs) for c in categories)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
