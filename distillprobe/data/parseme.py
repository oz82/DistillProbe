"""Minimal parser for the PARSEME `.cupt` format (CoNLL-U + 11th MWE column).

Plain 10-column CoNLL-U files are accepted too (the MWE column is then read as
`*` for every token), so UD treebanks without MWE annotation can feed the
probes that only need the UD layer (syntax, morphology).

The CUPT format is an extension of CoNLL-U with a single extra final column,
`PARSEME:MWE`, encoding multiword-expression annotations:

    *           — token participates in no MWE
    1:VID       — token begins MWE #1 of category VID
    1           — token is a non-initial member of MWE #1
    1:VID;2     — multi-MWE: starts MWE 1 and continues MWE 2

Multi-token range rows like `2-3` (Turkish has these because of agglutinative
suffix splitting) are kept on the side and skipped in MWE token indexing — the
component sub-token rows (`2`, `3`) carry the actual MWE labels.

This parser intentionally stays tolerant: malformed or partial blocks are
skipped with a logged warning rather than raising.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

log = logging.getLogger(__name__)


@dataclass
class Token:
    id: str  # original ID string ("3" or "2-3" for ranges)
    form: str
    lemma: str
    upos: str
    head: str
    deprel: str  # column 8: dependency relation to head
    mwe_field: str  # raw column 11
    feats: str = ""  # column 6: morphological features ("Number=Plur|Case=Loc|...")

    @property
    def is_range(self) -> bool:
        return "-" in self.id

    def feats_dict(self) -> dict[str, str]:
        if not self.feats or self.feats == "_":
            return {}
        out: dict[str, str] = {}
        for kv in self.feats.split("|"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                out[k] = v
        return out


@dataclass
class MWE:
    """A single MWE occurrence inside a sentence."""

    mwe_id: int  # within-sentence MWE id (1-based)
    category: str  # VID | LVC.full | LVC.cause | IRV | VPC.full | VPC.semi | IAV | MVC | LS.ICV
    token_indices: list[int] = field(default_factory=list)  # indices into Sentence.tokens

    def phrase(self, sent: "Sentence") -> str:
        return " ".join(sent.tokens[i].form for i in self.token_indices)

    def lemma_phrase(self, sent: "Sentence") -> str:
        return " ".join(sent.tokens[i].lemma for i in self.token_indices)


@dataclass
class Sentence:
    sent_id: str | None
    text: str | None
    tokens: list[Token] = field(default_factory=list)
    mwes: list[MWE] = field(default_factory=list)


def _parse_block(lines: list[str]) -> Sentence | None:
    sent_id: str | None = None
    text: str | None = None
    tokens: list[Token] = []
    mwe_by_id: dict[int, MWE] = {}

    for raw in lines:
        line = raw.rstrip("\n")
        if not line:
            continue
        if line.startswith("#"):
            stripped = line[1:].strip()
            if stripped.startswith("text") and "=" in stripped:
                text = stripped.split("=", 1)[1].strip()
            elif stripped.startswith("sent_id") and "=" in stripped:
                sent_id = stripped.split("=", 1)[1].strip()
            elif stripped.startswith("source_sent_id") and "=" in stripped and sent_id is None:
                sent_id = stripped.split("=", 1)[1].strip()
            continue

        cols = line.split("\t")
        if len(cols) < 10:
            continue  # malformed
        # Plain CoNLL-U (10 columns, e.g. a UD treebank) has no PARSEME:MWE
        # column; treat every token as MWE-free so the UD layer stays usable.
        tok = Token(
            id=cols[0],
            form=cols[1],
            lemma=cols[2],
            upos=cols[3],
            head=cols[6],
            deprel=cols[7],
            mwe_field=cols[10] if len(cols) >= 11 else "*",
            feats=cols[5],
        )
        token_index = len(tokens)
        tokens.append(tok)

        if tok.is_range:
            continue  # range rows don't carry their own MWE
        if tok.mwe_field in ("*", "_", ""):
            continue

        for entry in tok.mwe_field.split(";"):
            entry = entry.strip()
            if not entry:
                continue
            if ":" in entry:
                left, cat = entry.split(":", 1)
                try:
                    mid = int(left)
                except ValueError:
                    log.warning("bad MWE start in sent=%s: %r", sent_id, entry)
                    continue
                mwe = mwe_by_id.setdefault(mid, MWE(mwe_id=mid, category=cat.strip()))
                # First mention defines category; later starts with same id are unusual
                if mwe.category != cat.strip():
                    log.debug("MWE id %d redefined %s→%s in %s", mid, mwe.category, cat, sent_id)
                mwe.token_indices.append(token_index)
            else:
                try:
                    mid = int(entry)
                except ValueError:
                    log.warning("bad MWE continuation in sent=%s: %r", sent_id, entry)
                    continue
                if mid not in mwe_by_id:
                    log.debug("continuation %d before start in %s", mid, sent_id)
                    mwe_by_id[mid] = MWE(mwe_id=mid, category="<unknown>")
                mwe_by_id[mid].token_indices.append(token_index)

    if not tokens:
        return None
    mwes = [mwe_by_id[k] for k in sorted(mwe_by_id)]
    return Sentence(sent_id=sent_id, text=text, tokens=tokens, mwes=mwes)


def iter_cupt(path: str | Path) -> Iterator[Sentence]:
    """Yield Sentence objects, one per blank-line-separated block in the file."""
    p = Path(path)
    block: list[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                if block:
                    sent = _parse_block(block)
                    if sent is not None:
                        yield sent
                    block = []
            else:
                block.append(line)
    if block:
        sent = _parse_block(block)
        if sent is not None:
            yield sent


def load_cupt(path: str | Path) -> list[Sentence]:
    return list(iter_cupt(path))


def iter_many_cupt(paths: Iterable[str | Path]) -> Iterator[Sentence]:
    for p in paths:
        yield from iter_cupt(p)
