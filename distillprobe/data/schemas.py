from __future__ import annotations

from pydantic import BaseModel, Field


class StemFormPair(BaseModel):
    stem: str
    form: str
    note: str | None = None


class MorphologyCategory(BaseModel):
    id: str = Field(..., description="Stable identifier, e.g. 'plural_lAr'")
    label: str = Field(..., description="Human-readable name, e.g. 'plural -lAr'")
    description: str | None = None
    pairs: list[StemFormPair]


class MorphologyDataset(BaseModel):
    language: str = "tr"
    version: str
    categories: list[MorphologyCategory]
    distractor_pool: list[str] = Field(
        default_factory=list,
        description="Word pool for the random-pair baseline (unrelated stems/forms).",
    )


# ---------------------------------------------------------------- idiomaticity


class IdiomItem(BaseModel):
    """One annotated MWE occurrence in its source sentence."""

    sent_id: str | None = None
    sentence_text: str
    sentence_tokens: list[str] = Field(
        default_factory=list,
        description="Word-level (PARSEME-token-level) surface forms of the source sentence, "
        "excluding multi-token range rows. Required for subword-level (V2) probing; "
        "empty in V1 datasets.",
    )
    phrase: str
    lemma_phrase: str
    category: str = Field(
        ...,
        description="MWE category (PARSEME inventory, e.g. VID, LVC.full, LVC.cause, IRV, MVC). "
        "Reserved value 'LITERAL' is used for compositional baselines.",
    )
    token_count: int
    phrase_token_indices: list[int] = Field(
        default_factory=list,
        description="0-based indices into `sentence_tokens` marking the phrase span. "
        "Empty in V1 datasets.",
    )


class IdiomaticityDataset(BaseModel):
    language: str = "tr"
    version: str
    source: str
    items: list[IdiomItem]


# ---------------------------------------------------------------- polysemy


class PolysemySense(BaseModel):
    """One sense of a polysemic word."""

    sense_id: str = Field(..., description="Unique id within a word (e.g. 'tdk:1234')")
    homograph_id: str | None = Field(
        default=None,
        description="If the word has multiple madde records (homographs), which one this sense belongs to.",
    )
    definition: str
    pos: str | None = None
    examples: list[str] = Field(default_factory=list)


class PolysemyWord(BaseModel):
    word: str
    senses: list[PolysemySense]


class PolysemyDataset(BaseModel):
    language: str = "tr"
    version: str
    source: str
    words: list[PolysemyWord]


# ---------------------------------------------------------------- syntax


class SyntaxItem(BaseModel):
    """One occurrence of a target lemma in a syntactic role."""

    word: str
    role: str = Field(..., description="Universal Dependencies relation (e.g. 'nsubj', 'obj')")
    sentence_text: str
    sentence_tokens: list[str]
    target_token_index: int


class SyntaxDataset(BaseModel):
    language: str = "tr"
    version: str
    source: str
    target_roles: list[str] = Field(default_factory=lambda: ["nsubj", "obj"])
    items: list[SyntaxItem]


# ---------------------------------------------------------------- xlingual


class XLingualPair(BaseModel):
    """One parallel sentence aligned across multiple languages."""

    pair_id: str
    text_by_lang: dict[str, str]


class XLingualDataset(BaseModel):
    version: str
    source: str
    languages: list[str]
    pairs: list[XLingualPair]


# ---------------------------------------------------------------- negation


class NegationTriplet(BaseModel):
    """A (positive, negation, antonym) triplet for the polarity-flip probe."""

    positive: str
    negation: str
    antonym: str
    note: str | None = None


class NegationDataset(BaseModel):
    language: str = "tr"
    version: str
    source: str
    triplets: list[NegationTriplet]


# ---------------------------------------------------------------- STS (downstream)


class STSPair(BaseModel):
    """One sentence pair with a continuous similarity score."""

    sentence1: str
    sentence2: str
    score: float = Field(..., description="Gold similarity in [0, 5] (STS-B convention)")


class STSDataset(BaseModel):
    language: str = "tr"
    version: str
    source: str
    pairs: list[STSPair]


# ---------------------------------------------------------------- clustering (downstream)


class ClusteringSet(BaseModel):
    """One clustering instance: texts with gold category labels (MTEB layout)."""

    sentences: list[str]
    labels: list[str]


class ClusteringDataset(BaseModel):
    language: str = "tr"
    version: str
    source: str
    sets: list[ClusteringSet]


# ---------------------------------------------------------------- retrieval (downstream)


class RetrievalDoc(BaseModel):
    id: str
    title: str = ""
    text: str


class RetrievalQuery(BaseModel):
    id: str
    text: str


class RetrievalDataset(BaseModel):
    """BEIR-style ad-hoc retrieval: corpus, queries and graded relevance judgements."""

    language: str = "tr"
    version: str
    source: str
    corpus: list[RetrievalDoc]
    queries: list[RetrievalQuery]
    qrels: dict[str, dict[str, float]] = Field(
        ..., description="query id -> {doc id -> relevance grade}; only judged queries are scored"
    )
