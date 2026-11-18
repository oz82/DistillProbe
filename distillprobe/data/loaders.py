from __future__ import annotations

import json
from pathlib import Path

from .schemas import (
    ClusteringDataset,
    IdiomaticityDataset,
    MorphologyDataset,
    NegationDataset,
    PolysemyDataset,
    RetrievalDataset,
    STSDataset,
    SyntaxDataset,
    XLingualDataset,
)


def load_morphology_dataset(path: str | Path) -> MorphologyDataset:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return MorphologyDataset(**raw)


def load_idiomaticity_dataset(path: str | Path) -> IdiomaticityDataset:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return IdiomaticityDataset(**raw)


def load_polysemy_dataset(path: str | Path) -> PolysemyDataset:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return PolysemyDataset(**raw)


def load_syntax_dataset(path: str | Path) -> SyntaxDataset:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return SyntaxDataset(**raw)


def load_xlingual_dataset(path: str | Path) -> XLingualDataset:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return XLingualDataset(**raw)


def load_negation_dataset(path: str | Path) -> NegationDataset:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return NegationDataset(**raw)


def load_sts_dataset(path: str | Path) -> STSDataset:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return STSDataset(**raw)


def load_clustering_dataset(path: str | Path) -> ClusteringDataset:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return ClusteringDataset(**raw)


def load_retrieval_dataset(path: str | Path) -> RetrievalDataset:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return RetrievalDataset(**raw)
