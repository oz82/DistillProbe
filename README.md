# DistillProbe

> A modular probing framework that maps the **loss spectrum** of distilled static
> embeddings — what linguistic structure survives when a contextual sentence
> transformer is collapsed into a bag-of-words lookup table, and what does not.

Distilled static embeddings such as **Model2Vec** encode a text as the *uniform
mean* of its token vectors: a permutation-invariant bag-of-words operation with
no attention and no positional information. They are fast and GPU-free, but what
do they lose relative to their teacher? DistillProbe answers that question
quantitatively across six linguistic phenomena and five typologically diverse
languages.

This repository is the code-and-data release accompanying the article
*DistillProbe: A Multi-Axis Loss-Spectrum Analysis of Distilled Static
Embeddings*.

## The loss spectrum

Five axes are measured in the distilled student, ordered from what a bag of
words can natively express to what it structurally cannot:

| Axis | Retention | Note |
|---|---|---|
| Cross-lingual alignment | **0.66** | P@1 retention, 17 languages (FLORES-200); 0.54 on the harder Tatoeba bitext benchmark; 0.74 under CSLS |
| Polysemy sense ranking | **ρ = 0.334** | leak-free, cross-encoder agreement; permutation control ≈ 0, *p* = 0.001 |
| Morphology | **0.25** | mean difference-vector consistency; category-dependent (fastText beats Model2Vec on the genitive) |
| Syntactic role (positional) | **+0.052** | teacher − student gap; overall student accuracy stays above chance (0.63) via collocation |
| Negation polarity flip | **0/98** | on genuine lexical antonyms; LaBSE 11/98, fastText 0/98 |

A sixth probe, **idiomaticity**, is *teacher-internal*: it contrasts a contextual
encoder's post-attention and pre-attention representations of the same phrase.
It measures the compositional signal attention adds — signal a bag-of-words
student structurally cannot reproduce — and is therefore **not** a point on the
student-retention scale.

**Human ceiling.** Four native Turkish speakers rated the salience of 346 word
senses. They agree with each other moderately (Krippendorff's α = 0.48; mean
pairwise per-word ρ ≈ 0.50), yet *neither* encoder tracks human sense salience
(ρ ≈ 0.05–0.10, confidence intervals containing zero). The cross-encoder
agreement above therefore reflects **shared embedding-space structure between
the two models, not human-like sense dominance**.

**Typology gradient (hypothesis-generating).** The teacher − student positional
gap is numerically ~50% larger in the three fusional than in the two
agglutinative languages (+0.091 vs +0.061), but this is not significant at five
languages (exact permutation *p* = 0.10) and is confounded with language family
and resourcedness. It is reported as a hypothesis for future family-balanced
study, not a confirmed interaction.

## Encoders

| Encoder | Role | Model |
|---|---|---|
| `Model2VecEncoder` | distilled static student (256-d BoW lookup) | `minishlab/potion-multilingual-128M`, distilled from `BAAI/bge-m3` via Tokenlearn |
| `TransformerEncoder` | reference contextual encoder | `sentence-transformers/LaBSE` (768-d) |
| `FastTextEncoder` | static subword baseline | `cc.tr.300.bin` (300-d) |

LaBSE is deliberately called a *reference* encoder, not the teacher: it is **not**
the teacher of `potion-multilingual-128M` (that is bge-m3). It is the teacher
only of the older `M2V_multilingual_output` variant used in the ablation.

Every `Encoder` satisfies the same contract — `name`, `dim`,
`encode(text) -> ndarray` (L2-normalized), `tokenize(text) -> list[str]` — so
adding a backend means implementing one ABC.

## Install

```bash
pip install -e ".[teacher,dev,annotation]"
```

Python ≥ 3.10. CUDA is optional but LaBSE inference is roughly 10× faster on GPU.
fastText's Turkish vectors (`cc.tr.300.bin`, ~6.8 GB) are needed only for the
morphology probe and go under `data/fasttext/`.

## Quick start

```bash
distillprobe list-modules
distillprobe describe negation

# The negation probe runs straight from this archive: its dataset is included
# and both encoders download themselves from HuggingFace on first use.
distillprobe run --config configs/negation_v3_clean_raw.yaml
```

The morphology probe additionally needs fastText's `cc.tr.300.bin` (~6.8 GB,
placed under `data/fasttext/`), and the idiomaticity and syntax probes need the
PARSEME corpora, which are not redistributed here — see *Data* below.

Every run writes a self-describing directory:

```
results/<module>/<UTC-timestamp>_<hash>/
├── run_manifest.json    # config + git revision + dependency versions + seed
├── metrics.json         # full numerical result
├── per_category.csv     # human-readable table
└── plots/*.png
```

## Reproducing the paper

All reported numbers come from a single set of 22 canonical runs executed at one
committed revision (`git_rev 8c4a0f0`). The mapping from each configuration to
the run behind each number is in [`paper/canonical_runs.md`](paper/canonical_runs.md).

```bash
bash scripts/rerun_canonical.sh        # all 22 configs (~15 min with CUDA)

python scripts/audit_canonical.py      # verifies each headline figure against its run
python scripts/holm_family.py          # 17/18 significant, max corrected p = 2.0e-3
python scripts/typology_permutation.py # exact permutation p = 0.10
python scripts/score_polysemy_annotation.py   # human ceiling: alpha, LOO, human-human rho
python -m pytest -q                    # 47 tests
```

`scripts/score_polysemy_annotation.py` runs from this archive alone: it falls
back to the released sense manifest for word and sense ordering when the
gloss-bearing dataset is absent.

## Data: what is here and what is not

Several sources are used under terms that do not permit redistribution, so they
ship as **regeneration recipes** rather than as data.

| Source | Status |
|---|---|
| TDK GTS (Turkish dictionary) | **cited, not redistributed.** The polysemy dataset rebuilds deterministically from a sense-identifier manifest |
| PARSEME 1.2, UD-Turkish | **not redistributed.** Obtain from the PARSEME distribution; pass the `.cupt` paths to the build script |
| KeNet (Turkish WordNet) | **not redistributed.** Obtain the XML release; pass its path to `scripts/extract_kenet_polysemic.py` |
| FLORES-200, STS/sentiment sources | downloaded automatically on first use |
| Human sense-salience ratings | **included**, de-identified as R1–R4, with no dictionary text |

Rebuilding the polysemy dataset:

```bash
python scripts/build_polysemy_dataset.py \
  --manifest data/polysemy/tr_polysemy.v3.sense_manifest.json \
  --out data/polysemy/tr_polysemy.v3.json
```

This replays the recorded TDK sense identifiers in order, applies the recorded
manual edits, and **aborts** if the dictionary has drifted since the manifest was
taken — a silently different rebuild would be worse than none. Verify the result
against `data/polysemy/tr_polysemy.v3.sha256`.

## Integrity

A per-file SHA-256 manifest accompanies the archived release. After unpacking:

```bash
sha256sum -c CHECKSUMS.sha256
```

## Limitations

- **One distillation family.** Two Model2Vec variants from `minishlab`. Other
  paradigms (SBERT distillation, TinyBERT, BERT-of-Theseus) may differ.
- **Five languages.** Broader than typical single-language probing work, but
  narrow relative to Model2Vec's 100+ language coverage.
- **Annotation-scheme sensitivity.** PARSEME annotation practice affects
  category orderings (a Basque outlier is discussed in the paper).
- **Modest human-annotation scale.** Four annotators at moderate agreement. The
  qualitative claim holds — inter-annotator agreement exceeds human–encoder
  agreement five- to ten-fold — but this is a bounded human-ceiling analysis, not
  a definitive sense-salience gold standard.
- **Partial downstream validation.** Only the negation axis is confirmed against
  a real task; the spectrum is otherwise a hypothesis-generating deployment
  guide.

## Practical implication

Distilled static embeddings are a reasonable production choice for retrieval,
clustering and cross-lingual similarity at the ~66% retention level (on FLORES;
~54% on Tatoeba), and are **ill-suited to negation-sensitive tasks** (sentiment
with negation, NLI involving "not") and **positionally discriminative tasks**
(e.g. relation extraction). There the loss is structural, not gradual.

## License

The framework source is released under the **MIT** license.

Datasets derived from the Turkish WordNet **KeNet** (polysemy candidate lists,
negation antonym triplets) are redistributed under **GPL-3.0** to honour KeNet's
copyleft; a verbatim copy is in [`COPYING.GPL-3.0`](COPYING.GPL-3.0). The STS-Tr
test split is redistributed from the public `emrecan/stsb-mt-turkish` dataset
under its own terms. PARSEME 1.2 and FLORES-200 retain their original CC-BY-SA
terms and are not redistributed. See [`LICENSE`](LICENSE) for the full statement.
