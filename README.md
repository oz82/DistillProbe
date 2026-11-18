# DistillProbe

> A modular probing framework that measures what **static sentence encoders**
> retain of a contextual reference encoder — what linguistic structure survives
> when a sentence transformer is collapsed into a bag-of-words lookup table, and
> what does not — on one common, teacher-relative scale.

Distilled static embeddings such as **Model2Vec** encode a text as the *uniform
mean* of its token vectors: a permutation-invariant bag-of-words operation with
no attention and no positional information. They are fast and GPU-free, but what
do they lose relative to their teacher? DistillProbe answers that question
quantitatively across five student-measured axes (plus a teacher-internal
idiomaticity probe), in two typologically distinct full-battery languages
(Turkish, English), with the corpus-based axes replicated in six languages, for
five static models spanning three Model2Vec recipes, a non-Model2Vec distillation
family and a non-distilled contrastive model.

This repository is the code-and-data release (version 1.1.0) accompanying the
article *DistillProbe: A Multi-Axis Loss-Spectrum Analysis of Distilled Static
Embeddings*.

## The loss spectrum on a common scale

Every axis is reported as **teacher-relative retention**
`R = (student − chance) / (reference − chance)`: 1 means the static student
matches the contextual reference's above-chance signal, 0 means no signal.
Primary student `potion-multilingual-128M`, reference LaBSE:

| Axis | Turkish | English | Raw quantity |
|---|---|---|---|
| Cross-lingual alignment | **0.66** | (shared 17-language run) | P@1 retention on FLORES-200; 0.54 on the harder Tatoeba benchmark |
| Syntactic role | **0.71** | **0.61** | LOO role-classifier accuracy above chance; the *positional* gap (teacher − student) is +0.052 (TR) and +0.103 (EN) |
| Morphology | **0.74** | **0.31** | difference-vector consistency of suffix operators (fastText beats the student on the Turkish genitive and on five of six English categories) |
| Polysemy sense ranking | **0.31** | **0.61** | leak-free cross-encoder Spearman agreement; permutation control ≈ 0 |
| Negation polarity flip | **0** | **0** | 0/98 (TR) and 0/70 (EN) flips on genuine lexical antonyms; LaBSE 11/98 and 3/70 |

A sixth probe, **idiomaticity**, is *teacher-internal*: it contrasts a contextual
encoder's post-attention and pre-attention representations of the same phrase
(VID > LVC.full > LITERAL in 5/6 languages) and is **not** a point on the
student-retention scale.

**Invariant vs. recipe-dependent axes.** Across all five static models in both
languages, negation polarity flip is 0 and the positional gap is positive — no
tested recipe or teacher changes them, as mean pooling would lead one to expect.
Cross-lingual alignment, morphology and sense ranking swing by a factor of two to
three across recipes and families: holding the teacher (bge-m3) fixed, replacing
Tokenlearn by the plain PCA/SIF recipe halves cross-lingual retention on FLORES
(0.66 → 0.32) and *raises* morphology retention (0.74 → 0.86); on Tatoeba the
teacher matters more than the recipe.

**Task level.** On retrieval (TQuAD, SciFact EN/TR) and clustering (Turkish
abstracts and columns, 20 Newsgroups) the primary student reaches 0.86–1.18 of
LaBSE's score but only 0.56–0.86 of multilingual-E5's on five of the six sets
(1.07 on the Turkish opinion columns; across all four multilingual students
0.55–1.30 and 0.36–1.11); every bag-of-words model
loses more sentiment accuracy on negated reviews than the contextual encoders,
in both languages.

**Human pilot.** On the earlier 50-noun Turkish polysemy set (v3), four native
Turkish speakers rated the salience of its 346 senses (Krippendorff's α = 0.48);
*neither* encoder tracks human sense salience (ρ ≈ 0.05–0.10). The pilot also
validates dictionary sense order as a salience proxy (ρ = 0.60), and against it the
encoders are only weakly aligned on the rebuilt sets in both languages
(ρ = 0.12–0.22), far below their agreement with each other: the cross-encoder
agreement mostly reflects shared embedding-space structure.

**Typology gradient (hypothesis only).** The positional gap is larger in the four
Indo-European languages than in the two agglutinative ones, but this is not
significant at six languages (exact permutation *p* = 0.07) and is confounded
with family and resource level.

## Encoders

Every encoder in a run configuration declares a `role`: `student`, `reference`
or `baseline`.

| Config type | Model | Role |
|---|---|---|
| `model2vec` | `minishlab/potion-multilingual-128M` (bge-m3 teacher, Tokenlearn) | student |
| `model2vec` | `minishlab/M2V_multilingual_output` (LaBSE teacher, PCA + Zipf) | student |
| `model2vec` | `models/m2v_bgem3_pca256_sif` — bge-m3 + plain PCA/SIF, distilled by us (`model2vec.distill`; separate Zenodo upload) | student |
| `transformer` | `sentence-transformers/static-similarity-mrl-multilingual-v1` (no teacher; contrastive static table) | student |
| `static_wordvec` | Wada et al. (2025) English static word embeddings (GTE-base → PCA → distillation) | student |
| `fasttext` | `cc.tr.300.bin`, `cc.en.300.bin` | baseline |
| `transformer` | `sentence-transformers/LaBSE`, `intfloat/multilingual-e5-base` | reference |

LaBSE is deliberately called a *reference* encoder, not the teacher: it is **not**
the teacher of `potion-multilingual-128M` (that is bge-m3).

Every `Encoder` satisfies the same contract — `name`, `dim`,
`encode(text) -> ndarray` (L2-normalized), `tokenize(text) -> list[str]`, `role`
— so adding a backend means implementing one ABC.

## Install

```bash
pip install -e ".[teacher,dev,annotation]"
pip install "model2vec[distill]==0.8.1"   # only to re-distil the bge-m3 + PCA/SIF student
```

Two student models are not on the Hugging Face Hub and go into the repository:

```bash
# bge-m3 + PCA/SIF student (separate Zenodo upload), or re-distil it with
# python scripts/distill_bgem3_pca_sif.py --out models/m2v_bgem3_pca256_sif
mkdir -p models && tar -xzf m2v_bgem3_pca256_sif-1.1.0.tar.gz -C models

# Wada et al. (2025) English static word embeddings (Apache-2.0), English only
git clone https://github.com/twadada/swe4semantics external/swe4semantics
unzip external/swe4semantics/embeddings/swe_gte256_en.txt.zip -d external/swe4semantics/embeddings
```

Python ≥ 3.10. CUDA is optional but LaBSE/mE5 inference is roughly 10× faster on
GPU. fastText's `cc.tr.300.bin` / `cc.en.300.bin` (~7 GB each) are needed only for
the morphology and negation baselines and go under `data/fasttext/`.

## Quick start

```bash
distillprobe list-modules
distillprobe describe negation

# Runs straight from this archive: the dataset is included and both encoders
# download themselves from HuggingFace on first use.
distillprobe run --config configs/polysemy_en_primary.yaml
```

Every run writes a self-describing directory:

```
results/<module>/<UTC-timestamp>_<hash>/
├── run_manifest.json    # config + git revision + dependency versions + seed
├── metrics.json         # full numerical result
├── per_category.csv     # human-readable table
└── plots/*.png
```

## Reproducing the paper

All reported numbers come from a single set of **78 canonical runs** executed at
one committed revision (`git_rev b84e9e6`). The mapping from each configuration
to the run behind each number is in
[`paper/canonical_runs.md`](paper/canonical_runs.md). The curated run artefacts
(`run_manifest.json` + `metrics.json` per run, plus `per_category.csv` where the
module writes one) are a separate Zenodo upload, `results_canonical-1.1.0.tar.gz`;
unpack it in the repository root, where it creates `results/`, and the analysis
scripts below read it as is. A full re-run additionally needs the datasets that
are not redistributed, built into `data/` from their sources (next section).

```bash
bash scripts/rerun_canonical.sh        # all 78 configs (~1 h with CUDA)

python scripts/audit_canonical.py      # prints every headline figure from its run
python scripts/spectrum_scale.py       # teacher-relative retention per student/language/axis
python scripts/holm_family.py          # 23/24 significant, max corrected p = 3.0e-3
python scripts/typology_permutation.py # six-language exact permutation p = 0.07
python scripts/score_polysemy_annotation.py   # human pilot (v3 set): alpha, LOO, human-human rho
python scripts/polysemy_sense_count_check.py  # TR-EN polysemy gap within sense-count strata
python scripts/dictionary_order_proxy.py      # dictionary order as salience proxy (pilot + rebuilt sets)
python scripts/negation_provenance_subset.py  # polarity flips on lexicon-derived items only
python scripts/tokenizer_fragmentation.py     # subwords/word per Turkic language (needs FLORES data)
python scripts/summarize_revision.py   # one Markdown table per (module, language)
python paper/figures/make_figures.py   # the four paper figures, read live from the runs
python -m pytest -q                    # 58 tests
```

## Data: what is here and what is not

Several sources are used under terms that do not permit redistribution, so they
ship as **regeneration recipes** rather than as data.

| Source | Status |
|---|---|
| TDK GTS (Turkish dictionary) | **cited, not redistributed.** The Turkish polysemy datasets rebuild deterministically from sense-identifier manifests (below) |
| WordNet 3.0 (English) | **derived data included** (`en_polysemy.v2`, `en_negation.*`); WordNet licence |
| PARSEME 1.2 (TR/EU/PL/IT/EL), PARSEME 1.3 (EN) | **not redistributed.** Obtain from PARSEME; pass the `.cupt` paths to `scripts/build_idiomaticity_dataset.py --language <xx>` / `build_syntax_dataset.py` |
| UD English-EWT, UniMorph `eng` | **not redistributed.** Obtain from their repositories; pass to `scripts/build_morphology_dataset_en.py` / `build_syntax_dataset.py` |
| KeNet (Turkish WordNet) | **not redistributed.** Obtain the XML release; pass its path to the KeNet-based build scripts (`--lexicon` for morphology v3) |
| FLORES-200, STS-B, sentiment, retrieval and clustering sources | downloaded automatically on first use (`scripts/build_xlingual_dataset.py`, `build_downstream_dataset.py`, `build_negation_sentiment*.py`, `build_sts_dataset.py`) |
| Human sense-salience ratings | **included**, de-identified as R1–R4, with no dictionary text (pilot, collected on the v3 set) |

Rebuilding the Turkish polysemy datasets (v4 is the primary set; v3 is the pilot
set the human ratings refer to):

```bash
python scripts/build_polysemy_dataset_v4.py \
  --cache-dir data/polysemy/tdk_cache \
  --manifest data/polysemy/tr_polysemy.v4.sense_manifest.json \
  --out data/polysemy/tr_polysemy.v4.json
python scripts/augment_polysemy_with_parseme.py \
  --in data/polysemy/tr_polysemy.v4.json --out data/polysemy/tr_polysemy.v4_aug.json \
  --parseme <PARSEME-tr train/dev/test .cupt> --version v4-aug-parseme-2026-11-11
python scripts/build_polysemy_dataset.py \
  --manifest data/polysemy/tr_polysemy.v3.sense_manifest.json \
  --out data/polysemy/tr_polysemy.v3.json
```

Each replay queries only the recorded headwords, keeps the recorded TDK sense
identifiers in order (v3 also applies its two recorded manual edits, stored as
rules: an antecedent-sense substitution and an example typo fix), and
**aborts** if the dictionary has drifted since the manifest was taken. Verify the
results against `data/polysemy/tr_polysemy.v4.sha256` and `tr_polysemy.v3.sha256`.
(The copy of v3 used for the runs also carried a free-text curation log field,
`review_notes`, which the replay does not regenerate and the probe does not read;
the v3 checksum is that of the replayed file.)

## Citation

Cite the article and this archive. This release, version 1.1.0, is
[10.5281/zenodo.24345356](https://doi.org/10.5281/zenodo.24345356); the concept DOI
[10.5281/zenodo.21821401](https://doi.org/10.5281/zenodo.21821401) always resolves to
the latest version.

## Integrity

A per-file SHA-256 manifest, `CHECKSUMS.sha256`, is uploaded next to the source
archive. With both in the same folder:

```bash
tar -xzf distillprobe-1.1.0.tar.gz
cd distillprobe-1.1.0 && sha256sum -c ../CHECKSUMS.sha256
```

## Limitations

- **Scope.** Five static models (three Model2Vec recipes, one other distillation
  family, one non-distilled contrastive model). Contextual compression methods
  (SBERT distillation, TinyBERT, BERT-of-Theseus) are out of scope.
- **Languages.** Two full-battery languages; six for the corpus-based axes. The
  Turkish–English contrast confounds typology with resource level.
- **English morphology is thin by construction** (six inflections, two of them
  with 41–46 pairs).
- **Reference dependence.** For the primary student, task-level retention is
  0.86–1.18 against LaBSE but 0.56–0.86 against multilingual-E5 (on five of six
  sets); a retention figure is only meaningful with its reference.
- **The human analysis is a pilot.** Four annotators at moderate agreement, on the
  earlier v3 set rather than the rebuilt v4 set; the English polysemy set has no
  human study.

## License

The framework source is released under the **MIT** license.

Datasets derived from the Turkish WordNet **KeNet** (polysemy candidate lists,
negation antonym triplets) are redistributed under **GPL-3.0** to honour KeNet's
copyleft; a verbatim copy is in [`COPYING.GPL-3.0`](COPYING.GPL-3.0). The English
morphology set (UD English-EWT, CC-BY-SA 4.0, filtered with UniMorph) and the
WordNet-derived polysemy and negation sets (WordNet licence) ship under those
terms; the English syntax set (UD English-EWT) and idiomaticity set (PARSEME 1.3
English) are not redistributed and are rebuilt by the build scripts. The
Turkish morphology pair lists extracted from the PARSEME 1.2 Turkish UD layer
(`tr_categories.v2`/`v3`) are redistributed under CC-BY-NC-SA 4.0, the licence of
their source; the author-curated `tr_categories.v1` and `tr_negation.v1` are under
CC-BY 4.0. The STS test splits are redistributed from the public `emrecan/stsb-mt-turkish` and
`mteb/stsbenchmark-sts` datasets under their own terms. PARSEME 1.2 (CC-BY-NC-SA
4.0; layer-dependent for Polish) and FLORES-200 (CC-BY-SA) are not redistributed.
See [`LICENSE`](LICENSE) for the full statement.
