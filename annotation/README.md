# Polysemy human annotation

This directory holds the material for the **human pilot** of the polysemy
probe: four native speakers of Turkish rated how readily each dictionary sense of
a polysemous noun comes to mind for that word in isolation. The pilot was run on
the earlier 50-noun Turkish set (v3, 346 senses), not on the primary v4 set.

## Files

- **`polysemy_annotation_GUIDE.md`** — the annotator guide. It was written for a
  panel of up to five (forms A–E); four annotators took part. **Kept in Turkish
  verbatim**: it is the document the annotators actually read, so it is evidence
  of what participants were told, not documentation to be translated. An English
  summary is at the top of that file.
- **`polysemy_form_annotator_*.xlsx`** — blank annotation
  workbooks. They contain no personal data, **but their *Form* sheet embeds all
  346 TDK definitions and example sentences verbatim**, so they are excluded from
  the public release for copyright reasons.
  Regenerate them from the released sense manifest with
  `python scripts/build_polysemy_form.py -n <number of annotators>`.
- **`polysemy_ratings_anon.csv`** — the de-identified 0–3 ratings (R1–R4). **This
  is the published data artefact.**
- **`responses/`** — raw filled workbooks and the name↔code map. Contains
  personal data; **gitignored, never released**.
- Related: `data/polysemy/tr_polysemy.v3.sense_manifest.json` (+ `.sha256`) — the
  regeneration recipe that rebuilds the dataset without redistributing TDK text.

The workbook has two sheets: *Yönerge* (instructions plus annotator information
and consent fields) and *Form* (the rating table). The `PUAN_0_3` column is a
constrained 0–3 dropdown, so out-of-range values cannot be entered.

## Procedure

1. Four native speakers read the guide, completed the information and consent
   fields, and then filled in their own workbook **independently** — independence
   is required for the agreement measurement to mean anything.
2. Filled workbooks go to `annotation/responses/`, which is gitignored: the
   instruction sheet carries annotator names and ages.
3. **De-identification.** The anonymized `polysemy_ratings_anon.csv` is derived
   from the raw workbooks. Columns are `R1..R4` (no names, no initials), keyed by
   `kelime_no, anlam_no`. **The TDK gloss column is absent** — the text is
   copyrighted and joined back in from the manifest when needed. This CSV is the
   scientific artefact that is committed and released. The name↔code mapping is
   kept offline by the researcher and is not released.
4. **Scoring:**
   ```bash
   python scripts/score_polysemy_annotation.py \
     --encoder-metrics results/polysemy/<polysemy_v3_primary run>/metrics.json
   ```
   It reports the human ceiling (Krippendorff's ordinal α, plus
   leave-one-annotator-out α and the mean pairwise per-word rank correlation), a
   per-sense gold (median or mean), and each encoder's agreement with that gold
   (Spearman ρ and winner agreement, with bootstrap confidence intervals). Run
   without `--encoder-metrics` to get the human-ceiling figures alone. Requires
   `pip install -e ".[annotation]"`.

## Result (pilot, four annotators)

α = 0.477 (human ceiling; leave-one-out 0.463–0.499; mean pairwise per-word
ρ = 0.514). Gold-vs-encoder: LaBSE ρ = 0.10, Model2Vec ρ = 0.06 — both
indistinguishable from zero.

The annotators agree with each other five- to ten-fold more than either encoder
agrees with them, so the cross-encoder agreement reported for the polysemy axis
reflects shared embedding-space structure rather than human sense salience. See
the "Human pilot: gold-vs-encoder alignment" paragraph in the article.

## Notes

- Annotators must work **independently**; the agreement statistic is meaningless
  otherwise.
- An optional calibration round — everyone rates the same five words first to
  align on what 0–3 mean, then proceeds independently — is recommended.
