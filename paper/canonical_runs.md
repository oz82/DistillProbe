# Canonical run manifest (IEEE Access revision, November 2026)

All numbers in the manuscript come from these runs at **`git_rev = b84e9e6`** (`b84e9e67b89c14217c73103b4313c94b15671985`), 2026-11-11. `results/` is not committed; the curated `results_canonical` upload on Zenodo holds each run's `run_manifest.json` and `metrics.json` (plus `per_category.csv` where the module writes one), and `scripts/audit_canonical.py` reads this list to print every headline figure. Earlier manifests are superseded: the first submission (22 runs at `8c4a0f0`, 2026-07-02) and the first revision draft (75 runs at `103047a`, 2026-09-14, before the polysemy sets were rebuilt); their run ids are kept in `results/_canonical_runs_8c4a0f0.txt` and `results/_canonical_runs_103047a.txt` in the development repository. Against `103047a`, the 68 runs whose data did not change reproduce field by field.

78 runs. Section, table and figure numbers refer to the revised manuscript.

| Config | Module | run_id | Manuscript |
|---|---|---|---|
| morph_v2_corpus | morphology | 20261111_163336_558a61 | Table 3 (TR), §IV-A |
| idiom_v2 | idiomaticity_v2 | 20261111_163340_1e9b72 | §IV-B, Table 6 (TR) |
| idiom_v2_eu | idiomaticity_v2 | 20261111_163420_57c597 | Table 6, §V-A |
| idiom_v2_pl | idiomaticity_v2 | 20261111_163445_091b08 | Table 6, §V-A |
| idiom_v2_it | idiomaticity_v2 | 20261111_163509_78c6d7 | Table 6, §V-A |
| idiom_v2_el | idiomaticity_v2 | 20261111_163531_b113b4 | Table 6, §V-A |
| idiom_v2_me5_teacher | idiomaticity_v2 | 20261111_163603_445c58 | Table 10, §VI-C |
| polysemy_v3_primary | polysemy | 20261111_163651_6be790 | §IV-C human pilot (first-submission set), §III-F |
| syntax_v2 | syntax_v2 | 20261111_163704_3d1cf8 | §IV-D, Table 7 (TR), Fig. 1–2 |
| syntax_v2_eu | syntax_v2 | 20261111_163727_7eefe3 | Table 7, Fig. 2, §V-B |
| syntax_v2_pl | syntax_v2 | 20261111_163744_52930b | Table 7, Fig. 2, §V-B |
| syntax_v2_it | syntax_v2 | 20261111_163811_5c8378 | Table 7, Fig. 2, §V-B |
| syntax_v2_el | syntax_v2 | 20261111_163830_8617e9 | Table 7, Fig. 2, §V-B |
| xlingual_v3 | xlingual | 20261111_163920_54f69f | §IV-E, Table 4, Fig. 1 |
| xlingual_v3_bgem3 | xlingual | 20261111_164006_feb233 | §VI-A (reference-robustness) |
| xlingual_v4_ablation | xlingual | 20261111_164113_6718ca | §VI-A (two off-the-shelf variants) |
| negation_v3_clean_raw_baselines | negation | 20261111_164238_121a24 | Table 5 (TR), §IV-F |
| negation_v3_clean_carrier_baselines | negation | 20261111_164317_139992 | Table 5 (TR), §IV-F |
| sts_v1 | sts | 20261111_164338_f491ac | Table 11 (STS), §VI-D |
| downstream_negation | downstream_negation | 20261111_164502_7895c4 | Table 11 (neg_drop), §VI-D |
| tatoeba | tatoeba | 20261111_164517_f73b81 | §IV-E (Tatoeba), Table 8 |
| polysemy_v4_primary | polysemy | 20261111_164557_6c68f9 | §IV-C PRIMARY (leak-free), Fig. 1 |
| polysemy_v4 | polysemy | 20261111_164625_305967 | §IV-C secondary (augmented) |
| negation_v2_pool_raw | negation | 20261111_164655_d5b4c4 | §IV-F *Selection robustness*, §III-F |
| negation_v2_pool_carrier | negation | 20261111_164710_5cf67b | §IV-F *Selection robustness*, §III-F |
| idiom_v2_en | idiomaticity_v2 | 20261111_164716_6c0812 | Table 6, §V-A |
| morph_en_all_encoders | morphology | 20261111_164827_af15bc | Table 3 Ref column, Eq. 1 denominators; Fig. 1, Fig. 3 |
| morph_en_corpus | morphology | 20261111_164857_fa7206 | Table 3 (EN), §IV-A |
| negation_en_clean_carrier_baselines | negation | 20261111_164934_455414 | Table 5 (EN), §IV-F |
| negation_en_clean_raw_baselines | negation | 20261111_165012_a847bb | Table 5 (EN), §IV-F |
| polysemy_en_primary | polysemy | 20261111_165026_b3d38f | §IV-C (EN), Fig. 1 |
| sts_en | sts | 20261111_165101_614f79 | Table 11 (STS), §VI-D |
| sts_en_all_encoders | sts | 20261111_165240_649e8b | Table 11 (STS, all encoders), Fig. 4 |
| syntax_v2_en | syntax_v2 | 20261111_165542_5f2ec3 | §IV-D, Table 7 (EN), Fig. 1–2 |
| morph_tr_all_encoders | morphology | 20261111_165643_ce892d | Table 3 Ref column, Eq. 1 denominators; Fig. 1, Fig. 3 |
| sts_tr_all_encoders | sts | 20261111_165735_a798aa | Table 11 (STS, all encoders), Fig. 4 |
| ablation_morph_m2v_bgem3pca | morphology | 20261111_170002_b11e0f | Table 8, Fig. 3, §VI-A |
| ablation_morph_m2v_output | morphology | 20261111_170030_579c8f | Table 8, Fig. 3, §VI-A |
| ablation_negation_carrier_m2v_bgem3pca | negation | 20261111_170042_b8b960 | Table 8, Fig. 3, §VI-A |
| ablation_negation_carrier_m2v_output | negation | 20261111_170055_f39977 | Table 8, Fig. 3, §VI-A |
| ablation_negation_raw_m2v_bgem3pca | negation | 20261111_170108_05e630 | Table 8, Fig. 3, §VI-A |
| ablation_negation_raw_m2v_output | negation | 20261111_170121_0ee826 | Table 8, Fig. 3, §VI-A |
| ablation_polysemy_m2v_bgem3pca | polysemy | 20261111_170134_5a28f8 | Table 8, Fig. 3, §VI-A |
| ablation_polysemy_m2v_output | polysemy | 20261111_170201_6982b3 | Table 8, Fig. 3, §VI-A |
| ablation_syntax_m2v_bgem3pca | syntax_v2 | 20261111_170218_f616e5 | Table 8, Fig. 3, §VI-A |
| ablation_syntax_m2v_output | syntax_v2 | 20261111_170239_89d532 | Table 8, Fig. 3, §VI-A |
| ablation_tatoeba | tatoeba | 20261111_170312_2095dd | Table 8, Fig. 3, §VI-A |
| ablation_xlingual_v5 | xlingual | 20261111_170404_630c09 | Table 8, Fig. 3, §VI-A |
| family_en_downstream_negation_ststatic | downstream_negation | 20261111_170534_42c4e9 | Table 11 (neg_drop), §VI-B |
| family_en_downstream_negation_wada | downstream_negation | 20261111_170621_85828b | Table 11 (neg_drop), §VI-B |
| family_en_morph_ststatic | morphology | 20261111_170655_007cb7 | Table 9, Fig. 3, §VI-B |
| family_en_morph_wada | morphology | 20261111_170726_998100 | Table 9, Fig. 3, §VI-B |
| family_en_negation_carrier_ststatic | negation | 20261111_170739_69519f | Table 9, Fig. 3, §VI-B |
| family_en_negation_carrier_wada | negation | 20261111_170755_2c2bc8 | Table 9, Fig. 3, §VI-B |
| family_en_negation_raw_ststatic | negation | 20261111_170808_d45d7b | Table 9, Fig. 3, §VI-B |
| family_en_negation_raw_wada | negation | 20261111_170824_a1e4be | Table 9, Fig. 3, §VI-B |
| family_en_polysemy_ststatic | polysemy | 20261111_170838_4f67dc | Table 9, Fig. 3, §VI-B |
| family_en_polysemy_wada | polysemy | 20261111_170909_3a2558 | Table 9, Fig. 3, §VI-B |
| family_en_sts_ststatic | sts | 20261111_170943_d987a9 | Table 9, Fig. 3, §VI-B |
| family_en_sts_wada | sts | 20261111_171027_d39965 | Table 9, Fig. 3, §VI-B |
| family_en_syntax_ststatic | syntax_v2 | 20261111_171057_3de0b1 | Table 9, Fig. 3, §VI-B |
| family_en_syntax_wada | syntax_v2 | 20261111_171112_c856a6 | Table 9, Fig. 3, §VI-B |
| family_tatoeba_ststatic | tatoeba | 20261111_171134_11e4ab | Table 9, Fig. 3, §VI-B |
| family_tr_downstream_negation_ststatic | downstream_negation | 20261111_171249_081ba6 | Table 11 (neg_drop), §VI-B |
| family_tr_morph_ststatic | morphology | 20261111_171323_9f3ca2 | Table 9, Fig. 3, §VI-B |
| family_tr_negation_carrier_ststatic | negation | 20261111_171337_53dbbe | Table 9, Fig. 3, §VI-B |
| family_tr_negation_raw_ststatic | negation | 20261111_171351_733954 | Table 9, Fig. 3, §VI-B |
| family_tr_polysemy_ststatic | polysemy | 20261111_171405_7d5c9b | Table 9, Fig. 3, §VI-B |
| family_tr_sts_ststatic | sts | 20261111_171437_453bf1 | Table 9, Fig. 3, §VI-B |
| family_tr_syntax_ststatic | syntax_v2 | 20261111_171508_0e6306 | Table 9, Fig. 3, §VI-B |
| family_xlingual_ststatic | xlingual | 20261111_171542_9a72fe | Table 9, Fig. 3, §VI-B |
| downstream_clustering_en_20ng | downstream_clustering | 20261111_171703_9af300 | Table 11, Fig. 4, §VI-D |
| downstream_clustering_tr_abstracts | downstream_clustering | 20261111_171947_fee48d | Table 11, Fig. 4, §VI-D |
| downstream_clustering_tr_columns | downstream_clustering | 20261111_172233_003f5a | Table 11, Fig. 4, §VI-D |
| downstream_negation_en | downstream_negation | 20261111_172346_cad3c4 | Table 11 (neg_drop), §VI-D |
| downstream_retrieval_en_scifact | downstream_retrieval | 20261111_172435_657366 | Table 11, Fig. 4, §VI-D |
| downstream_retrieval_tr_scifact | downstream_retrieval | 20261111_172547_f92ae3 | Table 11, Fig. 4, §VI-D |
| downstream_retrieval_tr_tquad | downstream_retrieval | 20261111_172700_f8678a | Table 11, Fig. 4, §VI-D |
