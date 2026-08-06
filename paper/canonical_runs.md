# Canonical run manifest (Phase 5)

All numbers in `main_ieee.tex` come from these runs at **`git_rev = 8c4a0f0`** (`8c4a0f0a24f83e6a085f21164f8f30e63748120b`), 2026-07-02. `results/` not committed; audit confirms every figure.

| Config | Module | run_id | Manuscript |
|---|---|---|---|
| morph_v2_corpus | morphology | 20260702_161528_43b3f5 | Table 1, §4.1 |
| idiom_v2 | idiomaticity_v2 | 20260702_161532_116d9f | §4.2 |
| idiom_v2_eu | idiomaticity_v2 | 20260702_161610_dcf60e | §5.1 |
| idiom_v2_pl | idiomaticity_v2 | 20260702_161636_1b5b15 | §5.1 |
| idiom_v2_it | idiomaticity_v2 | 20260702_161700_4c0b9e | §5.1 |
| idiom_v2_el | idiomaticity_v2 | 20260702_161723_d70d72 | §5.1 |
| idiom_v2_me5_teacher | idiomaticity_v2 | 20260702_161756_76b77e | §6.2 |
| polysemy_v3_primary | polysemy | 20260702_161840_97cb46 | §4.3 PRIMARY |
| polysemy_v3 | polysemy | 20260702_161900_28855e | §4.3 secondary |
| syntax_v2 | syntax_v2 | 20260702_161914_5a6a89 | §4.4 |
| syntax_v2_eu | syntax_v2 | 20260702_161938_ea1d8b | §5.2 |
| syntax_v2_pl | syntax_v2 | 20260702_161954_8de6c6 | §5.2 |
| syntax_v2_it | syntax_v2 | 20260702_162021_7b0ab8 | §5.2 |
| syntax_v2_el | syntax_v2 | 20260702_162041_23e1b4 | §5.2 |
| xlingual_v3 | xlingual | 20260702_162127_e4ddb5 | §4.5 |
| xlingual_v3_bgem3 | xlingual | 20260702_162206_2e9547 | §6.1 |
| xlingual_v4_ablation | xlingual | 20260702_162306_0f1c4d | §6.1 |
| negation_v3_clean_raw_baselines | negation | 20260702_162430_f31b6f | §4.6 |
| negation_v3_clean_carrier_baselines | negation | 20260702_162514_8e2296 | §4.6 |
| sts_v1 | sts | 20260702_162532_d03c04 | §6.3 |
| downstream_negation | downstream_negation | 20260702_162703_0d42ed | §6.4 |
| tatoeba | tatoeba | 20260702_162717_f062ae | §4.5 |

**Audit:** 22/22 ok, all `git_rev=8c4a0f0`. Includes the syntax subword-alignment fix (occurrences with an unaligned/truncated target token are dropped from both encoders; affects PL only: student 0.647→0.648, p 8.5e-16→1.0e-15; typology means +0.091/+0.061 and all headline figures unchanged). Polysemy leak-free ρ 0.334, augmented 0.520.
