#!/usr/bin/env bash
# Canonical re-run of every config that feeds the IEEE Access manuscript
# (first-submission set plus the revision's English battery, ablations and task-level probes).
# Runs each to results/ (so manifests carry the final committed git_rev), and
# records run_id per config into results/_canonical_runs.txt for the numeric audit.
set -u
cd "$(dirname "$0")/.."

SUMMARY=results/_canonical_runs.txt
mkdir -p results
# Keep the previous run list (e.g. the released one) instead of overwriting it.
[ -s "$SUMMARY" ] && cp "$SUMMARY" "results/_canonical_runs.prev.txt"
rev=$(git rev-parse --short HEAD 2>/dev/null || echo "not-a-git-checkout")
: > "$SUMMARY"
echo "canonical re-run @ git_rev=${rev}  $(date -u)" >> "$SUMMARY"

CONFIGS=(
  # --- first-submission canonical set (21; the augmented polysemy_v3 run is retired) ---
  morph_v2_corpus
  idiom_v2
  idiom_v2_eu
  idiom_v2_pl
  idiom_v2_it
  idiom_v2_el
  idiom_v2_me5_teacher
  polysemy_v3_primary
  syntax_v2
  syntax_v2_eu
  syntax_v2_pl
  syntax_v2_it
  syntax_v2_el
  xlingual_v3
  xlingual_v3_bgem3
  xlingual_v4_ablation
  negation_v3_clean_raw_baselines
  negation_v3_clean_carrier_baselines
  sts_v1
  downstream_negation
  tatoeba
  # --- revision 2: polysemy rebuild (TR v4 / EN v2) + negation selection robustness ---
  polysemy_v4_primary
  polysemy_v4
  negation_v2_pool_raw
  negation_v2_pool_carrier
  # --- English full battery ---
  idiom_v2_en
  morph_en_all_encoders
  morph_en_corpus
  negation_en_clean_carrier_baselines
  negation_en_clean_raw_baselines
  polysemy_en_primary
  sts_en
  sts_en_all_encoders
  syntax_v2_en
  # --- all-encoder morphology / STS (common-scale denominators) ---
  morph_tr_all_encoders
  sts_tr_all_encoders
  # --- controlled distillation ablation ---
  ablation_morph_m2v_bgem3pca
  ablation_morph_m2v_output
  ablation_negation_carrier_m2v_bgem3pca
  ablation_negation_carrier_m2v_output
  ablation_negation_raw_m2v_bgem3pca
  ablation_negation_raw_m2v_output
  ablation_polysemy_m2v_bgem3pca
  ablation_polysemy_m2v_output
  ablation_syntax_m2v_bgem3pca
  ablation_syntax_m2v_output
  ablation_tatoeba
  ablation_xlingual_v5
  # --- distillation-family ablation ---
  family_en_downstream_negation_ststatic
  family_en_downstream_negation_wada
  family_en_morph_ststatic
  family_en_morph_wada
  family_en_negation_carrier_ststatic
  family_en_negation_carrier_wada
  family_en_negation_raw_ststatic
  family_en_negation_raw_wada
  family_en_polysemy_ststatic
  family_en_polysemy_wada
  family_en_sts_ststatic
  family_en_sts_wada
  family_en_syntax_ststatic
  family_en_syntax_wada
  family_tatoeba_ststatic
  family_tr_downstream_negation_ststatic
  family_tr_morph_ststatic
  family_tr_negation_carrier_ststatic
  family_tr_negation_raw_ststatic
  family_tr_polysemy_ststatic
  family_tr_sts_ststatic
  family_tr_syntax_ststatic
  family_xlingual_ststatic
  # --- task-level evaluation ---
  downstream_clustering_en_20ng
  downstream_clustering_tr_abstracts
  downstream_clustering_tr_columns
  downstream_negation_en
  downstream_retrieval_en_scifact
  downstream_retrieval_tr_scifact
  downstream_retrieval_tr_tquad
)

for cfg in "${CONFIGS[@]}"; do
  echo "=== [$(date -u +%H:%M:%S)] running $cfg ==="
  out=$(distillprobe run --config "configs/${cfg}.yaml" 2>&1 | tee /dev/stderr | grep -oE "results/[a-z0-9_]+/[0-9]{8}_[0-9]{6}_[0-9a-f]+" | tail -1)
  if [ -n "$out" ]; then
    rev=$(python -c "import json;print(json.load(open('${out}/run_manifest.json')).get('git_rev'))" 2>/dev/null)
    echo "${cfg}	${out}	git_rev=${rev}" >> "$SUMMARY"
  else
    echo "${cfg}	FAILED" >> "$SUMMARY"
  fi
done

echo "=== ALL DONE ==="
echo "--- summary ---"; cat "$SUMMARY"
