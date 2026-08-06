#!/usr/bin/env bash
# Faz 5: canonical re-run of every config that feeds the IEEE Access manuscript.
# Runs each to results/ (so manifests carry the final committed git_rev), and
# records run_id per config into results/_canonical_runs.txt for the numeric audit.
set -u
cd "$(dirname "$0")/.."

SUMMARY=results/_canonical_runs.txt
: > "$SUMMARY"
echo "canonical re-run @ git_rev=$(git rev-parse --short HEAD)  $(date -u)" >> "$SUMMARY"

CONFIGS=(
  morph_v2_corpus
  idiom_v2 idiom_v2_eu idiom_v2_pl idiom_v2_it idiom_v2_el
  idiom_v2_me5_teacher
  polysemy_v3_primary polysemy_v3
  syntax_v2 syntax_v2_eu syntax_v2_pl syntax_v2_it syntax_v2_el
  xlingual_v3 xlingual_v3_bgem3 xlingual_v4_ablation
  negation_v3_clean_raw_baselines negation_v3_clean_carrier_baselines
  sts_v1
  downstream_negation
  tatoeba
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
