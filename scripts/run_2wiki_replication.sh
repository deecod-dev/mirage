#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"

stage="${1:-pilot}"
if [[ "$stage" == "pilot" ]]; then
  python -u scripts/run_server_experiment.py \
    --config configs/experiments/E1_2wiki_pilot.yaml --seed 20260823 --resume
  python -u scripts/run_natural_labels.py --experiment E1_2wiki_pilot \
    --output results/2wiki_pilot_natural_labels.json
  exit 0
fi

if [[ "$stage" != "full" ]]; then
  echo "usage: $0 [pilot|full]" >&2
  exit 2
fi

pilot="results/2wiki_pilot_natural_labels.json"
if [[ ! -f "$pilot" ]]; then
  echo "pilot result missing; run '$0 pilot' first" >&2
  exit 2
fi

expected_traces=7000
while true; do
  present=$(find artifacts/E4_2wiki_v3 -name 'trace_*.json' 2>/dev/null | wc -l)
  if (( present >= expected_traces )); then
    echo "generation complete: ${present}/${expected_traces} traces"
    break
  fi
  echo "generation incomplete: ${present}/${expected_traces}; resuming"
  python -u scripts/run_server_experiment.py \
    --config configs/experiments/E4_2wiki_v3.yaml --seed 20260823 --resume
done
python -u scripts/run_natural_labels.py --experiment E4_2wiki_v3 \
  --emit-counterfactual-plan results/2wiki_counterfactual_plan_v3.json \
  --output results/2wiki_natural_labels_v3.json
python -u scripts/run_baselines.py --experiment E4_2wiki_v3 \
  --baselines nli nli-large judge --device cuda:0 --output results/2wiki_baselines_v3.json
python -u scripts/run_c2st_matched.py --experiment E4_2wiki_v3 \
  --output results/2wiki_c2st_matched_v3.json
python -u scripts/run_decision_validity.py --experiment E4_2wiki_v3 \
  --output results/2wiki_decision_validity_v3.json
python -u scripts/run_counterfactual.py --experiment E4_2wiki_v3 \
  --config configs/experiments/E4_2wiki_v3.yaml --device cuda:0 \
  --output results/2wiki_counterfactual_v3.json
