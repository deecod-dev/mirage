#!/usr/bin/env bash
# C6b --- does the T2 decision-validity failure survive a change of retrieval depth?
#
# The first k=3 run injected only T1, so the paper's T2 selection-validity result rests on
# a single operating point (k=2) that was itself chosen after inspecting natural-failure
# rates. This chain adds the missing tiers at k=3 and recomputes decision validity there.
#
# Three stages, sequential because each needs the previous one's output:
#   1. inject T0 and T2 at k=3 (clean traces are reused -- same generation_hash)
#   2. score the external detectors on the k=3 traces, so the candidate pool matches k=2
#   3. recompute family-matched decision validity, selection regret and both candidate sets
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== [1/3] injecting T0 and T2 at k=3 ==="
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_experiment.py \
    --config configs/experiments/E4_k3_full.yaml --seed 11 --resume

echo "=== [2/3] scoring external detectors on k=3 traces ==="
python -u scripts/run_baselines.py --experiment E4_k3_v2 \
    --baselines nli nli-large judge mirage trivial \
    --device cuda:0 --output results/k3_baselines.json

echo "=== [3/3] decision validity at k=3 ==="
python -u scripts/run_decision_validity.py --experiment E4_k3_v2 \
    --output results/k3_decision_validity.json

echo "=== COMPARISON: does T2 behave the same at both depths? ==="
python - <<'PY'
import json
for depth, path in (("k=2", "results/decision_validity_v2.json"),
                    ("k=3", "results/k3_decision_validity.json")):
    d = json.load(open(path))
    print(f"\n{depth}  ({len(d['natural_detection'])} detectors, "
          f"natural best = {d['natural_best']})")
    for name, cs in d.get("candidate_sets", {}).items():
        print(f"  candidate set {name}")
        for tier, v in cs["by_tier"].items():
            mark = "holds" if v["selection_validity"] else "FAILS"
            print(f"    {tier}  selection {mark:6s} picks {v['synthetic_best']:16s} "
                  f"regret {v['regret']:+.4f}")
PY
echo "DONE_K3T2"
