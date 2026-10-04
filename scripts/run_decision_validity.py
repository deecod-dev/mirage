#!/usr/bin/env python
"""C1 --- does a synthetic benchmark pick the detector natural failures would pick?

The published version of this analysis averaged each detector's AUROC over all four fault
families before ranking. Two of those families --- `tool_invocation` and `grounding` ---
have no natural counterpart in this pipeline, so their scores cannot be compared against
anything natural. Including them is what produced the tidy "all three tiers agree" result.

Here the comparison is restricted to families that have a natural reference, and the single
"decision validity" verdict is split into the two questions practitioners actually ask:

* **selection validity** --- does the benchmark pick the same winner? This is what matters
  if you are choosing one detector to deploy.
* **ranking validity** --- does it preserve the whole ordering? This is what matters if you
  are reading a leaderboard.

These can disagree, and reporting one number hides that.

    python scripts/run_decision_validity.py --experiment E4_fingerprint_v2 \
        --output results/decision_validity_v2.json

See docs/EXPERIMENT_PLAN.md C1 for the registered interpretation of each outcome.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from mirage.analysis import ARMS, dedup, read_rows
from mirage.artifacts import atomic_json
from mirage.probe import raw_matrix
from mirage.validity import (binary_auroc, bootstrap_selection_stability,
                             cluster_bootstrap, oof_binary)

EXTERNAL = ("nli", "nli-large", "judge")
ARMS_REPORTED = ("decoupled", "all_features", "semantic_only", "trivial_baselines")

#: Only these two families have a natural counterpart: a clean trace can lose a gold
#: paragraph, but nothing in this pipeline naturally emits a malformed tool call.
COMPARABLE = ("query_retrieval", "source_degradation")
NO_NATURAL_REFERENCE = ("tool_invocation", "grounding")

#: Detectors whose feature set contains a write-then-read path to an injector. V1 rejects
#: them, so a second candidate set excludes them: it is incoherent to let a configuration
#: this paper has already declared invalid define the "correct" deployment choice.
COUPLED_DETECTORS = ("all_features", "trivial_baselines")
CANDIDATE_SETS = {"A_all": None, "B_validity_clean": COUPLED_DETECTORS}


def trace_key(row: dict) -> str:
    return f"{row['query_id']}|{row.get('fault_source')}|{row.get('injector_seed')}"


def load_external(root: Path, experiment: str) -> dict[str, dict[str, float]]:
    out = {}
    for name in EXTERNAL:
        path = root / experiment / "baseline_scores" / f"{name}.json"
        if path.exists():
            out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out


def detector_matrix(name: str, rows, cache) -> np.ndarray | None:
    if name in ARMS:
        return raw_matrix(rows, ARMS[name])
    scores = cache.get(name)
    if not scores or any(trace_key(r) not in scores for r in rows):
        return None
    return np.asarray([[scores[trace_key(r)]] for r in rows], dtype=float)


def score_all(rows, labels, detectors, cache, folds, bootstrap, seed,
              *, return_predictions=False):
    groups = np.asarray([str(r["query_id"]) for r in rows])
    table = {}
    predictions = {}
    for name in detectors:
        matrix = detector_matrix(name, rows, cache)
        if matrix is None:
            continue
        oof = oof_binary(matrix, labels, groups, folds, seed)
        predictions[name] = oof
        table[name] = cluster_bootstrap(lambda idx: binary_auroc(labels, oof, idx),
                                        groups, bootstrap, seed)
    if return_predictions:
        return table, {"labels": np.asarray(labels), "groups": groups,
                       "scores": predictions}
    return table


def bootstrap_tau(synthetic: dict[str, float], natural: dict[str, float],
                  repeats: int, seed: int) -> dict[str, float]:
    """Interval for Kendall's tau by resampling the detectors being ranked.

    With only seven detectors the tau is itself a small-sample statistic; resampling the
    ranked items is the honest way to show how unstable it is.
    """
    names = [n for n in synthetic if n in natural]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        picked = rng.choice(names, size=len(names), replace=True)
        if len(set(picked)) < 3:
            continue
        tau, _ = kendalltau([synthetic[n] for n in picked], [natural[n] for n in picked])
        if np.isfinite(tau):
            draws.append(tau)
    tau, p = kendalltau([synthetic[n] for n in names], [natural[n] for n in names])
    low, high = (np.percentile(draws, [2.5, 97.5]) if draws else (np.nan, np.nan))
    return {"tau": float(tau), "p_value": float(p), "ci_low": float(low),
            "ci_high": float(high), "n_resamples": len(draws)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--experiment", default="E4_fingerprint_v2")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.artifacts).resolve()
    everything = dedup(read_rows(root, args.experiment))
    cache = load_external(root, args.experiment)
    clean = [r for r in everything if str(r.get("fault_source", "")) == "clean"]
    detectors = list(ARMS_REPORTED) + [n for n in EXTERNAL if n in cache]
    print(f"{len(everything)} traces, {len(clean)} clean, detectors: {detectors}")

    # Reference ranking: detection of naturally degraded clean traces.
    natural_labels = np.asarray([int(bool(r.get("natural_retrieval_failure"))) for r in clean])
    natural_table, natural_predictions = score_all(
        clean, natural_labels, detectors, cache, args.folds, args.bootstrap, args.seed,
        return_predictions=True)
    natural_rank = {k: v["point"] for k, v in natural_table.items()}
    natural_best = max(natural_rank, key=natural_rank.get)
    print(f"\nnatural reference: best = {natural_best} ({natural_rank[natural_best]:.3f})")

    result = {"analysis": "family_matched_decision_validity", "experiment": args.experiment,
              "comparable_families": list(COMPARABLE),
              "families_without_natural_reference": list(NO_NATURAL_REFERENCE),
              "natural_detection": natural_table, "natural_best": natural_best,
              "by_tier": {}, "by_tier_all_families_deprecated": {}}

    per_condition: dict[str, dict[str, float]] = {}
    condition_predictions = {}
    for condition in sorted({str(r["fault_source"]) for r in everything} - {"clean"}):
        faulty = [r for r in everything if str(r.get("fault_source")) == condition]
        rows = clean + faulty
        labels = np.asarray([0] * len(clean) + [1] * len(faulty))
        table, predictions = score_all(
            rows, labels, detectors, cache, args.folds, args.bootstrap, args.seed,
            return_predictions=True)
        per_condition[condition] = {k: v["point"] for k, v in table.items()}
        condition_predictions[condition] = predictions

    # Selection regret: how much natural-fault performance is actually lost by deploying
    # the detector the synthetic benchmark chose. A changed argmax between two detectors
    # that perform identically on natural faults is a nominal failure, not a costly one.
    def regret(chosen: str) -> float:
        return float(natural_rank[natural_best] - natural_rank[chosen])

    result["candidate_sets"] = {}
    for set_name, excluded in CANDIDATE_SETS.items():
        pool = [d for d in natural_rank if not excluded or d not in excluded]
        nat_best_pool = max(pool, key=lambda d: natural_rank[d])
        entry = {"detectors": pool, "natural_best": nat_best_pool, "by_tier": {}}
        for tier in ("T0", "T1", "T2"):
            conditions = [c for c in per_condition if c.startswith(tier)
                          and c.split(":", 1)[1] in COMPARABLE]
            if not conditions:
                continue
            pooled = {d: float(np.mean([per_condition[c][d] for c in conditions
                                        if d in per_condition[c]])) for d in pool}
            best = max(pooled, key=pooled.get)
            entry["by_tier"][tier] = {
                "synthetic_best": best, "natural_best": nat_best_pool,
                "selection_validity": bool(best == nat_best_pool),
                "regret": float(natural_rank[nat_best_pool] - natural_rank[best]),
                "natural_auroc_of_synthetic_choice": float(natural_rank[best]),
                "natural_auroc_of_natural_choice": float(natural_rank[nat_best_pool])}
            selected = [condition_predictions[c] for c in conditions]
            stability = bootstrap_selection_stability(
                natural_predictions["labels"], natural_predictions["scores"],
                natural_predictions["groups"],
                [(p["labels"], p["scores"], p["groups"]) for p in selected],
                pool, args.bootstrap, args.seed + {"T0": 0, "T1": 1, "T2": 2}[tier])
            entry["by_tier"][tier]["bootstrap_stability"] = stability
        result["candidate_sets"][set_name] = entry

    for tier in ("T0", "T1", "T2"):
        for label, families in (("by_tier", COMPARABLE),
                                ("by_tier_all_families_deprecated", None)):
            conditions = [c for c in per_condition if c.startswith(tier)
                          and (families is None or c.split(":", 1)[1] in families)]
            if not conditions:
                continue
            pooled = {d: float(np.mean([per_condition[c][d] for c in conditions
                                        if d in per_condition[c]]))
                      for d in natural_rank}
            best = max(pooled, key=pooled.get)
            entry = {"n_conditions": len(conditions), "synthetic_ranking": pooled,
                     "synthetic_best": best, "natural_best": natural_best,
                     "selection_validity": bool(best == natural_best),
                     "ranking_validity": bootstrap_tau(pooled, natural_rank,
                                                       args.bootstrap, args.seed)}
            result[label][tier] = entry

    atomic_json(Path(args.output), result)

    print(f"\n{'tier':6s}{'selection':>12s}{'picks':>18s}{'tau':>8s}{'95% CI':>18s}")
    for tier, entry in result["by_tier"].items():
        r = entry["ranking_validity"]
        mark = "SAME" if entry["selection_validity"] else "DIFFERENT"
        print(f"{tier:6s}{mark:>12s}{entry['synthetic_best']:>18s}"
              f"{r['tau']:>8.3f}   [{r['ci_low']:+.3f},{r['ci_high']:+.3f}]")

    print("\nfor contrast, the deprecated all-family pooling that the paper published:")
    for tier, entry in result["by_tier_all_families_deprecated"].items():
        mark = "SAME" if entry["selection_validity"] else "DIFFERENT"
        print(f"  {tier}  {mark:10s} picks {entry['synthetic_best']:18s} "
              f"tau={entry['ranking_validity']['tau']:+.3f}")

    print("\nselection regret --- how costly is the benchmark's choice?")
    for set_name, entry in result["candidate_sets"].items():
        print(f"  candidate set {set_name} ({len(entry['detectors'])} detectors, "
              f"natural best = {entry['natural_best']})")
        for tier, v in entry["by_tier"].items():
            mark = "SAME" if v["selection_validity"] else "DIFFERENT"
            print(f"    {tier}  {mark:10s} picks {v['synthetic_best']:18s} "
                  f"regret {v['regret']:+.4f}  "
                  f"({v['natural_auroc_of_synthetic_choice']:.3f} vs "
                  f"{v['natural_auroc_of_natural_choice']:.3f})")

    disagreements = [t for t, e in result["by_tier"].items()
                     if not e["selection_validity"]]
    print("\nPOINT-ESTIMATE VERDICT: " +
          ("selection agrees at every tier" if not disagreements else
           f"selection disagrees at {', '.join(disagreements)}"))
    print("Bootstrap winner-agreement probabilities and regret intervals above are "
          "the inferential result; a point winner alone is not a stable pass/fail.")
    print(f"{', '.join(NO_NATURAL_REFERENCE)}: N/A (no natural reference distribution)")
    print(f"\nwritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
