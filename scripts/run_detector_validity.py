#!/usr/bin/env python
"""Detector comparison, redone under the paper's own rules.

Three defects in the original head-to-head, all raised in review, all fixed here.

1. **The coupled configuration was the only one reported.** Tables comparing MIRAGE to
   external detectors used all fourteen features, including the three the audit had
   already identified as write-then-read tautologies. A paper arguing that coupled
   features inflate results cannot present its own headline in coupled form. Every
   comparison below is reported per arm, and `decoupled` is primary.

2. **The intervals were not intervals.** The union analysis took percentiles over 25
   overlapping random halves of the same queries. Those differences are strongly
   positively correlated, so the spread understates uncertainty. Replaced with a paired
   query-level cluster bootstrap on pooled out-of-fold predictions, which evaluates both
   methods on identical resampled rows.

3. **Only AUROC was compared, never the decision.** Benchmarks are used to *choose* a
   detector, so the question that matters is whether a synthetic benchmark picks the same
   detector natural faults would. We rank detectors under each tier and under natural
   failures, and report Kendall's tau between those rankings. A benchmark that ranks
   detectors differently from reality is misleading even where its AUROCs look fine.

    python scripts/run_detector_validity.py --experiment E4_fingerprint_v2 \
        --output results/detector_validity_v2.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from mirage.analysis import ARMS, FEATURES, dedup, read_rows
from mirage.artifacts import atomic_json
from mirage.probe import raw_matrix
from mirage.validity import (binary_auroc, cluster_bootstrap, oof_binary,
                             paired_cluster_bootstrap)

EXTERNAL = ("nli", "nli-large", "judge")
ARMS_REPORTED = ("decoupled", "all_features", "semantic_only", "trivial_baselines")


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
    """Feature matrix for one detector. Arms use MIRAGE metrics; externals a single score."""
    if name in ARMS:
        return raw_matrix(rows, ARMS[name])
    scores = cache.get(name)
    if not scores or any(trace_key(r) not in scores for r in rows):
        return None
    return np.asarray([[scores[trace_key(r)]] for r in rows], dtype=float)


def evaluate(rows, labels, detectors, cache, folds, bootstrap, seed):
    """Pooled out-of-fold scores and bootstrapped AUROC for every detector."""
    groups = np.asarray([str(r["query_id"]) for r in rows])
    oof, table = {}, {}
    for name in detectors:
        matrix = detector_matrix(name, rows, cache)
        if matrix is None:
            continue
        scores = oof_binary(matrix, labels, groups, folds, seed)
        oof[name] = scores
        table[name] = cluster_bootstrap(lambda idx: binary_auroc(labels, scores, idx),
                                        groups, bootstrap, seed)
    return groups, oof, table


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
    print(f"{len(everything)} traces, {len(clean)} clean; detectors: {detectors}")

    result = {"analysis": "detector_validity", "experiment": args.experiment,
              "detectors": detectors, "primary_arm": "decoupled",
              "by_condition": {}, "union": {}, "ranking_stability": {}}

    # ---- detection AUROC and unions, per condition ---------------------------------
    conditions = sorted({str(r["fault_source"]) for r in everything} - {"clean"})
    per_condition_rank: dict[str, dict[str, float]] = {}

    for condition in conditions:
        faulty = [r for r in everything if str(r.get("fault_source")) == condition]
        rows = clean + faulty
        labels = np.asarray([0] * len(clean) + [1] * len(faulty))
        groups, oof, table = evaluate(rows, labels, detectors, cache,
                                      args.folds, args.bootstrap, args.seed)
        result["by_condition"][condition] = table
        per_condition_rank[condition] = {k: v["point"] for k, v in table.items()}

        # Union of the primary arm with each external, tested against the better member.
        unions = {}
        for external in [n for n in EXTERNAL if n in oof]:
            base = detector_matrix("decoupled", rows, cache)
            extra = detector_matrix(external, rows, cache)
            joint = oof_binary(np.hstack([base, extra]), labels, groups,
                               args.folds, args.seed)
            best = max(("decoupled", external), key=lambda n: table[n]["point"])
            unions[f"decoupled+{external}"] = {
                "auroc": cluster_bootstrap(
                    lambda idx: binary_auroc(labels, joint, idx),
                    groups, args.bootstrap, args.seed),
                "vs": best,
                "paired": paired_cluster_bootstrap(
                    lambda idx: binary_auroc(labels, joint, idx),
                    lambda idx: binary_auroc(labels, oof[best], idx),
                    groups, args.bootstrap, args.seed)}
        result["union"][condition] = unions

        cells = "  ".join(f"{n}={table[n]['point']:.3f}" for n in table)
        print(f"  {condition:26s} {cells}")

    # ---- decision validity: does a synthetic benchmark pick the right detector? -----
    natural_rows = clean
    natural_labels = np.asarray([int(bool(r.get("natural_retrieval_failure")))
                                 for r in clean])
    _, _, natural_table = evaluate(natural_rows, natural_labels, detectors, cache,
                                   args.folds, args.bootstrap, args.seed)
    result["natural_detection"] = natural_table
    natural_rank = {k: v["point"] for k, v in natural_table.items()}

    shared = [d for d in detectors if d in natural_rank]
    natural_vector = [natural_rank[d] for d in shared]
    for tier in ("T0", "T1", "T2"):
        # Average a detector's AUROC over the families measurable at this tier, then ask
        # whether that ordering agrees with the ordering natural failures produce.
        pooled = {d: float(np.mean([v[d] for c, v in per_condition_rank.items()
                                    if c.startswith(tier) and d in v]))
                  for d in shared}
        tau, p = kendalltau([pooled[d] for d in shared], natural_vector)
        result["ranking_stability"][tier] = {
            "kendall_tau": float(tau), "p_value": float(p),
            "synthetic_ranking": sorted(pooled, key=pooled.get, reverse=True),
            "natural_ranking": sorted(natural_rank, key=natural_rank.get, reverse=True),
            "synthetic_best": max(pooled, key=pooled.get),
            "natural_best": max(natural_rank, key=natural_rank.get),
            "picks_same_detector": max(pooled, key=pooled.get) == max(natural_rank,
                                                                     key=natural_rank.get)}

    atomic_json(Path(args.output), result)

    print("\nnatural-failure detection (clean traces, gold paragraph missing)")
    for name, v in sorted(natural_table.items(), key=lambda kv: -kv[1]["point"]):
        print(f"  {name:20s} {v['point']:.3f} [{v['ci_low']:.3f},{v['ci_high']:.3f}]")

    print("\ndecision validity --- would this synthetic tier pick the right detector?")
    for tier, v in result["ranking_stability"].items():
        mark = "same choice" if v["picks_same_detector"] else "DIFFERENT choice"
        print(f"  {tier}  tau={v['kendall_tau']:+.3f} (p={v['p_value']:.3f})  "
              f"synthetic picks {v['synthetic_best']}, natural picks {v['natural_best']} "
              f"-> {mark}")

    print("\nunions with a proper paired cluster bootstrap (primary arm = decoupled)")
    for condition, unions in result["union"].items():
        for name, v in unions.items():
            p = v["paired"]
            if p["significant"]:
                print(f"  {condition:26s} {name:22s} vs {v['vs']:10s} "
                      f"{p['delta']:+.3f} [{p['ci_low']:+.3f},{p['ci_high']:+.3f}] "
                      f"{p['direction']}")
    print(f"\nwritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
