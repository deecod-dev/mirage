#!/usr/bin/env python
"""Can a probe tell WHICH INJECTOR produced a trace, with the fault family held fixed?

This is the experiment the fingerprint claim actually requires, and the one the original
analysis did not run. `mirage/analysis.py:family_label` strips the tier prefix, so the
published probe predicted one of five fault *families* and never distinguished any two of
the twelve injectors. Family attribution is fault localisation --- a thing a detector is
supposed to do well --- so high accuracy there is not evidence of contamination.

Here the family is held constant and the target is the tier:

    given only `grounding` traces, predict T0 vs T1 vs T2

Above-chance accuracy means detector features carry information about the *implementation*
that produced a fault, not merely about the fault itself.

    python scripts/run_tier_discrimination.py --experiment E4_fingerprint_v2 \
        --output results/tier_discrimination_v2.json

WHAT THIS EXPERIMENT CANNOT SETTLE

Our three tiers are not three implementations of one fault. The T0 tool injector calls an
unregistered tool, T1 emits an expression that errors, T2 returns a plausible wrong
number: three different severities with genuinely different downstream consequences. A
probe separating them may be reading severity, not manufacturing signature. So a positive
result here is necessary but not sufficient for the fingerprint claim; settling it needs
a matched-effect design in which several implementations target an identical semantic
effect. We report the diagnostic and state the limit rather than overclaiming from it.

The null is a within-query permutation of tier labels: same queries, same class balance,
same rows, association destroyed. Pairwise tier contrasts are reported alongside the
three-way macro, because T0-vs-T1 (both published-style) is a different question from
T1-vs-T2 (published vs hardened).
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from mirage.analysis import ARMS, FEATURES, dedup, read_rows
from mirage.artifacts import atomic_json
from mirage.probe import new_probe, raw_matrix
from mirage.validity import (binary_auroc, cluster_bootstrap, macro_ovr_auroc,
                             oof_binary, oof_multiclass, permutation_null, prepare)

#: Reported for every arm, so a positive result cannot be an artifact of the coupled
#: features the audit already flagged.
REPORTED_ARMS = ("all_features", "decoupled", "semantic_only", "trivial_baselines")


def top_features(matrix: np.ndarray, labels: np.ndarray, columns: list[str],
                 k: int = 4) -> list[dict]:
    """Largest standardised coefficients, to show WHAT separates the tiers.

    If tier discrimination is driven by a coupled feature, that is a very different
    story from discrimination driven by semantic ones, and the reader should be able
    to see which it is.
    """
    everything = np.arange(len(labels))
    ready = prepare(matrix, everything)
    model = new_probe()
    model.fit(ready, labels)
    weight = np.abs(model[-1].coef_).max(axis=0)
    names = columns + [f"{c}__defined" for c in columns][:ready.shape[1] - len(columns)]
    order = np.argsort(weight)[::-1][:k]
    return [{"feature": names[i], "abs_coef": float(weight[i])} for i in order]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--experiment", default="E4_fingerprint_v2")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.artifacts).resolve()
    rows = dedup(read_rows(root, args.experiment))
    rows = [r for r in rows if str(r.get("fault_source", "")) != "clean"]
    families = sorted({str(r["fault_family"]) for r in rows})
    print(f"{len(rows)} injected traces across {len(families)} families")

    result: dict = {"analysis": "within_family_tier_discrimination",
                    "experiment": args.experiment, "folds": args.folds,
                    "seed": args.seed, "by_family": {}}

    for family in families:
        subset = [r for r in rows if str(r["fault_family"]) == family]
        groups = np.asarray([str(r["query_id"]) for r in subset])
        tiers = np.asarray([str(r["fault_tier"]) for r in subset])
        print(f"\n=== {family}: {len(subset)} traces, {len(np.unique(groups))} queries, "
              f"tiers {dict(zip(*np.unique(tiers, return_counts=True)))}")
        entry: dict = {"n_traces": len(subset), "n_queries": int(len(np.unique(groups)))}

        for arm in REPORTED_ARMS:
            columns = ARMS[arm]
            matrix = raw_matrix(subset, columns)

            proba, classes = oof_multiclass(matrix, tiers, groups, args.folds, args.seed)
            stat = lambda idx: macro_ovr_auroc(tiers, proba, classes, idx)
            overall = cluster_bootstrap(stat, groups, args.bootstrap, args.seed)

            # The null refits the probe on permuted labels, so it measures what this
            # pipeline yields from noise rather than assuming 0.5.
            def null_stat(shuffled):
                p, c = oof_multiclass(matrix, shuffled, groups, args.folds, args.seed)
                return macro_ovr_auroc(shuffled, p, c)
            null = permutation_null(null_stat, tiers, groups,
                                    max(args.permutations // 5, 5), args.seed)

            pairs = {}
            for a, b in itertools.combinations(sorted(set(tiers)), 2):
                mask = np.where((tiers == a) | (tiers == b))[0]
                binary = (tiers[mask] == b).astype(int)
                scores = oof_binary(matrix[mask], binary, groups[mask],
                                    args.folds, args.seed)
                pairs[f"{a}_vs_{b}"] = cluster_bootstrap(
                    lambda idx: binary_auroc(binary, scores, idx),
                    groups[mask], args.bootstrap, args.seed)

            entry[arm] = {"macro_auroc": overall, "permutation_null": null,
                          "pairwise": pairs, "n_features": len(columns),
                          "top_features": top_features(matrix, tiers, list(columns))}

            flag = "ABOVE NULL" if overall["ci_low"] > null["max"] else "within null"
            print(f"  {arm:20s} macro {overall['point']:.3f} "
                  f"[{overall['ci_low']:.3f},{overall['ci_high']:.3f}]  "
                  f"null mean {null['mean']:.3f} max {null['max']:.3f}  {flag}")
            print(f"  {'':20s} " + "  ".join(
                f"{k} {v['point']:.3f}" for k, v in pairs.items()))

        result["by_family"][family] = entry

    atomic_json(Path(args.output), result)

    print("\n" + "=" * 78)
    print("VERDICT (decoupled arm --- the pre-registered one)")
    for family, entry in result["by_family"].items():
        arm = entry["decoupled"]
        m, null = arm["macro_auroc"], arm["permutation_null"]
        verdict = ("injector-identifying signal" if m["ci_low"] > null["max"]
                   else "not distinguishable from the permutation null")
        print(f"  {family:22s} {m['point']:.3f} [{m['ci_low']:.3f},{m['ci_high']:.3f}] "
              f"vs null max {null['max']:.3f} -> {verdict}")
    print(f"\nwritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
