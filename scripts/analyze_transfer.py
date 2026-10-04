"""Transfer-gap analysis (E5) — the experiment that carries the paper's claim.

For each injector tier, a detector is trained to separate *injected* retrieval faults
from clean traces, then evaluated twice:

  in-distribution   held-out queries, same injector          -> what the literature reports
  natural transfer  held-out queries, real retrieval failure -> what the detector is for

The gap between them is the quantity the paper is about. A detector that scores well
in-distribution and near chance on natural faults has learned the injector.

Natural faults come from inside the same pipeline: HotpotQA annotates
supporting facts, so "an annotated supporting paragraph is absent from the evidence the
generator saw" is a free, automatically verifiable label. Clean traces that carry that
label are the positives; clean traces that do not are the negatives, so both classes are
produced by the untouched pipeline and nothing about the comparison is synthetic.

Scope (docs/PLAN.md §1): retrieval-family faults only. Natural grounding faults are NOT
claimed here — that needs independent human annotation.
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np

from mirage.analysis import (ARMS, BinaryDetector, assert_query_disjoint, binary_auroc,
                             cluster_bootstrap, dedup, fit_binary, read_rows)
from mirage.artifacts import atomic_json

RETRIEVAL_FAMILIES = ("query_retrieval", "source_degradation")
#: K3 in docs/PLAN.md §14: below this many natural traces the transfer arm is not reportable.
MIN_NATURAL = 150


def is_clean(row) -> bool:
    return row.get("fault_source") == "clean"


def natural_positive(row) -> bool:
    return bool(row.get("natural_retrieval_failure"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--train-experiment", required=True)
    parser.add_argument("--test-experiment", required=True)
    parser.add_argument("--families", nargs="+", default=list(RETRIEVAL_FAMILIES))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--min-natural", type=int, default=MIN_NATURAL)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.artifacts).resolve()
    train_rows = dedup(read_rows(root, args.train_experiment))
    test_rows = dedup(read_rows(root, args.test_experiment))
    assert_query_disjoint(train_rows, test_rows)

    families = set(args.families)
    train_clean = [r for r in train_rows if is_clean(r)]
    test_clean = [r for r in test_rows if is_clean(r)]

    # Natural arm. Negatives must be clean traces that are *not* natural failures.
    natural_test = test_clean
    n_natural = sum(natural_positive(r) for r in natural_test)
    natural_available = n_natural >= args.min_natural and (len(natural_test) - n_natural) >= args.min_natural

    tiers = sorted({str(r.get("fault_tier")) for r in train_rows} - {"None", "null"})
    results: dict[str, dict] = {}

    for tier in tiers:
        injected_train = [r for r in train_rows
                          if r.get("fault_family") in families and str(r.get("fault_tier")) == tier]
        injected_test = [r for r in test_rows
                         if r.get("fault_family") in families and str(r.get("fault_tier")) == tier]
        # Clean negatives for the in-distribution arm exclude natural failures, so the
        # "clean" class really is fault-free.
        train_split = injected_train + [r for r in train_clean if not natural_positive(r)]
        test_split = injected_test + [r for r in test_clean if not natural_positive(r)]
        if not injected_train or not injected_test:
            continue

        per_arm = {}
        for arm, columns in ARMS.items():
            # One detector, two evaluation sets. Only the evaluation changes between
            # the in-distribution and transfer arms — never the model.
            detector = BinaryDetector(columns).fit(train_split, lambda r: not is_clean(r))
            labels = np.asarray([int(not is_clean(r)) for r in test_split])
            in_dist = cluster_bootstrap(partial(binary_auroc, labels, detector.score(test_split)),
                                        test_split, args.seed, args.bootstrap_repeats)
            entry = {"in_distribution": in_dist}
            if natural_available:
                nat_labels = np.asarray([int(natural_positive(r)) for r in natural_test])
                transfer = cluster_bootstrap(
                    partial(binary_auroc, nat_labels, detector.score(natural_test)),
                    natural_test, args.seed, args.bootstrap_repeats)
                entry["natural_transfer"] = transfer
                entry["transfer_gap"] = in_dist["point"] - transfer["point"]
            per_arm[arm] = entry
        results[tier] = per_arm

    # Natural-supervised ceiling: how separable natural failures are at all, when the
    # detector is allowed to train on them. Bounds how much of the gap is "hard task"
    # rather than "wrong training distribution".
    ceiling = {}
    if natural_available and any(natural_positive(r) for r in train_clean):
        for arm, columns in ARMS.items():
            labels, scores = fit_binary(train_clean, natural_test, columns, natural_positive)
            ceiling[arm] = cluster_bootstrap(partial(binary_auroc, labels, scores),
                                             natural_test, args.seed, args.bootstrap_repeats)

    result = {
        "analysis": "injected_to_natural_transfer_gap",
        "train_experiment": args.train_experiment, "test_experiment": args.test_experiment,
        "families": sorted(families), "tiers": tiers,
        "n_clean_test": len(test_clean), "n_natural_failures_test": n_natural,
        "n_natural_failures_train": sum(natural_positive(r) for r in train_clean),
        "min_natural_required": args.min_natural,
        "natural_arm_available": natural_available,
        "kill_condition_K3": not natural_available,
        "by_tier": results, "natural_supervised_ceiling": ceiling,
        "seed": args.seed, "bootstrap_repeats": args.bootstrap_repeats,
    }
    atomic_json(Path(args.output), result)

    if not natural_available:
        print(f"K3 FIRED: only {n_natural} natural retrieval failures among {len(test_clean)} "
              f"clean test traces (need {args.min_natural} per class). Transfer arm not reportable.")
    print(f"\n{'tier':6s} {'arm':20s} {'in-distribution':>24s} {'natural transfer':>24s} {'gap':>8s}")
    for tier, per_arm in results.items():
        for arm, entry in per_arm.items():
            a = entry["in_distribution"]
            if "natural_transfer" in entry:
                b = entry["natural_transfer"]
                print(f"{tier:6s} {arm:20s} {a['point']:.3f} [{a['ci_low']:.3f},{a['ci_high']:.3f}]  "
                      f"{b['point']:.3f} [{b['ci_low']:.3f},{b['ci_high']:.3f}]  {entry['transfer_gap']:+.3f}")
            else:
                print(f"{tier:6s} {arm:20s} {a['point']:.3f} [{a['ci_low']:.3f},{a['ci_high']:.3f}]  "
                      f"{'unavailable':>24s}")
    if ceiling:
        print("\nnatural-supervised ceiling (trained on natural failures):")
        for arm, ci in ceiling.items():
            print(f"       {arm:20s} {ci['point']:.3f} [{ci['ci_low']:.3f},{ci['ci_high']:.3f}]")
    print(json.dumps({"n_natural_failures_test": n_natural,
                      "natural_arm_available": natural_available}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
