#!/usr/bin/env python
"""How many real failures do you need before synthetic evaluation can be trusted?

The transfer experiment says detectors trained on injected faults lose ground on natural
ones, and a "natural-supervised reference" of 0.763 suggests the natural task is simply
harder. Both statements are about a fixed, unequal budget: thousands of injected traces
against 61 natural positives. Neither answers the question a practitioner actually has,
which is not "is synthetic data biased" but "how much real data do I need alongside it".

So we vary the natural-label budget and compare three regimes at each budget:

* `synthetic_only`  --- train on injected faults, never see a natural label.
* `natural_only`    --- train on n natural failures alone.
* `synthetic_plus_n`--- train on injected faults, then refit with the n natural labels
                        appended, which is how a practitioner would actually spend a small
                        annotation budget.

Reading the curves: where `natural_only` overtakes `synthetic_only` is the point at which
synthetic data stops helping. Where `synthetic_plus_n` beats both is the regime in which
synthetic pretraining plus a little real calibration is the right recipe --- the design
ARES already adopts with roughly 150 human-labelled examples, and the reason the honest
recommendation is "validate with real data", not "abandon synthetic data".

    python scripts/run_learning_curves.py --output results/learning_curves_v2.json

Budgets are drawn at the QUERY level and evaluated on held-out queries, so a trace never
informs a prediction about its own query. Each budget is repeated over many draws because
at n=20 the variance between draws is the dominant source of uncertainty.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mirage.analysis import ARMS, dedup, read_rows
from mirage.artifacts import atomic_json
from mirage.probe import new_probe, raw_matrix
from mirage.validity import binary_auroc, prepare

BUDGETS = (20, 40, 80, 160)


def fit_score(train_matrix, train_labels, test_matrix) -> np.ndarray:
    """Fit on one matrix and score another, imputing from the training rows only."""
    stacked = np.vstack([train_matrix, test_matrix])
    ready = prepare(stacked, np.arange(len(train_matrix)))
    model = new_probe()
    model.fit(ready[:len(train_matrix)], train_labels)
    positive = list(model.classes_).index(1)
    return model.predict_proba(ready[len(train_matrix):])[:, positive]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--train-experiment", default="E3_positive_control_v2")
    parser.add_argument("--test-experiment", default="E4_fingerprint_v2")
    parser.add_argument("--arm", default="decoupled", choices=list(ARMS))
    parser.add_argument("--tier", default="T1")
    parser.add_argument("--draws", type=int, default=40)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.artifacts).resolve()
    columns = ARMS[args.arm]

    # Synthetic training pool: injected faults of the chosen tier, plus clean controls.
    train_rows = dedup(read_rows(root, args.train_experiment))
    synth = [r for r in train_rows
             if str(r.get("fault_tier")) == args.tier or str(r.get("fault_source")) == "clean"]
    synth_matrix = raw_matrix(synth, columns)
    synth_labels = np.asarray([int(str(r.get("fault_source")) != "clean") for r in synth])

    # Natural pool: clean test traces, positive where a gold paragraph is missing.
    clean = [r for r in dedup(read_rows(root, args.test_experiment))
             if str(r.get("fault_source", "")) == "clean"]
    nat_matrix = raw_matrix(clean, columns)
    nat_labels = np.asarray([int(bool(r.get("natural_retrieval_failure"))) for r in clean])
    nat_queries = np.asarray([str(r["query_id"]) for r in clean])
    positives = np.unique(nat_queries[nat_labels == 1])
    negatives = np.unique(nat_queries[nat_labels == 0])
    print(f"synthetic pool {len(synth)} rows; natural pool {len(clean)} traces "
          f"({len(positives)} positive queries, {len(negatives)} negative)")

    rng = np.random.default_rng(args.seed)
    result = {"analysis": "natural_label_learning_curves", "arm": args.arm,
              "tier": args.tier, "draws": args.draws, "budgets": list(BUDGETS),
              "n_natural_positive_queries": int(len(positives)), "by_budget": {}}

    for budget in BUDGETS:
        if budget > len(positives):
            print(f"  n={budget}: skipped, only {len(positives)} positive queries exist")
            continue
        scores: dict[str, list[float]] = {"synthetic_only": [], "natural_only": [],
                                          "synthetic_plus_n": []}
        for _ in range(args.draws):
            # Sample a labelled budget at query level; everything else is the test set.
            pos = rng.choice(positives, size=budget, replace=False)
            neg = rng.choice(negatives, size=min(budget * 2, len(negatives)), replace=False)
            labelled = set(pos) | set(neg)
            train_mask = np.isin(nat_queries, list(labelled))
            test_mask = ~train_mask
            if len(set(nat_labels[test_mask])) < 2 or len(set(nat_labels[train_mask])) < 2:
                continue

            y_test = nat_labels[test_mask]
            scores["synthetic_only"].append(binary_auroc(
                y_test, fit_score(synth_matrix, synth_labels, nat_matrix[test_mask])))
            scores["natural_only"].append(binary_auroc(
                y_test, fit_score(nat_matrix[train_mask], nat_labels[train_mask],
                                  nat_matrix[test_mask])))
            scores["synthetic_plus_n"].append(binary_auroc(
                y_test, fit_score(np.vstack([synth_matrix, nat_matrix[train_mask]]),
                                  np.concatenate([synth_labels, nat_labels[train_mask]]),
                                  nat_matrix[test_mask])))

        entry = {k: {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v)),
                     "n_draws": int(np.isfinite(v).sum())}
                 for k, v in scores.items() if v}
        result["by_budget"][str(budget)] = entry
        print(f"  n={budget:4d}  " + "  ".join(
            f"{k}={entry[k]['mean']:.3f}+-{entry[k]['sd']:.3f}" for k in entry))

    atomic_json(Path(args.output), result)

    print("\ncrossover: the budget at which real labels overtake synthetic ones")
    crossed = None
    for budget, entry in result["by_budget"].items():
        if "natural_only" in entry and entry["natural_only"]["mean"] > entry["synthetic_only"]["mean"]:
            crossed = budget
            break
    print(f"  {'n=' + crossed if crossed else 'not reached within the budgets tested'}")
    result["crossover_budget"] = crossed
    atomic_json(Path(args.output), result)
    print(f"written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
