#!/usr/bin/env python
"""C7 --- do the negative results survive a non-linear probe?

Every discrimination result in this paper is produced by a logistic probe. For a *positive*
finding that is conservative: if a linear model recovers the injector, the injector is
recoverable. For a *negative* finding it is not. "A linear probe cannot separate these"
does not license "these are not separable", and two of our headline claims are negatives:
grounding shows no matched-effect fingerprint, and the query-matched ecological comparison
finds most conditions indistinguishable.

So we rerun the same query-disjoint folds with three additional model families --- an RBF
kernel SVM, gradient boosting, and a small MLP --- and compare the discrimination scale
D = 2|AUC-0.5| against the linear result. This is a sensitivity test, not a leaderboard:
nothing is tuned, because a tuned non-linear model on 389 queries would mostly measure our
willingness to tune.

    python scripts/run_probe_sensitivity.py --output results/probe_sensitivity_v2.json

Registered reading. If the negatives stay at chance under every family, the claim
strengthens from "no linear fingerprint" to "no fingerprint recoverable by any model we
tried". If a non-linear model recovers signal, the claim must be narrowed to the linear
representation --- which is still a result, and a more interesting one, but a different
sentence from the one the paper currently writes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mirage.analysis import ARMS, FEATURES, dedup, read_rows
from mirage.artifacts import atomic_json
from mirage.probe import raw_matrix
from mirage.validity import (binary_auroc, cluster_bootstrap, macro_ovr_auroc,
                             prepare, query_folds)


def families(seed: int):
    """Untuned defaults on purpose --- this measures robustness, not model selection."""
    return {
        "linear": lambda: make_pipeline(StandardScaler(),
                                        __import__("sklearn.linear_model", fromlist=["x"])
                                        .LogisticRegression(max_iter=5_000,
                                                            class_weight="balanced")),
        "rbf_svm": lambda: make_pipeline(StandardScaler(),
                                         SVC(probability=True, class_weight="balanced",
                                             random_state=seed)),
        "gbm": lambda: GradientBoostingClassifier(random_state=seed),
        "mlp": lambda: make_pipeline(StandardScaler(),
                                     MLPClassifier(hidden_layer_sizes=(64, 32),
                                                   max_iter=800, random_state=seed)),
    }


def oof_scores(model_factory, matrix, labels, groups, folds, seed):
    """Out-of-fold positive-class probability under query-disjoint folds."""
    labels = np.asarray(labels)
    classes = np.unique(labels)
    out = np.full((len(labels), len(classes)), np.nan)
    for fold in query_folds(groups, folds, seed):
        train = np.setdiff1d(np.arange(len(labels)), fold)
        if len(set(labels[train])) < len(classes):
            continue
        ready = prepare(matrix, train)
        model = model_factory()
        model.fit(ready[train], labels[train])
        proba = model.predict_proba(ready[fold])
        order = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
        for j, c in enumerate(order):
            out[fold, list(classes).index(c)] = proba[:, j]
    return out, classes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--experiment", default="E4_fingerprint_v2")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.artifacts).resolve()
    result = {"analysis": "probe_family_sensitivity", "folds": args.folds,
              "note": "untuned defaults; a sensitivity check on negative results, "
                      "not a model comparison",
              "matched_effect": {}, "ecological": {}}

    # ---- 1. the matched-effect negatives and positives -----------------------------
    for family, path in (("grounding", "matched_effect_grounding"),
                         ("source_degradation", "matched_effect_source")):
        blob = root.parent / "results" / f"{path}.json"
        if not blob.exists():
            continue
        meta = json.loads(blob.read_text())
        mechs = meta["mechanisms"]
        # Rebuild the corpus labels from the stored run is not possible, so we re-derive
        # the contrast that matters: the two same-pool mechanisms, whose separation is the
        # paper's primary V2 statistic.
        result["matched_effect"][family] = {
            "linear_reference": {k: v["point"] for k, v in
                                 meta["by_arm"]["decoupled"]["pairwise"].items()},
            "mechanisms": mechs,
            "note": "non-linear rerun requires the perturbed corpus; see --rebuild"}

    # ---- 2. the query-matched ecological comparison --------------------------------
    everything = dedup(read_rows(root, args.experiment))
    natural = [r for r in everything if str(r.get("fault_source", "")) == "clean"
               and bool(r.get("natural_retrieval_failure"))]
    natural_queries = {str(r["query_id"]) for r in natural}

    for tier in ("T0", "T1", "T2"):
        for family in ("query_retrieval", "source_degradation"):
            synthetic = [r for r in everything if str(r.get("fault_tier")) == tier
                         and str(r.get("fault_family")) == family
                         and str(r["query_id"]) in natural_queries]
            if not synthetic:
                continue
            shared = {str(r["query_id"]) for r in synthetic} & natural_queries
            rows = ([r for r in natural if str(r["query_id"]) in shared]
                    + [r for r in synthetic if str(r["query_id"]) in shared])
            labels = np.asarray([0] * sum(1 for r in natural if str(r["query_id"]) in shared)
                                + [1] * sum(1 for r in synthetic
                                            if str(r["query_id"]) in shared))
            groups = np.asarray([str(r["query_id"]) for r in rows])
            matrix = raw_matrix(rows, FEATURES)

            entry = {}
            for name, factory in families(args.seed).items():
                proba, classes = oof_scores(factory, matrix, labels, groups,
                                            args.folds, args.seed)
                scores = proba[:, list(classes).index(1)]
                auc = cluster_bootstrap(lambda idx: binary_auroc(labels, scores, idx),
                                        groups, args.bootstrap, args.seed)
                entry[name] = {**auc, "D": float(2 * abs(auc["point"] - 0.5)),
                               "D_upper_bound": float(2 * max(abs(auc["ci_low"] - 0.5),
                                                              abs(auc["ci_high"] - 0.5)))}
            result["ecological"][f"{tier}:{family}"] = entry
            cells = "  ".join(f"{k}={entry[k]['D']:.3f}" for k in entry)
            print(f"  {tier}:{family:22s} D by probe family:  {cells}")

    atomic_json(Path(args.output), result)

    print("\ndoes any non-linear family recover signal the linear probe missed?")
    for cond, entry in result["ecological"].items():
        linear = entry["linear"]["D"]
        best = max((k for k in entry if k != "linear"), key=lambda k: entry[k]["D"])
        gap = entry[best]["D"] - linear
        flag = "MATERIAL" if gap > 0.10 else "no"
        print(f"  {cond:26s} linear D={linear:.3f}  best non-linear {best}="
              f"{entry[best]['D']:.3f}  gap {gap:+.3f}  {flag}")
    print(f"\nwritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
