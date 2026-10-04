"""Held-out injector-fingerprint analysis (E4).

Reports every pre-registered arm and every tier, not a single macro number. The
interesting comparison for the paper is not the headline AUROC but its profile:

* if the **coupled_only** arm alone reproduces the headline, the probe is reading
  back fields the injectors wrote and the result is an artefact;
* if **decoupled** holds up on T1 and T2 while collapsing on T0, that is exactly
  the claim — naive injectors carry an implementation fingerprint, and the
  fingerprint shrinks as the injector is hardened.

G-4 is evaluated on the `decoupled` arm, not on `all_features`, because a gate that
a tautology can pass is not a gate.
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from mirage.analysis import (ARMS, assert_query_disjoint, cluster_bootstrap, dedup,
                             dedup_strict,
                             family_label, fit_multiclass, multiclass_auroc, read_rows)
from mirage.artifacts import atomic_json

GATE_ARM = "decoupled"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--train-experiment", required=True)
    parser.add_argument("--test-experiment", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--gate", type=float, default=0.65)
    parser.add_argument("--no-indicators", action="store_true",
                        help="drop the <metric>__defined columns. Measured on v1 artifacts "
                             "they are nearly free (decoupled 0.7003 with, 0.6988 without); "
                             "the flag exists so the table can be reported both ways.")
    parser.add_argument("--strict-dedup", action="store_true",
                        help="collapse injector-seed replicates: one row per "
                             "(query_id, fault_source). Everything else identical, so "
                             "the difference isolates the deduplication key.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.artifacts).resolve()
    collapse = dedup_strict if args.strict_dedup else dedup
    train = collapse(read_rows(root, args.train_experiment))
    test = collapse(read_rows(root, args.test_experiment))
    assert_query_disjoint(train, test)

    tiers = sorted({str(r.get("fault_tier")) for r in test} - {"None", "null"})
    arms = {}
    for name, columns in ARMS.items():
        labels, probabilities, classes = fit_multiclass(
            train, test, columns, with_indicators=not args.no_indicators)
        overall = cluster_bootstrap(partial(multiclass_auroc, labels, probabilities),
                                    test, args.seed, args.bootstrap_repeats)
        per_class = {c: float(roc_auc_score((labels == c).astype(int), probabilities[:, i]))
                     for i, c in enumerate(classes)}
        per_tier = {}
        for tier in tiers:
            keep = np.asarray([i for i, r in enumerate(test)
                               if str(r.get("fault_tier")) == tier or r.get("fault_source") == "clean"])
            subset = [test[i] for i in keep]
            sub_labels, sub_probabilities = labels[keep], probabilities[keep]
            per_tier[tier] = cluster_bootstrap(
                partial(multiclass_auroc, sub_labels, sub_probabilities),
                subset, args.seed, args.bootstrap_repeats)
        arms[name] = {"features": columns, "overall": overall,
                      "per_class": per_class, "per_tier": per_tier}

    gate = arms[GATE_ARM]["overall"]
    headline = arms["all_features"]["overall"]["point"]
    result = {
        "analysis": "held_out_injector_fingerprint",
        "train_experiment": args.train_experiment, "test_experiment": args.test_experiment,
        "n_train": len(train), "n_test": len(test),
        "train_queries": len({r["query_id"] for r in train}),
        "test_queries": len({r["query_id"] for r in test}),
        "classes": sorted({family_label(r) for r in train}), "tiers": tiers,
        "definedness_indicators": not args.no_indicators,
        "arms": arms,
        "gate_arm": GATE_ARM, "gate_threshold": args.gate,
        "gate_g4": gate["ci_low"] > args.gate,
        "macro_auroc": headline,
        "coupling_share_of_above_chance_signal":
            float((headline - arms[GATE_ARM]["overall"]["point"]) / (headline - 0.5))
            if headline > 0.5 else None,
        "seed": args.seed, "bootstrap_repeats": args.bootstrap_repeats,
    }
    atomic_json(Path(args.output), result)

    print(f"{'arm':20s} {'overall':>22s}   " + "  ".join(f"{t:>16s}" for t in tiers))
    for name, arm in arms.items():
        overall = arm["overall"]
        cells = "  ".join(f"{arm['per_tier'][t]['point']:.3f} [{arm['per_tier'][t]['ci_low']:.3f}]"
                          for t in tiers)
        print(f"{name:20s} {overall['point']:.4f} [{overall['ci_low']:.4f},{overall['ci_high']:.4f}]   {cells}")
    print(f"\nG-4 evaluated on '{GATE_ARM}': lower CI {gate['ci_low']:.4f} vs floor {args.gate} "
          f"-> {'PASS' if result['gate_g4'] else 'FAIL'}")
    print(json.dumps({"macro_auroc": headline,
                      "coupling_share": result["coupling_share_of_above_chance_signal"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
