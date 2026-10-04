#!/usr/bin/env python
"""C3 --- is a synthetic tier *practically equivalent* to natural failures, or just underpowered?

Two defects in how the paper currently reads its C2ST numbers.

**Failure to reject is not equivalence.** The paper reads a discrimination AUROC near 0.5 as
"statistically indistinguishable from natural". That is backwards: not finding a difference
is what a small sample looks like. Establishing similarity requires an equivalence test
against a margin declared in advance. We use the discrimination scale

    D = 2 * |AUC - 0.5|          0 = identical, 1 = perfectly separable

and declare practical equivalence only when the **upper** one-sided 95% bound on D falls
below a predeclared delta (default 0.05, with sensitivity at 0.025 and 0.075).

**A trained discriminator below chance is a symptom, not a result.** The headline realism
cell (T2 source, surface features) scores 0.457 with an interval that excludes 0.5 from
below. For a fixed score, AUC and 1-AUC carry the same information and orienting is
harmless. For a probe *trained* and scored out-of-fold it is not: significantly-below-chance
means the fit anti-generalises across folds, which is a defect in the measurement. Reporting
max(AUC, 1-AUC) would convert a broken measurement into an apparent finding. So we keep the
sign and diagnose instead, with three checks:

    shuffled   labels permuted within query --- should sit at 0.5; if the below-chance
               behaviour persists, the fold structure is at fault, not the fit
    holdout    a single query-disjoint split instead of 5 folds --- isolates fold-count
    balanced   synthetic downsampled to the natural count --- isolates class imbalance

    python scripts/run_c2st_equivalence.py --experiment E4_fingerprint_v2 \
        --output results/c2st_equivalence_v2.json

See docs/EXPERIMENT_PLAN.md C3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mirage.analysis import FEATURES, dedup, read_rows
from mirage.artifacts import atomic_json
from mirage.probe import raw_matrix
from mirage.validity import (binary_auroc, cluster_bootstrap, oof_binary,
                             permutation_null, query_folds, prepare)
from mirage.probe import new_probe

COMPARABLE = ("query_retrieval", "source_degradation")
EXTERNAL = ("nli", "nli-large", "judge")
DELTAS = (0.025, 0.05, 0.075)


def trace_key(row: dict) -> str:
    return f"{row['query_id']}|{row.get('fault_source')}|{row.get('injector_seed')}"


def surface_features(row: dict) -> list[float]:
    """Text statistics, sharing no code with the metric suite (see run_realism_c2st.py)."""
    from collections import Counter
    answer = str(row.get("answer") or "")
    passages = [str(p.get("text") or "") for p in row.get("selected_context") or []]
    context = " ".join(passages)
    question = str(row.get("question") or "")
    a_tok, c_tok, q_tok = answer.lower().split(), context.lower().split(), question.lower().split()
    a_set, c_set, q_set = set(a_tok), set(c_tok), set(q_tok)
    return [float(len(answer)), float(len(a_tok)), float(len(passages)), float(len(context)),
            float(np.mean([len(p) for p in passages]) if passages else 0.0),
            len(a_set & c_set) / len(a_set) if a_set else 0.0,
            len(q_set & c_set) / len(q_set) if q_set else 0.0,
            len(a_set) / len(a_tok) if a_tok else 0.0,
            max(Counter(a_tok).values()) / len(a_tok) if a_tok else 0.0]


def build(rows, kind, cache):
    if kind == "surface":
        return np.asarray([surface_features(r) for r in rows], dtype=float)
    if kind == "mirage":
        return raw_matrix(rows, FEATURES)
    usable = [n for n in EXTERNAL if n in cache and all(trace_key(r) in cache[n] for r in rows)]
    return (np.asarray([[cache[n][trace_key(r)] for n in usable] for r in rows], dtype=float)
            if usable else None)


def equivalence(point: float, ci_low: float, ci_high: float) -> dict:
    """Upper one-sided bound on D = 2|AUC-0.5|, tested against each margin.

    The bound is driven by whichever end of the AUROC interval sits further from 0.5,
    because that is the most separable value the data are consistent with.
    """
    d_point = 2 * abs(point - 0.5)
    d_upper = 2 * max(abs(ci_low - 0.5), abs(ci_high - 0.5))
    out = {"D": float(d_point), "D_upper_bound": float(d_upper)}
    for delta in DELTAS:
        out[f"equivalent_at_{delta}"] = bool(d_upper < delta)
    out["verdict"] = ("practically equivalent" if d_upper < 0.05 else
                      "underpowered: no difference detected but equivalence not established"
                      if ci_low <= 0.5 <= ci_high else
                      "distinguishable")
    return out


def diagnose(rows, labels, groups, matrix, folds, bootstrap, seed) -> dict:
    """Why is a trained out-of-fold discriminator scoring below chance?"""
    out = {}

    def auroc_from(oof):
        return cluster_bootstrap(lambda idx: binary_auroc(labels, oof, idx),
                                 groups, bootstrap, seed)

    out["as_run"] = auroc_from(oof_binary(matrix, labels, groups, folds, seed))

    # 1. Permuted labels: a correct pipeline sits at 0.5 here by construction.
    out["shuffled"] = permutation_null(
        lambda y: binary_auroc(y, oof_binary(matrix, y, groups, folds, seed)),
        labels, groups, repeats=10, seed=seed)

    # 2. Single query-disjoint holdout instead of K folds.
    fold = query_folds(groups, 2, seed)[0]
    train = np.setdiff1d(np.arange(len(labels)), fold)
    ready = prepare(matrix, train)
    model = new_probe().fit(ready[train], labels[train])
    positive = list(model.classes_).index(1)
    scores = np.full(len(labels), np.nan)
    scores[fold] = model.predict_proba(ready[fold])[:, positive]
    out["single_holdout"] = auroc_from(scores)

    # 3. Balanced counts: synthetic downsampled to the natural sample size.
    rng = np.random.default_rng(seed)
    pos, neg = np.where(labels == 1)[0], np.where(labels == 0)[0]
    keep = np.sort(np.concatenate([rng.choice(pos, size=min(len(pos), len(neg)),
                                              replace=False), neg]))
    sub_oof = oof_binary(matrix[keep], labels[keep], groups[keep], folds, seed)
    out["balanced"] = cluster_bootstrap(
        lambda idx: binary_auroc(labels[keep], sub_oof, idx),
        groups[keep], bootstrap, seed)

    below = out["as_run"]["ci_high"] < 0.5
    null_ok = abs(out["shuffled"]["mean"] - 0.5) < 0.03
    out["interpretation"] = (
        "not below chance --- nothing to diagnose" if not below else
        "anti-generalising fit: the permuted-label control is centred at 0.5, so the "
        "structure is sound and the probe is learning a fold-specific direction that "
        "reverses out of fold. Report as a measurement failure, not as realism."
        if null_ok else
        "pipeline defect: even permuted labels depart from 0.5, so the fold or grouping "
        "structure is at fault and this condition's C2ST cannot be interpreted at all.")
    return out


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
    cache = {}
    for name in EXTERNAL:
        path = root / args.experiment / "baseline_scores" / f"{name}.json"
        if path.exists():
            cache[name] = json.loads(path.read_text(encoding="utf-8"))

    natural = [r for r in everything if str(r.get("fault_source", "")) == "clean"
               and bool(r.get("natural_retrieval_failure"))]
    print(f"{len(natural)} naturally degraded traces; margins {DELTAS}")

    result = {"analysis": "c2st_equivalence", "experiment": args.experiment,
              "n_natural": len(natural), "margins": list(DELTAS),
              "scale": "D = 2*|AUC-0.5|; equivalence requires the upper bound below delta",
              "by_condition": {}, "diagnosis": {}}

    for family in COMPARABLE:
        for tier in ("T0", "T1", "T2"):
            synthetic = [r for r in everything if str(r.get("fault_tier")) == tier
                         and str(r.get("fault_family")) == family]
            if not synthetic:
                continue
            rows = natural + synthetic
            labels = np.asarray([0] * len(natural) + [1] * len(synthetic))
            groups = np.asarray([str(r["query_id"]) for r in rows])
            entry = {}
            for kind in ("surface", "external", "mirage"):
                matrix = build(rows, kind, cache)
                if matrix is None:
                    continue
                oof = oof_binary(matrix, labels, groups, args.folds, args.seed)
                auc = cluster_bootstrap(lambda idx: binary_auroc(labels, oof, idx),
                                        groups, args.bootstrap, args.seed)
                entry[kind] = {**auc, **equivalence(auc["point"], auc["ci_low"],
                                                    auc["ci_high"])}
                # Diagnose any cell whose interval sits wholly below chance.
                if auc["ci_high"] < 0.5:
                    print(f"  diagnosing {tier}:{family}/{kind} (below chance) ...")
                    result["diagnosis"][f"{tier}:{family}/{kind}"] = diagnose(
                        rows, labels, groups, matrix, args.folds,
                        max(args.bootstrap // 5, 200), args.seed)
            result["by_condition"][f"{tier}:{family}"] = entry

    atomic_json(Path(args.output), result)

    print(f"\n{'condition':26s}{'rep':10s}{'AUROC':>18s}{'D upper':>9s}  verdict")
    for cond, entry in result["by_condition"].items():
        for kind, v in entry.items():
            if kind == "mirage":
                continue          # shown in the paper only as the non-held-out contrast
            print(f"{cond:26s}{kind:10s}"
                  f"{v['point']:>7.3f} [{v['ci_low']:.3f},{v['ci_high']:.3f}]"
                  f"{v['D_upper_bound']:>9.3f}  {v['verdict']}")

    equivalent = [f"{c}/{k}" for c, e in result["by_condition"].items()
                  for k, v in e.items() if k != "mirage" and v["equivalent_at_0.05"]]
    print("\n" + "=" * 74)
    print("practically equivalent to natural at delta=0.05: "
          + (", ".join(equivalent) if equivalent else "NONE"))
    for name, d in result["diagnosis"].items():
        print(f"\ndiagnosis {name}: {d['interpretation']}")
        print(f"  as run {d['as_run']['point']:.3f}  permuted {d['shuffled']['mean']:.3f}  "
              f"holdout {d['single_holdout']['point']:.3f}  "
              f"balanced {d['balanced']['point']:.3f}")
    print(f"\nwritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
