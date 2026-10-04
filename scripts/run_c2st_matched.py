#!/usr/bin/env python
"""C3b --- ecological validity with the query held fixed.

The unmatched C2ST compares synthetic faulty traces against the $232$ naturally degraded
ones, and those $232$ are not a random sample of queries: they are precisely the queries on
which the retrieval stack failed. They are plausibly harder, more multi-hop, less lexically
aligned. If the synthetic class is drawn from the wider query population while the natural
class comes from that special subset, a discriminator can score well by learning
\"this is one of the hard queries\" rather than anything about how the fault was made.

Our own diagnostic pointed at this: most queries contributed traces of only one class, which
is also what made the within-query permutation null weak.

The fix needs no new generation. Restrict both classes to queries that have a natural
failure, so every query contributes a natural trace *and* its synthetic counterpart. The
comparison becomes

    P(X_synthetic | F, Q=q)   versus   P(X_natural | F, Q=q)

and the query can no longer carry the signal. It also licenses the strong conditional null:
permuting the synthetic/natural label *within* a query is now a real permutation, because
both labels are present.

    python scripts/run_c2st_matched.py --experiment E4_fingerprint_v2 \
        --output results/c2st_matched_v2.json

Either outcome is informative. If the distances survive matching, the ecological result
stands on much firmer ground. If they collapse, then the unmatched C2ST was measuring the
natural-failure selection process rather than the injector --- which, in a paper about
measurement validity, is a finding rather than an embarrassment.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from mirage.analysis import FEATURES, dedup, read_rows
from mirage.artifacts import atomic_json
from mirage.probe import raw_matrix
from mirage.validity import binary_auroc, cluster_bootstrap, oof_binary, permutation_null

COMPARABLE = ("query_retrieval", "source_degradation")
EXTERNAL = ("nli", "nli-large", "judge")
DELTAS = (0.05, 0.10, 0.15, 0.20)


def trace_key(row: dict) -> str:
    return f"{row['query_id']}|{row.get('fault_source')}|{row.get('injector_seed')}"


def surface_features(row: dict) -> list[float]:
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


def equivalence(point: float, lo: float, hi: float) -> dict:
    d_upper = 2 * max(abs(lo - 0.5), abs(hi - 0.5))
    out = {"D": float(2 * abs(point - 0.5)), "D_upper_bound": float(d_upper)}
    for delta in DELTAS:
        out[f"equivalent_at_{delta}"] = bool(d_upper < delta)
    out["verdict"] = ("practically equivalent" if d_upper < DELTAS[0] else
                      "underpowered" if lo <= 0.5 <= hi else "distinguishable")
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
    natural_queries = {str(r["query_id"]) for r in natural}
    print(f"{len(natural)} natural traces over {len(natural_queries)} queries")

    result = {"analysis": "query_matched_c2st", "experiment": args.experiment,
              "n_natural": len(natural), "margins": list(DELTAS),
              "design": "both classes restricted to queries that have a natural failure, "
                        "so every query contributes a natural and a synthetic trace",
              "by_condition": {}}

    for family in COMPARABLE:
        for tier in ("T0", "T1", "T2"):
            synthetic = [r for r in everything
                         if str(r.get("fault_tier")) == tier
                         and str(r.get("fault_family")) == family
                         # the matching: synthetic traces only from natural-failure queries
                         and str(r["query_id"]) in natural_queries]
            if not synthetic:
                continue
            shared = {str(r["query_id"]) for r in synthetic} & natural_queries
            rows = [r for r in natural if str(r["query_id"]) in shared] + \
                   [r for r in synthetic if str(r["query_id"]) in shared]
            labels = np.asarray([0] * sum(1 for r in natural if str(r["query_id"]) in shared)
                                + [1] * sum(1 for r in synthetic if str(r["query_id"]) in shared))
            groups = np.asarray([str(r["query_id"]) for r in rows])
            entry = {"n_queries": len(shared), "n_synthetic": int((labels == 1).sum()),
                     "n_natural": int((labels == 0).sum())}

            for kind in ("surface", "external", "mirage"):
                matrix = build(rows, kind, cache)
                if matrix is None:
                    continue
                oof = oof_binary(matrix, labels, groups, args.folds, args.seed)
                auc = cluster_bootstrap(lambda idx: binary_auroc(labels, oof, idx),
                                        groups, args.bootstrap, args.seed)
                # Now that every query carries both classes, permuting the label inside a
                # query is a genuine null rather than a no-op.
                null = permutation_null(
                    lambda y: binary_auroc(y, oof_binary(matrix, y, groups, args.folds,
                                                         args.seed)),
                    labels, groups, repeats=10, seed=args.seed)
                entry[kind] = {**auc, **equivalence(auc["point"], auc["ci_low"],
                                                    auc["ci_high"]),
                               "permutation_null": null}
            result["by_condition"][f"{tier}:{family}"] = entry
            cells = "  ".join(f"{k}={entry[k]['point']:.3f}" for k in
                              ("surface", "external", "mirage") if k in entry)
            print(f"  {tier}:{family:22s} {len(shared):4d} queries  {cells}")

    atomic_json(Path(args.output), result)

    print(f"\n{'condition':26s}{'rep':10s}{'AUROC':>18s}{'D upper':>9s}{'null':>7s}  verdict")
    for cond, entry in result["by_condition"].items():
        for kind in ("surface", "external"):
            v = entry.get(kind)
            if not v:
                continue
            print(f"{cond:26s}{kind:10s}{v['point']:>7.3f} "
                  f"[{v['ci_low']:.3f},{v['ci_high']:.3f}]{v['D_upper_bound']:>9.3f}"
                  f"{v['permutation_null']['mean']:>7.3f}  {v['verdict']}")

    print("\nequivalence under increasing margins (external representation):")
    header = "  " + f"{'condition':26s}" + "".join(f"{'d=' + str(d):>9s}" for d in DELTAS)
    print(header)
    for cond, entry in result["by_condition"].items():
        v = entry.get("external")
        if v:
            print(f"  {cond:26s}" + "".join(
                f"{'pass' if v[f'equivalent_at_{d}'] else 'fail':>9s}" for d in DELTAS))
    print(f"\nwritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
