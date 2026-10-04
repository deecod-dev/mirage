#!/usr/bin/env python
"""Are hardened synthetic faults actually distributed like natural ones?

The injector ladder calls T2 "hardened", and the paper has been reading that as "more
realistic". It does not mean that. T2 was designed knowing the MIRAGE metric suite, so
what it demonstrably is is *hard to detect with our chosen measurements* --- which is a
statement about our metrics, not about the world. A reviewer is entitled to say: of course
AUROC falls at T2, you built T2 to defeat that detector.

This script gives realism an operational definition via a classifier two-sample test. For
each tier we ask an independent discriminator to separate synthetic faulty traces from
naturally degraded ones:

    AUROC = 0.5  ->  indistinguishable from natural (ecologically valid)
    AUROC = 1.0  ->  trivially recognisable as manufactured

If hardening genuinely buys realism, discrimination should fall monotonically T0 -> T1 ->
T2. If instead T2 is merely invisible to our metrics, discrimination can stay high while
detection collapses --- and those two outcomes have opposite implications for anyone
using injected faults as a proxy.

    python scripts/run_realism_c2st.py --experiment E4_fingerprint_v2 \
        --output results/realism_c2st_v2.json

THE MEASUREMENT MUST BE HELD OUT. Scoring realism with the same features T2 was hardened
against would be circular in exactly the way this project exists to complain about. So the
discriminator never sees a MIRAGE metric. It uses two independent families:

* `surface` --- lengths, passage counts, lexical overlap, type-token ratio. Computed here
  from raw trace text, sharing no code with `mirage/metrics.py`.
* `external` --- cached NLI and LLM-judge scores from `run_baselines.py`, which are
  third-party models with no knowledge of our injectors.

The MIRAGE arm is reported too, but only as a contrast: a large gap between "invisible to
MIRAGE" and "still obvious to a held-out discriminator" is itself the finding.
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
from mirage.validity import binary_auroc, cluster_bootstrap, oof_binary

#: Natural degradation only exists for retrieval-side faults, so only these two families
#: have a natural counterpart to be compared against.
COMPARABLE = ("query_retrieval", "source_degradation")
EXTERNAL = ("nli", "nli-large", "judge")


def surface_features(row: dict) -> list[float]:
    """Cheap text statistics, deliberately sharing no code with the metric suite."""
    answer = str(row.get("answer") or "")
    passages = [str(p.get("text") or "") for p in row.get("selected_context") or []]
    context = " ".join(passages)
    question = str(row.get("question") or "")

    a_tok = answer.lower().split()
    c_tok = context.lower().split()
    q_tok = question.lower().split()
    a_set, c_set, q_set = set(a_tok), set(c_tok), set(q_tok)

    grounded = len(a_set & c_set) / len(a_set) if a_set else 0.0
    q_cover = len(q_set & c_set) / len(q_set) if q_set else 0.0
    ttr = len(a_set) / len(a_tok) if a_tok else 0.0
    repeat = max(Counter(a_tok).values()) / len(a_tok) if a_tok else 0.0
    return [float(len(answer)), float(len(a_tok)), float(len(passages)),
            float(len(context)), float(np.mean([len(p) for p in passages]) if passages else 0.0),
            grounded, q_cover, ttr, repeat]


def load_external(root: Path, experiment: str) -> dict[str, dict[str, float]]:
    out = {}
    for name in EXTERNAL:
        path = root / experiment / "baseline_scores" / f"{name}.json"
        if path.exists():
            out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out


def trace_key(row: dict) -> str:
    return f"{row['query_id']}|{row.get('fault_source')}|{row.get('injector_seed')}"


def build(rows, kind: str, cache) -> np.ndarray | None:
    if kind == "surface":
        return np.asarray([surface_features(r) for r in rows], dtype=float)
    if kind == "mirage":
        return raw_matrix(rows, FEATURES)
    usable = [n for n in EXTERNAL
              if n in cache and all(trace_key(r) in cache[n] for r in rows)]
    if not usable:
        return None
    return np.asarray([[cache[n][trace_key(r)] for n in usable] for r in rows], dtype=float)


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
    print(f"external score caches available: {sorted(cache)}")

    natural = [r for r in everything
               if str(r.get("fault_source", "")) == "clean"
               and bool(r.get("natural_retrieval_failure"))]
    print(f"{len(natural)} naturally degraded traces as the reference sample")

    result = {"analysis": "realism_classifier_two_sample_test",
              "experiment": args.experiment, "n_natural": len(natural),
              "representations": ["surface", "external", "mirage"],
              "note": "AUROC near 0.5 means the synthetic tier is indistinguishable from "
                      "natural degradation under that representation; 1.0 means trivially "
                      "recognisable as manufactured",
              "by_condition": {}}

    for family in COMPARABLE:
        for tier in ("T0", "T1", "T2"):
            synthetic = [r for r in everything
                         if str(r.get("fault_tier")) == tier
                         and str(r.get("fault_family")) == family]
            if not synthetic:
                continue
            rows = natural + synthetic
            labels = np.asarray([0] * len(natural) + [1] * len(synthetic))
            groups = np.asarray([str(r["query_id"]) for r in rows])
            entry = {"n_synthetic": len(synthetic)}

            for kind in ("surface", "external", "mirage"):
                matrix = build(rows, kind, cache)
                if matrix is None:
                    entry[kind] = {"skipped": "no cached scores covering these traces"}
                    continue
                scores = oof_binary(matrix, labels, groups, args.folds, args.seed)
                entry[kind] = cluster_bootstrap(
                    lambda idx: binary_auroc(labels, scores, idx),
                    groups, args.bootstrap, args.seed)
            result["by_condition"][f"{tier}:{family}"] = entry

            cells = "  ".join(
                f"{k}={entry[k]['point']:.3f}" if "point" in entry.get(k, {}) else f"{k}=--"
                for k in ("surface", "external", "mirage"))
            print(f"  {tier}:{family:20s} n={len(synthetic):4d}  {cells}")

    atomic_json(Path(args.output), result)

    print("\n" + "=" * 74)
    print("Does hardening buy realism? (held-out representations only)")
    for family in COMPARABLE:
        for kind in ("surface", "external"):
            row = []
            for tier in ("T0", "T1", "T2"):
                e = result["by_condition"].get(f"{tier}:{family}", {}).get(kind, {})
                row.append(e.get("point", float("nan")))
            trend = ("falls with hardening" if row[0] > row[1] > row[2]
                     else "NOT monotone --- hardening does not make traces more natural")
            print(f"  {family:20s} {kind:9s} "
                  f"T0 {row[0]:.3f} -> T1 {row[1]:.3f} -> T2 {row[2]:.3f}   {trend}")
    print(f"\nwritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
