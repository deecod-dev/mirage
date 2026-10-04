#!/usr/bin/env python
"""Does MIRAGE + an external detector beat either one alone?

The head-to-head table (`run_baselines.py`) shows a split decision: MIRAGE wins on the
three pipeline-structural fault families, `nli-large` wins on grounding. That pattern is
only interesting if the two are *complementary* — if combining them beats both. If instead
the combination just tracks whichever member is stronger, they are redundant and the
paper should say so.

This script answers that question and nothing else. It runs on CPU in about a minute
because every input already exists: MIRAGE's 14 metrics live on the traces, and the
external scores are in `artifacts/<experiment>/baseline_scores/`. Nothing is recomputed
and no model is loaded.

    python scripts/run_union.py --experiment E4_fingerprint_v2 \
        --output results/union_v2.json

WHAT WOULD MAKE THIS AN HONEST RESULT

A probe with more features cannot do worse *in-sample*, so "the union scored higher" is
not by itself evidence of anything. Three things make the comparison real:

* every combination is scored on the **same 25 query-disjoint splits**, so the gain is
  measured out-of-sample on queries the probe never saw;
* the comparison is **paired** — the per-split difference is computed on identical test
  rows, then summarised across splits, so split-to-split difficulty cancels;
* the interval is reported, and a gain whose interval spans zero is reported as no gain.

The interval here is the 2.5/97.5 percentile of the paired per-split differences. It
describes spread across random query halves. It is deliberately not called a bootstrap CI:
the project's headline numbers use a 1000x query-level cluster bootstrap and this is a
cheaper, different quantity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from mirage.analysis import ARMS, FEATURES, dedup, read_rows
from mirage.artifacts import atomic_json
from mirage.probe import new_probe, raw_matrix

#: Single-score external detectors, read from the cache `run_baselines.py` wrote.
EXTERNAL = ("nli", "nli-large", "judge", "hhem")

#: Combinations to score. Each value is (mirage_features, external_score_names).
#: `mirage` and the externals alone are the references every union is measured against;
#: `external_only` answers "what do you get without MIRAGE at all?".
COMBOS: dict[str, tuple[Sequence[str], Sequence[str]]] = {
    "trivial":                  (ARMS["trivial_baselines"], ()),
    "mirage":                   (FEATURES, ()),
    "nli-large":                ((), ("nli-large",)),
    "judge":                    ((), ("judge",)),
    "external_only":            ((), ("nli-large", "judge")),
    "mirage+nli-large":         (FEATURES, ("nli-large",)),
    "mirage+judge":             (FEATURES, ("judge",)),
    "mirage+nli-large+judge":   (FEATURES, ("nli-large", "judge")),
}

#: Unions are judged against the best of these, not against a hand-picked weak member.
REFERENCES = ("mirage", "nli-large", "judge", "external_only")


def trace_key(row: dict[str, Any]) -> str:
    return f"{row['query_id']}|{row.get('fault_source')}|{row.get('injector_seed')}"


def load_external(root: Path, experiment: str, rows: Sequence[dict]) -> dict[str, np.ndarray]:
    """Read cached single-score baselines, keeping only those covering every row."""
    found: dict[str, np.ndarray] = {}
    for name in EXTERNAL:
        path = root / experiment / "baseline_scores" / f"{name}.json"
        if not path.exists():
            continue
        cached = json.loads(path.read_text(encoding="utf-8"))
        missing = sum(1 for r in rows if trace_key(r) not in cached)
        if missing:
            print(f"  {name}: SKIPPED — {missing}/{len(rows)} traces unscored. "
                  f"Run run_baselines.py --baselines {name} first.")
            continue
        found[name] = np.asarray([cached[trace_key(r)] for r in rows], dtype=float)
        print(f"  {name}: {len(rows)} cached scores")
    return found


def query_splits(rows: Sequence[dict], repeats: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Identical construction to run_baselines.py, so the two tables are comparable."""
    queries = np.unique([str(r["query_id"]) for r in rows])
    owner = np.asarray([str(r["query_id"]) for r in rows])
    rng = np.random.default_rng(seed)
    splits = []
    for _ in range(repeats):
        shuffled = rng.permutation(queries)
        half = set(shuffled[:len(shuffled) // 2])
        mask = np.asarray([q in half for q in owner])
        splits.append((np.where(mask)[0], np.where(~mask)[0]))
    return splits


def prepare(matrix: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Impute with the TRAINING median and append definedness indicators."""
    with np.errstate(all="ignore"):
        medians = np.nan_to_num(np.nanmedian(matrix[train_idx], axis=0), nan=0.0)
    defined = np.isfinite(matrix).astype(float)
    filled = np.where(np.isfinite(matrix), matrix, medians)
    return np.hstack([filled, defined]) if defined.min() == 0 else filled


def per_split_auroc(matrix: np.ndarray, labels: np.ndarray, splits) -> np.ndarray:
    """AUROC on each split's held-out half. NaN where a split is degenerate."""
    out = []
    for train_idx, test_idx in splits:
        if len(set(labels[train_idx])) < 2 or len(set(labels[test_idx])) < 2:
            out.append(np.nan)
            continue
        prepared = prepare(matrix, train_idx)
        model = new_probe()
        model.fit(prepared[train_idx], labels[train_idx])
        positive = list(model.classes_).index(1)
        out.append(roc_auc_score(labels[test_idx],
                                 model.predict_proba(prepared[test_idx])[:, positive]))
    return np.asarray(out, dtype=float)


def paired_delta(union: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Summarise the per-split difference between two combinations."""
    delta = union - reference
    ok = np.isfinite(delta)
    if not ok.any():
        return {}
    delta = delta[ok]
    low, high = np.percentile(delta, [2.5, 97.5])
    return {"delta": float(delta.mean()), "lo": float(low), "hi": float(high),
            "wins": int((delta > 0).sum()), "n_splits": int(delta.size),
            # A gain is only called real when the whole interval clears zero.
            "significant": bool(low > 0)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.artifacts).resolve()
    rows = dedup(read_rows(root, args.experiment))
    rows = [r for r in rows if r.get("selected_context") and r.get("answer")]
    print(f"{len(rows)} traces, {len({r['query_id'] for r in rows})} queries\n"
          f"reading cached external scores:")
    external = load_external(root, args.experiment, rows)

    combos = {name: (feats, tuple(e for e in ext if e in external))
              for name, (feats, ext) in COMBOS.items()}
    combos = {name: value for name, value in combos.items()
              if value[0] or value[1]}                 # drop combos with nothing to score
    mirage_matrix = raw_matrix(rows, FEATURES)
    trivial_matrix = raw_matrix(rows, ARMS["trivial_baselines"])

    def build(features: Sequence[str], names: Sequence[str]) -> np.ndarray:
        parts = []
        if features:
            parts.append(mirage_matrix if list(features) == list(FEATURES) else trivial_matrix)
        parts.extend(external[n].reshape(-1, 1) for n in names)
        return np.hstack(parts)

    clean = [r for r in rows if r.get("fault_source") == "clean"]
    conditions = sorted({str(r["fault_source"]) for r in rows} - {"clean"})
    index = {trace_key(r): i for i, r in enumerate(rows)}

    table: dict[str, Any] = {}
    for condition in conditions:
        subset = clean + [r for r in rows if str(r.get("fault_source")) == condition]
        picked = np.asarray([index[trace_key(r)] for r in subset])
        labels = np.asarray([int(r.get("fault_source") != "clean") for r in subset])
        splits = query_splits(subset, args.repeats, args.seed)

        curves = {name: per_split_auroc(build(*value)[picked], labels, splits)
                  for name, value in combos.items()}
        entry: dict[str, Any] = {
            name: {"auroc": float(np.nanmean(curve)), "sd": float(np.nanstd(curve)),
                   "n_features": int(build(*combos[name]).shape[1])}
            for name, curve in curves.items()}

        # Every union is scored against the strongest reference available, so a union
        # cannot look good merely by beating its own weakest member.
        for name in combos:
            if "+" not in name:
                continue
            best = max((r for r in REFERENCES if r in curves),
                       key=lambda r: float(np.nanmean(curves[r])))
            entry[name]["vs_best_single"] = best
            entry[name]["gain"] = paired_delta(curves[name], curves[best])
        table[condition] = entry

    result = {"analysis": "union_of_detectors", "experiment": args.experiment,
              "n_traces": len(rows), "n_queries": len({r["query_id"] for r in rows}),
              "repeats": args.repeats, "seed": args.seed,
              "externals_used": sorted(external), "by_condition": table,
              "interval": "2.5/97.5 percentile of paired per-split differences across "
                          f"{args.repeats} query-disjoint splits; not a cluster bootstrap"}
    atomic_json(Path(args.output), result)

    names = list(combos)
    width = max(len(n) for n in names) + 2
    print(f"\n{'condition':24s}" + "".join(f"{n:>{width}s}" for n in names))
    for condition, entry in table.items():
        print(f"{condition:24s}" + "".join(f"{entry[n]['auroc']:>{width}.3f}" for n in names))

    print("\npaired gain of each union over the best single detector on that condition")
    print(f"{'condition':24s} {'union':24s} {'vs':16s} {'gain':>7s}  {'interval':>16s}")
    for condition, entry in table.items():
        for name in names:
            gain = entry.get(name, {}).get("gain")
            if not gain:
                continue
            mark = "  <- real" if gain["significant"] else ""
            print(f"{condition:24s} {name:24s} {entry[name]['vs_best_single']:16s} "
                  f"{gain['delta']:+7.3f}  [{gain['lo']:+.3f},{gain['hi']:+.3f}]{mark}")

    print(f"\nwritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
