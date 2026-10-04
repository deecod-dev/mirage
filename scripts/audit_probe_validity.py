"""Validity audit for the E4 fingerprint probe.

Answers the question "how much of the held-out AUROC is the probe recognising a
fault, and how much is a metric reading back a field the injector just wrote?"

Everything is recomputed from the trace artifacts on disk; nothing is read from
results/. Reports, in order:

  1. seed determinism      -- are the N seeds real replicates or duplicates?
  2. tier collapse         -- do T0/T1/T2 actually differ, per family?
  3. exact-value signatures-- does any condition pin a metric to one exact float?
  4. metric redundancy     -- pairs with |r| > 0.95 on finite rows (the E2 audit)
  5. single-feature AUROC  -- best lone feature per class
  6. feature ablations     -- macro AUROC with coupled/trivial groups removed
  7. cluster bootstrap CI  -- resampling whole queries, not rows
  8. permutation null      -- shuffled labels must land at chance

Usage:
    python scripts/audit_probe_validity.py \
        --train-experiment E3_positive_control \
        --test-experiment E4_fingerprint \
        --output results/E4_validity_audit.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mirage.analysis import COUPLED, FEATURES, TRIVIAL
from mirage.artifacts import atomic_json
from mirage.probe import FORBIDDEN_FEATURES

# Legacy runs also carried SEO, which E2 removed as an exact duplicate of TCS. Audit
# whatever the artifacts actually contain so old and new runs are both readable.
LEGACY_FEATURES = ["SEO"]

NAN_SENTINEL = -999.0


def resolve_features(root: Path, experiment: str) -> list[str]:
    """Audit whichever metrics the artifacts actually carry.

    Runs predating the E2 de-duplication also contain SEO. Rather than hard-coding a
    suite that only matches one generation of artifacts, take the intersection of the
    canonical list plus known legacy columns with what is on disk. FEATURES is module
    level because the worker pool reads it.
    """
    global FEATURES
    sample = next(iter(sorted((root / experiment).rglob("trace_*.json"))), None)
    if sample is None:
        raise FileNotFoundError(f"no trace artifacts found for {experiment} under {root}")
    present = set(json.loads(sample.read_text(encoding="utf-8")).get("metrics") or {})
    FEATURES = [f for f in list(FEATURES) + LEGACY_FEATURES if f in present]
    if not FEATURES:
        raise ValueError(f"no known metrics present in {sample}")
    return FEATURES


def _load(path: str) -> tuple:
    row = json.loads(Path(path).read_text(encoding="utf-8"))
    metrics = row.get("metrics") or {}
    return (str(row.get("query_id")), str(row.get("fault_source")), str(row.get("fault_tier")),
            str(row.get("fault_family")), int(row.get("seed", -1)),
            bool(row.get("natural_retrieval_failure", False)),
            sorted(row.get("supporting_titles", [])),
            [c["title"] for c in row.get("selected_context", [])],
            [float(metrics.get(f, np.nan)) for f in FEATURES])


def read_experiment(root: Path, experiment: str) -> dict[str, Any]:
    files = sorted(str(p) for p in (root / experiment).rglob("trace_*.json"))
    if not files:
        raise FileNotFoundError(f"no trace artifacts found for {experiment} under {root}")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        rows = list(pool.map(_load, files, chunksize=200))
    return {
        "qid": np.array([r[0] for r in rows]), "src": np.array([r[1] for r in rows]),
        "tier": np.array([r[2] for r in rows]), "family": np.array([r[3] for r in rows]),
        "seed": np.array([r[4] for r in rows]), "natural": np.array([r[5] for r in rows]),
        "supporting": [r[6] for r in rows], "selected": [r[7] for r in rows],
        "X": np.asarray([r[8] for r in rows], dtype=float),
    }


def probe_label(source: str) -> str:
    return "clean" if source == "clean" else source.split(":", 1)[-1]


def dedup(data: dict[str, Any]) -> np.ndarray:
    """Indices of the first occurrence of each (query_id, fault_source) pair."""
    seen: set[tuple[str, str]] = set()
    keep = []
    for i, (q, s) in enumerate(zip(data["qid"], data["src"])):
        if (q, s) not in seen:
            seen.add((q, s))
            keep.append(i)
    return np.asarray(keep)


def fit_probe(columns: list[str], x_train, y_train, x_test) -> tuple[np.ndarray, np.ndarray]:
    leaked = FORBIDDEN_FEATURES.intersection(columns)
    if leaked:
        raise ValueError(f"prohibited probe features: {sorted(leaked)}")
    idx = [FEATURES.index(c) for c in columns]
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=5_000, class_weight="balanced"))
    model.fit(x_train[:, idx], y_train)
    return model.predict_proba(x_test[:, idx]), model.classes_


def macro_auroc(y, probabilities) -> float:
    return float(roc_auc_score(y, probabilities, multi_class="ovr", average="macro"))


def cluster_bootstrap(y, probabilities, queries, seed: int, repeats: int) -> dict[str, float]:
    """Resample whole queries. Row-level resampling treats traces from one query as
    independent draws and reports a CI that is far too narrow."""
    unique = np.unique(queries)
    by_query = {q: np.where(queries == q)[0] for q in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_query[q] for q in picked])
        if len(set(y[idx])) < 2:
            continue
        values.append(macro_auroc(y[idx], probabilities[idx]))
    return {"point": macro_auroc(y, probabilities),
            "ci_low": float(np.quantile(values, .025)),
            "ci_high": float(np.quantile(values, .975))}


def audit_determinism(data: dict[str, Any]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[int]] = {}
    for i, (q, s) in enumerate(zip(data["qid"], data["src"])):
        groups.setdefault((q, s), []).append(i)
    replicated = {k: v for k, v in groups.items() if len(v) > 1}
    spread = 0.0
    identical = 0
    for idx in replicated.values():
        block = data["X"][idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            delta = np.nanmax(np.nanmax(block, axis=0) - np.nanmin(block, axis=0))
        delta = 0.0 if not np.isfinite(delta) else float(delta)
        spread = max(spread, delta)
        identical += delta < 1e-9
    return {"n_traces": len(data["src"]), "n_unique_conditions": len(groups),
            "n_unique_queries": int(len(set(data["qid"].tolist()))),
            "seeds": sorted(set(data["seed"].tolist())),
            "replicated_groups": len(replicated), "byte_identical_groups": identical,
            "max_across_seed_spread": spread,
            "verdict": "seeds are duplicates — no stochasticity in the pipeline"
                       if replicated and identical == len(replicated) else "seeds vary"}


def audit_tier_collapse(data: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for family in sorted({f for f in data["family"] if f != "None"}):
        per_tier = {}
        for tier in ("T0", "T1", "T2"):
            mask = data["src"] == f"{tier}:{family}"
            per_tier[tier] = {q: row for q, row in zip(data["qid"][mask], data["X"][mask])}
        shared = sorted(set.intersection(*(set(v) for v in per_tier.values())) if all(per_tier.values()) else set())
        if not shared:
            continue
        pairs = {}
        for a, b in itertools.combinations(("T0", "T1", "T2"), 2):
            same = sum(np.max(np.abs(np.nan_to_num(per_tier[a][q], nan=NAN_SENTINEL)
                                     - np.nan_to_num(per_tier[b][q], nan=NAN_SENTINEL))) < 1e-9
                       for q in shared)
            pairs[f"{a}=={b}"] = int(same)
        out[family] = {"n_queries": len(shared), "identical_pairs": pairs,
                       "collapsed": all(v == len(shared) for v in pairs.values())}
    return out


def audit_exact_signatures(data: dict[str, Any], threshold: float = 0.95) -> dict[str, Any]:
    out = {}
    for condition in sorted(set(data["src"].tolist())):
        block = data["X"][data["src"] == condition]
        tells = {}
        for j, name in enumerate(FEATURES):
            column = block[:, j][np.isfinite(block[:, j])]
            if not len(column):
                continue
            values, counts = np.unique(np.round(column, 12), return_counts=True)
            share = counts.max() / len(column)
            if share > threshold:
                tells[name] = {"value": float(values[counts.argmax()]), "share": float(share)}
        if tells:
            out[condition] = tells
    return out


def audit_redundancy(data: dict[str, Any], threshold: float = 0.95) -> list[dict[str, Any]]:
    out = []
    for i, j in itertools.combinations(range(len(FEATURES)), 2):
        mask = np.isfinite(data["X"][:, i]) & np.isfinite(data["X"][:, j])
        if mask.sum() < 100:
            continue
        r = float(np.corrcoef(data["X"][mask, i], data["X"][mask, j])[0, 1])
        if abs(r) > threshold:
            delta = float(np.abs(data["X"][mask, i] - data["X"][mask, j]).max())
            out.append({"a": FEATURES[i], "b": FEATURES[j], "r": r, "max_abs_diff": delta,
                        "identical": delta < 1e-12})
    return out


def audit_natural_faults(data: dict[str, Any]) -> dict[str, Any]:
    per_query = {}
    for q, src, sup, sel in zip(data["qid"], data["src"], data["supporting"], data["selected"]):
        if src == "clean" and q not in per_query:
            per_query[q] = (set(sup), sel)
    n = len(per_query)
    if not n:
        return {"n_clean_queries": 0}
    no_gold = sum(1 for sup, sel in per_query.values() if not sup & set(sel))
    incomplete = sum(1 for sup, sel in per_query.values() if len(sup & set(sel)) < len(sup))
    sizes = [len(sel) for _, sel in per_query.values()]
    return {"n_clean_queries": n,
            "no_gold_title_in_selected": no_gold,
            "not_all_gold_titles_present": incomplete,
            "mean_selected_context_size": float(np.mean(sizes)),
            "max_selected_context_size": int(np.max(sizes)),
            "note": "HotpotQA distractor ships 10 candidate paragraphs per query; a mean "
                    "selected size at or near 10 means retrieval filters nothing."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--train-experiment", required=True)
    parser.add_argument("--test-experiment", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--bootstrap-repeats", type=int, default=1_000)
    parser.add_argument("--permutation-repeats", type=int, default=20)
    parser.add_argument("--gate", type=float, default=0.65, help="G-4 lower-CI floor")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.artifacts).resolve()
    resolve_features(root, args.train_experiment)
    train, test = read_experiment(root, args.train_experiment), read_experiment(root, args.test_experiment)

    overlap = set(train["qid"].tolist()) & set(test["qid"].tolist())
    if overlap:
        raise ValueError(f"query overlap between probe splits: {sorted(overlap)[:5]}")

    keep = dedup(test)
    x_train = np.nan_to_num(train["X"], nan=NAN_SENTINEL)
    y_train = np.asarray([probe_label(s) for s in train["src"]])
    x_test = np.nan_to_num(test["X"], nan=NAN_SENTINEL)[keep]
    y_test = np.asarray([probe_label(s) for s in test["src"]])[keep]
    q_test = test["qid"][keep]

    # --- single-feature AUROC: the best lone column per class ---------------
    single = {}
    for label in sorted(set(y_train)):
        binary = (y_test == label).astype(int)
        scored = [(max(a, 1 - a), f) for j, f in enumerate(FEATURES)
                  if np.std(x_test[:, j]) > 0
                  for a in [roc_auc_score(binary, x_test[:, j])]]
        scored.sort(reverse=True)
        single[label] = {"best_feature": scored[0][1], "auroc": float(scored[0][0])}

    # --- ablations ----------------------------------------------------------
    arms = {
        "all_features": FEATURES,
        "minus_AGS": [c for c in FEATURES if c != "AGS"],
        "minus_tool_coupling": [c for c in FEATURES if c not in ("TIHD_health", "TRTR_route")],
        "minus_all_coupled": [c for c in FEATURES if c not in COUPLED],
        "minus_coupled_and_trivial": [c for c in FEATURES if c not in set(COUPLED) | set(TRIVIAL)],
        "trivial_baselines_only": [c for c in TRIVIAL if c in FEATURES],
        "coupled_only": [c for c in COUPLED if c in FEATURES],
        "novel_metrics_only": [c for c in FEATURES if c not in TRIVIAL],
    }
    ablations = {}
    for tag, columns in arms.items():
        probabilities, classes = fit_probe(columns, x_train, y_train, x_test)
        ci = cluster_bootstrap(y_test, probabilities, q_test, args.seed, args.bootstrap_repeats)
        ablations[tag] = {
            "features": columns, "macro_auroc": ci["point"], "macro_auroc_ci": ci,
            "gate_pass": ci["ci_low"] > args.gate,
            "per_class": {c: float(roc_auc_score((y_test == c).astype(int), probabilities[:, i]))
                          for i, c in enumerate(classes)},
        }

    # --- permutation null ---------------------------------------------------
    rng = np.random.default_rng(args.seed)
    nulls = []
    for _ in range(args.permutation_repeats):
        probabilities, _ = fit_probe(FEATURES, x_train, rng.permutation(y_train), x_test)
        nulls.append(macro_auroc(y_test, probabilities))

    headline = ablations["all_features"]["macro_auroc"]
    decoupled = ablations["minus_all_coupled"]["macro_auroc"]
    result = {
        "analysis": "probe_validity_audit",
        "train_experiment": args.train_experiment, "test_experiment": args.test_experiment,
        "n_train": len(y_train), "n_test_rows_on_disk": len(test["src"]), "n_test_deduplicated": len(y_test),
        "features": FEATURES, "coupled_features": COUPLED, "trivial_baselines": TRIVIAL,
        "determinism": {"train": audit_determinism(train), "test": audit_determinism(test)},
        "tier_collapse": audit_tier_collapse(test),
        "exact_value_signatures": audit_exact_signatures(test),
        "metric_redundancy": audit_redundancy(test),
        "natural_faults": audit_natural_faults(test),
        "single_feature_auroc": single,
        "ablations": ablations,
        "permutation_null": {"mean": float(np.mean(nulls)), "max": float(np.max(nulls)),
                             "repeats": args.permutation_repeats},
        "coupling_share_of_above_chance_signal":
            float((headline - decoupled) / (headline - 0.5)) if headline > 0.5 else None,
        "seed": args.seed, "bootstrap_repeats": args.bootstrap_repeats,
    }
    atomic_json(Path(args.output), result)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in {"exact_value_signatures", "ablations"}}, indent=2))
    print("\nablations (macro AUROC, cluster-bootstrap CI over queries):")
    for tag, arm in ablations.items():
        ci = arm["macro_auroc_ci"]
        flag = "PASS" if arm["gate_pass"] else "FAIL"
        print(f"  {tag:28s} {ci['point']:.4f}  [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]  G-4 {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
