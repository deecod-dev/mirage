#!/usr/bin/env python
"""C7 external V1 and strongest defensible released-data V2 audit on AEGIS."""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from mirage.artifacts import atomic_json
from mirage.external_audit import (DIRECT_BOOKKEEPING_PATHS, available_text,
                                   single_injection, task_group, trajectory_text,
                                   unsafe_all_fields_text)
from mirage.validity import (binary_auroc, cluster_bootstrap, paired_cluster_bootstrap,
                             query_folds)


PRIMARY_MODES = ("FM-1.1", "FM-2.2")


def oof_text(texts, labels, groups, folds, seed):
    labels = np.asarray(labels, int)
    out = np.full(len(labels), np.nan)
    for test in query_folds(groups, folds, seed):
        train = np.setdiff1d(np.arange(len(labels)), test)
        if len(np.unique(labels[train])) < 2:
            continue
        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20_000,
                            sublinear_tf=True),
            LogisticRegression(max_iter=5000, class_weight="balanced"))
        model.fit([texts[i] for i in train], labels[train])
        out[test] = model.predict_proba([texts[i] for i in test])[:, 1]
    return out


def text_auc(texts, labels, groups, folds, bootstrap, seed):
    scores = oof_text(texts, labels, groups, folds, seed)
    interval = cluster_bootstrap(lambda i: binary_auroc(labels, scores, i), groups,
                                 bootstrap, seed)
    return {**interval, "raw_predictions": [float(x) for x in scores]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="external/AEGIS")
    parser.add_argument("--data", default="external/AEGIS/evaluation/data/aegis.jsonl")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--output", default="results/external_aegis_audit_v3.json")
    args = parser.parse_args()

    repo, data = Path(args.repo), Path(args.data)
    if not data.exists():
        raise SystemExit(f"released AEGIS sample missing: {data}")
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    eligible = []
    for row in rows:
        injection = single_injection(row)
        if injection and (row.get("ground_truth") or {}).get("is_injection_successful"):
            eligible.append((row, injection))

    # V1: two predeclared, well-represented error modes. Each is audited one-vs-rest on
    # every released trace, including valid multi-injection traces. The primary FULL/AVAILABLE arm
    # excludes targets; unsafe_all_fields is a diagnostic showing what would happen if a
    # consumer accidentally passed output/ground_truth objects to the evaluator.
    groups = np.asarray([task_group(row) for row in rows])
    arms = {"available": [available_text(row) for row in rows],
            "decoupled": [trajectory_text(row) for row in rows],
            "unsafe_all_fields_diagnostic": [unsafe_all_fields_text(row) for row in rows]}
    v1_results = {}
    for mode_index, mode in enumerate(PRIMARY_MODES):
        labels = np.asarray([
            int(mode in {str(x.get("error_type") or "") for x in
                         ((row.get("ground_truth") or {}).get("injected_agents") or [])})
            for row in rows])
        mode_arms = {name: text_auc(texts, labels, groups, args.folds, args.bootstrap,
                                    args.seed + 10 * mode_index + i)
                     for i, (name, texts) in enumerate(arms.items())}
        available_scores = np.asarray(mode_arms["available"]["raw_predictions"])
        decoupled_scores = np.asarray(mode_arms["decoupled"]["raw_predictions"])
        paired_loss = paired_cluster_bootstrap(
            lambda i: binary_auroc(labels, available_scores, i),
            lambda i: binary_auroc(labels, decoupled_scores, i),
            groups, args.bootstrap, args.seed + 100 + mode_index)
        delta = paired_loss["delta"]
        v1_results[mode] = {
            "n": len(rows), "n_positive": int(labels.sum()), "arms": mode_arms,
            "available_minus_decoupled": float(delta),
            "paired_available_minus_decoupled": paired_loss,
            "verdict": (
                "fail: material direct-bookkeeping loss established"
                if paired_loss["ci_low"] >= 0.05 else
                "pass: upper confidence bound is below the material-loss margin"
                if paired_loss["ci_high"] < 0.05 else
                "inconclusive: point loss is below 0.05 but its interval crosses the margin")}

    # Released-data V2: AEGIS exposes prompt and response mechanisms, but its 600-row
    # sample has no adequately powered same-mode/same-task matched contrast. Restrict to
    # frameworks that contain both strategies and label this weaker test explicitly.
    frameworks = {}
    for row, injection in eligible:
        framework = str((row.get("metadata") or {}).get("framework"))
        frameworks.setdefault(framework, set()).add(injection["strategy"])
    mixed_frameworks = {f for f, strategies in frameworks.items()
                        if {"prompt_injection", "response_corruption"} <= strategies}
    v2 = [(row, injection) for row, injection in eligible
          if str((row.get("metadata") or {}).get("framework")) in mixed_frameworks
          and injection["strategy"] in {"prompt_injection", "response_corruption"}]
    v2_labels = np.asarray([int(injection["strategy"] == "response_corruption")
                            for _, injection in v2])
    v2_groups = np.asarray([task_group(row) for row, _ in v2])
    v2_result = text_auc([trajectory_text(row) for row, _ in v2], v2_labels,
                         v2_groups, args.folds, args.bootstrap, args.seed + 20)

    result = {
        "analysis": "external_aegis_validity_audit",
        "analysis_status": "confirmatory V1; diagnostic weak V2",
        "aegis_commit": commit, "data_file": str(data), "n_released_rows": len(rows),
        "schema_audit": {
            "evaluator_visible": ["input.query", "input.conversation_history",
                                  "input.final_output", "metadata.framework",
                                  "metadata.benchmark", "metadata.model",
                                  "metadata.num_agents", "metadata.task_type"],
            "direct_injection_bookkeeping_paths": list(DIRECT_BOOKKEEPING_PATHS),
            "primary_available_excludes_targets": True,
        },
        "v1": {
            "modes": list(PRIMARY_MODES), "by_mode": v1_results,
        },
        "v2": {
            "status": "weak mechanism contrast; NOT effect-matched",
            "reason": ("the released 600-row subset has no adequately powered contrast "
                       "holding error mode, source task, framework and severity fixed"),
            "frameworks": sorted(mixed_frameworks), "n": len(v2),
            "class_counts": dict(Counter(i["strategy"] for _, i in v2)),
            "mechanism_auroc": v2_result,
        },
        "negative_control_assertions": {
            "targets_excluded_from_available": True,
            "task_disjoint_folds": True,
            "explicit_label_arm_is_diagnostic_only": True,
        },
    }
    atomic_json(Path(args.output), result)
    summary = {
        "v1": {mode: {"n": value["n"], "n_positive": value["n_positive"],
                       "available": value["arms"]["available"]["point"],
                       "decoupled": value["arms"]["decoupled"]["point"],
                       "delta": value["available_minus_decoupled"],
                       "delta_ci": [value["paired_available_minus_decoupled"]["ci_low"],
                                    value["paired_available_minus_decoupled"]["ci_high"]],
                       "verdict": value["verdict"]}
               for mode, value in result["v1"]["by_mode"].items()},
        "v2": {"status": result["v2"]["status"], "n": result["v2"]["n"],
               "mechanism_auroc": result["v2"]["mechanism_auroc"]["point"]},
    }
    print(json.dumps(summary, indent=2))
    print(f"written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
