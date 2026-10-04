"""Shared analysis primitives for the probe experiments.

Three rules are enforced here rather than repeated in every script:

* **Deduplicate before scoring.** Under greedy decoding a clean trace is identical
  across seeds; only the injectors are stochastic. Scoring the raw file list counts
  the same clean row once per seed and reports a CI that is far too narrow.
* **Resample queries, not rows.** Every query contributes many traces, so a row-level
  bootstrap treats correlated rows as independent draws.
* **Report pre-registered arms, never one number.** docs/PLAN.md M-D0 requires trivial
  baselines in every detection table, and the injector-coupled metrics have to be
  separable from the rest or the headline cannot be interpreted.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from mirage.probe import FeatureSpace, new_probe

#: The metric suite after E2 removed the SEO/TCS duplicate.
FEATURES: list[str] = ["QCM", "ICRR", "SCR", "CDI_max", "MUT", "IPS", "TCS", "GSA",
                       "SRS", "AGS", "TRTR_route", "TIHD_health", "token_count", "answer_length"]

#: docs/PLAN.md M-D0 trivial baselines that exist inside the metric suite.
TRIVIAL: list[str] = ["token_count", "answer_length", "TRTR_route"]

#: Metrics a T0 injector can write into directly: grounding assigns `answer` (AGS),
#: tool faults assign `ok` (TIHD_health) and `name` (TRTR_route). Reporting an arm
#: without these is what separates "the probe found a fault" from "the probe read
#: back the assignment".
COUPLED: list[str] = ["AGS", "TIHD_health", "TRTR_route"]

ARMS: dict[str, list[str]] = {
    "trivial_baselines": TRIVIAL,
    "coupled_only": COUPLED,
    "semantic_only": [c for c in FEATURES if c not in TRIVIAL],
    "decoupled": [c for c in FEATURES if c not in COUPLED],
    "all_features": FEATURES,
}


def read_rows(root: Path, experiment: str) -> list[dict[str, Any]]:
    """Every trace under an experiment, including the shared clean-trace store."""
    rows = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((root / experiment).rglob("trace_*.json"))]
    if not rows:
        raise FileNotFoundError(f"no trace artifacts found for {experiment} under {root}")
    return rows


def dedup(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (query_id, fault_source, injector_seed)."""
    seen: set[tuple] = set()
    out = []
    for row in rows:
        key = (str(row.get("query_id")), str(row.get("fault_source")), row.get("injector_seed"))
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def dedup_strict(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (query_id, fault_source), collapsing injector-seed replicates.

    The registered key keeps seeds, on the grounds that a seed replicate is a genuinely
    different trace once injection is stochastic. The conservative reading is that five
    traces of the same (query, fault) are near-replicates whose retention lets the probe
    fit condition-specific noise. Reporting both is the only honest option, and the two
    must differ ONLY in the key --- comparing across scripts that also differ in their
    imputation confounds the two choices.
    """
    seen: set[tuple] = set()
    out = []
    for row in rows:
        key = (str(row.get("query_id")), str(row.get("fault_source")))
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def family_label(row: dict[str, Any]) -> str:
    source = str(row.get("fault_source", ""))
    return "clean" if source == "clean" else source.split(":", 1)[-1]


def queries(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    return np.asarray([str(row["query_id"]) for row in rows])


def cluster_bootstrap(score, rows: Sequence[dict[str, Any]], seed: int, repeats: int) -> dict[str, float]:
    """Bootstrap `score(indices)` by resampling whole queries."""
    query_ids = queries(rows)
    unique = np.unique(query_ids)
    grouped = {q: np.where(query_ids == q)[0] for q in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        picked = rng.choice(unique, size=len(unique), replace=True)
        try:
            values.append(score(np.concatenate([grouped[q] for q in picked])))
        except ValueError:
            continue
    if not values:
        raise ValueError("bootstrap could not form a scorable resample")
    return {"point": float(score(np.arange(len(rows)))),
            "ci_low": float(np.quantile(values, .025)),
            "ci_high": float(np.quantile(values, .975)),
            "n_resamples": len(values)}


def multiclass_auroc(labels: np.ndarray, probabilities: np.ndarray, indices: np.ndarray) -> float:
    subset = labels[indices]
    if len(set(subset)) < 2:
        raise ValueError("resample is single-class")
    return float(roc_auc_score(subset, probabilities[indices], multi_class="ovr", average="macro"))


def binary_auroc(labels: np.ndarray, scores: np.ndarray, indices: np.ndarray) -> float:
    subset = labels[indices]
    if len(set(subset)) < 2:
        raise ValueError("resample is single-class")
    return float(roc_auc_score(subset, scores[indices]))


def fit_multiclass(train: Sequence[dict], test: Sequence[dict], columns: Sequence[str],
                   label_of=family_label, *, with_indicators: bool = True
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (test labels, predicted probabilities, class order).

    `with_indicators` controls whether `<metric>__defined` columns join the matrix.
    Measured on the v1 artifacts they are nearly free: the `decoupled` arm scores 0.7003
    with them and 0.6988 without. The arm's rise from the audit's 0.6494 comes from
    replacing the -999.0 sentinel with median imputation, not from the indicators — the
    sentinel was distorting standardisation across every column at once. Keep the flag
    so the table can still be reported both ways.
    """
    space = FeatureSpace(columns, with_indicators=with_indicators).fit(train)
    y_train = np.asarray([label_of(r) for r in train])
    y_test = np.asarray([label_of(r) for r in test])
    unseen = set(y_test) - set(y_train)
    if unseen:
        raise ValueError(f"test contains labels absent from train: {sorted(unseen)}")
    model = new_probe()
    model.fit(space.transform(train), y_train)
    return y_test, model.predict_proba(space.transform(test)), model.classes_


class BinaryDetector:
    """A fitted binary probe that can be applied to more than one evaluation set.

    E5 needs exactly this: one detector, trained on injected faults, scored against
    both a held-out injected set and a natural-fault set. Refitting per evaluation
    would silently change the detector between the two arms.
    """

    def __init__(self, columns: Sequence[str]) -> None:
        self.space = FeatureSpace(columns)
        self.model = new_probe()
        self._positive = 1

    def fit(self, train: Sequence[dict], label_of) -> "BinaryDetector":
        self.space.fit(train)
        y_train = np.asarray([int(bool(label_of(r))) for r in train])
        if len(set(y_train)) < 2:
            raise ValueError("training split is single-class")
        self.model.fit(self.space.transform(train), y_train)
        self._positive = list(self.model.classes_).index(1)
        return self

    def score(self, rows: Sequence[dict]) -> np.ndarray:
        return self.model.predict_proba(self.space.transform(rows))[:, self._positive]


def fit_binary(train: Sequence[dict], test: Sequence[dict], columns: Sequence[str],
               label_of) -> tuple[np.ndarray, np.ndarray]:
    detector = BinaryDetector(columns).fit(train, label_of)
    return np.asarray([int(bool(label_of(r))) for r in test]), detector.score(test)


def assert_query_disjoint(train: Iterable[dict], test: Iterable[dict]) -> None:
    overlap = {str(r["query_id"]) for r in train} & {str(r["query_id"]) for r in test}
    if overlap:
        raise ValueError(f"query overlap between splits: {sorted(overlap)[:5]}")
