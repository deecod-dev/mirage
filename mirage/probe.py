"""Probe training with mandatory source-label leakage prevention.

Missing metrics are imputed with the *training* median and paired with an explicit
`<metric>__defined` indicator. The first implementation mapped NaN to -999.0, which
turns "this metric was inapplicable" into an extreme, perfectly separable feature
value — a leak whenever inapplicability correlates with a condition (it did: the
trajectory metrics were undefined on 9.3% of most conditions but 16.0% of
T1:query_retrieval). An indicator column keeps that information available to the
probe as a fact about the trace, without letting it masquerade as a metric value.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FORBIDDEN_FEATURES = frozenset({"fault_source", "fault_family", "fault_tier", "query_id",
                                "injector_seed", "natural_retrieval_failure"})


def _assert_clean(columns: Sequence[str]) -> None:
    leaked = FORBIDDEN_FEATURES.intersection(columns)
    if leaked:
        raise ValueError(f"prohibited probe features: {sorted(leaked)}")


def raw_matrix(rows: Iterable[dict], columns: Sequence[str]) -> np.ndarray:
    """Metric values with NaN preserved. Never feed this to an estimator directly."""
    _assert_clean(columns)
    return np.asarray([[float(row["metrics"].get(c, np.nan)) for c in columns] for row in rows],
                      dtype=float)


class FeatureSpace:
    """Median imputation plus definedness indicators, fitted on training rows only."""

    def __init__(self, columns: Sequence[str], *, with_indicators: bool = True) -> None:
        _assert_clean(columns)
        self.columns = list(columns)
        self.with_indicators = with_indicators
        self.medians: np.ndarray | None = None

    @property
    def names(self) -> list[str]:
        if not self.with_indicators:
            return list(self.columns)
        return list(self.columns) + [f"{c}__defined" for c in self.columns]

    def fit(self, rows: Iterable[dict]) -> "FeatureSpace":
        matrix = raw_matrix(rows, self.columns)
        with np.errstate(all="ignore"):
            medians = np.nanmedian(matrix, axis=0)
        self.medians = np.nan_to_num(medians, nan=0.0)
        return self

    def transform(self, rows: Iterable[dict]) -> np.ndarray:
        if self.medians is None:
            raise RuntimeError("FeatureSpace.fit must be called before transform")
        matrix = raw_matrix(rows, self.columns)
        defined = np.isfinite(matrix).astype(float)
        filled = np.where(np.isfinite(matrix), matrix, self.medians)
        return np.hstack([filled, defined]) if self.with_indicators else filled

    def fit_transform(self, rows: list[dict]) -> np.ndarray:
        return self.fit(rows).transform(rows)


def feature_matrix(rows: Iterable[dict], columns: Sequence[str]) -> np.ndarray:
    """Backwards-compatible single-split matrix; imputes with its own median."""
    rows = list(rows)
    return FeatureSpace(columns, with_indicators=False).fit_transform(rows)


def new_probe() -> "object":
    """The one estimator every experiment uses. Standardised, so imputed medians
    sit at zero rather than dominating the coefficient scale."""
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5_000, class_weight="balanced"))


def source_probe(train: list[dict], test: list[dict], columns: Sequence[str]) -> float:
    space = FeatureSpace(columns).fit(train)
    model = new_probe()
    model.fit(space.transform(train), np.asarray([row["fault_source"] for row in train]))
    probabilities = model.predict_proba(space.transform(test))
    return float(roc_auc_score(np.asarray([row["fault_source"] for row in test]),
                               probabilities, multi_class="ovr", average="macro"))
