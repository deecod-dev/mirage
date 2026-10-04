"""Shared machinery for the validity audit experiments.

Every analysis added in response to the construct-validity review uses the same three
primitives, so that no result differs from another because of an incidental analysis
choice --- the exact failure mode the review caught in the original dedup sensitivity,
where a change of key and a change of imputation were reported as one effect.

The primitives are:

* `prepare` --- train-median imputation plus explicit definedness indicators. Never fit
  imputation statistics on rows that will be scored.
* `oof_*` --- out-of-fold predictions under **query-disjoint** K-folds. Each row is scored
  exactly once, by a model that never saw its query. Pooled out-of-fold scores are what
  make a paired bootstrap possible at all: two methods are compared on identical rows.
* `cluster_bootstrap` --- resamples whole queries, never rows. Multiple traces share a
  query, so row-level resampling would treat correlated traces as independent evidence
  and produce intervals that are far too narrow.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from mirage.probe import new_probe


def prepare(matrix: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """Impute with the TRAINING median and append definedness indicators.

    The indicator columns matter: a metric that is undefined for a whole condition is
    itself a signal, and silently imputing it would either hide that signal or (with a
    sentinel value) distort standardisation across every column at once.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    with np.errstate(all="ignore"):
        medians = np.nan_to_num(np.nanmedian(matrix[train_idx], axis=0), nan=0.0)
    defined = np.isfinite(matrix).astype(float)
    filled = np.where(np.isfinite(matrix), matrix, medians)
    # Only carry indicators when something is actually undefined, so that fully-defined
    # feature sets are not padded with constant columns.
    return np.hstack([filled, defined]) if defined.min() == 0 else filled


def query_folds(groups: Sequence[str], n_folds: int, seed: int) -> list[np.ndarray]:
    """Partition row indices into K folds that never split a query across folds."""
    groups = np.asarray([str(g) for g in groups])
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    assignment = {q: i % n_folds for i, q in enumerate(shuffled)}
    owner = np.asarray([assignment[g] for g in groups])
    return [np.where(owner == k)[0] for k in range(n_folds)]


def oof_binary(matrix: np.ndarray, labels: np.ndarray, groups: Sequence[str],
               n_folds: int = 5, seed: int = 11,
               sample_weight: np.ndarray | None = None) -> np.ndarray:
    """Out-of-fold P(label=1) for every row, under query-disjoint folds."""
    labels = np.asarray(labels)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if weights is not None and len(weights) != len(labels):
        raise ValueError("sample_weight must have one value per row")
    out = np.full(len(labels), np.nan)
    for fold in query_folds(groups, n_folds, seed):
        train = np.setdiff1d(np.arange(len(labels)), fold)
        if len(set(labels[train])) < 2:
            continue
        ready = prepare(matrix, train)
        model = new_probe()
        fit_kwargs = ({"logisticregression__sample_weight": weights[train]}
                      if weights is not None else {})
        model.fit(ready[train], labels[train], **fit_kwargs)
        positive = list(model.classes_).index(1)
        out[fold] = model.predict_proba(ready[fold])[:, positive]
    return out


def oof_multiclass(matrix: np.ndarray, labels: np.ndarray, groups: Sequence[str],
                   n_folds: int = 5, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold class probabilities and the class order they follow."""
    labels = np.asarray(labels)
    classes = np.unique(labels)
    out = np.full((len(labels), len(classes)), np.nan)
    for fold in query_folds(groups, n_folds, seed):
        train = np.setdiff1d(np.arange(len(labels)), fold)
        if len(set(labels[train])) < len(classes):
            continue
        ready = prepare(matrix, train)
        model = new_probe()
        model.fit(ready[train], labels[train])
        # Fold models may order classes differently; map into the global order.
        for j, c in enumerate(model.classes_):
            out[fold, list(classes).index(c)] = model.predict_proba(ready[fold])[:, j]
    return out, classes


def macro_ovr_auroc(labels: np.ndarray, proba: np.ndarray, classes: np.ndarray,
                    subset: np.ndarray | None = None) -> float:
    """Macro one-vs-rest AUROC, averaging only over classes that are present."""
    labels = np.asarray(labels)
    idx = np.arange(len(labels)) if subset is None else np.asarray(subset)
    scores = []
    for j, c in enumerate(classes):
        target = (labels[idx] == c).astype(int)
        column = proba[idx, j]
        ok = np.isfinite(column)
        if target[ok].sum() in (0, ok.sum()):
            continue
        scores.append(roc_auc_score(target[ok], column[ok]))
    return float(np.mean(scores)) if scores else float("nan")


def binary_auroc(labels: np.ndarray, scores: np.ndarray,
                 subset: np.ndarray | None = None) -> float:
    labels = np.asarray(labels)
    idx = np.arange(len(labels)) if subset is None else np.asarray(subset)
    ok = np.isfinite(scores[idx])
    target, column = labels[idx][ok], scores[idx][ok]
    if len(set(target)) < 2:
        return float("nan")
    return float(roc_auc_score(target, column))


def cluster_bootstrap(statistic: Callable[[np.ndarray], float], groups: Sequence[str],
                      repeats: int = 1000, seed: int = 11) -> dict[str, float]:
    """Bootstrap `statistic(row_indices)` by resampling whole queries with replacement.

    Returns the point estimate on the observed data plus a percentile interval. NaN
    resamples (a draw with only one class present, say) are dropped rather than
    propagated.
    """
    groups = np.asarray([str(g) for g in groups])
    unique = np.unique(groups)
    members = {q: np.where(groups == q)[0] for q in unique}
    rng = np.random.default_rng(seed)
    point = statistic(np.arange(len(groups)))
    draws = []
    for _ in range(repeats):
        picked = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([members[q] for q in picked])
        value = statistic(rows)
        if np.isfinite(value):
            draws.append(value)
    if not draws:
        return {"point": float(point), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_resamples": 0}
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"point": float(point), "ci_low": float(low), "ci_high": float(high),
            "n_resamples": len(draws)}


def paired_cluster_bootstrap(stat_a: Callable[[np.ndarray], float],
                             stat_b: Callable[[np.ndarray], float],
                             groups: Sequence[str], repeats: int = 1000,
                             seed: int = 11) -> dict[str, Any]:
    """Bootstrap the difference `stat_a - stat_b` on identical resampled rows.

    This is the correct replacement for taking percentiles over overlapping random
    half-splits: the two statistics are evaluated on the *same* resampled queries, so
    query difficulty cancels, and resampling is at the clustering unit.
    """
    groups = np.asarray([str(g) for g in groups])
    unique = np.unique(groups)
    members = {q: np.where(groups == q)[0] for q in unique}
    rng = np.random.default_rng(seed)
    everything = np.arange(len(groups))
    point = stat_a(everything) - stat_b(everything)
    draws = []
    for _ in range(repeats):
        picked = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([members[q] for q in picked])
        delta = stat_a(rows) - stat_b(rows)
        if np.isfinite(delta):
            draws.append(delta)
    if not draws:
        return {"delta": float(point), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_resamples": 0, "significant": False}
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"delta": float(point), "ci_low": float(low), "ci_high": float(high),
            "n_resamples": len(draws),
            # Two-sided at the 5% level: the interval must not straddle zero.
            "significant": bool(low > 0 or high < 0),
            "direction": "gain" if point > 0 else "regression"}


def permutation_null(statistic: Callable[[np.ndarray], float], labels: np.ndarray,
                     groups: Sequence[str], repeats: int = 50,
                     seed: int = 11) -> dict[str, float]:
    """Shuffle labels *within each query* and rerun the statistic.

    For the tier experiments each query contributes one row per tier, so permuting
    labels inside a query preserves the design exactly --- same queries, same class
    balance, same number of rows --- and destroys only the association between a trace
    and the tier that produced it. That is precisely the null we want to reject.

    `statistic` is called with a permuted label vector and returns a scalar.
    """
    groups = np.asarray([str(g) for g in groups])
    labels = np.asarray(labels)
    blocks = [np.where(groups == q)[0] for q in np.unique(groups)]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        shuffled = labels.copy()
        for rows in blocks:
            shuffled[rows] = rng.permutation(labels[rows])
        value = statistic(shuffled)
        if np.isfinite(value):
            values.append(value)
    return {"mean": float(np.mean(values)) if values else float("nan"),
            "max": float(np.max(values)) if values else float("nan"),
            "repeats": len(values)}


def bootstrap_selection_stability(
        natural_labels: np.ndarray,
        natural_scores: dict[str, np.ndarray],
        natural_groups: Sequence[str],
        synthetic_conditions: Sequence[tuple[np.ndarray, dict[str, np.ndarray], Sequence[str]]],
        detectors: Sequence[str],
        repeats: int = 1000,
        seed: int = 11) -> dict[str, Any]:
    """Query-bootstrap detector selection agreement and deployment regret.

    Every replicate resamples the same query IDs for the natural reference and all
    synthetic conditions, recomputes each AUROC, chooses both winners, and evaluates the
    synthetic winner on the resampled natural data. This captures the discontinuity of
    ``argmax``; bootstrapping detector identities or marginal AUROC intervals does not.
    """
    natural_labels = np.asarray(natural_labels)
    natural_groups = np.asarray([str(g) for g in natural_groups])
    detectors = [d for d in detectors if d in natural_scores]
    if len(detectors) < 2:
        raise ValueError("selection stability requires at least two detectors")
    if not synthetic_conditions:
        raise ValueError("selection stability requires at least one synthetic condition")

    common = set(np.unique(natural_groups))
    prepared = []
    for labels, scores, groups in synthetic_conditions:
        labels = np.asarray(labels)
        groups = np.asarray([str(g) for g in groups])
        common &= set(np.unique(groups))
        prepared.append((labels, scores, groups))
    unique = np.asarray(sorted(common))
    if len(unique) < 2:
        raise ValueError("fewer than two query groups are shared across conditions")

    def members(groups: np.ndarray) -> dict[str, np.ndarray]:
        return {q: np.where(groups == q)[0] for q in unique}

    natural_members = members(natural_groups)
    condition_members = [members(groups) for _, _, groups in prepared]

    def rows_for(picked: np.ndarray, table: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([table[str(q)] for q in picked])

    def one(picked: np.ndarray) -> tuple[str, str, float] | None:
        nidx = rows_for(picked, natural_members)
        natural = {d: binary_auroc(natural_labels, np.asarray(natural_scores[d]), nidx)
                   for d in detectors}
        if not all(np.isfinite(list(natural.values()))):
            return None
        synthetic_by_detector = {d: [] for d in detectors}
        for (labels, scores, _), table in zip(prepared, condition_members, strict=True):
            idx = rows_for(picked, table)
            for d in detectors:
                value = binary_auroc(labels, np.asarray(scores[d]), idx)
                if np.isfinite(value):
                    synthetic_by_detector[d].append(value)
        if any(not values for values in synthetic_by_detector.values()):
            return None
        synthetic = {d: float(np.mean(values))
                     for d, values in synthetic_by_detector.items()}
        natural_winner = max(detectors, key=lambda d: natural[d])
        synthetic_winner = max(detectors, key=lambda d: synthetic[d])
        regret = natural[natural_winner] - natural[synthetic_winner]
        return natural_winner, synthetic_winner, float(regret)

    point = one(unique)
    if point is None:
        raise ValueError("observed data do not contain both classes in every condition")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repeats):
        value = one(rng.choice(unique, size=len(unique), replace=True))
        if value is not None:
            draws.append(value)
    if not draws:
        raise ValueError("every bootstrap replicate was undefined")

    natural_counts = {d: 0 for d in detectors}
    synthetic_counts = {d: 0 for d in detectors}
    regrets = []
    agreements = []
    for natural_winner, synthetic_winner, regret in draws:
        natural_counts[natural_winner] += 1
        synthetic_counts[synthetic_winner] += 1
        regrets.append(regret)
        agreements.append(natural_winner == synthetic_winner)
    low, high = np.percentile(regrets, [2.5, 97.5])
    return {
        "natural_winner": point[0], "synthetic_winner": point[1],
        "point_agreement": bool(point[0] == point[1]),
        "point_regret": float(point[2]),
        "selection_agreement_probability": float(np.mean(agreements)),
        "natural_winner_probability": {d: natural_counts[d] / len(draws) for d in detectors},
        "synthetic_winner_probability": {d: synthetic_counts[d] / len(draws) for d in detectors},
        "regret": {"point": float(point[2]), "bootstrap_mean": float(np.mean(regrets)),
                   "ci_low": float(low), "ci_high": float(high),
                   "n_resamples": len(draws)},
        "n_queries": len(unique),
    }
