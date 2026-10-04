"""Observed severity controls for matched-effect injector comparisons."""
from __future__ import annotations

import re
import string
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEVERITY_NAMES = (
    "answer_similarity", "answer_length_change", "answer_char_change",
    "changed_token_count", "answer_correctness_change", "answer_token_f1_change",
    "context_token_change", "question_context_overlap",
)


def invalidate_metric_cache(row: dict) -> dict:
    """Remove evaluator outputs that became stale after a trajectory intervention."""
    row.pop("metrics", None)
    return row


def _tokens(text: str) -> list[str]:
    table = str.maketrans("", "", string.punctuation)
    return text.lower().translate(table).split()


def contains_gold(answer: str, gold: str) -> bool:
    a, g = " ".join(_tokens(answer)), " ".join(_tokens(gold))
    return bool(g) and g in a


def token_f1(answer: str, gold: str) -> float:
    a, g = _tokens(answer), _tokens(gold)
    if not a or not g:
        return 0.0
    overlap = sum((Counter(a) & Counter(g)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(a), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def severity_vector(clean: dict, perturbed: dict) -> np.ndarray:
    """Declared observed severity vector; no injector identity or bookkeeping fields."""
    clean_answer, answer = str(clean.get("answer") or ""), str(perturbed.get("answer") or "")
    a, b = clean_answer.split(), answer.split()
    gold = str(clean.get("gold_answer") or "")
    clean_context = " ".join(str(p.get("text") or "")
                             for p in clean.get("selected_context") or [])
    context = " ".join(str(p.get("text") or "")
                       for p in perturbed.get("selected_context") or [])
    question_words = set(_tokens(str(clean.get("question") or "")))
    context_words = set(_tokens(context))
    overlap = len(question_words & context_words) / max(len(question_words), 1)
    return np.asarray([
        SequenceMatcher(None, clean_answer, answer).ratio(),
        len(b) - len(a), len(answer) - len(clean_answer),
        sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b)),
        float(contains_gold(answer, gold)) - float(contains_gold(clean_answer, gold)),
        token_f1(answer, gold) - token_f1(clean_answer, gold),
        len(context.split()) - len(clean_context.split()), overlap,
    ], dtype=float)


def propensity_weights(matrix: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                       folds: int = 5, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fitted, clipped inverse-propensity weights for a binary mechanism contrast."""
    matrix, labels = np.asarray(matrix, float), np.asarray(labels, int)
    groups = np.asarray([str(g) for g in groups])
    probabilities = np.full(len(labels), np.nan)
    splitter = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    # GroupKFold is deterministic; seed is accepted to keep the analysis interface fixed.
    del seed
    for train, test in splitter.split(matrix, labels, groups):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        model.fit(matrix[train], labels[train])
        probabilities[test] = model.predict_proba(matrix[test])[:, 1]
    probabilities = np.clip(probabilities, 0.05, 0.95)
    prevalence = float(np.mean(labels))
    weights = np.where(labels == 1, prevalence / probabilities,
                       (1 - prevalence) / (1 - probabilities))
    lo, high = np.percentile(weights, [1, 99])
    return np.clip(weights, lo, high), probabilities


def standardised_mean_differences(matrix: np.ndarray, labels: np.ndarray,
                                  weights: np.ndarray | None = None) -> np.ndarray:
    """Absolute SMD per severity variable, optionally under sample weights."""
    matrix, labels = np.asarray(matrix, float), np.asarray(labels, int)
    weights = np.ones(len(labels)) if weights is None else np.asarray(weights, float)
    out = []
    for column in matrix.T:
        means, variances = [], []
        for value in (0, 1):
            mask = labels == value
            w, x = weights[mask], column[mask]
            mean = np.average(x, weights=w)
            means.append(mean)
            variances.append(np.average((x - mean) ** 2, weights=w))
        pooled = np.sqrt((variances[0] + variances[1]) / 2)
        out.append(abs(means[1] - means[0]) / pooled if pooled > 0 else 0.0)
    return np.asarray(out)
