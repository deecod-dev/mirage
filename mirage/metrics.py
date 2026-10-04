"""Metric suite evaluated on real trace artifacts.

E2 (metric-independence audit) removed ``SEO`` from the suite: it was defined as
``_mean_pairwise(trajectory)``, the identical expression already used for ``TCS``.
Across the 39,000 E4 traces the two agreed to r = 1.0000 with max|TCS - SEO| = 0.
This is the same failure an earlier draft had for SCR/GSA, so
the duplicate is dropped rather than redefined, and the suite is 14 metrics.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _mean_pairwise(matrix: np.ndarray) -> float:
    if len(matrix) < 2:
        return float("nan")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-12)
    sim = normalized @ normalized.T
    return float(sim[np.triu_indices(len(matrix), k=1)].mean())


def compute_metrics(trace: dict[str, Any], embedder: Any) -> dict[str, float]:
    """Compute metrics; NaN means a metric was inapplicable, never zero by default."""
    selected = trace["selected_context"]
    rejected = trace["rejected_context"]
    if not selected:
        raise ValueError("trace has no real selected context")
    query = trace["question"]
    step_contexts = trace.get("step_contexts", [])
    subgoals = trace.get("subgoals", [])
    # One encoder call per trace. Encoding each subgoal and each step separately cost
    # ~6 small GPU round-trips per trace, which at 40k traces is pure overhead.
    step_texts = [p["text"] for step in step_contexts for p in step]
    texts = [query, trace["answer"],
             *[p["text"] for p in selected], *[p["text"] for p in rejected],
             *[str(g["text"]) for g in subgoals], *step_texts]
    vectors = np.asarray(embedder.encode(texts, normalize_embeddings=True))
    q, answer = vectors[:2]
    cursor = 2
    chosen = vectors[cursor:cursor + len(selected)]; cursor += len(selected)
    rejected_vectors = vectors[cursor:cursor + len(rejected)]; cursor += len(rejected)
    subgoal_vectors = vectors[cursor:cursor + len(subgoals)]; cursor += len(subgoals)
    step_vectors = vectors[cursor:]
    query_chosen = chosen @ q
    query_rejected = rejected_vectors @ q if len(rejected_vectors) else np.array([np.nan])
    coverage = 1.0 - float(query_chosen.max())
    cdi = []
    for idx in range(len(chosen)):
        remaining = np.delete(chosen, idx, axis=0)
        cdi.append((1.0 - float((remaining @ q).max())) - coverage if len(remaining) else np.nan)
    step_centroids, offset = [], 0
    for step in step_contexts:
        if step:
            step_centroids.append(step_vectors[offset:offset + len(step)].mean(axis=0))
        offset += len(step)
    trajectory = np.asarray(step_centroids)
    tool_history = trace.get("tool_history", [])
    bad_tools = sum(call["name"] not in {"calculator", "document_lookup"} for call in tool_history)
    failed_tools = sum(not call["ok"] for call in tool_history)
    return {
        "QCM": float(query_chosen.mean() - query_rejected.mean()),
        "ICRR": _mean_pairwise(chosen),
        "SCR": coverage,
        "CDI_max": float(np.nanmax(cdi)) if any(np.isfinite(v) for v in cdi) else float("nan"),
        "MUT": float((1.0 - coverage) / max(1, sum(len(p["text"].split()) for p in selected))),
        "IPS": float(np.mean(subgoal_vectors @ q)) if len(subgoal_vectors) else float("nan"),
        "TCS": _mean_pairwise(trajectory),
        "GSA": float(np.mean([q @ c for c in trajectory])) if len(trajectory) else float("nan"),
        "SRS": float(np.mean([np.max(c @ chosen.T) for c in trajectory])) if len(trajectory) else float("nan"),
        "AGS": float(answer @ chosen.mean(axis=0)),
        "TRTR_route": float(bad_tools / max(1, len(tool_history))),
        "TIHD_health": float(failed_tools / max(1, len(tool_history))),
        "token_count": float(sum(len(p["text"].split()) for p in selected)),
        "answer_length": float(len(trace["answer"].split())),
    }
