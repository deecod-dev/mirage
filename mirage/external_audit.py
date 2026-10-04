"""Representation and schema helpers for external injector audits."""
from __future__ import annotations

import hashlib
import json
from typing import Any


DIRECT_BOOKKEEPING_PATHS = (
    "output.faulty_agents", "ground_truth.injected_agents",
    "ground_truth.is_injection_successful", "metadata.num_injected_agents",
)


def task_group(row: dict[str, Any]) -> str:
    query = str(row.get("input", {}).get("query") or row.get("id") or "")
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


def trajectory_text(row: dict[str, Any]) -> str:
    """Evaluator-visible task and trajectory, excluding all label/bookkeeping objects."""
    source = row.get("input") or {}
    turns = []
    for turn in source.get("conversation_history") or []:
        turns.append(" ".join([
            str(turn.get("agent_name") or ""), str(turn.get("phase") or ""),
            str(turn.get("content") or "")]))
    return "\n".join([str(source.get("query") or ""), *turns,
                      str(source.get("final_output") or "")])


def available_text(row: dict[str, Any]) -> str:
    """Fields a benchmark consumer might plausibly pass to an evaluator."""
    metadata = row.get("metadata") or {}
    safe_metadata = {k: metadata.get(k) for k in
                     ("framework", "benchmark", "model", "num_agents", "task_type")}
    return json.dumps(safe_metadata, sort_keys=True) + "\n" + trajectory_text(row)


def unsafe_all_fields_text(row: dict[str, Any]) -> str:
    """Diagnostic only: demonstrates the direct path if target fields are consumed."""
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def single_injection(row: dict[str, Any]) -> dict[str, str] | None:
    injected = (row.get("ground_truth") or {}).get("injected_agents") or []
    if len(injected) != 1:
        return None
    item = injected[0]
    strategy = str(item.get("injection_strategy") or "")
    if strategy == "prompt":
        strategy = "prompt_injection"
    return {"error_type": str(item.get("error_type") or ""),
            "strategy": strategy, "agent_name": str(item.get("agent_name") or "")}
