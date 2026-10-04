"""Immutable, resumable experiment artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def atomic_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


class ArtifactStore:
    def __init__(self, root: Path, experiment_id: str, config_hash: str, seed: int,
                 subdir: str | None = None) -> None:
        # `subdir` lets seed-independent artifacts (clean traces, which are fully
        # determined by the generation config under greedy decoding) live once per
        # experiment instead of being regenerated for every seed.
        self.root = root / experiment_id / (subdir or f"{config_hash}_seed{seed}")
        self.traces = self.root / "traces"
        self.root.mkdir(parents=True, exist_ok=True)
        self.traces.mkdir(exist_ok=True)

    def trace_paths(self, query_id: str, condition: str) -> list[Path]:
        prefix = stable_hash({"query_id": query_id, "condition": condition})
        return sorted(self.traces.glob(f"trace_{prefix}_*.json"))

    def load_trace(self, query_id: str, condition: str) -> dict[str, Any]:
        paths = self.trace_paths(query_id, condition)
        if not paths:
            raise FileNotFoundError(f"no trace for query={query_id} condition={condition}")
        return json.loads(paths[0].read_text(encoding="utf-8"))

    def has_trace(self, query_id: str, condition: str) -> bool:
        return bool(self.trace_paths(query_id, condition))

    def write_trace(self, trace: dict[str, Any], condition: str) -> Path:
        payload = dict(trace)
        payload["written_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        content_hash = stable_hash({k: v for k, v in payload.items() if k != "written_at"})
        payload["artifact_hash"] = content_hash
        prefix = stable_hash({"query_id": str(payload["query_id"]), "condition": condition})
        path = self.traces / f"trace_{prefix}_{content_hash}.json"
        if path.exists():
            return path
        existing = self.trace_paths(str(payload["query_id"]), condition)
        if existing:
            # Same content hash means another worker wrote an identical trace — that is
            # convergence, not a conflict, and it happens routinely when two GPUs share
            # a clean-trace store. Only a *different* hash is a real immutability breach.
            if any(p.name.endswith(f"_{content_hash}.json") for p in existing):
                return existing[0]
            raise FileExistsError(
                f"immutable trace already exists for query={payload['query_id']} "
                f"condition={condition}; refusing to overwrite {existing[0]}"
            )
        return atomic_json(path, payload)

    def iter_traces(self) -> Iterable[dict[str, Any]]:
        for path in sorted(self.traces.glob("trace_*.json")):
            yield json.loads(path.read_text(encoding="utf-8"))

    def write_result(self, result: dict[str, Any]) -> Path:
        payload = dict(result)
        payload["artifact_hash"] = stable_hash(payload)
        return atomic_json(self.root / "result.json", payload)
