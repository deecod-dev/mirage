"""Result persistence. Every experiment MUST call write_result() before returning.

The status generator reads only these files, so an experiment that does not
call this leaves no record under results/.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

Status = Literal["pass", "fail", "partial", "blocked"]


def config_hash(config: dict[str, Any]) -> str:
    """Stable 12-char hash of a config dict. Used in every artifact filename."""
    blob = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def write_result(
    experiment_id: str,
    status: Status,
    config: dict[str, Any],
    seed: int,
    metrics: dict[str, Any] | None = None,
    gate: str | None = None,
    gate_passed: bool | None = None,
    kill_condition_fired: str | None = None,
    artifacts: list[str] | None = None,
    notes: str = "",
) -> Path:
    """Persist one experiment result.

    metrics values should be dicts of the form
    {"point": float, "ci_low": float, "ci_high": float} — bare floats are
    permitted only for counts.
    """
    if gate is not None and gate_passed is None:
        raise ValueError(f"{experiment_id} declares gate {gate} but no gate_passed")

    payload = {
        "experiment_id": experiment_id,
        "status": status,
        "gate": gate,
        "gate_passed": gate_passed,
        "kill_condition_fired": kill_condition_fired,
        "seed": seed,
        "config": config,
        "config_hash": config_hash(config),
        "metrics": metrics or {},
        "artifacts": artifacts or [],
        "notes": notes,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }

    out_dir = RESULTS_DIR / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"result_{payload['config_hash']}_seed{seed}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path
