"""Compatibility shim. The authoritative engine is `mirage.server_runner`.

This module used to be a second, independent execution path. It fell behind: it
scored faulted traces against the pre-fault answer, used the stricter
natural-retrieval-failure definition that never fired, and ran one prompt at a time.
Keeping two runners is how those divergences happen, so `run()` now delegates and
there is exactly one place where a trace is produced.

`scripts/run_experiment.py` and `scripts/run_server_experiment.py` are therefore
equivalent; prefer the latter (see README).
"""
from __future__ import annotations

from typing import Any

from mirage.config import ExperimentConfig
from mirage.server_runner import run_server


def run(config: ExperimentConfig, *, resume: bool = True) -> dict[str, Any]:
    return run_server(config, resume=resume)
