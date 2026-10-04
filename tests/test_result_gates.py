"""Every persisted result that claims a gate must record whether it passed."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_result_declares_gate_outcome():
    """A result claiming a gate must say whether it passed. See mirage/record.py."""
    for p in (ROOT / "results").rglob("result_*.json"):
        d = json.loads(p.read_text())
        if d.get("gate") is not None:
            assert d.get("gate_passed") is not None, (
                f"{p.name} declares gate {d['gate']} without gate_passed"
            )
