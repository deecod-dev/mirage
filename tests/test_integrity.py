from pathlib import Path

import pytest

from mirage.artifacts import ArtifactStore
from mirage.config import ExperimentConfig
from mirage.faults import inject
from mirage.probe import feature_matrix
from mirage.runtime import Runtime


def test_headline_requires_held_out_size():
    with pytest.raises(ValueError, match="300"):
        ExperimentConfig(experiment_id="E4", stage="E4", hotpot={"train": 500, "validation": 200, "test": 299, "reserve": 1})


def test_probe_rejects_fault_source():
    with pytest.raises(ValueError, match="prohibited"):
        feature_matrix([{"metrics": {"QCM": 1.0}}], ["QCM", "fault_source"])


def test_grounding_requires_real_donor():
    trace = {"query_id": "same", "answer": "a"}
    with pytest.raises(ValueError, match="donor"):
        inject(trace, "T1", "grounding")


def test_artifacts_are_content_addressed(tmp_path: Path):
    store = ArtifactStore(tmp_path, "P0", "abc", 11)
    first = store.write_trace({"query_id": "q", "answer": "a"}, "clean")
    assert first.name.startswith("trace_")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        store.write_trace({"query_id": "q", "answer": "b"}, "clean")


def test_runtime_is_not_kaggle_specific(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MIRAGE_ROOT", str(tmp_path))
    assert Runtime.from_environment("server").use_4bit is False
    assert Runtime.from_environment("kaggle_t4").use_4bit is True
