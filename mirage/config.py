"""Validated, fully specified experiment configuration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import ClassVar, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class Models(BaseModel):
    embedder: str = "BAAI/bge-base-en-v1.5"
    reranker: str = "BAAI/bge-reranker-base"
    generator: str = "Qwen/Qwen2.5-7B-Instruct"
    judge: str = "Qwen/Qwen2.5-7B-Instruct"
    revision: str = "main"
    embedder_revision: str | None = None
    reranker_revision: str | None = None
    generator_revision: str | None = None
    judge_revision: str | None = None

    def revision_for(self, component: str) -> str:
        value = getattr(self, f"{component}_revision")
        return value or self.revision

    model_config = {"extra": "forbid"}


class DataSplit(BaseModel):
    train: int = Field(gt=0)
    validation: int = Field(gt=0)
    test: int = Field(gt=0)
    reserve: int = Field(ge=0)

    model_config = {"extra": "forbid"}


class ExperimentConfig(BaseModel):
    experiment_id: str
    stage: Literal["P0", "E1", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10"]
    runtime: Literal["server", "kaggle_t4"] = "server"
    partition: Literal["pilot", "train", "validation", "test", "reserve"] = "validation"
    dataset_name: str = "hotpotqa/hotpot_qa"
    dataset_config: str = "distractor"
    dataset_revision: str | None = None
    device: str = "auto"
    seed: int = 11
    max_examples: int | None = Field(default=None, gt=0)
    #: Queries to generate *clean* traces for. The E5 natural arm is a label on clean
    #: traces, so it scales independently of injection: one extra query costs one
    #: pipeline run here, against twelve for an injected query. Defaults to
    #: `max_examples`; raise it when the natural-failure rate is too low to reach K3.
    natural_examples: int | None = Field(default=None, gt=0)
    #: Per-subgoal retrieval width — how many passages each subgoal pulls into the pool.
    top_k: int = Field(default=5, ge=2, le=10)
    #: Evidence that actually reaches the generator, after cross-encoder consolidation.
    #: Must stay below `top_k` or retrieval is a no-op: the union of per-subgoal
    #: retrievals already covers the whole HotpotQA distractor candidate set.
    answer_top_k: int = Field(default=3, ge=1, le=10)
    #: Prompts per generation batch. The dominant throughput knob on one GPU.
    batch_size: int = Field(default=16, ge=1, le=128)
    max_input_tokens: int = Field(default=3072, ge=512, le=8192)
    model: Models = Field(default_factory=Models)
    hotpot: DataSplit = Field(default_factory=lambda: DataSplit(train=500, validation=200, test=300, reserve=200))
    gsm8k_validation: int = 300
    gsm8k_test: int = 300
    injector_tiers: list[Literal["T0", "T1", "T2"]] = Field(default_factory=lambda: ["T0", "T1"])
    injector_families: list[Literal["query_retrieval", "source_degradation",
                                    "tool_invocation", "grounding"]] = Field(
        default_factory=lambda: ["query_retrieval", "source_degradation",
                                 "tool_invocation", "grounding"])
    bootstrap_repeats: int = 1000
    checkpoint_every: int = 25
    max_job_hours: float = 7.5
    max_free_disk_gb: int = 20

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_counts(self) -> "ExperimentConfig":
        if self.stage in {"E4", "E5", "E7"} and self.hotpot.test < 300:
            raise ValueError("headline experiments require 300 held-out HotpotQA queries")
        if self.checkpoint_every > 25:
            raise ValueError("checkpoint_every may not exceed 25")
        if self.answer_top_k >= self.top_k:
            raise ValueError(
                "answer_top_k must be strictly smaller than top_k, otherwise consolidation "
                "selects the whole candidate pool and retrieval cannot fail naturally")
        return self

    def canonical(self) -> dict:
        return self.model_dump(mode="json")

    @property
    def hash(self) -> str:
        raw = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()[:12]

    #: Fields that change what a *clean* trace looks like. Decoding is greedy, so a
    #: clean trace depends on none of seed, injector_tiers, or the analysis settings —
    #: it can therefore be generated once and shared by every seed of an experiment.
    GENERATION_FIELDS: ClassVar[tuple[str, ...]] = (
        "dataset_name", "dataset_config", "dataset_revision", "stage", "partition",
        "max_examples", "natural_examples", "top_k", "answer_top_k", "max_input_tokens",
        "model", "hotpot", "runtime")

    @property
    def clean_examples(self) -> int | None:
        """How many queries need a clean trace: the larger of the two budgets."""
        if self.max_examples is None or self.natural_examples is None:
            return self.natural_examples or self.max_examples
        return max(self.max_examples, self.natural_examples)

    @property
    def generation_hash(self) -> str:
        payload = {k: self.canonical()[k] for k in self.GENERATION_FIELDS}
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()[:12]


def load_config(path: str | Path, *, seed: int | None = None) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if seed is not None:
        data["seed"] = seed
    return ExperimentConfig.model_validate(data)
