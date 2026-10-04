"""Authoritative server runner with fixed query-disjoint partitions and rerun semantics.

Execution is organised in two phases so that a single GPU stays busy:

  Phase 1  every clean trace for the partition, generated in batches. Clean traces
           are seed-independent under greedy decoding, so they are written once to a
           shared store keyed by `config.generation_hash` and reused by every seed.
           This also supplies grounding donors, which previously cost a full extra
           pipeline run per query.

  Phase 2  every (tier, family) condition. Faults are injected for the whole
           partition first, then grouped by how much pipeline they need re-run
           (`Injection.rerun`) so each group is executed as one batch.

Nothing downstream of a perturbation is ever scored against a stale value.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Sequence

from mirage.artifacts import ArtifactStore, stable_hash
from mirage.config import ExperimentConfig
from mirage.data.hotpot import HotpotExample, Passage
from mirage.faults import Injection, inject
from mirage.metrics import compute_metrics
from mirage.pipeline import RealAgenticRAG
from mirage.runtime import Runtime
from mirage.stats import bootstrap_mean

FAMILIES = ("query_retrieval", "source_degradation", "tool_invocation", "grounding")

#: Stages that inject faults. Everything else records clean traces only.
INJECTING_STAGES = frozenset({"E3", "E4", "E5", "E6", "E7", "E9", "E10"})


def fixed_hotpot_partition(config: ExperimentConfig) -> list[HotpotExample]:
    """Create fixed train/validation/test/reserve slices from the real validation split."""
    from datasets import load_dataset
    kwargs: dict[str, Any] = {"split": "validation"}
    if config.dataset_revision:
        kwargs["revision"] = config.dataset_revision
    raw = load_dataset(config.dataset_name, config.dataset_config, **kwargs)
    indices = list(range(len(raw)))
    import random
    random.Random(20260816).shuffle(indices)
    counts = config.hotpot
    offsets = {
        "P0": (0, config.max_examples or 100), "E1": (counts.train, counts.validation),
        "E3": (counts.train, counts.validation), "E4": (counts.train + counts.validation, counts.test),
        "E5": (counts.train + counts.validation, counts.test), "E6": (counts.train + counts.validation, counts.test),
        "E7": (counts.train + counts.validation, counts.test), "E8": (counts.train + counts.validation, counts.test),
        "E9": (counts.train + counts.validation, counts.test), "E10": (counts.train + counts.validation, counts.test),
    }
    offset, count = offsets[config.stage]
    # Clean traces may cover more queries than injection does — see `clean_examples`.
    if config.clean_examples is not None:
        count = min(count, config.clean_examples)
    examples = []
    for idx in indices[offset:offset + count]:
        row = raw[idx]
        passages = tuple(
            Passage(f"{row['id']}:{i}", title, " ".join(sentences).strip())
            for i, (title, sentences) in enumerate(zip(row["context"]["title"], row["context"]["sentences"], strict=True))
            if " ".join(sentences).strip()
        )
        examples.append(HotpotExample(str(row["id"]), row["question"], row["answer"], passages,
                                      frozenset(row["supporting_facts"]["title"])))
    return examples


def label_natural_failure(trace: dict[str, Any]) -> dict[str, Any]:
    """Attach the automatically verifiable natural-retrieval-failure labels.

    docs/PLAN.md defines natural retrieval failure as "an annotated supporting paragraph
    is absent from top-k" — i.e. *incomplete* gold support, which for a two-hop
    HotpotQA question is the failure that actually degrades the answer. The first
    implementation required *no* gold paragraph at all, a far stricter event that
    never fired once in 500 queries.
    """
    gold = set(trace.get("supporting_titles", []))
    titles = {p["title"] for p in trace.get("selected_context", [])}
    found = gold & titles
    trace["natural_retrieval_failure"] = bool(gold - titles)
    trace["natural_retrieval_failure_severe"] = bool(gold) and not found
    trace["natural_support_recall"] = (len(found) / len(gold)) if gold else float("nan")
    return trace


def _chunks(items: Sequence[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


# ── phase 1: clean traces ────────────────────────────────────────────────────
def build_clean_traces(agent: RealAgenticRAG, examples: list[HotpotExample],
                       store: ArtifactStore, config: ExperimentConfig,
                       deadline: float) -> dict[str, dict[str, Any]]:
    """Load cached clean traces; generate and persist whatever is missing, in batches."""
    clean: dict[str, dict[str, Any]] = {}
    pending: list[HotpotExample] = []
    for example in examples:
        if store.has_trace(example.query_id, "clean"):
            clean[example.query_id] = store.load_trace(example.query_id, "clean")
        else:
            pending.append(example)
    for batch in _chunks(pending, config.batch_size):
        if time.monotonic() > deadline:
            break
        for trace in agent.run_many(batch):
            trace.update({"fault_source": "clean", "fault_tier": None, "fault_family": None,
                          "injector_seed": None})
            label_natural_failure(trace)
            trace["metrics"] = compute_metrics(trace, agent._load_embedder())
            store.write_trace(trace, "clean")
            clean[trace["query_id"]] = trace
    return clean


# ── phase 2: faulted traces ──────────────────────────────────────────────────
def _execute_reruns(agent: RealAgenticRAG, pending: list[tuple[Injection, HotpotExample]],
                    config: ExperimentConfig) -> list[dict[str, Any]]:
    """Re-execute exactly as much pipeline as each injection declared it needs."""
    grouped: dict[str, list[tuple[Injection, HotpotExample]]] = defaultdict(list)
    for injection, example in pending:
        grouped[injection.rerun].append((injection, example))
    done: list[dict[str, Any]] = []

    for injection, _ in grouped.get("none", []):
        done.append(injection.trace)

    verify_items = grouped.get("verify", [])
    if verify_items:
        results = agent.verify_many([(i.trace["question"], i.trace["answer"], i.trace["selected_context"])
                                     for i, _ in verify_items])
        for (injection, _), verification in zip(verify_items, results):
            injection.trace["verification"] = verification
            done.append(injection.trace)

    downstream = grouped.get("downstream", [])
    if downstream:
        answers = agent.answer_many([(i.trace["question"], i.trace["selected_context"], i.trace["tool_history"])
                                     for i, _ in downstream])
        verifications = agent.verify_many([(i.trace["question"], a, i.trace["selected_context"])
                                           for (i, _), a in zip(downstream, answers)])
        for (injection, _), answer, verification in zip(downstream, answers, verifications):
            injection.trace["answer"] = answer
            injection.trace["verification"] = verification
            done.append(injection.trace)

    full = grouped.get("full", [])
    if full:
        rerun_examples = [replace(example, question=injection.trace["question"])
                          for injection, example in full]
        for (injection, _), fresh in zip(full, agent.run_many(rerun_examples)):
            carried = {k: injection.trace[k] for k in
                       ("fault_source", "fault_tier", "fault_family", "injector_seed")}
            fresh.update(carried)
            fresh["gold_answer"] = injection.trace["gold_answer"]
            fresh["supporting_titles"] = injection.trace["supporting_titles"]
            done.append(fresh)
    return done


def run_condition(agent: RealAgenticRAG, examples: list[HotpotExample],
                  clean: dict[str, dict[str, Any]], tier: str, family: str,
                  store: ArtifactStore, config: ExperimentConfig, deadline: float) -> int:
    """Inject and materialise one (tier, family) condition across the partition."""
    condition = f"{tier}:{family}"
    order = [e for e in examples if e.query_id in clean and not store.has_trace(e.query_id, condition)]
    written = 0
    for batch in _chunks(order, config.batch_size):
        if time.monotonic() > deadline:
            break
        pending: list[tuple[Injection, HotpotExample]] = []
        for example in batch:
            neighbours = [q for q in clean if q != example.query_id]
            # stable_hash, not hash(): str hashing is salted per process, which would
            # make donor choice differ between an initial run and its resume.
            donor = (clean[neighbours[int(stable_hash([config.seed, example.query_id]), 16)
                                      % len(neighbours)]] if neighbours else None)
            injection = inject(clean[example.query_id], tier, family, seed=config.seed,
                               donor=donor,
                               generate=(lambda s, u: agent.chat_many([(s, u)], max_new_tokens=256)[0]))
            pending.append((injection, example))
        for trace in _execute_reruns(agent, pending, config):
            label_natural_failure(trace)
            trace["metrics"] = compute_metrics(trace, agent._load_embedder())
            store.write_trace(trace, f"{trace['fault_tier']}:{trace['fault_family']}")
            written += 1
    return written


def run_server(config: ExperimentConfig, *, resume: bool = True) -> dict[str, Any]:
    """Run one experiment seed.

    `resume` is accepted for CLI compatibility but resumption is now unconditional:
    artifacts are immutable and content-addressed, so anything already on disk is
    reused and nothing is ever recomputed or overwritten. Re-issuing an interrupted
    command is always safe and always cheap.
    """
    del resume
    runtime = Runtime.from_environment(config.runtime)
    runtime.ensure_layout()
    runtime.assert_free_space(config.max_free_disk_gb)

    clean_store = ArtifactStore(runtime.artifacts, config.experiment_id, config.hash, config.seed,
                                subdir=f"clean_{config.generation_hash}")
    store = ArtifactStore(runtime.artifacts, config.experiment_id, config.hash, config.seed)

    examples = fixed_hotpot_partition(config)
    agent = RealAgenticRAG(config)
    started = time.monotonic()
    deadline = started + config.max_job_hours * 3600

    clean = build_clean_traces(agent, examples, clean_store, config, deadline)

    # Injection covers only `max_examples`; the surplus clean traces exist to give the
    # E5 natural arm enough positives without paying twelve conditions for each one.
    injectable = examples[:config.max_examples] if config.max_examples else examples

    written = 0
    if config.stage in INJECTING_STAGES:
        for tier in config.injector_tiers:
            for family in config.injector_families:
                written += run_condition(agent, injectable, clean, tier, family, store, config, deadline)

    faulted = list(store.iter_traces())
    traces = list(clean.values()) + faulted
    expected_clean = min(config.clean_examples, len(examples))
    expected_faulted = (len(injectable) * len(config.injector_tiers) *
                        len(config.injector_families)
                        if config.stage in INJECTING_STAGES else 0)
    qcm = [row["metrics"]["QCM"] for row in traces if row.get("fault_source") == "clean"]
    natural = [bool(row.get("natural_retrieval_failure")) for row in traces if row.get("fault_source") == "clean"]
    result = {
        "experiment_id": config.experiment_id, "stage": config.stage, "seed": config.seed,
        "config": config.canonical(), "config_hash": config.hash,
        "generation_hash": config.generation_hash,
        "n_traces": len(traces), "n_clean": len(clean), "n_faulted_written": written,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # A deadline exit is a resumable checkpoint, not successful completion.
        "complete": (len(clean) == expected_clean and len(faulted) == expected_faulted),
        "expected_traces": expected_clean + expected_faulted,
        "status": "pass" if qcm else "partial",
        "natural_retrieval_failure_rate": (sum(natural) / len(natural)) if natural else None,
        "n_natural_retrieval_failures": sum(natural),
        "metrics": {"clean_QCM": bootstrap_mean(qcm, config.seed, config.bootstrap_repeats)} if qcm else {},
    }
    store.write_result(result)
    return result
