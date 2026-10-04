"""E10 negative controls (docs/PLAN.md §11).

These are the assertions that would have caught the defects the validity audit found:
a collapsed tier ladder, seed-invariant injection, duplicate metrics, a leaking NaN
sentinel, and label leakage into the feature matrix. They run on real trace artifacts
when they exist and skip when they do not, so a fresh clone still passes.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from mirage.analysis import ARMS, COUPLED, FEATURES, TRIVIAL, dedup, family_label
from mirage.config import ExperimentConfig
from mirage.faults import inject
from mirage.probe import FORBIDDEN_FEATURES, FeatureSpace, raw_matrix

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FAMILIES = ("query_retrieval", "source_degradation", "tool_invocation", "grounding")


def _clean_traces(limit: int = 2) -> list[dict]:
    found: list[dict] = []
    if not ARTIFACTS.exists():
        return found
    for path in sorted(ARTIFACTS.rglob("trace_*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("fault_source") == "clean" and row.get("selected_context"):
            found.append(row)
        if len(found) >= limit:
            break
    return found


@pytest.fixture(scope="module")
def clean_traces() -> list[dict]:
    traces = _clean_traces(6)
    if len(traces) < 2:
        pytest.skip("no clean trace artifacts on disk yet")
    return traces


@pytest.fixture(scope="module")
def clean_pair(clean_traces) -> tuple[dict, dict]:
    base, donor = clean_traces[0], clean_traces[1]
    return base, {**donor, "query_id": f"donor-{donor['query_id']}"}


def _stub_generate(system: str, user: str) -> str:
    """Test-only stand-in so T2 grounding is exercisable without loading a 7B model.
    Never used by mirage/ at runtime — the runner always passes the real generator.

    It echoes a digest of the prompt, so a test that asserts output variance is really
    asserting that the *prompt* varies, which is where T2 grounding's seed-sensitivity
    has to live: decoding is greedy, so an identical prompt gives an identical answer.
    """
    return f"Rewritten answer {abs(hash(user)) % 10 ** 9}."


def _payload(trace: dict) -> str:
    keys = ("question", "answer", "selected_context", "rejected_context", "tool_history")
    return json.dumps({k: trace.get(k) for k in keys}, sort_keys=True)


# ── the ladder is the paper ───────────────────────────────────
@pytest.mark.parametrize("family", FAMILIES)
def test_tiers_are_not_collapsed(clean_pair, family):
    base, donor = clean_pair
    payloads = {tier: _payload(inject(base, tier, family, seed=11, donor=donor,
                                      generate=_stub_generate).trace)
                for tier in ("T0", "T1", "T2")}
    for a, b in itertools.combinations(payloads, 2):
        assert payloads[a] != payloads[b], (
            f"{family}: tiers {a} and {b} produce identical traces — the T0/T1/T2 "
            "ladder is fictional for this family")


@pytest.mark.parametrize("family", FAMILIES)
def test_injection_varies_with_seed(clean_traces, family):
    """Greedy decoding makes the pipeline deterministic, so if injection is also
    deterministic every seed-based confidence interval is vacuous.

    Checked over the population rather than a single trace: an individual trace can
    legitimately be seed-invariant (an answer with one entity has one entity to swap),
    but a family that never varies anywhere is a broken injector.
    """
    donor = {**clean_traces[-1], "query_id": "donor"}
    seeds = (11, 23, 37, 53, 71)
    varied = sum(
        len({_payload(inject(base, "T2", family, seed=s, donor=donor,
                             generate=_stub_generate).trace) for s in seeds}) > 1
        for base in clean_traces if base["query_id"] != donor["query_id"])
    assert varied > 0, f"T2:{family} is seed-invariant on every available trace"


def test_hardened_tool_fault_leaves_no_health_signal(clean_pair):
    """The T2 tool fault must be invisible to TRTR_route and TIHD_health: it executes
    a valid call that returns a real but wrong result."""
    base, donor = clean_pair
    trace = dict(base)
    trace["tool_history"] = [{"name": "calculator", "arguments": {"expression": "12 + 7 * 3"},
                              "output": "33.0", "ok": True}]
    call = inject(trace, "T2", "tool_invocation", seed=11, donor=donor).trace["tool_history"][0]
    assert call["ok"] is True, "hardened tool fault must not fail the call"
    assert call["name"] == "calculator", "hardened tool fault must not violate the registry"
    assert call["output"] != "33.0", "hardened tool fault must change the result"


def test_source_degradation_does_not_pin_qcm_to_zero(clean_pair):
    """T1 must *exchange* selected and rejected evidence, not copy one over the other:
    identical sets make QCM = mean(q.sel) - mean(q.rej) a bit-exact 0.0, a signature no
    natural fault can produce."""
    base, _ = clean_pair
    trace = inject(base, "T1", "source_degradation", seed=11).trace
    selected = [p["passage_id"] for p in trace["selected_context"]]
    rejected = [p["passage_id"] for p in trace["rejected_context"]]
    assert set(selected) != set(rejected[:len(selected)]) or not rejected


# ── leakage guards ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("arm", sorted(ARMS))
def test_no_arm_contains_a_label(arm):
    assert not FORBIDDEN_FEATURES.intersection(ARMS[arm])


def test_feature_space_never_emits_a_sentinel(clean_pair):
    """NaN must be median-imputed with an explicit indicator, not mapped to -999.0,
    which turns 'inapplicable' into a perfectly separable value."""
    base, donor = clean_pair
    missing = {**base, "metrics": {**base["metrics"], "TCS": float("nan")}}
    rows = [base, donor, missing]
    space = FeatureSpace(FEATURES).fit(rows)
    matrix = space.transform(rows)
    column = FEATURES.index("TCS")
    assert np.isfinite(matrix).all()
    assert len(space.names) == 2 * len(FEATURES)

    observed = [base["metrics"]["TCS"], donor["metrics"]["TCS"]]
    imputed = matrix[2, column]
    assert min(observed) <= imputed <= max(observed), (
        f"TCS was imputed to {imputed}, outside the observed range — that is a sentinel, "
        "not an imputation")

    indicators = matrix[:, len(FEATURES):]
    assert set(np.unique(indicators)).issubset({0.0, 1.0})
    assert indicators[2, column] == 0.0 and indicators[0, column] == 1.0


def test_raw_matrix_preserves_nan():
    rows = [{"metrics": {"QCM": float("nan")}}]
    assert np.isnan(raw_matrix(rows, ["QCM"])[0, 0])


# ── suite hygiene ────────────────────────────────────────────────────────────
def test_no_duplicate_metrics_in_suite():
    assert len(FEATURES) == len(set(FEATURES))
    assert "SEO" not in FEATURES, "SEO duplicates TCS exactly; E2 removed it"


def test_trivial_baselines_are_separable_from_the_suite():
    """docs/PLAN.md M-D0: trivial baselines must be reportable on their own."""
    assert set(TRIVIAL).issubset(FEATURES)
    assert ARMS["semantic_only"] and not set(ARMS["semantic_only"]) & set(TRIVIAL)
    assert ARMS["decoupled"] and not set(ARMS["decoupled"]) & set(COUPLED)


def test_metrics_are_not_pairwise_identical():
    """The E2 audit, as a test. SCR/GSA in the prior draft and TCS/SEO in this one were
    both caught only after a full run had been paid for."""
    paths = sorted(ARTIFACTS.rglob("trace_*.json"))[:400] if ARTIFACTS.exists() else []
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    rows = [r for r in rows if r.get("metrics")]
    if len(rows) < 50:
        pytest.skip("not enough trace artifacts to audit metric independence")
    matrix = raw_matrix(rows, FEATURES)
    for i, j in itertools.combinations(range(len(FEATURES)), 2):
        mask = np.isfinite(matrix[:, i]) & np.isfinite(matrix[:, j])
        if mask.sum() < 25:
            continue
        assert not np.allclose(matrix[mask, i], matrix[mask, j]), (
            f"{FEATURES[i]} and {FEATURES[j]} are the same metric")


# ── config guards ────────────────────────────────────────────────────────────
def test_consolidation_must_actually_filter():
    with pytest.raises(ValueError, match="answer_top_k"):
        ExperimentConfig(experiment_id="x", stage="E4", top_k=5, answer_top_k=5,
                         hotpot={"train": 500, "validation": 200, "test": 300, "reserve": 200})


def test_dedup_collapses_seed_replicas():
    rows = [{"query_id": "q", "fault_source": "clean", "injector_seed": None} for _ in range(5)]
    assert len(dedup(rows)) == 1
    varied = [{"query_id": "q", "fault_source": "T2:grounding", "injector_seed": s} for s in (11, 23)]
    assert len(dedup(varied)) == 2


def test_family_label_never_returns_a_tier():
    assert family_label({"fault_source": "T2:grounding"}) == "grounding"
    assert family_label({"fault_source": "clean"}) == "clean"
