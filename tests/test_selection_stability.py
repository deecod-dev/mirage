import numpy as np

from mirage.validity import bootstrap_selection_stability


def test_selection_stability_recomputes_winners_and_regret_by_query():
    groups = np.asarray([f"q{i}" for i in range(12)])
    labels = np.asarray([0, 1] * 6)
    natural = {
        "good": np.asarray([0.1, 0.9] * 6),
        "bad": np.asarray([0.8, 0.2] * 6),
    }
    synthetic = [(
        labels,
        {"good": np.asarray([0.8, 0.2] * 6),
         "bad": np.asarray([0.1, 0.9] * 6)},
        groups,
    )]
    result = bootstrap_selection_stability(
        labels, natural, groups, synthetic, ["good", "bad"], repeats=100, seed=7)
    assert result["natural_winner"] == "good"
    assert result["synthetic_winner"] == "bad"
    assert result["selection_agreement_probability"] == 0.0
    assert result["regret"]["ci_low"] > 0.9


def test_selection_stability_requires_shared_queries():
    labels = np.asarray([0, 1])
    scores = {"a": np.asarray([0.1, 0.9]), "b": np.asarray([0.2, 0.8])}
    try:
        bootstrap_selection_stability(
            labels, scores, ["a", "b"],
            [(labels, scores, ["x", "y"])], ["a", "b"], repeats=10)
    except ValueError as exc:
        assert "shared" in str(exc)
    else:
        raise AssertionError("disjoint query sets must be rejected")
