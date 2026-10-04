import numpy as np

from mirage.severity import (SEVERITY_NAMES, invalidate_metric_cache,
                             propensity_weights, severity_vector,
                             standardised_mean_differences)


def test_severity_vector_has_declared_shape_and_no_labels():
    clean = {"answer": "Paris is in France", "gold_answer": "Paris",
             "question": "Where is Paris?",
             "selected_context": [{"text": "Paris is in France"}]}
    fault = {**clean, "answer": "Lyon is in France",
             "selected_context": [{"text": "Lyon is in France"}],
             "fault_source": "matched:lexical"}
    vector = severity_vector(clean, fault)
    assert vector.shape == (len(SEVERITY_NAMES),)
    assert np.isfinite(vector).all()
    assert vector[4] == -1.0


def test_propensity_weighting_reduces_observed_imbalance():
    rng = np.random.default_rng(5)
    x0 = rng.normal(0.0, 1.0, size=(100, 2))
    x1 = rng.normal(0.7, 1.0, size=(100, 2))
    matrix = np.vstack([x0, x1])
    labels = np.asarray([0] * 100 + [1] * 100)
    groups = np.asarray([f"q{i}" for i in range(100)] * 2)
    weights, probabilities = propensity_weights(matrix, labels, groups)
    assert np.isfinite(probabilities).all()
    assert standardised_mean_differences(matrix, labels, weights).mean() < \
        standardised_mean_differences(matrix, labels).mean()


def test_intervention_invalidates_clean_metric_cache():
    changed = invalidate_metric_cache({"answer": "changed", "metrics": {"AGS": 0.99}})
    assert "metrics" not in changed
