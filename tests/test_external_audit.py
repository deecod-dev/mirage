from mirage.external_audit import (available_text, single_injection, task_group,
                                   trajectory_text)


def _row():
    return {
        "id": "x", "metadata": {"framework": "dylan", "benchmark": "GSM8K",
                                  "model": "m", "num_agents": 2,
                                  "num_injected_agents": 1, "task_type": "math"},
        "input": {"query": "What is 2+2?", "conversation_history": [
            {"agent_name": "Solver", "phase": "reasoning", "content": "It is five"}],
            "final_output": "five"},
        "output": {"faulty_agents": [{"error_type": "FM-2.2"}]},
        "ground_truth": {"is_injection_successful": True, "injected_agents": [
            {"agent_name": "Solver", "error_type": "FM-2.2",
             "injection_strategy": "response_corruption"}]},
    }


def test_available_arm_excludes_explicit_targets_and_injection_count():
    text = available_text(_row())
    assert "FM-2.2" not in text
    assert "response_corruption" not in text
    assert "num_injected_agents" not in text
    assert "It is five" in text


def test_trajectory_group_and_single_injection_are_stable():
    row = _row()
    assert task_group(row) == task_group(row)
    assert single_injection(row)["strategy"] == "response_corruption"
    assert trajectory_text(row).startswith("What is 2+2?")
