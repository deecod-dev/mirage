#!/usr/bin/env python
"""Fail if the revision's new headline numbers drift from result JSON.

This covers every number added for the AEGIS, severity-residual, V4-bootstrap and
2Wiki replication revisions. Earlier campaign numbers retain their existing result files.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "extended" / "mirage.tex"


def read(name: str) -> dict:
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def close(actual: float, expected: float, tolerance: float = 5e-4) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{actual} != {expected} within {tolerance}")


def main() -> int:
    severity = read("matched_effect_source_v3.json")["severity_residual_robustness"]
    close(severity["severity_only_auroc"]["point"], .628)
    close(severity["decoupled_after_severity_reweighting_auroc"]["point"], .607)
    close(severity["balance"]["max_smd_before"], .374)
    close(severity["balance"]["max_smd_after"], .014)

    natural = read("2wiki_natural_labels_v3.json")
    assert natural["n_clean_traces"] == 1000
    assert natural["n_degraded"] == 521
    assert natural["n_consequential"] == 187

    causal = read("2wiki_counterfactual_v3.json")
    assert causal["n_intervened"] == 187
    assert causal["n_causally_verified"] == 50
    close(causal["rescue_rate_gold"]["point"], .332)
    close(causal["rescue_rate_placebo"]["point"], .107)
    close(causal["causal_effect"]["point"], .225)
    assert causal["mcnemar"]["rescued_by_gold_only"] == 50
    assert causal["mcnemar"]["rescued_by_placebo_only"] == 8

    hotpot_causal = read("counterfactual_v2.json")
    assert hotpot_causal["n_intervened"] == 90
    assert hotpot_causal["n_causally_verified"] == 19
    close(hotpot_causal["rescue_rate_gold"]["point"], .311)
    close(hotpot_causal["rescue_rate_placebo"]["point"], .100)
    close(hotpot_causal["causal_effect"]["point"], .211)
    assert hotpot_causal["mcnemar"]["rescued_by_gold_only"] == 19
    assert hotpot_causal["mcnemar"]["rescued_by_placebo_only"] == 0

    c2st = read("2wiki_c2st_matched_v3.json")["by_condition"]
    expected_d = {
        "T0:query_retrieval": .278, "T1:query_retrieval": .369,
        "T2:query_retrieval": .138, "T0:source_degradation": .141,
        "T1:source_degradation": .257, "T2:source_degradation": .230,
    }
    for condition, expected in expected_d.items():
        close(c2st[condition]["external"]["D_upper_bound"], expected)
        assert c2st[condition]["external"]["verdict"] == "distinguishable"

    hotpot_c2st = read("c2st_matched_v2.json")["by_condition"]
    expected_hotpot = {
        "T0:query_retrieval": (.508, .016, .158),
        "T1:query_retrieval": (.486, .029, .161),
        "T2:query_retrieval": (.460, .080, .166),
        "T0:source_degradation": (.558, .117, .198),
        "T1:source_degradation": (.552, .105, .297),
        "T2:source_degradation": (.459, .082, .220),
    }
    for condition, (auc, d_value, upper) in expected_hotpot.items():
        result = hotpot_c2st[condition]["external"]
        close(result["point"], auc)
        close(result["D"], d_value, tolerance=8e-4)
        close(result["D_upper_bound"], upper)
        assert hotpot_c2st[condition]["n_queries"] == 71

    decision = read("2wiki_decision_validity_v3.json")["candidate_sets"]
    for tier, agreement in zip(("T0", "T1", "T2"), (.963, .692, .755)):
        close(decision["A_all"]["by_tier"][tier]["bootstrap_stability"]
              ["selection_agreement_probability"], agreement)
    for tier in ("T0", "T2"):
        cell = decision["B_validity_clean"]["by_tier"][tier]
        close(cell["regret"], .088, tolerance=7e-4)
        close(cell["bootstrap_stability"]["selection_agreement_probability"], 0)

    aegis = read("external_aegis_audit_v3.json")["v1"]["by_mode"]
    close(aegis["FM-1.1"]["available_minus_decoupled"], .032)
    close(aegis["FM-2.2"]["available_minus_decoupled"], .022)
    assert all("inconclusive" in value["verdict"] for value in aegis.values())

    aegis_full = read("external_aegis_audit_v3.json")
    assert aegis["FM-1.1"]["n_positive"] == 139
    assert aegis["FM-2.2"]["n_positive"] == 98
    close(aegis["FM-1.1"]["arms"]["available"]["point"], .673)
    close(aegis["FM-1.1"]["arms"]["decoupled"]["point"], .641)
    close(aegis["FM-2.2"]["arms"]["available"]["point"], .631)
    close(aegis["FM-2.2"]["arms"]["decoupled"]["point"], .609)
    assert aegis_full["v2"]["n"] == 61
    assert aegis_full["v2"]["class_counts"] == {
        "prompt_injection": 26, "response_corruption": 35,
    }
    close(aegis_full["v2"]["mechanism_auroc"]["point"], .407)
    assert "NOT effect-matched" in aegis_full["v2"]["status"]

    if TEX.exists():
        text = TEX.read_text(encoding="utf-8")
        required = (
            "$7{,}000$ traces", "$521$ natural degradations", "$187$ are wrong",
            "$33.2\\%$", "$10.7\\%$", "$+0.225$", "$.963,.692,.755$",
            "regret is $0.088$", "AUROC $0.607$", "$0.032$", "$0.022$",
            "$139/600$", "$98/600$", "AUROC is $.407$", "$19/0$", "$50/8$",
            "Complete confirmatory statistics", "Representations, estimators and missing values",
        )
        missing = [snippet for snippet in required if snippet not in text]
        if missing:
            raise AssertionError(f"manuscript is missing result-backed snippets: {missing}")
    print("revision numbers verified against result JSON")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
