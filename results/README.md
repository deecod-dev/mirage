# Results

Every number in the paper comes from a JSON file in this directory. Files are versioned
(`_v2`, `_v3`) and never overwritten; the paper uses the latest version of each analysis.

| File(s) | Written by | Used for |
|---|---|---|
| `detector_validity_v2.json` | `scripts/run_detector_validity.py` | V1 independence, Fig. 1-2 |
| `E4_validity_audit_v2.json`, `E4_fingerprint_v2.json` | `scripts/run_plan.py` analyses | V1 coupling audit |
| `matched_effect_source_v3.json`, `matched_effect_grounding.json` | `scripts/run_matched_effect.py` | V2 mechanism signal, severity residual |
| `tier_discrimination_v2.json` | `scripts/run_tier_discrimination.py` | V2 tier contrasts |
| `c2st_matched_v2.json`, `c2st_equivalence_v2.json`, `realism_c2st_v2.json` | `scripts/run_c2st_*.py`, `run_realism_c2st.py` | V3 ecological validity, Fig. 3 |
| `decision_validity_v3.json`, `baselines_v2.json`, `baselines_judge.json` | `scripts/run_decision_validity.py`, `run_baselines.py` | V4 decision validity |
| `natural_labels_v2.json`, `counterfactual_v2.json` | `scripts/run_natural_labels.py`, `run_counterfactual.py` | Natural failures and gold-restoration causal check, Fig. 4 |
| `mixing_controls_v2.json`, `learning_curves_v2.json` | `scripts/run_mixing_controls.py`, `run_learning_curves.py` | Natural-data budget, Fig. 4 |
| `probe_sensitivity_v2.json`, `union_v2.json` | `scripts/run_probe_sensitivity.py`, `run_union.py` | Robustness appendices |
| `k3_*.json` | `scripts/run_k3_t2.sh` | Retrieval depth k=3 |
| `2wiki_*_v3.json` | `scripts/run_2wiki_replication.sh` | 2WikiMultiHopQA replication |
| `external_aegis_audit_v3.json` | `scripts/run_external_aegis_audit.py` | External AEGIS audit |
| `retrieval_sweep*.json` | `scripts/sweep_retrieval.py` | Operating-point choice |

Superseded versions (no suffix or `_v2` where a `_v3` exists) are kept for the audit trail.
`invalid_matched_effect_source_v2_stale_metrics.json` is a rerun that inherited stale clean
metrics. It is excluded from every figure and claim and kept only for transparency.

Check the paper's headline numbers against these files with `make verify`.
