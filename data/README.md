# Data

## Public datasets (downloaded automatically)

HotpotQA, GSM8K and 2WikiMultiHopQA are loaded through Hugging Face `datasets` on first use.
Set `HF_HOME` to choose the cache location.

## Raw agent traces

The ~31,000 generated traces (about 2.3 GB of JSON, 282 MB compressed) are attached to the
GitHub release as one tarball per experiment. They are not needed for `make reproduce`,
which runs from `results/`. Download them to rerun the analyses or the trace-level negative
controls:

```bash
bash scripts/fetch_traces.sh                       # everything
bash scripts/fetch_traces.sh E4_fingerprint_v2     # one experiment
```

Each archive is checked against `SHA256SUMS` before unpacking into `artifacts/`.

| Archive | Contents |
|---|---|
| `P0_pilot`, `P1_retrieval_check`, `P2_injector_check` | Pilot and gate checks |
| `E1_pipeline_v2` | 200 clean pipeline-validation traces |
| `E3_positive_control_v2` | 5,000 probe-training traces |
| `E4_fingerprint_v2` | 18,800 held-out test traces (main campaign) |
| `E4_k3_v2` | Retrieval depth k=3 |
| `E4_2wiki_v3`, `E1_2wiki_pilot` | 2Wiki replication (7,000) and pilot |
| `matched_effect_source_degradation_v2` | Matched-effect traces |
| `E1_pipeline`, `E3_positive_control`, `E4_fingerprint`, `E4_2wiki_v1`, `E4_2wiki_v2` | Earlier superseded runs |

## AEGIS

The external audit uses the AEGIS benchmark (MIT) at a pinned commit:
`bash scripts/fetch_aegis.sh`.
