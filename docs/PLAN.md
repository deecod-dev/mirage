# MIRAGE authoritative research plan

## Claim and scope

MIRAGE tests whether RAG/agent fault detectors learn implementation fingerprints of synthetic fault injectors rather than underlying faults. The paper may claim this only for a real agentic RAG pipeline, synthetic injector tiers, and automatically verifiable natural retrieval/tool outcomes. It must not claim natural grounding or broad hallucination detection without independent human annotation.

## Fixed system

- Primary data: HotpotQA. Natural retrieval failure means an annotated supporting paragraph is absent from top-k.
- Tool data: GSM8K using an executable calculator.
- Models: `BAAI/bge-base-en-v1.5`, `BAAI/bge-reranker-base`, and `Qwen/Qwen2.5-7B-Instruct` (generator/judge), with resolved revisions recorded.
- Agent: an LLM plans 2--4 subgoals; real document lookup/calculator tools execute; retrieval, reranking, generation and verification use real components.
- A6000 uses BF16; Kaggle T4 uses 4-bit generation. Hardware may not alter prompts, data, thresholds or metrics.

## Design and gates

Use fixed query-disjoint HotpotQA splits: 500 train / 200 validation / 300 test / 200 reserve. Use GSM8K 300 validation / 300 test. T0 is a positive-control injector, T1 a faithful published-style injector, T2 hardened without the held-out probe. Train probes only on train queries, tune on validation, and test once on held-out queries. `fault_source` is a target only and never a feature.

Report five seeds and 1,000x bootstrap CIs; report ten seeds for E4/E5/E7 headlines. Always include length, token-count, marker, and tool-registry baselines.

1. P0: 100-query real environment/persistence pilot. Stop if a component is fake, unavailable, does not fit, or cannot resume.
2. E1: validate real pipeline and supporting-fact retrieval; then freeze components and prompts.
3. E3: T0/T1 positive control. Halt/debug on a failed gate.
4. E4: held-out fingerprint probe. On failure, write the preplanned null-result paper; never tune after seeing test results.
5. E5--E10 run only after E4 passes.

## Artifact and Kaggle safety policy

All trace/result artifacts are content-hashed and immutable. Checkpoint every trace and seal a shard every 25 traces. Kaggle jobs cap themselves at 7.5 hours and halt below 20GB free disk. Shards are <=250MB compressed JSONL, logs <=25MB, upload bundles <=4GB. Never upload model caches, corpora or FAISS indexes. Credentials are environment secrets only. Use one Kaggle account.
