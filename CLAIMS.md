# Confirmatory claim registry

Frozen before interpreting the new external and replication results. New result files are
versioned and never replace the earlier analyses.

| ID | Claim | Evidence / primary statistic | Status | May new work change it? |
|---|---|---|---|---|
| C-V1 | Harness coupling can create apparent detector capability. | Paired query-bootstrap change in AUROC after causal-path decoupling; material-loss margin 0.05. | Confirmatory | External audit may establish pass, fail, or inconclusive per fault mode. |
| C-V2 | Manufacturing identity can remain recoverable after prespecified effect matching. | `D = 2*abs(AUC-0.5)` on rank-vs-lexical source mechanisms; margin 0.05. Severity-only and cross-fitted severity-weighted probes are robustness analyses. | Confirmatory + robustness | Robustness may weaken the source result; the donor contrast remains post-hoc. The invalid cache-inheriting v2 rerun is excluded. |
| C-V3 | Query matching can remove an apparent synthetic-to-natural realism trend. | Within-query permutation C2ST; equivalence only when the one-sided upper bound on D is below 0.05. | Confirmatory | 2Wiki is a confirmatory replication and may differ. |
| C-V4 | Decision validity depends jointly on the benchmark and candidate pool. | Query-bootstrap winner-agreement probability and natural-regret 95% CI, with all-detector and V1-clean candidate sets. | Confirmatory | Bootstrap uncertainty or 2Wiki may show the point winner is unstable or dataset-specific. |
| C-NAT | Missing annotated evidence is causally relevant to at least some natural failures. | Paired gold-restoration minus placebo rescue rate and interval. | Confirmatory | 2Wiki is a replication; subsets below the declared size are not used to train detectors. |
| C-EXT | The audit procedure transports to an independently developed injector. | AEGIS V1 on FM-1.1/FM-2.2; released-data V2 is diagnostic and not effect-matched. | Confirmatory V1 / diagnostic V2 | Direction-independent: pass, fail, or inconclusive is retained. |

Fixed analysis rules:

- Query/task-disjoint folds and query/task-clustered resampling.
- Practical discrimination/material-loss margin `delta = 0.05`.
- A V3 failure to establish equivalence is not proof of difference.
- A V1 result passes only when the paired interval is wholly below the material-loss margin,
  fails only when it is wholly above the margin, and is otherwise inconclusive.
- Pilot 2Wiki questions are excluded from the confirmatory partition.
- External AEGIS fault modes were selected before the confirmatory run and are not changed in
  response to their result.
