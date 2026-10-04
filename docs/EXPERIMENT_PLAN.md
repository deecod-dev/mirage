# Experiment registry

Written **before** the runs below execute. A paper about evaluation validity cannot
decide what a result means after seeing it, so each entry fixes the question, the
statistic, the resampling unit, and — most importantly — what each possible outcome will
be taken to show.

Rule for all entries: **the query is the statistical unit.** Bootstraps resample whole
queries, permutations shuffle within a query, folds are query-disjoint, and paired
comparisons are computed inside the same resample.

---

## C1 — Family-matched decision validity

**Question.** Does a synthetic tier pick the same detector that natural failures pick?

**Defect being fixed.** The published version averaged each detector's AUROC over all four
families, including `tool_invocation` and `grounding`, which have no natural counterpart in
this pipeline. Their inclusion is what produced "all three tiers agree".

**Design.** Restrict to `query_retrieval` and `source_degradation`. Report two distinct
quantities rather than one verdict:

* **selection validity** — does `argmax` agree with the natural `argmax`?
* **ranking validity** — Kendall's $\tau$ between full rankings, with a query-level
  bootstrap interval.

**Outcomes.**
* Selection agrees and $\tau$ high → the benchmark is a safe guide to detector choice.
* Selection disagrees at any tier → that tier misleads the decision, and this is reported
  as a headline result regardless of $\tau$.
* Tool and grounding are reported **N/A**, never as passes.

**Output.** `results/decision_validity_v2.json`

---

## C2 — Mixing controls

**Question.** Does adding synthetic data to natural data hurt because synthetic data is
*misleading*, or merely because thousands of synthetic rows *outnumber* twenty natural ones?

**Defect being fixed.** The published learning-curve result compares ~5{,}000 synthetic rows
against 20–160 natural ones and concludes synthetic data causes negative transfer. Domain
imbalance is an equally good explanation and was not controlled.

**Design.** At each budget $n \in \{20,40,80,160\}$, five protocols on identical folds:

| | protocol |
|---|---|
| M0 | synthetic only |
| M1 | natural only |
| M2 | naive concatenation (as published) |
| M3 | balanced domain weight — each domain contributes half the total sample weight |
| M4 | synthetic pre-fit, then refit on natural with the synthetic solution as the prior |
| M5 | matched-count — sample $n$ and $5n$ synthetic rows instead of all of them |

**Outcomes.**
* All mixed protocols still lose to M1 → synthetic information itself transfers negatively.
  The published claim stands and strengthens.
* M3/M4/M5 recover to M1 or better → the published claim was an imbalance artifact. It must
  be rewritten as "naive pooling harms; correct weighting does not", which is more useful
  and must be reported even though it weakens the headline.

**Output.** `results/mixing_controls_v2.json`

---

## C3 — Ecological equivalence

**Question.** Are synthetic faults *practically equivalent* to natural ones, rather than
merely not-proven-different?

**Defects being fixed.** (a) The paper reads a C2ST point estimate near 0.5 as
"indistinguishable", which is a failure to reject, not evidence of equivalence. (b) The
headline cell (T2 source, surface) is $0.457$ with an interval excluding 0.5 **from below**.
A trained out-of-fold discriminator scoring significantly below chance has anti-generalised;
it has not found that the samples match.

**Design.** Predeclare an equivalence margin $\delta = 0.05$ on the discrimination scale
$D = 2|\text{AUC} - 0.5| \in [0,1]$. Declare practical equivalence only when the **upper**
one-sided 95% bound on $D$ lies below $\delta$. Report sensitivity at
$\delta \in \{0.025, 0.05, 0.075\}$. Report the signed AUROC as the primary number and
never silently orient it, because the sign carries the diagnostic.

Diagnose the below-chance cell by re-running it with (i) shuffled labels, (ii) a
single-fold holdout, and (iii) balanced class counts, to distinguish an anti-generalising
fit from a fold-structure artifact.

**Outcomes.**
* Upper bound below $\delta$ → practical equivalence, stated as such.
* Interval straddles 0.5 but bound above $\delta$ → **underpowered**, stated as such. This
  is the expected outcome for most cells and is not a realism claim.
* Below-chance cell survives diagnosis → report as a measurement failure, not as realism.

**Output.** `results/realism_c2st_v2.json` (extended), `results/c2st_diagnosis_v2.json`

---

## C4 — Matched-effect injector identification

**Question.** With the semantic fault effect held fixed, is the *manufacturing mechanism*
still identifiable from detector features? Formally, is $I(\text{mechanism};Z \mid F,S) > 0$?

**Defect being fixed.** The published tier experiment conditions on the fault family $F$ but
not on the semantic effect $S$: T0, T1 and T2 produce different severities. A probe
separating them may be reading severity rather than manufacturing signature.

**Design.** Three mechanisms produce the **identical** semantic effect — exactly one entity
span in an otherwise byte-identical answer becomes unsupported:

| | mechanism |
|---|---|
| `dict` | replacement drawn from a fixed same-type entity bank |
| `distractor` | replacement drawn from a non-supporting passage of the *same* query |
| `donor` | replacement drawn from a *different* query's answer |

All three are deterministic given the seed and none invokes a language model, because an LLM
rewrite would reintroduce the style variance we are controlling for.

**Severity must be verified before the classifier is run.** We report, per mechanism:
answer token count, edit distance from the clean answer, embedding similarity to the clean
answer, and entity-type distribution. If these differ materially the experiment is invalid
and no classification result may be quoted from it.

**Family choice.** `source_degradation` is the **test** — its tier discrimination is $0.874$,
so it is where a signal exists to explain away. `grounding` is the **control** — its tier
discrimination is already $0.500$, so a chance result there confirms the design does not
manufacture signal.

**Outcomes, fixed in advance.**
* Macro AUROC significantly above the within-query permutation null → detector
  representations encode how a fault was manufactured even with the fault held fixed. The
  fingerprint claim is established.
* Macro AUROC at the null → the published tier result was severity, not signature. The
  paper says so, and the matched-effect control becomes the contribution: it is the reason
  the earlier reading was wrong.

Neither outcome is a failure. No injector may be redesigned after seeing a classifier score.

**Output.** `results/matched_effect_v2.json`

---

## C5 — Counterfactual gold restoration *(needs GPU)*

**Question.** Do natural retrieval degradations *cause* wrong answers, or merely accompany
them?

**Design.** For each of the 90 traces that are both degraded and wrong, build three
conditions and regenerate under identical decoding:

* **C0** original degraded context;
* **C1** one non-gold selected passage replaced by the missing gold paragraph;
* **C2** placebo — the same slot replaced by a different non-gold passage, matched on
  length and retrieval rank.

$k$, passage count and ordering are held fixed across conditions.

**Statistic.** Rescue rate $R = P(\text{wrong} \to \text{correct})$ under each condition;
the causal estimate is $R_{\text{gold}} - R_{\text{placebo}}$, with a paired query bootstrap
and McNemar as a secondary check.

**Outcomes.**
* $R_{\text{gold}} \gg R_{\text{placebo}}$ → the degradation is causally consequential, and
  the subset defines a causally-verified natural failure label.
* $R_{\text{gold}} \approx R_{\text{placebo}}$ → restoring evidence does not fix the answer,
  so the label marks correlation. The transfer results must then be described as being about
  degraded intermediate states, not agent failures.

**Stopping rule.** If the causally-verified subset is smaller than 50, we report the rescue
rates and do **not** fit a detector on it.

**Output.** `results/counterfactual_v2.json`

---

## C6 — Operating-point replication at $k=3$ *(needs GPU)*

**Question.** Do the natural-fault results survive the retrieval operating point that was
*not* chosen after inspecting failure rates?

**Design.** Regenerate clean traces at `answer_top_k=3` and reproduce four quantities only:
natural degradation rate, natural-fault detection AUROC, transfer direction, and
$P(\text{wrong}\mid\text{missing})$.

**Outcome.** The test is on the *sign*, not the magnitude. Same sign → operating-point
independent. Different sign → proxy validity depends on the operating point, which is itself
a validity finding and is reported as one.

**Output.** `results/k3_sensitivity_v2.json`

---

## Prohibited

* Redesigning any injector after seeing a classifier score.
* Reporting `max(AUC, 1-AUC)` for a trained discriminator without also reporting the sign.
* Describing a failure to reject as equivalence.
* Marking a dimension as passing where no natural reference exists.
* Quoting an annotation-cost claim that was not measured.

---

## C7 — External AEGIS audit

**Status.** Confirmatory V1 on the released evaluation sample; the released-data V2 is
explicitly diagnostic because it cannot hold error mode, source task, framework and severity
fixed at useful power. Repository commit and selected modes are frozen in
`configs/external/aegis_audit.yaml` before the result is read.

**V1 design.** Audit FM-1.1 and FM-2.2 one-vs-rest on all 600 released traces. Compare
`available` (task, trajectory, final output and non-injection metadata) with `decoupled`
(task, trajectory and final output). Explicit target objects are excluded from both. An
`unsafe_all_fields` arm is a diagnostic only and may not determine the verdict. A material
loss is absolute paired AUROC loss at least 0.05.

**V2 design.** On single-injection successful traces, classify prompt-side versus
response-side manufacture only in frameworks containing both. This is a weak mechanism
contrast, not an effect-matched test; it cannot upgrade or refute the internal V2 claim.

**Output.** `results/external_aegis_audit_v3.json` (`v1` is the initial point-delta run;
`v2` adds paired query-bootstrap intervals; `v3` applies the conservative interval-versus-
margin verdict without overwriting either diagnostic predecessor).

## C8 — V2 residual-severity robustness

**Status.** Confirmatory robustness analysis of the pre-existing rank--lexical source
contrast. The donor mechanism remains diagnostic and is excluded.

**Declared severity vector.** Answer similarity, answer token/character length change,
changed-token count, correctness change, token-F1 change, context-token change and
question--context overlap. Report a severity-only classifier and the decoupled classifier
after cross-fitted clipped inverse-propensity weighting. Report standardised mean differences
before and after weighting. The declared vector is observed severity, not latent semantic
equality.

**Output.** `results/matched_effect_source_v3.json` plus an immutable matched-row corpus.
The invalid diagnostic predecessor is retained as
`results/invalid_matched_effect_source_v2_stale_metrics.json`: clean metric caches were
inherited by regenerated traces. `v3` removes those caches, recomputes every metric and
persists generated answers to a staging cache before GPU metric computation.

## C9 — V4 selection stability and regret uncertainty

**Status.** Confirmatory reanalysis of existing k=2 and k=3 predictions.

Within every query-cluster bootstrap replicate, recompute all natural and synthetic detector
AUROCs, both argmax decisions and natural regret. Report per-detector winner probabilities,
selection-agreement probability and the 95% regret interval for candidate Sets A and B. The
query, not the detector identity, is the resampling unit.

**Outputs.** `results/decision_validity_v3.json` and
`results/k3_decision_validity_v2.json`.

## C10 — 2WikiMultiHopQA replication

**Status.** Complete confirmatory cross-dataset replication. The 200-query pilot is disjoint from the
main slice and is used only as a feasibility gate. Main N is capped at 1,000 clean queries;
proceed only if the pilot projects at least 250 naturally degraded traces at that cap.

Only retrieval and source families are generated. Reuse the query-matched V3, both V4
candidate sets, and gold-restoration/placebo procedure without changing their thresholds.
The dataset is `framolfese/2WikiMultihopQA`, a script-free mirror retaining the Hotpot-style
context and supporting-fact annotations.

All 1,000 confirmatory queries receive every declared synthetic condition; this is required so
every naturally degraded query has its same-query synthetic counterpart. The final corrected
run is versioned as `E4_2wiki_v3`. Pre-fix/undersized `v1` and `v2` artifacts are retained for
provenance but excluded from every inference.

**Outputs.** `results/2wiki_*_v3.json`; pilot queries are stored under
`E1_2wiki_pilot`, main queries under the disjoint `E4_2wiki_v3` partition.

**Frozen outcome.** All 7,000 traces completed. V3 establishes no equivalence at
`delta=0.05`; all six external-representation cells are distinguishable. In V4 Set B, T0 and
T2 misselect with regret 0.088 and intervals excluding zero, while T1 agrees unstably. Gold
restoration rescues 0.332 versus 0.107 for placebo, a paired effect of 0.225
`[0.150, 0.294]`.
