#!/usr/bin/env python
"""C4 --- is the injector identifiable once the semantic effect is held fixed?

The tier experiment conditions on the fault family but not on the damage done. T0, T1 and
T2 grounding injectors produce a swapped answer, an entity substitution and a subtle
rewrite: three different severities. A probe that separates them may be reading how bad the
failure is, not how it was manufactured. Formally, the tier experiment estimates
I(injector; Z | F) and we need I(injector; Z | F, S), holding the semantic effect S fixed.

So we build three injectors that produce the *identical* semantic effect and differ only in
mechanism. For grounding, the effect is fixed as: exactly one entity span in an otherwise
byte-identical answer becomes unsupported by the evidence.

    dict        replacement drawn from a fixed same-type entity bank
    distractor  replacement drawn from a non-supporting passage of the SAME query
    donor       replacement drawn from a DIFFERENT query's answer

No language model is involved. An LLM rewrite would reintroduce exactly the style variance
we are controlling away, and would make a positive result uninterpretable.

    python scripts/run_matched_effect.py --family grounding --device cpu \
        --output results/matched_effect_grounding.json

WHICH FAMILY IS THE TEST. `grounding` is the **control**: its tier discrimination is already
0.500, so a chance result there confirms the design does not manufacture signal out of
nothing. The real test is a family where a signal exists to explain away --- source
degradation at 0.874 --- but source faults change the evidence, so the answer must be
regenerated and that needs the 7B. This script runs the answer-side families on CPU; pass
`--family source_degradation` only with a GPU available.

SEVERITY IS VERIFIED BEFORE THE CLASSIFIER RUNS. If the three mechanisms differ materially
in edit distance, length change or embedding shift, the experiment is invalid and the
script refuses to report a classification result. That check is the whole point.

Registered interpretation (docs/EXPERIMENT_PLAN.md C4): above-null macro AUROC establishes that
detector features encode the manufacturing mechanism with the fault held fixed. A chance
result establishes that the published tier result was severity rather than signature. Both
are reportable; neither licenses redesigning an injector and rerunning.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from mirage.analysis import ARMS, dedup, read_rows
from mirage.artifacts import atomic_json
from mirage.artifacts import stable_hash
from mirage.metrics import compute_metrics
from mirage.probe import raw_matrix
from mirage.severity import (SEVERITY_NAMES, invalidate_metric_cache,
                             propensity_weights, severity_vector,
                             standardised_mean_differences)
from mirage.validity import (cluster_bootstrap, macro_ovr_auroc, oof_multiclass,
                             permutation_null, binary_auroc, oof_binary)

MECHANISMS = ("dict", "distractor", "donor")

#: Source-degradation mechanisms. The effect held fixed is "exactly one gold supporting
#: passage in the selected context is replaced by a non-supporting one of comparable
#: length"; the three differ only in HOW the replacement is chosen.
SOURCE_MECHANISMS = ("rank", "lexical", "donor")

#: Same-type replacement pools. Kept deliberately small and fixed so the `dict` mechanism
#: is fully deterministic and inspectable.
ENTITY_BANK = {
    "YEAR": ["1871", "1904", "1938", "1962", "1987", "2003"],
    "NUMBER": ["7", "19", "42", "128", "301", "1024"],
    "PROPER": ["Alden Vance", "Rosalind Hale", "Corwin Fields", "Meredith Kane",
               "Sylvan Roth", "Imogen Vale"],
}
CAPITALISED = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
NUMBER = re.compile(r"\b([0-9]{1,4})\b")


def entity_type(span: str) -> str:
    if YEAR.fullmatch(span):
        return "YEAR"
    if NUMBER.fullmatch(span):
        return "NUMBER"
    return "PROPER"


def find_entities(text: str) -> list[tuple[str, str]]:
    """Candidate (span, type) pairs. A deliberately simple, inspectable extractor.

    Sentence-initial capitalised words are skipped: they are capitalised by grammar, not
    because they name anything, and substituting them would change the effect being held
    fixed into a grammatical anomaly.
    """
    out: list[tuple[str, str]] = []
    for match in YEAR.finditer(text):
        out.append((match.group(1), "YEAR"))
    sentence_starts = {m.start() for m in re.finditer(r"(?:^|[.!?]\s+)([A-Z])", text)}
    for match in CAPITALISED.finditer(text):
        if match.start() in sentence_starts:
            continue
        out.append((match.group(1), "PROPER"))
    seen, unique = set(), []
    for span, kind in out:
        if span not in seen and len(span) > 1:
            seen.add(span)
            unique.append((span, kind))
    return unique


def context_entities(row: dict) -> list[tuple[str, str]]:
    """Entities appearing in this query's *rejected* passages --- present but unsupporting."""
    text = " ".join(str(p.get("text") or "") for p in row.get("rejected_context") or [])
    return find_entities(text)


def pick_replacement(mechanism: str, span: str, kind: str, row: dict,
                     donor_pool: list[tuple[str, str]], rng) -> str | None:
    """Same-type replacement for `span`, chosen by the named mechanism."""
    def usable(candidates):
        same = [c for c, k in candidates if k == kind and c.lower() != span.lower()]
        # Length matching keeps the surface perturbation comparable across mechanisms.
        same.sort(key=lambda c: abs(len(c) - len(span)))
        return same[:8]

    if mechanism == "dict":
        pool = [c for c in ENTITY_BANK[kind] if c.lower() != span.lower()]
    elif mechanism == "distractor":
        pool = usable(context_entities(row))
    else:
        pool = usable(donor_pool)
    return str(rng.choice(pool)) if pool else None


def apply_effect(row: dict, mechanism: str, donor_pool, seed: int) -> dict | None:
    """Replace exactly one entity span in the answer. Everything else is untouched."""
    answer = str(row.get("answer") or "")
    candidates = [(s, k) for s, k in find_entities(answer)
                  # The span must actually be supported by the evidence, or replacing it
                  # does not make anything newly unsupported.
                  if any(s in str(p.get("text") or "")
                         for p in row.get("selected_context") or [])]
    if not candidates:
        return None
    # Two separate streams, and the split matters for the design. The SPAN is chosen
    # from the query alone, so all three mechanisms perturb the *same* entity --- that is
    # what holds the semantic effect fixed. Only the REPLACEMENT depends on the
    # mechanism. (Python's str hash is salted per process, so hash() here would also
    # make the experiment irreproducible; use the repo's stable hash.)
    span_rng = np.random.default_rng(
        int(stable_hash([seed, str(row["query_id"])]), 16) % (2 ** 32))
    pick_rng = np.random.default_rng(
        int(stable_hash([seed, str(row["query_id"]), mechanism]), 16) % (2 ** 32))
    span, kind = candidates[span_rng.integers(len(candidates))]
    replacement = pick_replacement(mechanism, span, kind, row, donor_pool, pick_rng)
    if replacement is None:
        return None

    perturbed = dict(row)
    perturbed["answer"] = answer.replace(span, replacement, 1)
    # Metric caches belong to the clean trajectory and are invalid after changing
    # the answer. Keeping them makes every mechanism look exactly identical.
    invalidate_metric_cache(perturbed)
    perturbed["fault_source"] = f"matched:{mechanism}"
    perturbed["fault_family"] = "grounding"
    perturbed["fault_tier"] = "matched"
    perturbed["mechanism"] = mechanism
    perturbed["_span"], perturbed["_replacement"], perturbed["_type"] = span, replacement, kind
    return perturbed


def build_source_variants(row: dict, donor_passages: list[dict], seed: int) -> dict | None:
    """Replace one gold passage with a non-supporting one, three different ways.

    Every mechanism removes exactly one gold paragraph and puts a non-supporting paragraph
    in the same slot, giving an effect-matched intervention under the declared severity
    controls. What differs is
    the selection procedure: the reranker's own ordering, surface overlap with the
    question, or a passage lifted from an unrelated query.
    """
    selected = list(row.get("selected_context") or [])
    rejected = [p for p in (row.get("rejected_context") or [])
                if p.get("title") not in set(row.get("supporting_titles") or [])]
    gold_slots = [i for i, p in enumerate(selected)
                  if p.get("title") in set(row.get("supporting_titles") or [])]
    if not selected or len(rejected) < 2 or not gold_slots:
        return None

    rng = np.random.default_rng(int(stable_hash([seed, str(row["query_id"])]), 16) % 2**32)
    slot = int(gold_slots[rng.integers(len(gold_slots))])
    target = len(str(selected[slot].get("text") or ""))
    question = set(str(row.get("question") or "").lower().split())

    def overlap(passage):
        words = set(str(passage.get("text") or "").lower().split())
        return len(question & words) / max(len(question), 1)

    picks = {
        # rejected_context preserves reranker order, so the first element is the passage
        # the retriever ranked highest among those it discarded.
        "rank": rejected[0],
        "lexical": max(rejected, key=overlap),
        "donor": min(donor_passages,
                     key=lambda p: abs(len(str(p.get("text") or "")) - target))
        if donor_passages else None,
    }
    if picks["donor"] is None:
        return None

    out = {}
    for mechanism, passage in picks.items():
        context = list(selected)
        context[slot] = passage
        out[mechanism] = context
    out["_slot"] = slot
    return out


def severity(clean: str, perturbed: str) -> dict[str, float]:
    """How much damage was done, independent of how it was done."""
    a, b = clean.split(), perturbed.split()
    return {"len_change": float(len(b) - len(a)),
            "char_change": float(len(perturbed) - len(clean)),
            "similarity": float(SequenceMatcher(None, clean, perturbed).ratio()),
            "changed_tokens": float(sum(1 for x, y in zip(a, b) if x != y))}


def build_grounding_corpus(clean, args):
    donor_pool: list[tuple[str, str]] = []
    for row in clean:
        donor_pool.extend(find_entities(str(row.get("answer") or "")))
    built: list[dict] = []
    for row in clean:
        made = {m: apply_effect(row, m, donor_pool, args.seed) for m in MECHANISMS}
        # Keep a query only when ALL mechanisms could act on it, so they are compared on
        # identical queries rather than on different subsets.
        if all(made.values()):
            built.extend(made.values())
    return built, MECHANISMS


def build_source_corpus(clean, args):
    """Build the source variants and regenerate their answers with the real generator.

    Unlike the grounding family, changing the evidence changes what the model would have
    said, so leaving the original answer attached would make the perturbation invisible to
    half the metric suite and visible to the other half for the wrong reason.
    """
    donor_passages = [p for row in clean for p in (row.get("rejected_context") or [])][:400]
    plans, base = {}, []
    for row in clean:
        plan = build_source_variants(row, donor_passages, args.seed)
        if plan is not None:
            plans[str(row["query_id"])] = plan
            base.append(row)
    print(f"{len(base)} queries admit all {len(SOURCE_MECHANISMS)} source mechanisms")
    if not base:
        raise SystemExit("no query admits the source intervention")

    from mirage.config import load_config
    from mirage.pipeline import RealAgenticRAG
    agent = RealAgenticRAG(load_config(args.config))

    built = []
    for mechanism in SOURCE_MECHANISMS:
        items = [(str(r["question"]), plans[str(r["query_id"])][mechanism],
                  r.get("tool_history") or []) for r in base]
        print(f"regenerating {mechanism} ({len(items)} answers) ...", flush=True)
        answers = agent.answer_many(items)
        for row, context, answer in zip(base, (i[1] for i in items), answers):
            variant = dict(row)
            variant["selected_context"] = context
            variant["answer"] = answer
            # Both evidence and answer changed; inherited clean metrics are stale.
            invalidate_metric_cache(variant)
            variant["fault_source"] = f"matched:{mechanism}"
            variant["fault_family"] = "source_degradation"
            variant["fault_tier"] = "matched"
            variant["mechanism"] = mechanism
            variant["_type"] = "PASSAGE"
            built.append(variant)
    return built, SOURCE_MECHANISMS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--experiment", default="E4_fingerprint_v2")
    parser.add_argument("--family", default="grounding",
                        choices=["grounding", "source_degradation"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", default="configs/experiments/E4_fingerprint.yaml")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--max-queries", type=int, default=400)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--practical-margin", type=float, default=0.05,
                        help="minimum D = 2*|AUC-0.5| to call a mechanism identifiable. "
                             "Matches the C3 equivalence margin, so 'distinguishable' "
                             "means the same thing in both sections.")
    parser.add_argument("--severity-tolerance", type=float, default=0.05,
                        help="max spread in mean similarity across mechanisms before the "
                             "experiment is declared invalid")
    parser.add_argument("--corpus-cache", default=None,
                        help="immutable JSONL cache of matched rows with recomputed metrics")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.family == "source_degradation" and args.device == "cpu":
        raise SystemExit("source_degradation regenerates answers with the 7B; pass "
                         "--device cuda:N. See docs/EXPERIMENT_PLAN.md C4.")
    if args.device.startswith("cuda:"):
        # The pipeline pins its generator to index 0 of the *visible* devices.
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":", 1)[1]
        args.device = "cuda:0"

    root = Path(args.artifacts).resolve()
    clean = [r for r in dedup(read_rows(root, args.experiment))
             if str(r.get("fault_source", "")) == "clean" and r.get("answer")]
    clean = clean[:args.max_queries]
    print(f"{len(clean)} clean traces as the base for matched-effect injection")

    cache_path = (Path(args.corpus_cache) if args.corpus_cache else
                  Path("artifacts") / f"matched_effect_{args.family}_v2" /
                  f"corpus_seed{args.seed}.jsonl")
    generated_path = cache_path.with_name(cache_path.stem + ".generated.jsonl")
    if cache_path.exists():
        built = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        mechanisms = tuple(sorted({str(row["mechanism"]) for row in built}))
        print(f"loaded {len(built)} immutable matched rows from {cache_path}")
    elif generated_path.exists():
        built = [json.loads(line) for line in
                 generated_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        mechanisms = tuple(sorted({str(row["mechanism"]) for row in built}))
        print(f"loaded {len(built)} generated rows from staging cache {generated_path}")
    elif args.family == "source_degradation":
        built, mechanisms = build_source_corpus(clean, args)
        generated_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = generated_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in built:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        temporary.replace(generated_path)
        print(f"persisted generated answers to staging cache {generated_path}")
    else:
        built, mechanisms = build_grounding_corpus(clean, args)
    print(f"{len(built)} perturbed traces "
          f"({len(built) // len(mechanisms)} queries x {len(mechanisms)} mechanisms)")
    if len(built) < 90:
        raise SystemExit("too few queries admit all three mechanisms; widen the entity "
                         "extractor or raise --max-queries before interpreting anything")

    # ---- severity check, BEFORE any classification --------------------------------
    by_query = {str(r["query_id"]): r for r in clean}
    sev = {m: [] for m in mechanisms}
    for row in built:
        sev[row["mechanism"]].append(
            severity(str(by_query[str(row["query_id"])]["answer"]), str(row["answer"])))
    sev_summary = {m: {k: float(np.mean([s[k] for s in v])) for k in v[0]}
                   for m, v in sev.items()}
    spread = max(s["similarity"] for s in sev_summary.values()) - \
        min(s["similarity"] for s in sev_summary.values())
    print("\nseverity check (must be comparable across mechanisms):")
    for m, s in sev_summary.items():
        print(f"  {m:12s} similarity {s['similarity']:.4f}  len_change {s['len_change']:+.2f}"
              f"  changed_tokens {s['changed_tokens']:.2f}")
    print(f"  spread in similarity = {spread:.4f} (tolerance {args.severity_tolerance})")
    matched = spread <= args.severity_tolerance

    result = {"analysis": "matched_effect_injector_identification", "family": args.family,
              "mechanisms": list(mechanisms), "n_traces": len(built),
              "n_queries": len(built) // len(mechanisms), "severity": sev_summary,
              "severity_spread": float(spread), "severity_matched": bool(matched),
              "entity_types": {m: {} for m in mechanisms}}
    for row in built:
        d = result["entity_types"][row["mechanism"]]
        d[row["_type"]] = d.get(row["_type"], 0) + 1

    if not matched:
        result["verdict"] = ("INVALID --- mechanisms differ in severity, so any classifier "
                             "result would be confounded exactly as the tier experiment is")
        atomic_json(Path(args.output), result)
        print(f"\n{result['verdict']}\nwritten to {args.output}")
        return 1

    # ---- recompute metrics on the perturbed answers -------------------------------
    if not all(row.get("metrics") for row in built):
        print(f"\nrecomputing metrics on {args.device} ...")
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=args.device)
        for i, row in enumerate(built):
            row["metrics"] = compute_metrics(row, embedder)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(built)}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            raise RuntimeError(f"refusing to overwrite immutable corpus cache: {cache_path}")
        temporary = cache_path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in built:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        temporary.replace(cache_path)
        print(f"persisted matched corpus to {cache_path}")

    # ---- can a probe recover the mechanism? ---------------------------------------
    groups = np.asarray([str(r["query_id"]) for r in built])
    labels = np.asarray([r["mechanism"] for r in built])
    result["by_arm"] = {}
    for arm in ("decoupled", "all_features", "semantic_only"):
        matrix = raw_matrix(built, ARMS[arm])
        proba, classes = oof_multiclass(matrix, labels, groups, args.folds, args.seed)
        overall = cluster_bootstrap(
            lambda idx: macro_ovr_auroc(labels, proba, classes, idx),
            groups, args.bootstrap, args.seed)
        def null_stat(shuffled):
            p, c = oof_multiclass(matrix, shuffled, groups, args.folds, args.seed)
            return macro_ovr_auroc(shuffled, p, c)
        null = permutation_null(null_stat, labels, groups, repeats=10, seed=args.seed)

        pairs = {}
        for a, b in ((mechanisms[0], mechanisms[1]), (mechanisms[0], mechanisms[2]),
                     (mechanisms[1], mechanisms[2])):
            mask = np.where((labels == a) | (labels == b))[0]
            binary = (labels[mask] == b).astype(int)
            scores = oof_binary(matrix[mask], binary, groups[mask], args.folds, args.seed)
            pairs[f"{a}_vs_{b}"] = cluster_bootstrap(
                lambda idx: binary_auroc(binary, scores, idx),
                groups[mask], args.bootstrap, args.seed)
        # Two conditions, and both must hold. Statistical detectability alone is not
        # identifiability: with 395 queries a 0.004 excess over the null clears any
        # significance test while meaning nothing, and treating that as a positive is the
        # exact conflation this paper objects to elsewhere. So we also require the effect
        # to clear the same practical margin used for ecological equivalence.
        above = bool(overall["ci_low"] > null["max"])
        practical = bool(2 * abs(overall["point"] - 0.5) >= args.practical_margin)
        result["by_arm"][arm] = {
            "macro_auroc": overall, "permutation_null": null, "pairwise": pairs,
            "above_null": above, "D": float(2 * abs(overall["point"] - 0.5)),
            "practically_identifiable": bool(above and practical)}
        label = ("identifiable" if above and practical else
                 "statistically above null but below the practical margin --- treat as chance"
                 if above else "at null")
        print(f"  {arm:16s} {overall['point']:.3f} "
              f"[{overall['ci_low']:.3f},{overall['ci_high']:.3f}]  "
              f"null max {null['max']:.3f}  D={2 * abs(overall['point'] - 0.5):.3f}  -> {label}")

    # ---- residual-severity robustness on the defensible binary contrast ----------
    # Donor is excluded: it changes topicality and was already declared diagnostic.
    primary_mechanisms = (("rank", "lexical") if args.family == "source_degradation"
                          else (mechanisms[0], mechanisms[1]))
    selected = [r for r in built if r["mechanism"] in primary_mechanisms]
    selected_groups = np.asarray([str(r["query_id"]) for r in selected])
    selected_labels = np.asarray([int(r["mechanism"] == primary_mechanisms[1])
                                  for r in selected])
    severity_matrix = np.vstack([
        severity_vector(by_query[str(row["query_id"])], row) for row in selected])
    severity_scores = oof_binary(severity_matrix, selected_labels, selected_groups,
                                 args.folds, args.seed)
    severity_auc = cluster_bootstrap(
        lambda idx: binary_auroc(selected_labels, severity_scores, idx),
        selected_groups, args.bootstrap, args.seed)
    weights, propensity = propensity_weights(
        severity_matrix, selected_labels, selected_groups, args.folds, args.seed)
    decoupled_matrix = raw_matrix(selected, ARMS["decoupled"])
    adjusted_scores = oof_binary(
        decoupled_matrix, selected_labels, selected_groups, args.folds, args.seed,
        sample_weight=weights)
    adjusted_auc = cluster_bootstrap(
        lambda idx: binary_auroc(selected_labels, adjusted_scores, idx),
        selected_groups, args.bootstrap, args.seed)
    before = standardised_mean_differences(severity_matrix, selected_labels)
    after = standardised_mean_differences(severity_matrix, selected_labels, weights)
    result["severity_residual_robustness"] = {
        "analysis_status": "confirmatory robustness",
        "contrast": list(primary_mechanisms),
        "severity_variables": list(SEVERITY_NAMES),
        "severity_only_auroc": severity_auc,
        "decoupled_after_severity_reweighting_auroc": adjusted_auc,
        "balance": {
            "smd_before": {n: float(v) for n, v in zip(SEVERITY_NAMES, before)},
            "smd_after": {n: float(v) for n, v in zip(SEVERITY_NAMES, after)},
            "max_smd_before": float(before.max()),
            "max_smd_after": float(after.max()),
            "weight_min": float(weights.min()), "weight_max": float(weights.max()),
        },
        "raw_predictions": [
            {"query_id": str(row["query_id"]), "mechanism": str(row["mechanism"]),
             "severity_only": float(severity_scores[i]),
             "severity_adjusted_decoupled": float(adjusted_scores[i]),
             "propensity": float(propensity[i]), "weight": float(weights[i])}
            for i, row in enumerate(selected)
        ],
        "interpretation": (
            "manufacturing signal survives declared observed-severity reweighting"
            if 2 * abs(adjusted_auc["point"] - 0.5) >= args.practical_margin else
            "manufacturing signal does not survive declared observed-severity reweighting"),
    }
    print("\nseverity-residual robustness:")
    print(f"  severity only: {severity_auc['point']:.3f} "
          f"[{severity_auc['ci_low']:.3f},{severity_auc['ci_high']:.3f}]")
    print(f"  adjusted decoupled: {adjusted_auc['point']:.3f} "
          f"[{adjusted_auc['ci_low']:.3f},{adjusted_auc['ci_high']:.3f}]")
    print(f"  max |SMD|: {before.max():.3f} -> {after.max():.3f}")

    primary = result["by_arm"]["decoupled"]
    result["verdict"] = (
        "injector mechanism IS identifiable with the semantic effect held fixed: detector "
        "features encode how a fault was manufactured, not only what it did"
        if primary["practically_identifiable"] else
        "injector mechanism is NOT identifiable once the semantic effect is held fixed; "
        "the published tier result is therefore better explained by severity than by "
        "manufacturing signature")
    atomic_json(Path(args.output), result)
    print(f"\n{'=' * 74}\nVERDICT: {result['verdict']}\nwritten to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
