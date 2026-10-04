#!/usr/bin/env python
"""Do our "natural retrieval failures" actually cause the agent to answer wrongly?

The transfer experiment labels a clean trace a natural failure when its selected context
omits at least one gold supporting paragraph. That is a degraded *intermediate state*, not
an observed end-to-end failure, and the paper's claim ("the injected fault-detection task
is easier than the natural one") is only about agent failures if the degradation actually
propagates to the answer.

Every trace already carries `gold_answer` and the generated `answer`, so this costs no GPU.
We measure answer correctness, cross-tabulate it against retrieval degradation, and emit a
stricter label:

    natural_consequential = gold paragraph missing AND final answer wrong

Two things this establishes. First, whether degradation and wrongness are associated at
all --- if a missing gold paragraph leaves the answer just as often correct, the label was
measuring plumbing rather than failure. Second, a supply of consequential positives that
the transfer analysis can target instead of the intermediate-state proxy.

    python scripts/run_natural_labels.py --experiment E4_fingerprint_v2 \
        --output results/natural_labels_v2.json

CORRECTNESS METRIC. HotpotQA gold answers are short spans while our generator emits prose,
so exact match is inapplicable. We report SQuAD-normalised containment (is the gold span
present in the answer) as the primary signal and token-F1 as a continuous companion, and
we report both so a reader can see the conclusion does not depend on the choice.

WHAT IS STILL MISSING. Association is not causation. The clean statement --- that this
retrieval event *caused* the wrong answer --- needs a counterfactual: restore the gold
paragraph, regenerate, and check the answer flips to correct. That requires generation and
is left for the GPU arm; `--emit-counterfactual-plan` writes the query list it would need.
"""
from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np

from mirage.analysis import dedup, read_rows
from mirage.artifacts import atomic_json
from mirage.validity import cluster_bootstrap


def normalise(text: str) -> str:
    """SQuAD normalisation: lowercase, drop articles, punctuation and extra whitespace."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def contains_gold(answer: str, gold: str) -> bool:
    a, g = normalise(answer), normalise(gold)
    return bool(g) and g in a


def token_f1(answer: str, gold: str) -> float:
    a, g = normalise(answer).split(), normalise(gold).split()
    if not a or not g:
        return 0.0
    shared = Counter(a) & Counter(g)
    same = sum(shared.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(a), same / len(g)
    return 2 * precision * recall / (precision + recall)


def rate(mask: np.ndarray, groups: np.ndarray, repeats: int, seed: int) -> dict:
    return cluster_bootstrap(lambda idx: float(np.mean(mask[idx])) if len(idx) else np.nan,
                             groups, repeats, seed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--experiment", default="E4_fingerprint_v2")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--emit-counterfactual-plan", default=None,
                        help="write the query ids a counterfactual regeneration would need")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.artifacts).resolve()
    rows = [r for r in dedup(read_rows(root, args.experiment))
            if str(r.get("fault_source", "")) == "clean"]
    rows = [r for r in rows if r.get("gold_answer") and r.get("answer")]
    print(f"{len(rows)} clean traces with a gold answer")

    groups = np.asarray([str(r["query_id"]) for r in rows])
    degraded = np.asarray([bool(r.get("natural_retrieval_failure")) for r in rows])
    severe = np.asarray([bool(r.get("natural_retrieval_failure_severe")) for r in rows])
    correct = np.asarray([contains_gold(r["answer"], r["gold_answer"]) for r in rows])
    f1 = np.asarray([token_f1(r["answer"], r["gold_answer"]) for r in rows])
    wrong = ~correct
    consequential = degraded & wrong

    table = {"degraded_and_wrong": int((degraded & wrong).sum()),
             "degraded_and_correct": int((degraded & correct).sum()),
             "intact_and_wrong": int((~degraded & wrong).sum()),
             "intact_and_correct": int((~degraded & correct).sum())}

    p_wrong_degraded = rate(wrong[degraded], groups[degraded], args.bootstrap, args.seed)
    p_wrong_intact = rate(wrong[~degraded], groups[~degraded], args.bootstrap, args.seed)

    # Paired on nothing --- these are disjoint groups --- so the interval on the
    # difference is computed by resampling queries across the whole set.
    def lift(idx):
        d, w = degraded[idx], wrong[idx]
        if d.sum() == 0 or (~d).sum() == 0:
            return np.nan
        return float(w[d].mean() - w[~d].mean())
    lift_ci = cluster_bootstrap(lift, groups, args.bootstrap, args.seed)

    result = {
        "analysis": "natural_failure_consequence",
        "experiment": args.experiment,
        "n_clean_traces": len(rows),
        "n_queries": int(len(np.unique(groups))),
        "contingency": table,
        "n_degraded": int(degraded.sum()),
        "n_severe": int(severe.sum()),
        "n_consequential": int(consequential.sum()),
        "answer_correct_overall": rate(correct, groups, args.bootstrap, args.seed),
        "token_f1_overall": rate(f1, groups, args.bootstrap, args.seed),
        "p_wrong_given_degraded": p_wrong_degraded,
        "p_wrong_given_intact": p_wrong_intact,
        "wrongness_lift": lift_ci,
        "token_f1_degraded": rate(f1[degraded], groups[degraded], args.bootstrap, args.seed),
        "token_f1_intact": rate(f1[~degraded], groups[~degraded], args.bootstrap, args.seed),
        "note": "correctness is SQuAD-normalised containment of the gold span in the "
                "generated answer; token_f1 is reported alongside as a continuous check",
    }
    atomic_json(Path(args.output), result)

    print(f"\n{'':22s}{'wrong':>8s}{'correct':>9s}")
    print(f"{'gold missing':22s}{table['degraded_and_wrong']:>8d}"
          f"{table['degraded_and_correct']:>9d}")
    print(f"{'gold present':22s}{table['intact_and_wrong']:>8d}"
          f"{table['intact_and_correct']:>9d}")
    print(f"\nP(wrong | gold missing) {p_wrong_degraded['point']:.3f} "
          f"[{p_wrong_degraded['ci_low']:.3f},{p_wrong_degraded['ci_high']:.3f}]")
    print(f"P(wrong | gold present) {p_wrong_intact['point']:.3f} "
          f"[{p_wrong_intact['ci_low']:.3f},{p_wrong_intact['ci_high']:.3f}]")
    print(f"lift                    {lift_ci['point']:+.3f} "
          f"[{lift_ci['ci_low']:+.3f},{lift_ci['ci_high']:+.3f}]")
    verdict = ("degradation is associated with wrong answers"
               if lift_ci["ci_low"] > 0 else
               "NO significant association --- the label marks plumbing, not failure")
    print(f"-> {verdict}")
    print(f"\nconsequential natural failures (missing AND wrong): {consequential.sum()}")

    if args.emit_counterfactual_plan:
        plan = {"queries": sorted({str(r["query_id"]) for r, d in zip(rows, degraded) if d}),
                "why": "regenerate each with the gold paragraph restored; the retrieval "
                       "event is causally consequential where the answer flips to correct"}
        Path(args.emit_counterfactual_plan).write_text(json.dumps(plan, indent=2))
        print(f"counterfactual plan -> {args.emit_counterfactual_plan}")

    print(f"written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
