#!/usr/bin/env python
"""C5 --- do natural retrieval failures *cause* wrong answers, or merely accompany them?

Section 9 of the paper establishes an association: when a gold supporting paragraph is
missing the answer is wrong 39% of the time, against 18% when it is present. Association is
not causation. The missing paragraph might be a symptom of a hard question that the model
would have answered wrongly regardless.

The intervention settles it. For every trace that is both degraded and wrong we rebuild the
context three ways and regenerate under identical decoding:

    C0  original degraded context (as generated)
    C1  gold restored --- the lowest-ranked selected passage is replaced by the missing
        gold paragraph
    C2  placebo       --- the same slot is replaced by a different NON-gold passage,
        matched on length

C2 is what makes this an experiment rather than a demonstration. Swapping any passage
perturbs the prompt, and some answers will flip for that reason alone; the placebo measures
that background flip rate so it can be subtracted. k, passage count and ordering are held
fixed across all three conditions, so the only thing that varies is whether the restored
passage is the one that was actually needed.

    python scripts/run_counterfactual.py --config configs/experiments/E4_fingerprint.yaml \\
        --device cuda:1 --output results/counterfactual_v2.json

Registered outcomes (docs/EXPERIMENT_PLAN.md C5): a rescue rate well above placebo establishes
that the degradation is causally consequential and defines a causally-verified natural
failure label. Rates that match means restoring the evidence does not fix the answer, and
the transfer results must then be described as being about degraded intermediate states
rather than agent failures. If the verified subset is smaller than 50 we report the rates
and do not fit a detector on it.
"""
from __future__ import annotations

import argparse
import os
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np

from mirage.analysis import dedup, read_rows
from mirage.artifacts import atomic_json, stable_hash
from mirage.validity import cluster_bootstrap


def normalise(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def contains_gold(answer: str, gold: str) -> bool:
    a, g = normalise(answer), normalise(gold)
    return bool(g) and g in a


def build_conditions(row: dict, seed: int) -> dict | None:
    """Return {C1: context, C2: context} or None if this trace cannot be intervened on."""
    selected = list(row.get("selected_context") or [])
    rejected = list(row.get("rejected_context") or [])
    gold_titles = set(row.get("supporting_titles") or [])
    if not selected or not rejected or not gold_titles:
        return None

    present = {p.get("title") for p in selected}
    missing_gold = [p for p in rejected if p.get("title") in gold_titles
                    and p.get("title") not in present]
    non_gold = [p for p in rejected if p.get("title") not in gold_titles]
    if not missing_gold or not non_gold:
        return None

    rng = np.random.default_rng(int(stable_hash([seed, str(row["query_id"])]), 16) % 2**32)
    gold = missing_gold[rng.integers(len(missing_gold))]

    # The placebo is matched on length so the two interventions perturb the prompt by a
    # comparable amount; without that, a shorter placebo would be an easier condition.
    target = len(str(gold.get("text") or ""))
    placebo = min(non_gold, key=lambda p: abs(len(str(p.get("text") or "")) - target))

    # Replace a NON-GOLD slot. Taking the last slot unconditionally would, whenever the
    # single retrieved gold passage happens to sit last, swap gold for gold and leave the
    # context still one paragraph short --- a weaker intervention than the one we claim to
    # be running. Prefer the lowest-ranked non-gold slot; fall back to the last slot only
    # when every selected passage is gold, which cannot happen for a degraded trace.
    non_gold_slots = [i for i, p in enumerate(selected) if p.get("title") not in gold_titles]
    slot = non_gold_slots[-1] if non_gold_slots else len(selected) - 1

    def swap(passage):
        replaced = list(selected)
        replaced[slot] = passage
        return replaced

    return {"C1": swap(gold), "C2": swap(placebo), "slot": slot,
            "replaced_was_gold": bool(selected[slot].get("title") in gold_titles),
            "gold_title": gold.get("title"), "placebo_title": placebo.get("title")}


def mcnemar(flip_a: np.ndarray, flip_b: np.ndarray) -> dict:
    """Exact McNemar on the discordant pairs: rescued by one condition but not the other."""
    from scipy.stats import binomtest
    only_a = int(np.sum(flip_a & ~flip_b))
    only_b = int(np.sum(~flip_a & flip_b))
    n = only_a + only_b
    p = binomtest(only_a, n, 0.5).pvalue if n else float("nan")
    return {"rescued_by_gold_only": only_a, "rescued_by_placebo_only": only_b,
            "discordant": n, "p_value": float(p)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--experiment", default="E4_fingerprint_v2")
    parser.add_argument("--config", default="configs/experiments/E4_fingerprint.yaml")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # The pipeline pins its generator to index 0 of the visible devices, so restricting
    # visibility here is what puts it on the card the user asked for.
    if args.device.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":", 1)[1]

    root = Path(args.artifacts).resolve()
    clean = [r for r in dedup(read_rows(root, args.experiment))
             if str(r.get("fault_source", "")) == "clean" and r.get("gold_answer")]
    target = [r for r in clean
              if bool(r.get("natural_retrieval_failure"))
              and not contains_gold(str(r["answer"]), str(r["gold_answer"]))]
    print(f"{len(clean)} clean traces; {len(target)} are degraded AND wrong")

    plans = {str(r["query_id"]): build_conditions(r, args.seed) for r in target}
    usable = [r for r in target if plans[str(r["query_id"])] is not None]
    print(f"{len(usable)} admit both a gold restoration and a length-matched placebo")
    if not usable:
        raise SystemExit("no trace admits the intervention")

    from mirage.config import load_config
    from mirage.pipeline import RealAgenticRAG
    agent = RealAgenticRAG(load_config(args.config))

    answers = {}
    for condition in ("C1", "C2"):
        items = [(str(r["question"]), plans[str(r["query_id"])][condition],
                  r.get("tool_history") or []) for r in usable]
        print(f"regenerating {condition} ({len(items)} traces) ...", flush=True)
        answers[condition] = agent.answer_many(items)

    gold_correct = np.asarray([contains_gold(a, str(r["gold_answer"]))
                               for a, r in zip(answers["C1"], usable)])
    placebo_correct = np.asarray([contains_gold(a, str(r["gold_answer"]))
                                  for a, r in zip(answers["C2"], usable)])
    groups = np.asarray([str(r["query_id"]) for r in usable])

    # Every trace in `usable` was wrong under C0 by construction, so "correct now" is
    # exactly "rescued".
    r_gold = cluster_bootstrap(lambda i: float(np.mean(gold_correct[i])),
                               groups, args.bootstrap, args.seed)
    r_placebo = cluster_bootstrap(lambda i: float(np.mean(placebo_correct[i])),
                                  groups, args.bootstrap, args.seed)
    delta = cluster_bootstrap(
        lambda i: float(np.mean(gold_correct[i]) - np.mean(placebo_correct[i])),
        groups, args.bootstrap, args.seed)

    causal = gold_correct & ~placebo_correct
    result = {
        "analysis": "counterfactual_gold_restoration",
        "experiment": args.experiment, "n_degraded_and_wrong": len(target),
        "n_intervened": len(usable),
        "rescue_rate_gold": r_gold, "rescue_rate_placebo": r_placebo,
        "causal_effect": delta, "mcnemar": mcnemar(gold_correct, placebo_correct),
        "n_causally_verified": int(causal.sum()),
        "causally_verified_queries": sorted(groups[causal].tolist()),
        "note": "every intervened trace was wrong under the original context, so a correct "
                "answer after restoration is a rescue",
    }
    verified = int(causal.sum())
    result["verdict"] = (
        "natural retrieval degradation is causally consequential: restoring the missing "
        "gold paragraph rescues answers well above the placebo rate"
        if delta["ci_low"] > 0 else
        "no causal effect detected: restoring the gold paragraph rescues no more answers "
        "than an arbitrary passage swap, so the natural label marks a degraded "
        "intermediate state rather than a caused failure")
    result["sufficient_for_detector_fit"] = bool(verified >= 50)
    atomic_json(Path(args.output), result)

    print(f"\nrescue rate, gold restored  {r_gold['point']:.3f} "
          f"[{r_gold['ci_low']:.3f},{r_gold['ci_high']:.3f}]")
    print(f"rescue rate, placebo swap   {r_placebo['point']:.3f} "
          f"[{r_placebo['ci_low']:.3f},{r_placebo['ci_high']:.3f}]")
    print(f"causal effect               {delta['point']:+.3f} "
          f"[{delta['ci_low']:+.3f},{delta['ci_high']:+.3f}]")
    print(f"McNemar                     {result['mcnemar']}")
    print(f"\ncausally verified failures  {verified}"
          f"  ({'enough' if verified >= 50 else 'too few'} to fit a detector)")
    print(f"\nVERDICT: {result['verdict']}")
    print(f"written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
