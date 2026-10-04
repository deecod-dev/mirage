"""Choose `answer_top_k` and the candidate-pool size from measurement, not taste.

Retrieval-depth decision. The natural-retrieval-failure rate is what makes E5
possible, and it is set by two knobs:

  answer_top_k   how much evidence reaches the generator
  pool_padding   how many paragraphs from *other* HotpotQA questions join the
                 candidate set, approximating open-corpus retrieval difficulty
                 without leaving the dataset or building a full-wiki index

Both are evaluated here with the cross-encoder alone — no 7B generator — by
replaying the consolidation step over trace artifacts that already exist. A sweep
therefore costs a minute instead of an experiment.

Usage:
    python scripts/sweep_retrieval.py --experiment P1_retrieval_check \
        --k 1 2 3 4 --padding 0 20 40 --output results/retrieval_sweep.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from mirage.artifacts import atomic_json
from mirage.config import load_config
from mirage.pipeline import RealAgenticRAG

#: Below this many traces per class the transfer arm is not reportable.
MIN_NATURAL = 150


def read_clean(root: Path, experiment: str) -> list[dict[str, Any]]:
    """One clean trace per query. Deduplication is not cosmetic here: legacy runs hold
    ten identical seed copies of every trace, which would fill the hard-negative bank
    with the same passage ten times over and leave the padded pool barely larger."""
    rows = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((root / experiment).rglob("trace_*.json"))]
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("fault_source") == "clean":
            unique.setdefault(str(row["query_id"]), row)
    if not unique:
        raise FileNotFoundError(f"no clean traces under {root / experiment}")
    return list(unique.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--config", default="configs/experiments/P1_retrieval_check.yaml")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--padding", type=int, nargs="+", default=[0, 20, 40])
    parser.add_argument("--hard-negatives", action="store_true",
                        help="pad with the paragraphs from other queries that are most "
                             "similar to this question, rather than random ones. Random "
                             "padding is discarded trivially by the cross-encoder and does "
                             "not reproduce open-corpus difficulty.")
    parser.add_argument("--target-queries", type=int, default=300,
                        help="held-out partition size the rate will be extrapolated to")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.artifacts).resolve()
    rows = read_clean(root, args.experiment)
    agent = RealAgenticRAG(load_config(args.config))
    reranker = agent._load_reranker()
    rng = random.Random(args.seed)

    # Every passage in the sample, used as the distractor bank for pool padding.
    bank = [(r["query_id"], p) for r in rows for p in r["rejected_context"] + r["selected_context"]]
    bank_owner = np.asarray([qid for qid, _ in bank])
    bank_vectors = None
    if args.hard_negatives:
        embedder = agent._load_embedder()
        bank_vectors = np.asarray(embedder.encode([p["text"] for _, p in bank],
                                                  normalize_embeddings=True, batch_size=256))
        question_vectors = np.asarray(embedder.encode([r["question"] for r in rows],
                                                      normalize_embeddings=True, batch_size=256))

    grid: list[dict[str, Any]] = []
    for padding in args.padding:
        # Score each query's padded pool once; the k sweep is then just a slice.
        ranked: list[tuple[set[str], list[str]]] = []
        for position, row in enumerate(rows):
            pool = {p["passage_id"]: p for p in row["selected_context"] + row["rejected_context"]}
            if padding and args.hard_negatives:
                similarity = bank_vectors @ question_vectors[position]
                similarity[bank_owner == row["query_id"]] = -np.inf
                extra = [bank[int(i)][1] for i in np.argsort(similarity)[::-1][:padding]]
                pool.update({p["passage_id"]: p for p in extra})
            elif padding:
                extra = [p for qid, p in rng.sample(bank, min(len(bank), padding * 3))
                         if qid != row["query_id"]][:padding]
                pool.update({p["passage_id"]: p for p in extra})
            passages = list(pool.values())
            scores = reranker.predict([(row["question"], p["text"]) for p in passages])
            order = np.argsort(np.asarray(scores))[::-1]
            ranked.append((set(row["supporting_titles"]),
                           [passages[int(i)]["title"] for i in order]))

        for k in args.k:
            incomplete = sum(1 for gold, titles in ranked if gold - set(titles[:k]))
            severe = sum(1 for gold, titles in ranked if not (gold & set(titles[:k])))
            recall = float(np.mean([len(gold & set(titles[:k])) / len(gold)
                                    for gold, titles in ranked if gold]))
            rate = incomplete / len(ranked)
            projected = round(rate * args.target_queries)
            grid.append({
                "answer_top_k": k, "pool_padding": padding,
                "pool_size": round(float(np.mean([len(t) for _, t in ranked])), 1),
                "natural_failure_rate": rate,
                "severe_rate": severe / len(ranked),
                "mean_support_recall": recall,
                "projected_failures_at_target": projected,
                "projected_successes_at_target": args.target_queries - projected,
                "meets_K3": min(projected, args.target_queries - projected) >= MIN_NATURAL,
            })

    usable = [g for g in grid if g["meets_K3"]]
    result = {"analysis": "retrieval_sweep", "experiment": args.experiment,
              "hard_negatives": args.hard_negatives,
              "n_queries_sampled": len(rows), "target_queries": args.target_queries,
              "min_natural_required": MIN_NATURAL, "grid": grid,
              "recommended": max(usable, key=lambda g: g["mean_support_recall"]) if usable else None}
    atomic_json(Path(args.output), result)

    print(f"sampled {len(rows)} clean queries; projecting to {args.target_queries}\n")
    print(f"{'k':>2} {'pad':>4} {'pool':>6} {'nat.fail':>9} {'severe':>7} {'recall':>7} "
          f"{'proj fail':>10} {'K3':>4}")
    for g in grid:
        print(f"{g['answer_top_k']:>2} {g['pool_padding']:>4} {g['pool_size']:>6.1f} "
              f"{g['natural_failure_rate']:>8.1%} {g['severe_rate']:>7.1%} "
              f"{g['mean_support_recall']:>7.3f} {g['projected_failures_at_target']:>10} "
              f"{'yes' if g['meets_K3'] else 'no':>4}")
    if result["recommended"]:
        r = result["recommended"]
        print(f"\nrecommended: answer_top_k={r['answer_top_k']} pool_padding={r['pool_padding']} "
              f"-> {r['natural_failure_rate']:.1%} natural failures "
              f"({r['projected_failures_at_target']} of {args.target_queries})")
    else:
        print(f"\nNo setting reaches {MIN_NATURAL} traces per class at {args.target_queries} "
              "queries. Widen --padding or reduce --k, else K3 fires and E5 drops to a limitation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
