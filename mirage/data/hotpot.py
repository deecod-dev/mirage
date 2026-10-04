"""HotpotQA adapter using real annotated contexts and supporting-fact labels."""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Passage:
    passage_id: str
    title: str
    text: str


@dataclass(frozen=True)
class HotpotExample:
    query_id: str
    question: str
    answer: str
    passages: tuple[Passage, ...]
    supporting_titles: frozenset[str]


def _id(question: str, title: str, index: int) -> str:
    return hashlib.sha256(f"{question}|{title}|{index}".encode()).hexdigest()[:16]


def load_hotpotqa(split: str, limit: int | None, seed: int) -> list[HotpotExample]:
    """Load the real HotpotQA distractor split; downloads only when explicitly called."""
    from datasets import load_dataset

    raw = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    indices = list(range(len(raw)))
    random.Random(seed).shuffle(indices)
    if limit is not None:
        indices = indices[:limit]
    examples: list[HotpotExample] = []
    for idx in indices:
        row = raw[idx]
        passages = []
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"], strict=True):
            text = " ".join(sentences).strip()
            if text:
                passages.append(Passage(_id(row["question"], title, len(passages)), title, text))
        support = frozenset(row["supporting_facts"]["title"])
        examples.append(HotpotExample(str(row["id"]), row["question"], row["answer"], tuple(passages), support))
    return examples


def missing_gold_support(example: HotpotExample, retrieved: Iterable[Passage]) -> bool:
    """Automatically verifiable natural-retrieval-failure label."""
    return not example.supporting_titles.intersection(p.title for p in retrieved)
