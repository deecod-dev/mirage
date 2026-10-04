"""Tiered fault injectors operating only on real trace artifacts.

The injector ladder is the core of the paper, so each tier must be a genuinely
different operation:

  T0  naive       a surface marker or registry violation. Meant to be trivially
                  detectable — it demonstrates the effect exists.
  T1  faithful    a faithful reimplementation of a published-style injector.
                  This is the tier where the result matters.
  T2  hardened    same underlying fault, but produced through the pipeline's own
                  machinery so that surface statistics look native.

Two invariants distinguish this module from the first implementation:

1. **No injector writes a field that a metric reads back directly.** Writing
   ``call["ok"] = False`` and then scoring ``TIHD_health = failed/total`` measures
   the assignment, not the fault. Tools here fail by *actually executing* and
   failing; grounding faults perturb the *generation*, not the answer field.
   The one deliberate exception is T0, whose entire job is to be a naive tell.

2. **Injection is stochastic and seeded.** With greedy decoding the pipeline is
   deterministic, so if injection is deterministic too, extra seeds produce
   byte-identical traces and every seed-based CI is vacuous. Each injector draws
   its choices from a per-trace RNG derived from (seed, query_id, tier, family).

Every injector returns an :class:`Injection` naming what the runner must re-run.
Nothing downstream of a perturbation may be scored against a stale value.
"""
from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal

from mirage.artifacts import stable_hash

Tier = Literal["T0", "T1", "T2"]
Family = Literal["query_retrieval", "source_degradation", "tool_invocation", "grounding"]
Rerun = Literal["none", "verify", "downstream", "full"]

REGISTERED_TOOLS = frozenset({"calculator", "document_lookup"})

#: Words that carry no retrieval signal, used when choosing a token to delete.
STOPWORDS = frozenset("""
a an the of in on at to for from by with and or but is are was were be been being
which who whom whose what when where why how that this these those it its as than
did does do have has had can could would should will shall may might must not no
""".split())

_ENTITY = re.compile(r"\b(?:[A-Z][\w'’-]*(?:\s+[A-Z][\w'’-]*)*|\d[\d,.]*)\b")


@dataclass(frozen=True)
class Injection:
    """A faulted trace plus the amount of pipeline the runner must re-execute.

    ``rerun`` values, cheapest first:

    ``none``        the perturbation is already terminal.
    ``verify``      re-run the verifier only (the answer changed, context did not).
    ``downstream``  re-run answer generation and verification.
    ``full``        re-run the whole pipeline from planning (the question changed).
    """

    trace: dict[str, Any]
    rerun: Rerun


def trace_rng(seed: int, query_id: str, tier: str, family: str) -> random.Random:
    """Deterministic per-trace RNG. Stable across resumes and process restarts."""
    return random.Random(int(stable_hash([seed, query_id, tier, family]), 16))


def _content_words(question: str) -> list[int]:
    tokens = question.split()
    return [i for i, t in enumerate(tokens)
            if t.strip("?.,;:'\"").lower() not in STOPWORDS and len(t.strip("?.,;:'\"")) > 3]


def _entities(text: str) -> list[str]:
    seen, out = set(), []
    for match in _ENTITY.finditer(text):
        value = match.group(0).strip()
        if len(value) > 1 and value.lower() not in STOPWORDS and value not in seen:
            seen.add(value)
            out.append(value)
    return out


# ── query_retrieval ───────────────────────────────────────────────────────────
def _query_retrieval(out: dict[str, Any], tier: Tier, rng: random.Random) -> Injection:
    """Degrade the question itself. Always re-plans and re-retrieves."""
    question = out["question"]
    tokens = question.split()
    if tier == "T0":
        # Naive: word order destroyed. Retrieval still works on bag-of-words
        # embeddings, so this is a weak fault with an obvious surface signature.
        out["question"] = " ".join(reversed(tokens))
    elif tier == "T1":
        # Published-style token dropout: delete a random ~50% of tokens.
        keep = [t for t in tokens if rng.random() > 0.5]
        out["question"] = " ".join(keep) if len(keep) >= 2 else " ".join(tokens[::2])
    else:
        # Hardened: delete one high-information content word. The question stays
        # grammatical and its length barely moves, so surface statistics do not
        # give the fault away — but the retrievable evidence genuinely changes.
        candidates = _content_words(question)
        if not candidates:
            out["question"] = " ".join(tokens[:-1]) or question
        else:
            ranked = sorted(candidates, key=lambda i: len(tokens[i]), reverse=True)[:3]
            drop = rng.choice(ranked)
            out["question"] = " ".join(t for i, t in enumerate(tokens) if i != drop)
    return Injection(out, "full")


# ── source_degradation ────────────────────────────────────────────────────────
def _source_degradation(out: dict[str, Any], tier: Tier, rng: random.Random) -> Injection:
    """Corrupt the evidence the generator is about to read."""
    selected, rejected = out["selected_context"], out.get("rejected_context", [])
    if tier == "T0":
        # Naive: a literal marker token in every passage. A regex baseline finds this.
        for passage in selected:
            passage["text"] = "OUTDATED " + passage["text"]
    elif tier == "T1":
        # Published-style full evidence swap. Selected and rejected are *exchanged*,
        # never copied: copying leaves the two sets identical, which pins
        # QCM = mean(q·selected) - mean(q·rejected) to bit-exact 0.0 and hands the
        # probe a signature no natural fault can produce.
        if not rejected:
            return Injection(out, "downstream")
        take = min(len(selected), len(rejected))
        promoted = [copy.deepcopy(p) for p in rejected[:take]]
        demoted = [copy.deepcopy(p) for p in selected[:take]]
        out["selected_context"] = promoted + selected[take:]
        out["rejected_context"] = demoted + rejected[take:]
    else:
        # Hardened: keep the top-ranked passage, replace the rest with the
        # near-miss distractors that just failed to make the cut. Partial, plausible
        # evidence — the kind of degradation a real reranker produces.
        if len(selected) < 2 or not rejected:
            return Injection(out, "downstream")
        keep = selected[:1]
        # Sample from the full near-miss pool. Shuffling a prefix of length one made
        # k=2 runs seed-invariant even when many rejected passages were available.
        take = min(len(selected) - 1, len(rejected))
        pool = [copy.deepcopy(p) for p in rng.sample(rejected, take)]
        demoted = [copy.deepcopy(p) for p in selected[1:1 + len(pool)]]
        out["selected_context"] = keep + pool
        out["rejected_context"] = demoted + rejected[len(pool):]
    return Injection(out, "downstream")


# ── tool_invocation ───────────────────────────────────────────────────────────
def _tool_invocation(out: dict[str, Any], tier: Tier, rng: random.Random) -> Injection:
    """Perturb tool use. T1 and T2 fail through real execution, never by assignment.

    Deferred to the runner, which owns the live tool objects; this function records
    the directive and the runner executes it. Writing ``ok = False`` here would make
    TIHD_health a read-back of our own assignment.
    """
    from mirage.tools import Calculator, DocumentLookup

    history = out.get("tool_history") or []
    if not history:
        history = [{"name": "document_lookup",
                    "arguments": {"query": out.get("question", "")},
                    "output": "", "ok": True}]
        out["tool_history"] = history
    index = rng.randrange(len(history))
    call = history[index]

    if tier == "T0":
        # Naive: an unregistered tool name. `tool_name not in registry` finds it.
        call["name"] = "unregistered_tool"
        return Injection(out, "downstream")

    if call["name"] == "calculator":
        expression = str(call.get("arguments", {}).get("expression", ""))
        numbers = re.findall(r"\d+(?:\.\d+)?", expression)
        if tier == "T1":
            # Published-style hard failure: a genuinely malformed expression, really
            # evaluated. ok=False is the calculator's verdict, not ours.
            broken = expression.replace("+", "+/", 1) if "+" in expression else expression + " /"
            executed = Calculator().execute(broken)
        elif len(numbers) >= 2:
            # Hardened: a *valid* expression with two operands transposed. It executes
            # cleanly and returns a real number that happens to be wrong, so neither
            # TIHD_health nor TRTR_route can see it at all.
            pairs = [(i, j) for i in range(len(numbers))
                     for j in range(i + 1, len(numbers))]
            rng.shuffle(pairs)
            executed = None
            for first, second in pairs:
                swapped, cursor = [], 0
                for position, match in enumerate(re.finditer(r"\d+(?:\.\d+)?", expression)):
                    swapped.append(expression[cursor:match.start()])
                    source = {first: second, second: first}.get(position, position)
                    swapped.append(numbers[source])
                    cursor = match.end()
                swapped.append(expression[cursor:])
                candidate = Calculator().execute("".join(swapped))
                # Operand transposition can preserve the value (for example swapping
                # the operands of a multiplication). A hardened fault must change the
                # result, not merely the expression string.
                if candidate.ok and candidate.output != str(call.get("output", "")):
                    executed = candidate
                    break
            if executed is None:
                # Degenerate commutative expressions still need a valid, changed result.
                match = next(re.finditer(r"\d+(?:\.\d+)?", expression))
                replacement = str(float(match.group()) + 1.0)
                changed = expression[:match.start()] + replacement + expression[match.end():]
                executed = Calculator().execute(changed)
        else:
            executed = Calculator().execute(expression + " * 1.1")
        call.update({"name": executed.name, "arguments": dict(executed.arguments),
                     "output": executed.output, "ok": executed.ok})
        return Injection(out, "downstream")

    # document_lookup: run the real tool against the wrong passage set.
    lookup = DocumentLookup()
    if tier == "T1":
        # Published-style retrieval-tool failure: the call really returns nothing.
        executed = lookup.execute(str(call.get("arguments", {}).get("query", "")), [])
    else:
        # Hardened: a successful call over the rejected passages. ok=True, plausible
        # prose, wrong evidence — invisible to every tool-health metric.
        from mirage.data.hotpot import Passage
        pool = out.get("rejected_context") or out.get("selected_context") or []
        chosen = [Passage(**p) for p in pool[:max(1, len(pool) // 2)]]
        executed = lookup.execute(str(call.get("arguments", {}).get("query", "")), chosen)
    call.update({"name": executed.name, "arguments": dict(executed.arguments),
                 "output": executed.output, "ok": executed.ok})
    return Injection(out, "downstream")


# ── grounding ─────────────────────────────────────────────────────────────────
def _grounding(out: dict[str, Any], tier: Tier, rng: random.Random,
               donor: dict[str, Any] | None, generate: Callable[[str, str], str] | None) -> Injection:
    """Make the answer unfaithful to the evidence.

    Only T0 overwrites the answer field wholesale. T1 edits the model's own answer
    in place, and T2 has the generator produce the unfaithful answer itself, so the
    length, register and token statistics stay native.
    """
    answer = out.get("answer", "")
    if tier == "T0":
        # Naive: a whole answer lifted from a different question. AGS collapses,
        # answer_length jumps — exactly the signature this tier is meant to expose.
        if donor is None or donor.get("query_id") == out.get("query_id"):
            raise ValueError("grounding fault requires a different real trace as donor")
        out["answer"] = donor["answer"]
        return Injection(out, "verify")

    if tier == "T1":
        # Published-style entity substitution (FactCC/SummEdits family): swap one
        # entity in the model's own answer for an entity drawn from evidence the
        # reranker rejected. Fluent, same length, single unsupported claim.
        targets = _entities(answer)
        pool_text = " ".join(p["text"] for p in out.get("rejected_context", [])) or \
                    " ".join(p["text"] for p in out.get("selected_context", []))
        replacements = [e for e in _entities(pool_text) if e not in targets]
        if targets and replacements:
            target = rng.choice(targets)
            out["answer"] = answer.replace(target, rng.choice(replacements), 1)
            return Injection(out, "verify")
        if donor is None:
            raise ValueError("grounding fault requires a different real trace as donor")
        out["answer"] = donor["answer"]
        return Injection(out, "verify")

    # T2 hardened: the generator writes the unfaithful answer. Nothing is pasted in,
    # so surface statistics are indistinguishable from a clean generation. The detail
    # to corrupt is drawn from the RNG rather than left to the model, both to control
    # what the fault is and so that seeds produce genuinely different traces — the
    # prompt is otherwise identical across seeds and greedy decoding would repeat it.
    if generate is None:
        raise ValueError("T2 grounding requires a generate callable")
    targets = _entities(answer)
    target = rng.choice(targets) if targets else None
    evidence = "\n\n".join(f"[{p['title']}] {p['text']}" for p in out["selected_context"])
    instruction = (
        f"Rewrite the draft answer so that the detail \"{target}\" is replaced by a different, "
        "plausible-sounding detail of the same kind that the evidence does not support."
        if target else
        "Rewrite the draft answer so that exactly one factual detail is not supported by the evidence.")
    produced = generate(
        "You rewrite answers for a robustness benchmark. Preserve the draft's wording, length and "
        "register as closely as possible, changing only what you are told to change. Do not flag, "
        "hedge, or mention the change. Return only the rewritten answer.",
        f"Question: {out['question']}\n\nEvidence:\n{evidence}\n\nDraft answer: {answer}\n\n{instruction}",
    ).strip()
    out["answer"] = produced or answer
    out["grounding_target"] = target
    return Injection(out, "verify")


# ── entry point ───────────────────────────────────────────────────────────────
def inject(trace: dict[str, Any], tier: Tier, family: Family, *,
           seed: int = 11,
           donor: dict[str, Any] | None = None,
           generate: Callable[[str, str], str] | None = None) -> Injection:
    """Return a faulted copy of ``trace`` plus the re-execution it requires."""
    if tier not in ("T0", "T1", "T2"):
        raise ValueError(f"unknown tier: {tier}")
    out = copy.deepcopy(trace)
    out["fault_tier"], out["fault_family"] = tier, family
    out["fault_source"] = f"{tier}:{family}"
    out["injector_seed"] = seed
    rng = trace_rng(seed, str(out.get("query_id", "")), tier, family)

    if family == "query_retrieval":
        return _query_retrieval(out, tier, rng)
    if family == "source_degradation":
        return _source_degradation(out, tier, rng)
    if family == "tool_invocation":
        return _tool_invocation(out, tier, rng)
    if family == "grounding":
        return _grounding(out, tier, rng, donor, generate)
    raise ValueError(f"unknown fault family: {family}")
