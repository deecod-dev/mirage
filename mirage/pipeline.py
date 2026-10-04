"""Real, lazy-loaded agentic RAG pipeline. No component has a mock fallback.

Two changes matter relative to the first implementation.

**Retrieval is selective.** Previously each subgoal retrieved top-k and the results
were unioned, so on HotpotQA distractor (10 candidate paragraphs, 2--4 subgoals,
k=5) the union was the entire candidate pool: mean |selected_context| was 10.0 out
of 10. Retrieval filtered nothing, every retrieval-sensitive metric scored a set
that always contained the answer, and `natural_retrieval_failure` was False for
every query in the corpus. Per-subgoal retrieval now builds a *candidate pool*
(kept as `step_contexts`, which is what the trajectory metrics need), and a final
cross-encoder consolidation selects the `answer_top_k` passages that actually
reach the generator.

**Generation is batched.** Greedy decoding one prompt at a time leaves an A6000
almost idle. Prompts are grouped into batches with left padding, which is where
essentially all of the throughput on a single GPU comes from.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from mirage.config import ExperimentConfig
from mirage.data.hotpot import HotpotExample, Passage
from mirage.tools import Calculator, DocumentLookup, ToolCall

PLANNER_SYSTEM = "You are a precise RAG-agent planner. Return JSON only."
ANSWER_SYSTEM = "Answer only from supplied evidence and tool outputs. If evidence is insufficient, say so."
VERIFY_SYSTEM = "Return JSON only with boolean verified and a short reason. Verify only against evidence."


@dataclass(frozen=True)
class Subgoal:
    text: str
    tool: str
    expression: str | None = None


def evidence_block(passages: Iterable[dict[str, Any] | Passage]) -> str:
    def field(passage, name):
        return passage[name] if isinstance(passage, dict) else getattr(passage, name)
    return "\n\n".join(f"[{field(p, 'title')}] {field(p, 'text')}" for p in passages)


def tool_block(calls: Iterable[dict[str, Any] | ToolCall]) -> str:
    def field(call, name):
        return call[name] if isinstance(call, dict) else getattr(call, name)
    return "\n".join(f"{field(c, 'name')}({field(c, 'arguments')}) => {field(c, 'output')}" for c in calls)


class RealAgenticRAG:
    """Loads actual Hugging Face/SentenceTransformers components on first use."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self._embedder = None
        self._reranker = None
        self._tokenizer = None
        self._generator = None
        self.calculator = Calculator()
        self.lookup = DocumentLookup()

    # ── components ───────────────────────────────────────────────────────────
    def _torch_device(self) -> str:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(
                self.config.model.embedder, revision=self.config.model.revision_for("embedder"),
                device=self._torch_device())
        return self._embedder

    def _load_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(
                self.config.model.reranker, revision=self.config.model.revision_for("reranker"),
                device=self._torch_device())
        return self._reranker

    def _load_generator(self):
        if self._generator is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            # Pin to a single visible device. `device_map="auto"` shards across every
            # visible GPU, which silently occupies hardware other jobs may be using;
            # select the device with CUDA_VISIBLE_DEVICES instead.
            kwargs: dict[str, Any] = {"revision": self.config.model.revision_for("generator"),
                                      "device_map": {"": 0} if torch.cuda.is_available() else "cpu"}
            if self.config.runtime == "kaggle_t4":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            else:
                kwargs["torch_dtype"] = torch.bfloat16
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.generator, revision=self.config.model.revision_for("generator"))
            # Decoder-only batched generation requires left padding, or short prompts
            # in a batch decode from padding tokens and produce garbage.
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            self._tokenizer = tokenizer
            self._generator = AutoModelForCausalLM.from_pretrained(self.config.model.generator, **kwargs)
            self._generator.eval()
        return self._tokenizer, self._generator

    # ── batched generation ───────────────────────────────────────────────────
    def chat_many(self, prompts: Sequence[tuple[str, str]], *, max_new_tokens: int = 384) -> list[str]:
        """Greedy-decode a list of (system, user) prompts in left-padded batches."""
        import torch
        if not prompts:
            return []
        tokenizer, model = self._load_generator()
        texts = [tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True) for system, user in prompts]
        outputs: list[str] = []
        for start in range(0, len(texts), self.config.batch_size):
            chunk = texts[start:start + self.config.batch_size]
            encoded = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True,
                                max_length=self.config.max_input_tokens).to(model.device)
            with torch.inference_mode():
                generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                           pad_token_id=tokenizer.pad_token_id)
            width = encoded["input_ids"].shape[1]
            outputs.extend(tokenizer.decode(row[width:], skip_special_tokens=True).strip()
                           for row in generated)
        return outputs

    def _chat(self, system: str, user: str, *, max_new_tokens: int = 384) -> str:
        return self.chat_many([(system, user)], max_new_tokens=max_new_tokens)[0]

    @staticmethod
    def _json_object(text: str) -> dict[str, Any]:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        candidate = match.group(1) if match else text[text.find("{"):text.rfind("}") + 1]
        if not candidate:
            raise ValueError("LLM did not return a JSON object")
        return json.loads(candidate)

    # ── planning ─────────────────────────────────────────────────────────────
    @staticmethod
    def _planner_prompt(question: str) -> tuple[str, str]:
        return PLANNER_SYSTEM, (
            "Create 2-4 sequential subgoals for this question. Each item must have text, tool, and "
            "optional expression. tool is exactly document_lookup or calculator. Use calculator only "
            "when an arithmetic expression is available. Schema: {\"subgoals\":[...]}.\nQuestion: " + question)

    def _parse_plan(self, response: str, question: str) -> list[Subgoal]:
        try:
            goals = self._json_object(response).get("subgoals", [])
            if not isinstance(goals, list) or not goals:
                goals = [{"text": question, "tool": "document_lookup"}]
        except Exception:
            goals = [{"text": question, "tool": "document_lookup"}]
        parsed: list[Subgoal] = []
        for goal in goals:
            if isinstance(goal, dict):
                tool = str(goal.get("tool", "document_lookup"))
                parsed.append(Subgoal(str(goal.get("text", question)),
                                      tool if tool in {"document_lookup", "calculator"} else "document_lookup",
                                      goal.get("expression")))
            elif isinstance(goal, str):
                parsed.append(Subgoal(goal, "document_lookup", None))
        return parsed or [Subgoal(question, "document_lookup", None)]

    def plan_many(self, questions: Sequence[str]) -> list[list[Subgoal]]:
        responses = self.chat_many([self._planner_prompt(q) for q in questions], max_new_tokens=300)
        return [self._parse_plan(r, q) for r, q in zip(responses, questions)]

    def plan(self, question: str) -> list[Subgoal]:
        return self.plan_many([question])[0]

    # ── retrieval ────────────────────────────────────────────────────────────
    def retrieve(self, query: str, passages: tuple[Passage, ...], *, k: int | None = None
                 ) -> tuple[list[Passage], list[float], list[Passage]]:
        """Dense recall into a pool, then cross-encoder rerank. Returns (top-k, scores, rest)."""
        if not passages:
            return [], [], []
        k = min(len(passages), k if k is not None else self.config.top_k)
        embedder = self._load_embedder()
        vectors = embedder.encode([query, *[p.text for p in passages]], normalize_embeddings=True)
        dense_scores = np.asarray(vectors[1:]) @ vectors[0]
        pool_size = min(len(passages), max(k, self.config.top_k * 2))
        pool = [passages[int(i)] for i in np.argsort(dense_scores)[::-1][:pool_size]]
        rerank_scores = self._load_reranker().predict([(query, p.text) for p in pool])
        order = np.argsort(np.asarray(rerank_scores))[::-1]
        selected = [pool[int(i)] for i in order[:k]]
        scores = [float(rerank_scores[int(i)]) for i in order[:k]]
        rejected = [pool[int(i)] for i in order[k:]]
        rejected += [p for p in passages if p not in pool]
        return selected, scores, rejected

    def consolidate(self, question: str, pool: list[Passage], everything: tuple[Passage, ...]
                    ) -> tuple[list[Passage], list[float], list[Passage]]:
        """Select the evidence that actually reaches the generator.

        Without this step the union of per-subgoal retrievals is the whole candidate
        set and retrieval is a no-op. `answer_top_k` is deliberately smaller than the
        per-step `top_k`: HotpotQA questions need 2 gold paragraphs, so a top-3 cut of
        a 10-paragraph pool makes "an annotated supporting paragraph is absent from
        top-k" (docs/PLAN.md) an event that genuinely occurs.
        """
        candidates = list(dict.fromkeys(pool)) or list(everything)
        if not candidates:
            return [], [], []
        k = min(len(candidates), self.config.answer_top_k)
        scores = self._load_reranker().predict([(question, p.text) for p in candidates])
        order = np.argsort(np.asarray(scores))[::-1]
        selected = [candidates[int(i)] for i in order[:k]]
        selected_scores = [float(scores[int(i)]) for i in order[:k]]
        rejected = [candidates[int(i)] for i in order[k:]]
        rejected += [p for p in everything if p not in candidates]
        return selected, selected_scores, rejected

    # ── answering and verification ───────────────────────────────────────────
    @staticmethod
    def _answer_prompt(question: str, evidence, calls) -> tuple[str, str]:
        return ANSWER_SYSTEM, f"Question: {question}\n\nEvidence:\n{evidence_block(evidence)}\n\nTools:\n{tool_block(calls)}"

    @staticmethod
    def _verify_prompt(question: str, answer: str, evidence) -> tuple[str, str]:
        return VERIFY_SYSTEM, f"Question: {question}\nAnswer: {answer}\nEvidence:\n{evidence_block(evidence)}"

    def answer_many(self, items: Sequence[tuple[str, Any, Any]]) -> list[str]:
        return self.chat_many([self._answer_prompt(*item) for item in items], max_new_tokens=256)

    def answer(self, question: str, evidence, calls) -> str:
        return self.answer_many([(question, evidence, calls)])[0]

    def _parse_verification(self, response: str) -> dict[str, Any]:
        try:
            payload = self._json_object(response)
            verified = payload.get("verified")
            if isinstance(verified, str):
                verified = verified.strip().lower() in {"true", "1", "yes"}
            elif not isinstance(verified, bool):
                verified = bool(verified)
            return {"verified": verified, "reason": str(payload.get("reason", ""))}
        except Exception as exc:
            return {"verified": False, "reason": f"verification parsing error: {exc}"}

    def verify_many(self, items: Sequence[tuple[str, str, Any]]) -> list[dict[str, Any]]:
        responses = self.chat_many([self._verify_prompt(*item) for item in items], max_new_tokens=128)
        return [self._parse_verification(r) for r in responses]

    def verify(self, question: str, answer: str, evidence) -> dict[str, Any]:
        return self.verify_many([(question, answer, evidence)])[0]

    # ── whole-trace execution ────────────────────────────────────────────────
    def _act(self, example: HotpotExample, goals: list[Subgoal]) -> dict[str, Any]:
        """Everything up to (but excluding) answer generation."""
        calls: list[ToolCall] = []
        pool: list[Passage] = []
        per_step: list[list[Passage]] = []
        for goal in goals:
            if goal.tool == "calculator":
                calls.append(self.calculator.execute(goal.expression or goal.text or ""))
                per_step.append([])
                continue
            step_selected, _, _ = self.retrieve(goal.text, example.passages)
            calls.append(self.lookup.execute(goal.text, step_selected))
            pool.extend(step_selected)
            per_step.append(step_selected)
        selected, scores, rejected = self.consolidate(example.question, pool, example.passages)
        return {"calls": calls, "selected": selected, "scores": scores,
                "rejected": rejected, "per_step": per_step}

    def run_many(self, examples: Sequence[HotpotExample]) -> list[dict[str, Any]]:
        """Plan, act, answer and verify a batch of examples."""
        plans = self.plan_many([e.question for e in examples])
        staged = [self._act(example, goals) for example, goals in zip(examples, plans)]
        answers = self.answer_many([(e.question, s["selected"], s["calls"])
                                    for e, s in zip(examples, staged)])
        verifications = self.verify_many([(e.question, a, s["selected"])
                                          for e, a, s in zip(examples, answers, staged)])
        traces = []
        for example, goals, stage, answer, verification in zip(examples, plans, staged, answers, verifications):
            traces.append({
                "query_id": example.query_id, "question": example.question, "gold_answer": example.answer,
                "supporting_titles": sorted(example.supporting_titles),
                "subgoals": [asdict(g) for g in goals],
                "selected_context": [asdict(p) for p in stage["selected"]],
                "rejected_context": [asdict(p) for p in stage["rejected"]],
                "rerank_scores": stage["scores"],
                "tool_history": [asdict(c) for c in stage["calls"]],
                "answer": answer, "verification": verification,
                "step_contexts": [[asdict(p) for p in step] for step in stage["per_step"]],
                "model": self.config.model.model_dump(), "seed": self.config.seed,
            })
        return traces

    def run(self, example: HotpotExample) -> dict[str, Any]:
        return self.run_many([example])[0]
