"""Real local tools used by the agent; no synthetic outputs are allowed."""
from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Callable

from mirage.data.hotpot import Passage


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, str]
    output: str
    ok: bool


class Calculator:
    name = "calculator"
    _ops: dict[type, Callable] = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    }

    def execute(self, expression: str) -> ToolCall:
        def evaluate(node: ast.AST) -> float:
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            if isinstance(node, ast.BinOp) and type(node.op) in self._ops:
                return self._ops[type(node.op)](evaluate(node.left), evaluate(node.right))
            if isinstance(node, ast.UnaryOp) and type(node.op) in self._ops:
                return self._ops[type(node.op)](evaluate(node.operand))
            raise ValueError("expression contains an unsupported operation")

        try:
            value = evaluate(ast.parse(expression, mode="eval").body)
            return ToolCall(self.name, {"expression": expression}, str(value), True)
        except (SyntaxError, ValueError, ZeroDivisionError) as exc:
            return ToolCall(self.name, {"expression": expression}, str(exc), False)


class DocumentLookup:
    name = "document_lookup"

    def execute(self, query: str, passages: list[Passage]) -> ToolCall:
        if not passages:
            return ToolCall(self.name, {"query": query}, "no retrieved passages", False)
        output = "\n\n".join(f"[{p.title}] {p.text}" for p in passages)
        return ToolCall(self.name, {"query": query}, output, True)
