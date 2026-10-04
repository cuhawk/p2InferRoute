"""Deterministic heuristic task classifier.

Classifies a chat request into ``chat | coding | reasoning | extraction |
summarization``. Rules are ordered and scored deterministically: identical input always
produces the identical task (no ML model involved — this is a router heuristic, and it
must be explainable in the routing trace).

A client can always override the classification explicitly via ``task`` in the request
body or the ``x-inferroute-task`` header; the classifier is the default.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from inferroute.config import TaskType

_CODE_SIGNALS = (
    "```",
    "def ",
    "class ",
    "function ",
    "import ",
    "select ",
    "insert into",
    "regex",
    "stack trace",
    "traceback",
    "compile",
    "runtime error",
    "debug",
    "refactor",
    "implement",
    "unit test",
    "typescript",
    "python",
    "javascript",
    "rust",
    "golang",
    "sql",
)
_REASONING_SIGNALS = (
    "step by step",
    "prove",
    "derive",
    "explain why",
    "walk through",
    "chain of thought",
    "analyze",
    "trade-off",
    "why does",
    "how does",
    "estimate",
    "calculate",
    "solve",
)
_EXTRACTION_SIGNALS = (
    "extract",
    "parse",
    "as json",
    "in json",
    "json object",
    "fields",
    "schema",
    "list all",
    "identify all",
    "output only",
)
_SUMMARIZATION_SIGNALS = (
    "summarize",
    "summary",
    "tl;dr",
    "tldr",
    "condense",
    "shorten",
    "in one paragraph",
    "bullet points",
    "key points",
)

# Code *syntax* markers only: braces, lambda/comparison operators, statement
# terminators. Bare keywords like `for`/`if` are too common in natural language.
_CODE_RE = re.compile(r"[{}]|\[\]|\(\)=>|=>|==|!=|::|;\s*$|^import |^from .+ import", re.MULTILINE)
_NUMBER_RE = re.compile(r"\d+\s*[\+\-\*/^]\s*\d+")


@dataclass(slots=True)
class TaskClassification:
    task: TaskType
    signals: dict[str, list[str]] = field(default_factory=dict)
    confidence: float = 0.0

    @property
    def explanation(self) -> str:
        matched = {k: v[:3] for k, v in self.signals.items() if v}
        return f"task={self.task.value} signals={matched or 'none'}"


def classify_task(messages: list[dict[str, str]]) -> TaskClassification:
    """Score heuristic signals over the conversation; highest wins, ties broken by a
    fixed priority: coding > reasoning > extraction > summarization > chat."""
    text = "\n".join(str(m.get("content") or "") for m in messages).lower()
    # Task type is influenced mostly by the final user message; the transcript gives context.
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = str(m.get("content") or "").lower()
            break
    focus = last_user or text
    input_length = max(len(text), 1)

    scores: dict[TaskType, list[str]] = {
        TaskType.coding: [],
        TaskType.reasoning: [],
        TaskType.extraction: [],
        TaskType.summarization: [],
        TaskType.chat: [],
    }

    for sig in _CODE_SIGNALS:
        if sig in focus or sig in text:
            scores[TaskType.coding].append(sig)
    if _CODE_RE.search(focus):
        scores[TaskType.coding].append("code-syntax")

    for sig in _REASONING_SIGNALS:
        if sig in focus:
            scores[TaskType.reasoning].append(sig)
    if _NUMBER_RE.search(focus):
        scores[TaskType.reasoning].append("arithmetic")

    for sig in _EXTRACTION_SIGNALS:
        if sig in focus:
            scores[TaskType.extraction].append(sig)

    for sig in _SUMMARIZATION_SIGNALS:
        if sig in focus:
            scores[TaskType.summarization].append(sig)
    # Long input with a short-output ask smells like summarization/compression.
    if input_length > 2000 and len(focus.split()) < 60:
        scores[TaskType.summarization].append("long-input-short-ask")

    # Priority-ordered pick: coding > reasoning > extraction > summarization > chat.
    priority = (
        TaskType.coding,
        TaskType.reasoning,
        TaskType.extraction,
        TaskType.summarization,
        TaskType.chat,
    )
    total = sum(len(v) for v in scores.values())
    if total == 0:
        best = TaskType.chat
    else:
        best = max(priority, key=lambda t: (len(scores[t]), -priority.index(t)))
    confidence = (len(scores[best]) / total) if total else 0.0
    return TaskClassification(
        task=best,
        signals={t.value: v for t, v in scores.items() if v},
        confidence=round(confidence, 3),
    )
