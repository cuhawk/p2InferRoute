"""Unit tests: task classifier determinism and category sanity."""

from __future__ import annotations

import pytest

from inferroute.config import TaskType
from inferroute.routing.task_classifier import classify_task


def msg(content: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": content}]


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("please summarize this article", TaskType.summarization),
        ("tl;dr the following", TaskType.summarization),
        ("def fix_bug(x):\n    return x", TaskType.coding),
        ("write a function that parses json in python", TaskType.coding),
        ("extract all names from this text as json", TaskType.extraction),
        ("explain why the sky is blue, step by step", TaskType.reasoning),
        ("hello there!", TaskType.chat),
        ("what is your name?", TaskType.chat),
    ],
)
def test_categories(prompt: str, expected: TaskType) -> None:
    assert classify_task(msg(prompt)).task == expected


def test_deterministic() -> None:
    prompt = msg("write code to implement a parser with error handling")
    first = classify_task(prompt)
    for _ in range(5):
        again = classify_task(prompt)
        assert again.task == first.task
        assert again.signals == first.signals


def test_no_signals_is_chat() -> None:
    result = classify_task(msg("greetings and salutations friend"))
    assert result.task == TaskType.chat
    assert result.confidence == 0.0


def test_code_fence_is_strong_signal() -> None:
    result = classify_task(msg("what does this do?\n```\ndef x(): pass\n```"))
    assert result.task == TaskType.coding


def test_natural_language_for_is_not_code() -> None:
    result = classify_task(msg("summarize this document for me please"))
    assert result.task == TaskType.summarization


def test_multiturn_uses_last_user_message() -> None:
    messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "here is a long document about databases" * 200},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "summarize it"},
    ]
    assert classify_task(messages).task == TaskType.summarization


def test_explanation_contains_task() -> None:
    result = classify_task(msg("extract the fields"))
    assert "task=extraction" in result.explanation
