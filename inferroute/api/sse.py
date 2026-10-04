"""OpenAI-compatible SSE encoding for chat completion chunks."""

from __future__ import annotations

import json
import time
from typing import Any

DONE_SENTINEL = "data: [DONE]\n\n"


def chunk_payload(
    completion_id: str,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage
        # OpenAI sends the usage chunk with an empty choices list
        payload["choices"] = []
    return payload


def format_sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# backwards-compatible alias
sse = format_sse


def error_payload(message: str, err_type: str, code: str | int | None = None) -> dict[str, Any]:
    return {"error": {"message": message, "type": err_type, "code": code}}
