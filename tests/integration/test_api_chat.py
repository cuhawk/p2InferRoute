"""Integration: OpenAI-compatible chat completions (streaming + non-streaming)."""

from __future__ import annotations

import json
import time

import httpx

from tests.conftest import CHAT_URL, chat_body, make_config


async def test_non_stream_response_shape(client: httpx.AsyncClient) -> None:
    resp = await client.post(CHAT_URL, json=chat_body())
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["model"]
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["finish_reason"] == "stop"
    usage = data["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    info = data["inferroute"]
    assert info["provider"] in {"mock-a", "mock-b"}
    assert info["gateway_model"] in {"small", "big"}
    assert info["strategy"] == "balanced"
    assert info["route_trace"]
    assert info["timings_ms"]["total"] >= 0
    assert info["simulated_inference"] is True


async def test_response_headers(client: httpx.AsyncClient) -> None:
    resp = await client.post(CHAT_URL, json=chat_body())
    assert resp.headers["x-inferroute-model"] in {"small", "big"}
    assert resp.headers["x-inferroute-provider"] in {"mock-a", "mock-b"}
    assert resp.headers["x-inferroute-cache"] == "DISABLED"
    assert resp.headers["x-inferroute-request-id"]


async def test_streaming_sse_shape(client: httpx.AsyncClient) -> None:
    body = chat_body(stream=True, max_tokens=6)
    chunks: list[dict] = []
    done_seen = False
    async with client.stream("POST", CHAT_URL, json=body) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done_seen = True
                break
            chunks.append(json.loads(payload))
    assert done_seen
    assert len(chunks) >= 9  # role + 6 tokens + finish + inferroute metadata
    first = chunks[0]
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"]["role"] == "assistant"
    content_chunks = [c for c in chunks if c["choices"] and c["choices"][0]["delta"].get("content")]
    assert len(content_chunks) == 6
    finish_chunks = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
    assert finish_chunks and finish_chunks[-1]["choices"][0]["finish_reason"] == "stop"
    meta_chunks = [c for c in chunks if c.get("inferroute")]
    assert meta_chunks, "expected an inferroute metadata chunk before [DONE]"
    assert meta_chunks[-1]["inferroute"]["provider"] in {"mock-a", "mock-b"}
    assert meta_chunks[-1]["inferroute"]["simulated_inference"] is True
    assert all(c["id"] == first["id"] for c in chunks)


async def test_streaming_inter_token_pacing(client_factory) -> None:
    """Regression: SSE streaming must preserve inter-token pacing.

    BaseHTTPMiddleware absorbs the stream generator's suspension points on
    this stack (all sleeps return instantly, chunks arrive bunched); the
    gateway therefore uses a pure ASGI middleware. If this test ever fails,
    a middleware/wrapper regression is collapsing streaming backpressure.
    """
    cfg = make_config()
    cfg.providers["mock-a"].options.update(ttft_ms=10, itl_ms=40)
    _, client = client_factory(cfg)
    body = chat_body(model="small", stream=True, max_tokens=6)
    started = time.perf_counter()
    async with client.stream("POST", CHAT_URL, json=body) as resp:
        assert resp.status_code == 200
        async for line in resp.aiter_lines():
            if line.startswith("data:") and "[DONE]" in line:
                break
    elapsed = time.perf_counter() - started
    # 5 inter-token gaps x 40ms = 200ms expected; conservative floor guards
    # against the collapsed-pacing regression while tolerating scheduler jitter.
    assert elapsed >= 0.12, f"stream finished in {elapsed * 1000:.1f}ms — token pacing lost"


async def test_streaming_usage_chunk_with_stream_options(client: httpx.AsyncClient) -> None:
    body = chat_body(stream=True, max_tokens=5, stream_options={"include_usage": True})
    usage_chunks: list[dict] = []
    async with client.stream("POST", CHAT_URL, json=body) as resp:
        async for line in resp.aiter_lines():
            if line.startswith("data:") and "[DONE]" not in line:
                chunk = json.loads(line[5:].strip())
                if chunk.get("usage"):
                    usage_chunks.append(chunk)
    assert len(usage_chunks) == 1
    usage = usage_chunks[0]["usage"]
    assert usage["completion_tokens"] == 5
    assert usage["estimated"] is False


async def test_explicit_model_selection(client: httpx.AsyncClient) -> None:
    resp = await client.post(CHAT_URL, json=chat_body(model="big"))
    assert resp.status_code == 200
    assert resp.json()["inferroute"]["gateway_model"] == "big"


async def test_unknown_model_404(client: httpx.AsyncClient) -> None:
    resp = await client.post(CHAT_URL, json=chat_body(model="no-such-model"))
    assert resp.status_code == 404
    err = resp.json()["error"]
    assert err["code"] == "model_not_found"
    assert "available models" in err["message"]


async def test_invalid_strategy_400(client: httpx.AsyncClient) -> None:
    resp = await client.post(CHAT_URL, json=chat_body(route={"strategy": "yolo"}))
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_strategy"


async def test_invalid_task_400(client: httpx.AsyncClient) -> None:
    resp = await client.post(CHAT_URL, json=chat_body(task="poetry"))
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_task"


async def test_empty_messages_400(client: httpx.AsyncClient) -> None:
    resp = await client.post(CHAT_URL, json={"model": "auto", "messages": []})
    assert resp.status_code == 400


async def test_bad_role_400(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        CHAT_URL, json=chat_body(messages=[{"role": "wizard", "content": "hi"}])
    )
    assert resp.status_code == 400


async def test_invalid_json_400(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        CHAT_URL, content=b"{nope", headers={"content-type": "application/json"}
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_json"


async def test_max_tokens_cap_400(client: httpx.AsyncClient) -> None:
    resp = await client.post(CHAT_URL, json=chat_body(max_tokens=1_000_000))
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "max_tokens_too_large"


async def test_task_override_header_and_body(client: httpx.AsyncClient) -> None:
    resp = await client.post(CHAT_URL, json=chat_body(task="reasoning"))
    data = resp.json()
    assert data["inferroute"]["task"] == "reasoning"
    assert data["inferroute"]["gateway_model"] == "big"  # only tier-4 model


async def test_tool_calling_via_capable_model(client: httpx.AsyncClient) -> None:
    cfg_tools = chat_body(
        model="big",
        tools=[{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
        tool_choice="required",
    )
    # big routes to mock-b; mock-b does not emit tool calls unless configured,
    # so this asserts passthrough + no error + capability filtering works.
    resp = await client.post(CHAT_URL, json=cfg_tools)
    assert resp.status_code == 200
    assert resp.json()["inferroute"]["gateway_model"] == "big"


async def test_tools_filtered_to_capable_models(client: httpx.AsyncClient) -> None:
    body = chat_body(
        tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}],
        tool_choice="auto",
    )
    resp = await client.post(CHAT_URL, json=body)
    assert resp.status_code == 200
    # small does not support tools -> must be routed to big
    assert resp.json()["inferroute"]["gateway_model"] == "big"


async def test_context_overflow_routed_to_bigger_model(client: httpx.AsyncClient) -> None:
    long_prompt = "word " * 3000  # ~3750 tokens; + 512 max_tokens exceeds small's 4096 window
    resp = await client.post(
        CHAT_URL,
        json=chat_body(max_tokens=512, messages=[{"role": "user", "content": long_prompt}]),
    )
    assert resp.status_code == 200
    info = resp.json()["inferroute"]
    assert info["gateway_model"] == "big"
    assert any(
        "context too short" in t["reason"] for t in info["route_trace"] if t["status"] == "rejected"
    )


async def test_privacy_constraint_via_route(client: httpx.AsyncClient) -> None:
    resp = await client.post(CHAT_URL, json=chat_body(route={"privacy": "local"}))
    assert resp.status_code == 200  # all mock models are local
