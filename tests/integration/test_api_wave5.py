"""Exercise Wave 5 through the normal OpenAI-compatible endpoint."""

from __future__ import annotations

import json

from conftest import make_config

URL = "/v1/chat/completions"


async def test_schema_cascade_promotes_only_invalid_results(client_factory) -> None:
    cfg = make_config()
    for model in cfg.models.values():
        model.supports_structured_output = True
    cfg.providers["mock-a"].options["structured_output_text"] = '{"answer":"42"}'
    cfg.providers["mock-b"].options["structured_output_text"] = '{"answer":42}'
    app, client = client_factory(cfg)
    body = {
        "messages": [{"role": "user", "content": "Return the integer answer."}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "integer"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        },
    }
    fallback = await client.post(URL, json=body)
    assert fallback.status_code == 200
    info = fallback.json()["inferroute"]
    assert info["gateway_model"] == "big"
    assert info["fallbacks"] == [
        {
            "from_provider": "mock-a",
            "to_provider": "mock-b",
            "reason": "completion failed JSON schema validation",
        }
    ]
    assert app.state.ctx.monitor.breaker_for("mock-a").allow_request()

    explicit = await client.post(URL, json={**body, "model": "small"})
    assert explicit.status_code == 502
    assert explicit.json()["error"]["code"] == "schema_validation_failed"

    app.state.ctx.providers["mock-a"].opts.structured_output_text = '{"answer":42}'
    first_tier = await client.post(URL, json=body)
    assert first_tier.status_code == 200
    assert first_tier.json()["inferroute"]["gateway_model"] == "small"
    assert first_tier.json()["inferroute"]["fallbacks"] == []

    body["response_format"]["json_schema"]["schema"]["type"] = "invalid"
    invalid = await client.post(URL, json=body)
    assert invalid.status_code == 400


async def test_sse_affinity_and_telemetry_for_conversation_extension(client_factory) -> None:
    cfg = make_config()
    for model in cfg.models.values():
        model.replica_group = "replicas"
        model.served_model_name = "shared"
        model.quality_tier = 2
    for provider in cfg.providers.values():
        provider.options["prefix_cache_entries"] = 32
    app, client = client_factory(cfg)
    body = {
        "messages": [
            {"role": "system", "content": "Shared instructions"},
            {"role": "user", "content": "First turn"},
        ],
        "route": {"strategy": "prefix_affinity"},
        "stream": True,
        "max_tokens": 3,
        "stream_options": {"include_usage": True},
    }
    first = await client.post(URL, json=body)
    assert first.status_code == 200
    body["messages"].append({"role": "assistant", "content": "First reply"})
    body["messages"].append({"role": "user", "content": "Next turn"})
    following = await client.post(URL, json=body)
    assert following.status_code == 200
    worker = first.headers["x-inferroute-provider"]
    assert following.headers["x-inferroute-provider"] == worker
    events = [
        json.loads(line[6:]) for line in following.text.splitlines() if line.startswith("data: {")
    ]
    usage = next(event["usage"] for event in events if "usage" in event)
    assert usage["prompt_tokens_details"]["cached_tokens"] == 4
    telemetry = events[-1]["inferroute"]["stream_telemetry"]
    assert telemetry["ttft_ms"] > 0
    assert len(telemetry["itl_ms"]) == 2
    assert telemetry["tpot_ms"] == sum(telemetry["itl_ms"]) / 2
    assert app.state.ctx.dispatcher.affinity.active[worker] == 0


async def test_stream_failure_releases_admission_and_never_counts_as_goodput(
    client_factory,
) -> None:
    cfg = make_config()
    cfg.gateway.max_concurrent_inference = 1
    cfg.gateway.max_queue_depth = 0
    cfg.providers["mock-a"].options.update(fail_every=1, fail_phase="first_token")
    app, client = client_factory(cfg)
    body = {
        "messages": [{"role": "user", "content": "Test stream."}],
        "model": "small",
        "stream": True,
        "max_tokens": 3,
    }
    rejected = await client.post(URL, json=body)
    assert rejected.status_code == 502
    assert app.state.ctx.queue_guard.active == 0
    assert app.state.ctx.dispatcher.affinity.active["mock-a"] == 0

    failed_provider = app.state.ctx.providers["mock-a"]
    failed_provider.opts.fail_phase = "mid_stream"
    interrupted = await client.post(URL, json=body)
    assert interrupted.status_code == 200
    assert "midstream_failure" in interrupted.text
    assert "stream_telemetry" not in interrupted.text
    assert app.state.ctx.queue_guard.active == 0
    assert app.state.ctx.dispatcher.affinity.active["mock-a"] == 0
    metrics = app.state.ctx.metrics.registry
    assert (
        metrics.get_sample_value(
            "inferroute_stream_slo_met_total", {"provider": "mock-a", "model": "small"}
        )
        is None
    )
