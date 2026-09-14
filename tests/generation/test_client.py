"""Tests for the OpenAI-compatible generation client."""

import json
from pathlib import Path

import httpx
import pytest

from danish_personas.generation.client import OpenAIClient, RequestBudgetExceeded
from danish_personas.generation.models import GenerationConfig


def test_client_enforces_total_request_budget_across_retries() -> None:
    """Transport retries cannot exceed the invocation-wide HTTP budget."""
    calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code=504)

    client = OpenAIClient(
        config=_config().model_copy(update={"maximum_total_requests": 1}),
        transport=httpx.MockTransport(handler=handler),
    )
    with pytest.raises(RequestBudgetExceeded):
        client.complete(
            system_prompt="Svar på dansk.",
            user_payload={"input": "test"},
            schema_name="probe",
            json_schema={"type": "object"},
        )
    client.close()
    assert calls == 1


def _config() -> GenerationConfig:
    return GenerationConfig(
        version=1,
        llm_generation_enabled=True,
        base_url="http://test/v1",
        model="gpt-test",
        api_key_env=None,
        timeout_seconds=10.0,
        maximum_http_attempts=2,
        maximum_validation_attempts=2,
        maximum_total_requests=5,
        retry_backoff_seconds=0.0,
        maximum_smoke_rows=5,
        max_tokens=None,
        response_format="json_schema",
        attributes_prompt=Path("attributes.md"),
        personas_prompt=Path("personas.md"),
    )


def test_client_sends_supported_schema_request() -> None:
    """The client sends strict JSON schema without unsupported sampling options."""
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-test",
                "model": "gpt-test",
                "choices": [{"message": {"content": '{"ok":true}'}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                    "estimated_cost": 0.00001,
                },
            },
            headers={"x-inference-provider": "test-provider"},
        )

    client = OpenAIClient(
        config=_config(), transport=httpx.MockTransport(handler=handler)
    )
    response = client.complete(
        system_prompt="Svar på dansk.",
        user_payload={"input": "test"},
        schema_name="probe",
        json_schema={"type": "object"},
    )
    client.close()
    assert response.total_tokens == 14
    assert response.estimated_cost_usd == 0.00001
    assert response.inference_provider == "test-provider"
    assert captured["model"] == "gpt-test"
    assert "temperature" not in captured
    assert "seed" not in captured
    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "probe", "strict": True, "schema": {"type": "object"}},
    }
