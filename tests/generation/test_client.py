"""Tests for the OpenAI-compatible generation client."""

import json
from pathlib import Path

import httpx
import pytest

from danish_personas.generation.client import OpenAIClient, RequestBudgetExceeded
from danish_personas.generation.models import GeneratedAttributes, GenerationConfig


def test_client_enforces_total_request_budget_across_retries() -> None:
    """Transport retries cannot exceed the invocation-wide HTTP budget."""
    calls = 0
    persisted: list[int] = []

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status_code=504)

    client = OpenAIClient(
        config=_config().model_copy(update={"maximum_total_requests": 1}),
        transport=httpx.MockTransport(handler=handler),
        record_request=persisted.append,
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
    assert persisted == [1]


def _config() -> GenerationConfig:
    return GenerationConfig(
        base_url="http://test/v1",
        model="gpt-test",
        api_key_env=None,
        timeout_seconds=10.0,
        maximum_http_attempts=2,
        maximum_validation_attempts=2,
        maximum_total_requests=5,
        retry_backoff_seconds=0.0,
        maximum_rows_per_shard=5,
        max_tokens=None,
        enable_thinking=None,
        reasoning_effort=None,
        response_format="json_schema",
        attributes_prompt=Path("attributes.md"),
        personas_prompt=Path("personas.md"),
        origin_label_contract=Path("config/folk2-ieland-labels-da.yaml"),
    )


def test_client_schema_is_accepted_by_strict_proxy_contract() -> None:
    """A strict proxy accepts the generated schema instead of returning its 500."""
    proxy_errors: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        response_format = body["response_format"]
        schema = response_format["json_schema"]["schema"]
        if _has_strict_schema_violation(schema):
            message = (
                "required must include every key in properties for strict JSON schema"
            )
            proxy_errors.append(message)
            return httpx.Response(status_code=500, json={"error": {"message": message}})
        return httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-strict-schema",
                "model": "gpt-5.6-sol",
                "choices": [{"message": {"content": "{}"}}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                    "estimated_cost": 0.0,
                },
            },
        )

    client = OpenAIClient(
        config=_config(), transport=httpx.MockTransport(handler=handler)
    )
    response = client.complete(
        system_prompt="Svar på dansk.",
        user_payload={"input": "test"},
        schema_name="generated_attributes",
        json_schema=GeneratedAttributes.model_json_schema(),
    )
    client.close()

    assert response.model == "gpt-5.6-sol"
    assert proxy_errors == []


def _has_strict_schema_violation(schema: object) -> bool:
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            properties = schema.get("properties", {})
            required = schema.get("required")
            if not isinstance(properties, dict) or not isinstance(required, list):
                return True
            if (
                set(required) != set(properties)
                or schema.get("additionalProperties") is not False
            ):
                return True
        return any(_has_strict_schema_violation(value) for value in schema.values())
    if isinstance(schema, list):
        return any(_has_strict_schema_violation(value) for value in schema)
    return False


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
        config=_config().model_copy(update={"reasoning_effort": "none"}),
        transport=httpx.MockTransport(handler=handler),
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
    assert captured["reasoning_effort"] == "none"
    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "probe", "strict": True, "schema": {"type": "object"}},
    }


@pytest.mark.parametrize(
    ("headers", "minimum_delay"),
    [({}, 60.0), ({"retry-after": "Fri, 31 Dec 9999 23:59:59 GMT"}, 61.0)],
)
def test_client_uses_rate_limit_backoff(
    monkeypatch: pytest.MonkeyPatch, headers: dict[str, str], minimum_delay: float
) -> None:
    """Rate-limit retries honour HTTP dates or use a conservative fallback."""
    calls = 0
    sleeps: list[float] = []

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(status_code=429, headers=headers)
        return httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-test",
                "model": "gpt-test",
                "choices": [{"message": {"content": '{"ok":true}'}}],
                "usage": {},
            },
        )

    monkeypatch.setattr("danish_personas.generation.client.time.sleep", sleeps.append)
    client = OpenAIClient(
        config=_config(), transport=httpx.MockTransport(handler=handler)
    )
    client.complete(
        system_prompt="Svar på dansk.",
        user_payload={"input": "test"},
        schema_name="probe",
        json_schema={"type": "object"},
    )
    client.close()
    assert len(sleeps) == 1
    assert sleeps[0] >= minimum_delay
