"""Offline tests for the private local proxy patch runner."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

from danish_personas.generation.models import GenerationConfig
from danish_personas.generation.prose_patch import ProsePatchResponse
from danish_personas.generation.proxy_budget import ProxyBudget
from danish_personas.generation.proxy_patch_runner import (
    ProxyPatchError,
    run_proxy_patch,
)


def _config(**overrides: object) -> GenerationConfig:
    values: dict[str, object] = {
        "base_url": "http://127.0.0.1:18080/v1",
        "model": "gpt-6-luna",
        "api_key_env": None,
        "timeout_seconds": 10.0,
        "maximum_http_attempts": 1,
        "maximum_total_requests": None,
        "retry_backoff_seconds": 0.0,
        "maximum_rows_per_shard": 1,
        "max_tokens": None,
        "enable_thinking": None,
        "reasoning_effort": "none",
        "prompt": Path("prompt.md"),
        "origin_label_contract": Path("config/folk2-ieland-labels-da.yaml"),
    }
    values.update(overrides)
    return GenerationConfig(**values)  # type: ignore[arg-type]


def _budget(tmp_path: Path) -> ProxyBudget:
    registry = tmp_path / "models.json"
    registry.write_text(
        json.dumps(
            {
                "openai-codex": {
                    "models": [
                        {
                            "id": "gpt-6-luna",
                            "maxTokens": 128_000,
                            "cost": {"input": "0.1", "output": "0.5"},
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    return ProxyBudget(
        ledger_path=tmp_path / "budget.jsonl",
        registry_path=registry,
        campaign="synthetic-test",
        source_hash="a" * 64,            prompt_hash=hashlib.sha256(
                "Ret kun den nødvendige lokale formulering.".encode()
            ).hexdigest(),
            schema_hash=hashlib.sha256(
                json.dumps(
                    ProsePatchResponse.provider_json_schema(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
cap_usd=Decimal("1"),
    )


def _row() -> dict[str, Any]:
    return {
        "record_id": "private-id",
        "source_sex": "private-sex",
        "municipality": "private municipality",
        "origin_country_da": "private origin",
        "persona_text": "Før ændring. " + "Dette er en syntetisk person. " * 12,
    }


def _transport(
    response_content: str,
    seen: list[httpx.Request],
    events: list[str] | None = None,
) -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if events is not None:
            events.append("network")
        assert "max_tokens" not in json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "response-1",
                "model": "gpt-6-luna",
                "choices": [{"message": {"content": response_content}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )

    return httpx.MockTransport(respond)


def _run(tmp_path: Path, transport: httpx.BaseTransport, **kwargs: object):
    return run_proxy_patch(
        row=kwargs.pop("row", _row()),
        changed_facts=kwargs.pop(
            "changed_facts", {"marital_status": {"old": "single", "new": "married"}}
        ),
        gender="kvinde",
        partner_gender="mand",
        prompt="Ret kun den nødvendige lokale formulering.",
        config=kwargs.pop("config", _config()),
        budget=_budget(tmp_path),
        checkpoint_path=tmp_path / "provisional.json",
        transport=transport,
        **kwargs,
    )


def test_payload_privacy_callback_before_network_and_restart_is_idempotent(
    tmp_path: Path,
) -> None:
    old = _row()["persona_text"]
    raw = json.dumps({"patches": [{"old_excerpt": "Før ændring", "new_excerpt": "Efter ændring"}]})
    requests: list[httpx.Request] = []
    events: list[str] = []
    budget = _budget(tmp_path)
    original_reserve = budget.reserve_attempt

    def reserve(request_id: str, request: dict[str, Any]) -> Decimal:
        events.append("reserved")
        return original_reserve(request_id, request)

    budget.reserve_attempt = reserve  # type: ignore[method-assign]
    result = run_proxy_patch(
        row=_row(),
        changed_facts={"marital_status": {"old": "single", "new": "married"}},
        gender="kvinde",
        partner_gender="mand",
        prompt="Ret kun den nødvendige lokale formulering.",
        config=_config(),
        budget=budget,
        checkpoint_path=tmp_path / "provisional.json",
        transport=_transport(raw, requests, events),
    )
    body = json.loads(requests[0].content)
    user_payload = json.loads(body["messages"][1]["content"])
    assert events == ["reserved", "network"]
    assert user_payload == {
        "persona_text": old,
        "changed_facts": {"marital_status": {"old": "single", "new": "married"}},
        "gender": "kvinde",
        "partner_gender": "mand",
    }
    assert all(secret not in requests[0].content.decode() for secret in ["private-id", "private-sex", "municipality", "origin_country_da"])
    assert result.persona_text == old.replace("Før ændring", "Efter ændring")
    assert result.persona_text[len("Efter ændring") :] == old[len("Før ændring") :]
    assert (tmp_path / "provisional.json").stat().st_mode & 0o777 == 0o600

    def must_not_send(_: httpx.Request) -> httpx.Response:
        raise AssertionError("a checkpoint restart must not make another request")

    restarted = run_proxy_patch(
        row=_row(),
        changed_facts={"marital_status": {"old": "single", "new": "married"}},
        gender="kvinde",
        partner_gender="mand",
        prompt="Ret kun den nødvendige lokale formulering.",
        config=_config(),
        budget=budget,
        checkpoint_path=tmp_path / "provisional.json",
        transport=httpx.MockTransport(must_not_send),
    )
    assert restarted == result


def test_rejects_bad_config_sensitive_original_and_non_allowlisted_fact(
    tmp_path: Path,
) -> None:
    with pytest.raises(ProxyPatchError):
        _run(tmp_path, httpx.MockTransport(lambda _: httpx.Response(500)), config=_config(max_tokens=100))
    row = _row()
    row["persona_text"] += " Seksual orientation."
    with pytest.raises(ProxyPatchError):
        _run(tmp_path, httpx.MockTransport(lambda _: httpx.Response(500)), row=row)
    with pytest.raises(ProxyPatchError):
        _run(
            tmp_path,
            httpx.MockTransport(lambda _: httpx.Response(500)),
            changed_facts={"municipality": {"old": "a", "new": "b"}},
        )


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps({"patches": [{"old_excerpt": "not present", "new_excerpt": "x"}]}),
        json.dumps({"patches": [{"old_excerpt": "Før ændring", "new_excerpt": "x", "extra": 1}]}),
    ],
)
def test_rejects_malformed_and_unmatched_patch(tmp_path: Path, content: str) -> None:
    requests: list[httpx.Request] = []
    with pytest.raises((ProxyPatchError, ValueError)):
        _run(tmp_path, _transport(content, requests))
    assert len(requests) == 1


def test_rejects_oversized_actual_http_body_before_network(tmp_path: Path) -> None:
    from danish_personas.generation.proxy_budget import ProxyBudgetError

    budget = _budget(tmp_path)
    # Force a reservation bound too small for the constructed HTTP body.
    budget.overhead = -1
    seen: list[httpx.Request] = []
    raw = json.dumps({"patches": [{"old_excerpt": "Før ændring", "new_excerpt": "Efter ændring"}]})
    with pytest.raises((ProxyPatchError, ProxyBudgetError)):
        run_proxy_patch(
            row=_row(),
            changed_facts={"marital_status": {"old": "single", "new": "married"}},
            gender=None,
            partner_gender=None,
            prompt="Ret kun den nødvendige lokale formulering.",
            config=_config(),
            budget=budget,
            checkpoint_path=tmp_path / "provisional.json",
            transport=_transport(raw, seen),
        )
    assert not seen
