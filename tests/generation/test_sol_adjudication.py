"""Offline tests for the private Sol adjudication engine."""

from __future__ import annotations

import hashlib
import json
import stat
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

import danish_personas.generation.proxy_budget as proxy_budget
from danish_personas.generation.models import GenerationConfig
from danish_personas.generation.proxy_budget import JSONValue, ProxyBudget
from danish_personas.generation.sol_adjudication import (
    SOL_MAX_OUTPUT_TOKENS,
    SolAdjudicationError,
    SolAdjudicationResponse,
    run_sol_adjudication,
    validate_sol_adjudication,
)

_PROMPT = "Vurder den danske persona konservativt og returnér kun JSON."
_PERSONA = (
    "Hun er 42 år og bor i Aarhus. "
    "Hun arbejder som lærer og beskrives i en rolig, hverdagsnær tekst. "
    "Personen har en stabil hverdag med kolleger, familie og fritidsinteresser."
)


@pytest.fixture(autouse=True)
def _private_budget_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep mock Sol reservations out of the user's real ledger."""
    monkeypatch.setattr(
        proxy_budget,
        "USER_SOL_ADJUDICATION_BUDGET_PATH",
        tmp_path / "proxy-sol-adjudication.jsonl",
    )


def test_consistent_evidence_checkpoints_privately(tmp_path: Path) -> None:
    """Accept exact grounded evidence and write a private checkpoint."""
    checkpoint = _checkpoint(tmp_path)

    result = run_sol_adjudication(
        original_persona=_PERSONA,
        candidate_row={"age": 42, "municipality": "Aarhus"},
        prompt=_PROMPT,
        config=_config(),
        budget=_budget(tmp_path),
        checkpoint_path=checkpoint,
        transport=_transport(
            {
                "disposition": "consistent",
                "reason": "Alder er nævnt i teksten.",
                "evidence": [
                    {"field": "age", "kind": "fact_present", "quote": "Hun er 42 år"}
                ],
                "patches": [],
            },
            [],
        ),
    )

    assert result.disposition == "consistent"
    assert result.proposed_text == _PERSONA
    assert stat.S_IMODE(checkpoint.stat().st_mode) == 0o600


def _budget(tmp_path: Path) -> ProxyBudget:
    registry = tmp_path / "models.json"
    registry.write_text(
        json.dumps(
            {
                "openai-codex": {
                    "models": [
                        {
                            "id": "gpt-6-sol",
                            "maxTokens": 128_000,
                            "cost": {"input": "2", "output": "10"},
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    return ProxyBudget(
        ledger_path=tmp_path / "ignored.jsonl",
        registry_path=registry,
        campaign="synthetic-sol-test",
        source_hash="a" * 64,
        prompt_hash=hashlib.sha256(_PROMPT.encode()).hexdigest(),
        schema_hash=hashlib.sha256(
            json.dumps(
                SolAdjudicationResponse.provider_json_schema(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        model="gpt-6-sol",
        input_usd_per_million="2",
        output_usd_per_million="10",
        cap_usd=Decimal("1"),
        uncapped=True,
        uncapped_purpose="sol_adjudication",
    )


def _checkpoint(tmp_path: Path, *, name: str = "sol.json") -> Path:
    parent = tmp_path / "private-checkpoints"
    parent.mkdir(mode=0o700, exist_ok=True)
    return parent / name


def _config(**overrides: object) -> GenerationConfig:
    values: dict[str, object] = {
        "base_url": "http://127.0.0.1:18080/v1",
        "model": "gpt-6-sol",
        "api_key_env": None,
        "timeout_seconds": 10.0,
        "maximum_http_attempts": 1,
        "maximum_total_requests": None,
        "retry_backoff_seconds": 0.0,
        "maximum_rows_per_shard": 1,
        "max_tokens": SOL_MAX_OUTPUT_TOKENS,
        "enable_thinking": None,
        "reasoning_effort": "none",
        "prompt": Path("config/persona-sol-adjudication-da.md"),
        "origin_label_contract": Path("config/folk2-ieland-labels-da.yaml"),
    }
    values.update(overrides)
    return GenerationConfig.model_validate(values)


def _transport(
    response_content: dict[str, Any],
    seen: list[httpx.Request],
    events: list[str] | None = None,
) -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if events is not None:
            events.append("network")
        body = json.loads(request.content)
        assert body["model"] == "gpt-6-sol"
        assert body["max_tokens"] == SOL_MAX_OUTPUT_TOKENS
        assert body["reasoning_effort"] == "none"
        return httpx.Response(
            200,
            json={
                "id": "response-1",
                "model": "gpt-6-sol",
                "choices": [{"message": {"content": json.dumps(response_content)}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )

    return httpx.MockTransport(respond)


def test_omits_private_fields_and_rejects_forbidden_text(tmp_path: Path) -> None:
    """Send only allowlisted facts and fail closed on identity terms."""
    requests: list[httpx.Request] = []

    result = run_sol_adjudication(
        original_persona=_PERSONA,
        candidate_row={
            "persona_id": "raw-persona-id",
            "origin_country": "private English origin",
            "origin_country_code": "private-code",
            "sidecar": {"raw": "private-sidecar"},
            "sexual_orientation": "private-orientation",
            "same_sex_partner_target": True,
            "partner_gender": "female",
            "age": 42,
            "municipality": "Aarhus",
            "job_title": "lærer",
        },
        prompt=_PROMPT,
        config=_config(),
        budget=_budget(tmp_path),
        checkpoint_path=_checkpoint(tmp_path),
        transport=_transport(
            {
                "disposition": "consistent",
                "reason": "Alder og by er direkte nævnt.",
                "evidence": [
                    {"field": "age", "kind": "fact_present", "quote": "Hun er 42 år"}
                ],
                "patches": [],
            },
            requests,
        ),
    )

    body = requests[0].content.decode()
    assert result.disposition == "consistent"
    assert "raw-persona-id" not in body
    assert "private English origin" not in body
    assert "private-code" not in body
    assert "private-sidecar" not in body
    assert "private-orientation" not in body
    assert "same_sex_partner_target" not in body
    assert "partner_gender" not in body

    with pytest.raises(SolAdjudicationError):
        run_sol_adjudication(
            original_persona="Hun beskrives som homoseksuel. " * 8,
            candidate_row={"age": 42},
            prompt=_PROMPT,
            config=_config(),
            budget=_budget(tmp_path),
            checkpoint_path=_checkpoint(tmp_path, name="blocked.json"),
            transport=httpx.MockTransport(lambda _: httpx.Response(500)),
        )


def test_rejects_nonunique_and_over_budget_patches() -> None:
    """Reject non-mechanical and too-large provider patch proposals."""
    with pytest.raises(SolAdjudicationError):
        validate_sol_adjudication(
            original_text="Hun bor i Aarhus. Hun arbejder som lærer.",
            candidate_facts={"municipality": "Aarhus"},
            response={
                "disposition": "patched",
                "reason": "Ikke entydig.",
                "evidence": [],
                "patches": [{"old_excerpt": "Hun", "new_excerpt": "Personen"}],
            },
        )

    with pytest.raises(SolAdjudicationError):
        validate_sol_adjudication(
            original_text=(
                "Hun bor i Aarhus og arbejder som lærer. Resten er kort neutral tekst."
            ),
            candidate_facts={"municipality": "Aarhus"},
            response={
                "disposition": "patched",
                "reason": "For stor ændring.",
                "evidence": [],
                "patches": [
                    {
                        "old_excerpt": "Hun bor i Aarhus og arbejder som lærer.",
                        "new_excerpt": "Hun bor i Aarhus.",
                    }
                ],
            },
        )


def test_rejects_stale_checkpoint_binding(tmp_path: Path) -> None:
    """Resume only when the bound candidate row is unchanged."""
    checkpoint = _checkpoint(tmp_path)
    run_sol_adjudication(
        original_persona=_PERSONA,
        candidate_row={"age": 42, "municipality": "Aarhus"},
        prompt=_PROMPT,
        config=_config(),
        budget=_budget(tmp_path),
        checkpoint_path=checkpoint,
        transport=_transport(
            {
                "disposition": "consistent",
                "reason": "Alder er nævnt.",
                "evidence": [
                    {"field": "age", "kind": "fact_present", "quote": "Hun er 42 år"}
                ],
                "patches": [],
            },
            [],
        ),
    )

    with pytest.raises(SolAdjudicationError):
        run_sol_adjudication(
            original_persona=_PERSONA,
            candidate_row={"age": 43, "municipality": "Aarhus"},
            prompt=_PROMPT,
            config=_config(),
            budget=_budget(tmp_path),
            checkpoint_path=checkpoint,
            transport=httpx.MockTransport(lambda _: httpx.Response(500)),
        )


def test_schema_invalid_records_usage_without_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Record successful HTTP usage before rejecting invalid JSON content."""
    events: list[str] = []
    budget = _budget(tmp_path)
    original_reserve = budget.reserve_attempt
    original_usage = budget.record_usage

    def reserve(
        request_id: str,
        request: dict[str, JSONValue],
        *,
        max_output_tokens: int | None = None,
    ) -> Decimal:
        assert max_output_tokens == SOL_MAX_OUTPUT_TOKENS
        events.append("reserved")
        return original_reserve(
            request_id, request, max_output_tokens=max_output_tokens
        )

    def usage(
        request_id: str, *, input_tokens: int, output_tokens: int, response_sha256: str
    ) -> None:
        events.append("usage")
        original_usage(
            request_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            response_sha256=response_sha256,
        )

    monkeypatch.setattr(budget, "reserve_attempt", reserve)
    monkeypatch.setattr(budget, "record_usage", usage)
    checkpoint = _checkpoint(tmp_path)

    with pytest.raises(SolAdjudicationError) as error:
        run_sol_adjudication(
            original_persona=_PERSONA,
            candidate_row={"persona_id": "raw-private-id", "age": 42},
            prompt=_PROMPT,
            config=_config(),
            budget=budget,
            checkpoint_path=checkpoint,
            transport=_transport({"disposition": "consistent"}, [], events),
        )

    assert events == ["reserved", "network", "usage"]
    assert not checkpoint.exists()
    assert "raw-private-id" not in str(error.value)
