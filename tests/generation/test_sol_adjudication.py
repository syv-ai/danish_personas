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
    preflight_sol_adjudication_payload,
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
    """Keep Sol adjudication tests on an isolated user-level ledger."""
    monkeypatch.setattr(
        proxy_budget,
        "USER_SOL_ADJUDICATION_BUDGET_PATH",
        tmp_path / "sol-adjudication.jsonl",
    )


_BANNED_IDENTITY_VARIANTS = (
    "seksuel orientering",
    "sexual orientation",
    "nonbinær",
    "non-binaer",
    "nonbinary",
    "panseksuel",
    "pansexual",
    "aseksuel",
    "asexual",
    "kønsidentitet",
    "koensidentitet",
    "gender identity",
    "transperson",
    "trans man",
    "transkvinde",
    "trans woman",
    "interkøn",
    "intersex",
    "variation in sex characteristics",
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
                            "maxTokens": proxy_budget.DEFAULT_MAX_TOKENS,
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
        max_tokens=proxy_budget.SOL_ADJUDICATION_LEDGER_MAX_TOKENS,
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
        "max_tokens": None,
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
        assert "max_tokens" not in body
        assert "max_completion_tokens" not in body
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


def test_permanent_http_400_resume_uses_unique_attempt_suffix(tmp_path: Path) -> None:
    """A failed reservation is kept and the resumed row uses the next suffix."""
    budget = _budget(tmp_path)
    checkpoint = _checkpoint(tmp_path)

    def reject(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "max_tokens" not in body
        assert "max_completion_tokens" not in body
        return httpx.Response(
            400,
            json={"error": {"message": "unsupported_parameter: max_tokens"}},
            request=request,
        )

    with pytest.raises(httpx.HTTPStatusError):
        run_sol_adjudication(
            original_persona=_PERSONA,
            candidate_row={"age": 42, "municipality": "Aarhus"},
            prompt=_PROMPT,
            config=_config(),
            budget=budget,
            checkpoint_path=checkpoint,
            transport=httpx.MockTransport(reject),
        )

    run_sol_adjudication(
        original_persona=_PERSONA,
        candidate_row={"age": 42, "municipality": "Aarhus"},
        prompt=_PROMPT,
        config=_config(),
        budget=budget,
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

    records = [
        json.loads(line)
        for line in budget.path.read_text(encoding="utf-8").splitlines()
    ]
    reservations = [record for record in records if record["type"] == "reservation"]
    usages = [record for record in records if record["type"] == "usage"]

    assert [
        record["request_id"].rsplit("-", maxsplit=1)[1] for record in reservations
    ] == ["1", "2"]
    assert usages[0]["request_id"] == reservations[1]["request_id"]


def test_preflight_allows_municipality_code_age_collision() -> None:
    """Permit a protected municipality code only as verified age text."""
    preflight_sol_adjudication_payload(
        original_persona="Hun er 101 år og bor i en dansk kommune.",
        candidate_row={
            "municipality_code": "101",
            "age": 101,
            "municipality": "København",
            "job_title": "lærer",
        },
        original_row={
            "municipality_code": "101",
            "age": 101,
            "municipality": "København",
            "job_title": "lærer",
        },
        prompt=_PROMPT,
    )


def test_preflight_blocks_embedded_raw_id_and_source_code(tmp_path: Path) -> None:
    """Raw IDs and source codes are token-matched in outbound prose and facts."""
    requests: list[httpx.Request] = []
    blocked_rows: list[dict[str, object]] = [
        {
            "persona_id": "opaque-persona-123",
            "origin_country_code": "SRC-777",
            "age": 42,
            "municipality": "Aarhus",
            "persona": "Neutral tekst med opaque-persona-123 som fejl.",
        },
        {
            "persona_id": "opaque-persona-123",
            "origin_country_code": "SRC-777",
            "age": 42,
            "municipality": "Aarhus",
            "job_title": "lærer SRC-777",
        },
    ]
    with pytest.raises(SolAdjudicationError):
        preflight_sol_adjudication_payload(
            original_persona=_PERSONA,
            candidate_row={
                "persona_id": "opaque-persona-123",
                "origin_country_code": "SRC-777",
                "age": 42,
                "job_title": "lærer",
            },
            prompt=_PROMPT,
            changed_fact_hints={"job_title": {"old": "SRC-777", "new": "lærer"}},
            original_row={"origin_country_code": "SRC-777", "job_title": "SRC-777"},
        )

    def unexpected_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500)

    for index, row in enumerate(blocked_rows):
        with pytest.raises(SolAdjudicationError):
            run_sol_adjudication(
                original_persona=str(row.get("persona", _PERSONA)),
                candidate_row=row,
                prompt=_PROMPT,
                config=_config(),
                budget=_budget(tmp_path),
                checkpoint_path=_checkpoint(tmp_path, name=f"blocked-{index}.json"),
                transport=httpx.MockTransport(unexpected_request),
            )

    preflight_sol_adjudication_payload(
        original_persona=_PERSONA,
        candidate_row={
            "persona_id": "opaque-persona-123",
            "origin_country_code": "SRC-777",
            "sexual_orientation": "pansexual",
            "transgender_status": "trans man",
            "sex_characteristics_variation": "intersex",
            "age": 42,
            "municipality": "Aarhus",
            "job_title": "lærer",
        },
        prompt=_PROMPT,
    )
    assert not requests


def test_preflight_rejects_all_banned_identity_variants() -> None:
    """Fail closed on identity variants before provider I/O."""
    for variant in _BANNED_IDENTITY_VARIANTS:
        with pytest.raises(SolAdjudicationError):
            preflight_sol_adjudication_payload(
                original_persona=f"Neutral tekst med {variant}.",
                candidate_row={"age": 42},
                prompt=_PROMPT,
            )


def test_preflight_rejects_municipality_age_collision_near_misses() -> None:
    """Keep protected codes blocked outside verified age representations."""
    base_candidate: dict[str, object] = {
        "municipality_code": "101",
        "age": 101,
        "municipality": "København",
        "job_title": "lærer",
    }
    base_original: dict[str, object] = {
        "municipality_code": "101",
        "age": 101,
        "municipality": "København",
        "job_title": "lærer",
    }

    with pytest.raises(SolAdjudicationError):
        preflight_sol_adjudication_payload(
            original_persona="Kommune 101 er nævnt i teksten.",
            candidate_row=base_candidate,
            original_row=base_original,
            prompt=_PROMPT,
        )

    with pytest.raises(SolAdjudicationError):
        preflight_sol_adjudication_payload(
            original_persona="Hun er 101 år og bor i en dansk kommune.",
            candidate_row=base_candidate,
            original_row={**base_original, "age": 100},
            prompt=_PROMPT,
        )

    with pytest.raises(SolAdjudicationError):
        preflight_sol_adjudication_payload(
            original_persona="Hun er 101 år og bor i en dansk kommune.",
            candidate_row={**base_candidate, "origin_country_code": "101"},
            original_row=base_original,
            prompt=_PROMPT,
        )


def test_preflight_rejects_municipality_code_in_unrelated_fact() -> None:
    """Reject the same numeric token when another fact carries it."""
    with pytest.raises(SolAdjudicationError):
        preflight_sol_adjudication_payload(
            original_persona="Hun er 101 år og bor i en dansk kommune.",
            candidate_row={
                "municipality_code": "101",
                "age": 101,
                "municipality": "København",
                "job_title": "lærer 101",
            },
            original_row={
                "municipality_code": "101",
                "age": 101,
                "municipality": "København",
                "job_title": "lærer",
            },
            prompt=_PROMPT,
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
    original_reserve = budget.reserve_next_attempt
    original_validate = budget.validate_request_body
    original_usage = budget.record_usage

    def reserve(
        request_id_prefix: str,
        request: dict[str, JSONValue],
        *,
        max_output_tokens: int | None = None,
        max_attempts: int,
    ) -> int:
        assert max_output_tokens == SOL_MAX_OUTPUT_TOKENS
        assert max_attempts == 10
        events.append("reserved")
        return original_reserve(
            request_id_prefix,
            request,
            max_output_tokens=max_output_tokens,
            max_attempts=max_attempts,
        )

    def validate_body(request_id: str, body: bytes) -> None:
        events.append("body")
        original_validate(request_id, body)

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

    monkeypatch.setattr(budget, "reserve_next_attempt", reserve)
    monkeypatch.setattr(budget, "validate_request_body", validate_body)
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

    assert events == ["reserved", "body", "network", "usage"]
    assert not checkpoint.exists()
    assert "raw-private-id" not in str(error.value)


def test_sol_row_lifetime_attempts_are_bounded(tmp_path: Path) -> None:
    """Stop retrying one row after the durable lifetime attempt limit."""
    budget = _budget(tmp_path)
    checkpoint = _checkpoint(tmp_path)

    def reject(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "permanent"}, request=request)

    for _ in range(10):
        with pytest.raises(httpx.HTTPStatusError):
            run_sol_adjudication(
                original_persona=_PERSONA,
                candidate_row={"age": 42, "municipality": "Aarhus"},
                prompt=_PROMPT,
                config=_config(),
                budget=budget,
                checkpoint_path=checkpoint,
                transport=httpx.MockTransport(reject),
            )

    with pytest.raises(proxy_budget.ProxyBudgetError, match="lifetime exhausted"):
        run_sol_adjudication(
            original_persona=_PERSONA,
            candidate_row={"age": 42, "municipality": "Aarhus"},
            prompt=_PROMPT,
            config=_config(),
            budget=budget,
            checkpoint_path=checkpoint,
            transport=httpx.MockTransport(reject),
        )


def test_uncapped_sol_records_output_overage(tmp_path: Path) -> None:
    """Record actual Sol output usage even when it exceeds the reservation."""
    budget = _budget(tmp_path)

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "max_tokens" not in body
        content = {
            "disposition": "consistent",
            "reason": "Alder er nævnt i teksten.",
            "evidence": [
                {"field": "age", "kind": "fact_present", "quote": "Hun er 42 år"}
            ],
            "patches": [],
        }
        return httpx.Response(
            200,
            json={
                "id": "response-overage",
                "model": "gpt-6-sol",
                "choices": [{"message": {"content": json.dumps(content)}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": SOL_MAX_OUTPUT_TOKENS + 17,
                },
            },
            request=request,
        )

    run_sol_adjudication(
        original_persona=_PERSONA,
        candidate_row={"age": 42, "municipality": "Aarhus"},
        prompt=_PROMPT,
        config=_config(),
        budget=budget,
        checkpoint_path=_checkpoint(tmp_path),
        transport=httpx.MockTransport(respond),
    )

    records = [
        json.loads(line)
        for line in budget.path.read_text(encoding="utf-8").splitlines()
    ]
    usage = next(record for record in records if record["type"] == "usage")
    assert usage["output_tokens"] == SOL_MAX_OUTPUT_TOKENS + 17
    assert usage["reserved_output_tokens"] == SOL_MAX_OUTPUT_TOKENS
    assert usage["output_tokens_over_reserved"] == 17
    assert usage["unbounded_output"] is True
