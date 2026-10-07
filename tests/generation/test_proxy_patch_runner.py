"""Offline tests for the private local proxy patch runner."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

import danish_personas.generation.proxy_budget as proxy_budget
from danish_personas.generation.models import GenerationConfig
from danish_personas.generation.prose_patch import ProsePatchResponse
from danish_personas.generation.proxy_budget import (
    JSONValue,
    ProxyBudget,
    ProxyBudgetError,
)
from danish_personas.generation.proxy_patch_runner import (
    ProxyPatchError,
    ProxyPatchProposal,
    run_proxy_patch,
)


@pytest.fixture(autouse=True)
def _private_budget_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep mock reservations out of the user's real cumulative budget ledger."""
    monkeypatch.setattr(proxy_budget, "USER_BUDGET_PATH", tmp_path / "budget.jsonl")


def test_payload_privacy_callback_before_network_and_restart_is_idempotent(
    tmp_path: Path,
) -> None:
    """Reserve before sending and reuse a validated checkpoint on restart."""
    old = _row()["persona"]
    raw = json.dumps(
        {"patches": [{"old_excerpt": "Før ændring", "new_excerpt": "Efter ændring"}]}
    )
    requests: list[httpx.Request] = []
    events: list[str] = []
    budget = _budget(tmp_path)
    original_reserve = budget.reserve_attempt

    def reserve(request_id: str, request: dict[str, JSONValue]) -> Decimal:
        events.append("reserved")
        return original_reserve(request_id, request)

    pytest.MonkeyPatch().setattr(budget, "reserve_attempt", reserve)
    result = run_proxy_patch(
        row=_row(),
        candidate_row=_candidate_row(
            {"marital_status": {"old": "single", "new": "married"}}
        ),
        changed_facts={"marital_status": {"old": "single", "new": "married"}},
        gender="woman",
        partner_gender="man",
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
        "persona": old,
        "changed_facts": {"marital_status": {"old": "single", "new": "married"}},
        "gender": "woman",
        "partner_gender": "man",
    }
    assert all(
        secret not in requests[0].content.decode()
        for secret in ["private-id", "private-sex", "municipality", "origin_country_da"]
    )
    assert result.persona_text == old.replace("Før ændring", "Efter ændring")
    assert result.changed_fraction == max(
        len("Før ændring"), len("Efter ændring")
    ) / len(old)
    assert result.persona_text[len("Efter ændring") :] == old[len("Før ændring") :]
    assert (tmp_path / "provisional.json").stat().st_mode & 0o777 == 0o600

    def must_not_send(_: httpx.Request) -> httpx.Response:
        raise AssertionError("a checkpoint restart must not make another request")

    restarted = run_proxy_patch(
        row=_row(),
        candidate_row=_candidate_row(
            {"marital_status": {"old": "single", "new": "married"}}
        ),
        changed_facts={"marital_status": {"old": "single", "new": "married"}},
        gender="woman",
        partner_gender="man",
        prompt="Ret kun den nødvendige lokale formulering.",
        config=_config(),
        budget=budget,
        checkpoint_path=tmp_path / "provisional.json",
        transport=httpx.MockTransport(must_not_send),
    )
    assert restarted == result


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
        source_hash="a" * 64,
        prompt_hash=hashlib.sha256(
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


def _candidate_row(
    facts: dict[str, dict[str, object]], row: dict[str, Any] | None = None
) -> dict[str, Any]:
    candidate = dict(_row() if row is None else row)
    candidate.update(
        {field: pair["new"] for field, pair in facts.items() if field in candidate}
    )
    return candidate


def _row() -> dict[str, Any]:
    return {
        "persona_id": "private-id",
        "record_id": "private-id",
        "source_sex": "private-sex",
        "municipality": "private municipality",
        "origin_country_da": "private origin",
        "persona": "Før ændring. " + "Dette er en syntetisk person. " * 12,
        "marital_status": "single",
        "skills_and_expertise": ["planlægning"] * 3,
        "hobbies_and_interests": ["cykling"] * 3,
    }


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
    return GenerationConfig.model_validate(values)


def _transport(
    response_content: str, seen: list[httpx.Request], events: list[str] | None = None
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


def test_real_schema_fact_fields_are_the_only_candidate_delta_sent(
    tmp_path: Path,
) -> None:
    """Send validated prose and allowlisted list-field deltas, never whole rows."""
    source = _row()
    candidate = dict(source)
    candidate["skills_and_expertise"] = ["analyse"] * 3
    candidate["hobbies_and_interests"] = ["vandring"] * 3
    facts = {
        "skills_and_expertise": {
            "old": source["skills_and_expertise"],
            "new": candidate["skills_and_expertise"],
        },
        "hobbies_and_interests": {
            "old": source["hobbies_and_interests"],
            "new": candidate["hobbies_and_interests"],
        },
    }
    raw = json.dumps(
        {"patches": [{"old_excerpt": "Før ændring", "new_excerpt": "Efter ændring"}]}
    )
    requests: list[httpx.Request] = []
    run_proxy_patch(
        row=source,
        candidate_row=candidate,
        changed_facts=facts,
        gender="unknown",
        partner_gender=None,
        prompt="Ret kun den nødvendige lokale formulering.",
        config=_config(),
        budget=_budget(tmp_path),
        checkpoint_path=tmp_path / "provisional.json",
        transport=_transport(raw, requests),
    )
    body = json.loads(requests[0].content)
    payload = json.loads(body["messages"][1]["content"])
    assert payload["changed_facts"] == facts
    assert "gender" not in payload
    assert "persona_id" not in requests[0].content.decode()
    assert "candidate_row" not in requests[0].content.decode()


def test_rejects_bad_config_sensitive_original_and_non_allowlisted_fact(
    tmp_path: Path,
) -> None:
    """Reject unsupported settings and identity data before network I/O."""
    with pytest.raises(ProxyPatchError):
        _run(
            tmp_path,
            httpx.MockTransport(lambda _: httpx.Response(500)),
            config=_config(max_tokens=100),
        )
    row = _row()
    row["persona"] += " Seksual orientation."
    with pytest.raises(ProxyPatchError):
        _run(tmp_path, httpx.MockTransport(lambda _: httpx.Response(500)), row=row)
    with pytest.raises(ProxyPatchError):
        _run(
            tmp_path,
            httpx.MockTransport(lambda _: httpx.Response(500)),
            changed_facts={"municipality": {"old": "a", "new": "b"}},
        )


def _run(
    tmp_path: Path,
    transport: httpx.BaseTransport,
    *,
    row: dict[str, Any] | None = None,
    changed_facts: dict[str, dict[str, object]] | None = None,
    config: GenerationConfig | None = None,
) -> ProxyPatchProposal:
    """Run one synthetic proposal with caller-selected negative-test inputs.

    Returns:
        The provisional patch proposal.
    """
    source = _row() if row is None else row
    facts = (
        {"marital_status": {"old": "single", "new": "married"}}
        if changed_facts is None
        else changed_facts
    )
    candidate = dict(source)
    candidate.update(
        {field: pair["new"] for field, pair in facts.items() if field in source}
    )
    return run_proxy_patch(
        row=source,
        candidate_row=candidate,
        changed_facts=facts,
        gender="woman",
        partner_gender="man",
        prompt="Ret kun den nødvendige lokale formulering.",
        config=_config() if config is None else config,
        budget=_budget(tmp_path),
        checkpoint_path=tmp_path / "provisional.json",
        transport=transport,
    )


def test_rejects_candidate_row_mismatch_before_network(tmp_path: Path) -> None:
    """Reject wrong persona bindings and fact deltas before network I/O."""
    facts: dict[str, dict[str, object]] = {
        "marital_status": {"old": "single", "new": "married"}
    }
    candidate = _candidate_row(facts)
    candidate["persona_id"] = "another-id"
    requests: list[httpx.Request] = []
    with pytest.raises(ProxyPatchError):
        run_proxy_patch(
            row=_row(),
            candidate_row=candidate,
            changed_facts=facts,
            gender="woman",
            partner_gender=None,
            prompt="Ret kun den nødvendige lokale formulering.",
            config=_config(),
            budget=_budget(tmp_path),
            checkpoint_path=tmp_path / "provisional.json",
            transport=_transport("{}", requests),
        )
    assert not requests


def test_rejects_checkpoint_with_matching_checksum_but_false_rewrite(
    tmp_path: Path,
) -> None:
    """Reapply evidence even when a forged checkpoint has a valid digest."""
    old = _row()["persona"]
    raw = json.dumps(
        {"patches": [{"old_excerpt": "Før ændring", "new_excerpt": "Efter ændring"}]}
    )
    requests: list[httpx.Request] = []
    _run(tmp_path, _transport(raw, requests))
    path = tmp_path / "provisional.json"
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    checkpoint["proposed_persona_text"] = old + " En skjult ændring."
    unsigned = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(ProxyPatchError):
        _run(tmp_path, httpx.MockTransport(lambda _: httpx.Response(500)))


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps({"patches": [{"old_excerpt": "not present", "new_excerpt": "x"}]}),
        json.dumps(
            {
                "patches": [
                    {"old_excerpt": "Før ændring", "new_excerpt": "x", "extra": 1}
                ]
            }
        ),
    ],
)
def test_rejects_malformed_and_unmatched_patch(tmp_path: Path, content: str) -> None:
    """Quarantine malformed and non-matching model patch suggestions."""
    requests: list[httpx.Request] = []
    with pytest.raises((ProxyPatchError, ValueError)):
        _run(tmp_path, _transport(content, requests))
    assert len(requests) == 1


def test_rejects_oversized_actual_http_body_before_network(tmp_path: Path) -> None:
    """Abort if the encoded request exceeds the reserved input bound."""
    budget = _budget(tmp_path)
    # Force a reservation bound too small for the constructed HTTP body.
    budget.overhead = -1
    seen: list[httpx.Request] = []
    raw = json.dumps(
        {"patches": [{"old_excerpt": "Før ændring", "new_excerpt": "Efter ændring"}]}
    )
    with pytest.raises((ProxyPatchError, ProxyBudgetError)):
        run_proxy_patch(
            row=_row(),
            candidate_row=_candidate_row(
                {"marital_status": {"old": "single", "new": "married"}}
            ),
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


@pytest.mark.parametrize(
    "identity_term",
    [
        "seksuel orientering",
        "seksual orientation",
        "homoseksuel",
        "biseksuel",
        "heteroseksuel",
        "lesbisk",
        "queer",
        "transkønnet",
        "interkønnet",
        "sexual_orientation",
    ],
)
def test_rejects_sensitive_fact_values(tmp_path: Path, identity_term: str) -> None:
    """Refuse sensitive values even when their fact key is allowlisted."""
    with pytest.raises(ProxyPatchError):
        _run(
            tmp_path,
            httpx.MockTransport(lambda _: httpx.Response(500)),
            changed_facts={
                "skills_and_expertise": {
                    "old": ["planlægning"] * 3,
                    "new": [identity_term],
                }
            },
        )


@pytest.mark.parametrize(
    ("prompt", "gender", "partner_gender"),
    [
        ("Use the term queer only as a private test.", "woman", None),
        ("Ret kun ændringen.", "transkønnet", None),
        ("Ret kun ændringen.", "woman", "transkønnet"),
    ],
)
def test_rejects_sensitive_or_unsupported_outbound_text_before_network(
    tmp_path: Path, prompt: str, gender: str, partner_gender: str | None
) -> None:
    """Preflight the prompt and synthetic-gender fields before transport."""
    facts: dict[str, dict[str, object]] = {
        "marital_status": {"old": "single", "new": "married"}
    }
    requests: list[httpx.Request] = []
    with pytest.raises(ProxyPatchError):
        run_proxy_patch(
            row=_row(),
            candidate_row=_candidate_row(facts),
            changed_facts=facts,
            gender=gender,
            partner_gender=partner_gender,
            prompt=prompt,
            config=_config(),
            budget=_budget(tmp_path),
            checkpoint_path=tmp_path / "provisional.json",
            transport=_transport("{}", requests),
        )
    assert not requests


def test_rejects_shared_checkpoint_directory_without_changing_its_mode(
    tmp_path: Path,
) -> None:
    """Never change the permissions of a caller-owned shared directory."""
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)
    with pytest.raises(ProxyPatchError, match="must be private"):
        run_proxy_patch(
            row=_row(),
            candidate_row=_candidate_row(
                {"marital_status": {"old": "single", "new": "married"}}
            ),
            changed_facts={"marital_status": {"old": "single", "new": "married"}},
            gender=None,
            partner_gender=None,
            prompt="Ret kun den nødvendige lokale formulering.",
            config=_config(),
            budget=_budget(tmp_path),
            checkpoint_path=shared / "checkpoint.json",
            transport=httpx.MockTransport(lambda _: httpx.Response(500)),
        )
    assert shared.stat().st_mode & 0o777 == 0o755
