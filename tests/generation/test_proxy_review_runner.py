"""Offline tests for the private local proxy review runner."""

from __future__ import annotations

import hashlib
import json
import stat
import threading
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest

import danish_personas.generation.proxy_budget as proxy_budget
import danish_personas.generation.proxy_review_runner as review_runner
from danish_personas.generation.models import GenerationConfig
from danish_personas.generation.prose_review import (
    ProseReviewResponse,
    ProseReviewResult,
)
from danish_personas.generation.proxy_budget import (
    JSONValue,
    ProxyBudget,
    ProxyBudgetError,
)
from danish_personas.generation.proxy_review_runner import (
    ProxyReviewError,
    run_proxy_review,
)

_PROMPT = "Vurder om den oprindelige persona kræver en lokal ændring."


def test_prepare_checkpoint_parent_allows_two_threads_for_same_new_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Create the same hash-prefix directory safely from concurrent workers."""
    parent = tmp_path / "checkpoints" / "ab"
    original_exists = Path.exists
    original_mkdir = Path.mkdir
    exists_barrier = threading.Barrier(parties=2)
    mkdir_lock = threading.Lock()
    prefix_exist_ok_values: list[bool] = []

    def coordinated_exists(self: Path) -> bool:
        exists = original_exists(self)
        if self == parent and not exists:
            exists_barrier.wait(timeout=5)
        return exists

    def recording_mkdir(
        self: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        if self == parent:
            with mkdir_lock:
                prefix_exist_ok_values.append(exist_ok)
        original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "exists", coordinated_exists)
    monkeypatch.setattr(Path, "mkdir", recording_mkdir)
    errors: list[BaseException] = []

    def prepare_parent() -> None:
        try:
            review_runner._prepare_checkpoint_parent(parent)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=prepare_parent) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert prefix_exist_ok_values == [True, True]
    assert stat.S_IMODE(parent.lstat().st_mode) == 0o700


def test_prepare_checkpoint_parent_rejects_symlink_prefix(tmp_path: Path) -> None:
    """Reject a hash-prefix path that resolves through a final symlink."""
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    parent = tmp_path / "checkpoints" / "ab"
    parent.parent.mkdir(mode=0o700)
    try:
        parent.symlink_to(target=target, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(ProxyReviewError, match="real directory"):
        review_runner._prepare_checkpoint_parent(parent)


def test_prepare_checkpoint_parent_rejects_public_prefix_mode(
    tmp_path: Path,
) -> None:
    """Reject an existing hash-prefix directory with group/world permissions."""
    parent = tmp_path / "checkpoints" / "ab"
    parent.mkdir(mode=0o700, parents=True)
    parent.chmod(0o755)

    with pytest.raises(ProxyReviewError, match="mode 0700"):
        review_runner._prepare_checkpoint_parent(parent)


@pytest.fixture(autouse=True)
def _private_budget_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep mock reservations out of the user's real cumulative budget ledger."""
    monkeypatch.setattr(proxy_budget, "USER_BUDGET_PATH", tmp_path / "budget.jsonl")


def test_accepts_verified_null_detail_quote_and_resumes(tmp_path: Path) -> None:
    """Accept contextual Danish category evidence without certifying release safety."""
    row, candidate_row, changed_facts = _null_detail_review_inputs("divorced")
    persona = "Hun er skilt. " + "Dette er en syntetisk person. " * 12
    row["persona"] = persona
    candidate_row["persona"] = persona
    requests: list[httpx.Request] = []

    result = run_proxy_review(
        row=row,
        candidate_row=candidate_row,
        changed_facts=changed_facts,
        prompt=_PROMPT,
        config=_config(),
        budget=_budget(tmp_path),
        checkpoint_path=tmp_path / "review.json",
        transport=_transport(
            json.dumps(
                {
                    "disposition": "unchanged_consistent",
                    "patches": [],
                    "unchanged_evidence": [
                        {
                            "field": "legal_status_detail",
                            "kind": "new_value_present",
                            "quote": "Hun er skilt",
                        }
                    ],
                    "manual_review_reason": None,
                }
            ),
            requests,
        ),
    )

    assert result.disposition == "unchanged_consistent"
    assert result.unchanged_consistent_note == (
        "provisional_classifier_not_independently_certified"
    )

    restarted = run_proxy_review(
        row=row,
        candidate_row=candidate_row,
        changed_facts=changed_facts,
        prompt=_PROMPT,
        config=_config(),
        budget=_budget(tmp_path),
        checkpoint_path=tmp_path / "review.json",
        transport=httpx.MockTransport(lambda _: httpx.Response(500)),
    )

    assert len(requests) == 1
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
        campaign="synthetic-review-test",
        source_hash="a" * 64,
        prompt_hash=hashlib.sha256(_PROMPT.encode()).hexdigest(),
        schema_hash=hashlib.sha256(
            json.dumps(
                ProseReviewResponse.provider_json_schema(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        cap_usd=Decimal("1"),
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
    return GenerationConfig.model_validate(values)


def _null_detail_review_inputs(
    marital_status: object,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, object]]]:
    row = _row()
    row["marital_status"] = marital_status
    row["legal_status_detail"] = "married"
    changed_facts: dict[str, dict[str, object]] = {
        "legal_status_detail": {"old": "married", "new": None}
    }
    return row, _candidate_row(changed_facts, row=row), changed_facts


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
        "age": 41,
        "skills_and_expertise": ["planlægning"] * 3,
        "hobbies_and_interests": ["cykling"] * 3,
    }


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


def test_accounting_failure_does_not_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail hard if durable accounting rejects observed usage."""
    budget = _budget(tmp_path)

    def fail_usage(_request_id: str, **_usage: object) -> None:
        raise ProxyBudgetError("ledger failure")

    monkeypatch.setattr(budget, "record_usage", fail_usage)

    with pytest.raises(ProxyBudgetError):
        _run(
            tmp_path,
            _transport(
                json.dumps(
                    {
                        "disposition": "needs_manual_review",
                        "patches": [],
                        "unchanged_evidence": [],
                        "manual_review_reason": "ambiguous",
                    }
                ),
                [],
            ),
            budget=budget,
        )

    assert not (tmp_path / "review.json").exists()


def _run(
    tmp_path: Path,
    transport: httpx.BaseTransport,
    *,
    changed_facts: dict[str, dict[str, object]] | None = None,
    budget: ProxyBudget | None = None,
) -> ProseReviewResult:
    source = _row()
    facts = (
        {"marital_status": {"old": "single", "new": "married"}}
        if changed_facts is None
        else changed_facts
    )
    return run_proxy_review(
        row=source,
        candidate_row=_candidate_row(facts, row=source),
        changed_facts=facts,
        prompt=_PROMPT,
        config=_config(),
        budget=_budget(tmp_path) if budget is None else budget,
        checkpoint_path=tmp_path / "review.json",
        transport=transport,
    )


@pytest.mark.parametrize(
    ("marital_status", "expected_label"),
    [("divorced", "skilt"), ("widowed", "enkestand"), ("never_married", "aldrig gift")],
)
def test_adds_minimal_null_detail_context_for_allowed_statuses(
    tmp_path: Path, marital_status: str, expected_label: str
) -> None:
    """Send only the allowlisted Danish target for supported null transitions."""
    row, candidate_row, changed_facts = _null_detail_review_inputs(marital_status)
    requests: list[httpx.Request] = []

    run_proxy_review(
        row=row,
        candidate_row=candidate_row,
        changed_facts=changed_facts,
        prompt=_PROMPT,
        config=_config(),
        budget=_budget(tmp_path),
        checkpoint_path=tmp_path / "review.json",
        transport=_transport(
            json.dumps(
                {
                    "disposition": "needs_manual_review",
                    "patches": [],
                    "unchanged_evidence": [],
                    "manual_review_reason": "ambiguous",
                }
            ),
            requests,
        ),
    )

    request_text = requests[0].content.decode()
    user_payload = _user_payload(requests[0])
    assert set(user_payload) == {
        "persona",
        "changed_facts",
        "legal_status_detail_null_context",
    }
    assert user_payload["legal_status_detail_null_context"] == {
        "target_marital_category_da": expected_label,
        "explanation": (
            "legal_status_detail = null betyder, at en ikke-understøttet fin "
            "detalje er fjernet. Den aktuelle kildeunderstøttede "
            "civilstandskategori her er målet; udled ingen nye personlige træk."
        ),
    }
    assert marital_status not in request_text
    assert "candidate_row" not in request_text
    assert all(
        secret not in request_text
        for secret in [
            "private-id",
            "private-sex",
            "private municipality",
            "private origin",
        ]
    )

    checkpoint = json.loads((tmp_path / "review.json").read_text(encoding="utf-8"))
    assert (
        checkpoint["payload_sha256"]
        == hashlib.sha256(
            json.dumps(
                user_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    )


def _user_payload(request: httpx.Request) -> dict[str, object]:
    request_body = json.loads(request.content)
    value = json.loads(request_body["messages"][1]["content"])
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize(
    ("provider_content", "expected_disposition"),
    [
        (
            {
                "disposition": "patched",
                "patches": [
                    {"old_excerpt": "Før ændring", "new_excerpt": "Efter ændring"}
                ],
                "unchanged_evidence": [],
                "manual_review_reason": None,
            },
            "patched",
        ),
        (
            {
                "disposition": "unchanged_consistent",
                "patches": [],
                "unchanged_evidence": [
                    {"field": "marital_status", "kind": "fact_not_stated", "quote": ""}
                ],
                "manual_review_reason": None,
            },
            "unchanged_consistent",
        ),
        (
            {
                "disposition": "needs_manual_review",
                "patches": [],
                "unchanged_evidence": [],
                "manual_review_reason": "ambiguous",
            },
            "needs_manual_review",
        ),
    ],
)
def test_all_dispositions_checkpoint_privately_and_resume_without_network(
    tmp_path: Path, provider_content: dict[str, object], expected_disposition: str
) -> None:
    """Persist exact bounded decisions, then revalidate them on restart."""
    requests: list[httpx.Request] = []
    result = run_proxy_review(
        row=_row(),
        candidate_row=_candidate_row(
            {"marital_status": {"old": "single", "new": "married"}}
        ),
        changed_facts={"marital_status": {"old": "single", "new": "married"}},
        prompt=_PROMPT,
        config=_config(),
        budget=_budget(tmp_path),
        checkpoint_path=tmp_path / "review.json",
        transport=_transport(json.dumps(provider_content), requests),
    )

    assert result.disposition == expected_disposition
    assert len(requests) == 1
    request_body = json.loads(requests[0].content)
    normal_payload = {
        "persona": _row()["persona"],
        "changed_facts": {"marital_status": {"old": "single", "new": "married"}},
    }
    assert request_body["messages"][1]["content"] == json.dumps(
        normal_payload, ensure_ascii=False
    )
    user_payload = json.loads(request_body["messages"][1]["content"])
    assert set(user_payload) == {"persona", "changed_facts"}
    assert "gender" not in user_payload
    assert "partner_gender" not in user_payload
    assert all(
        secret not in requests[0].content.decode()
        for secret in ["private-id", "municipality", "origin_country_da"]
    )

    checkpoint_path = tmp_path / "review.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint_path.stat().st_mode & 0o777 == 0o600
    assert tmp_path.stat().st_mode & 0o077 == 0
    assert "persona" not in checkpoint
    assert "proposed_text" not in checkpoint
    assert "completion" not in checkpoint
    assert checkpoint["disposition"] == expected_disposition

    def must_not_send(_: httpx.Request) -> httpx.Response:
        raise AssertionError("a checkpoint restart must not make another request")

    restarted = run_proxy_review(
        row=_row(),
        candidate_row=_candidate_row(
            {"marital_status": {"old": "single", "new": "married"}}
        ),
        changed_facts={"marital_status": {"old": "single", "new": "married"}},
        prompt=_PROMPT,
        config=_config(),
        budget=_budget(tmp_path),
        checkpoint_path=checkpoint_path,
        transport=httpx.MockTransport(must_not_send),
    )
    assert restarted == result


def test_failed_http_retry_gets_unique_reservations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never reuse a failed attempt's durable request ID on caller retry."""
    budget = _budget(tmp_path)
    original_reserve = budget.reserve_attempt
    request_ids: list[str] = []

    def reserve(request_id: str, request: dict[str, JSONValue]) -> Decimal:
        request_ids.append(request_id)
        return original_reserve(request_id, request)

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("local proxy unavailable", request=request)

    monkeypatch.setattr(budget, "reserve_attempt", reserve)
    for _ in range(2):
        with pytest.raises(httpx.ConnectError):
            _run(tmp_path, httpx.MockTransport(fail), budget=budget)

    assert len(request_ids) == 2
    assert len(set(request_ids)) == 2
    assert not (tmp_path / "review.json").exists()


def test_invalid_provider_metadata_does_not_checkpoint(tmp_path: Path) -> None:
    """Fail hard if a successful response is not bound to the pinned model."""

    def wrong_model(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "response-1",
                "model": "other-model",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "disposition": "needs_manual_review",
                                    "patches": [],
                                    "unchanged_evidence": [],
                                    "manual_review_reason": "ambiguous",
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )

    with pytest.raises(ProxyReviewError, match="model"):
        _run(tmp_path, httpx.MockTransport(wrong_model))

    assert not (tmp_path / "review.json").exists()


def test_mismatched_quote_abstains_and_resumes_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Store one manual checkpoint when local evidence validation fails."""
    requests: list[httpx.Request] = []
    events: list[str] = []
    budget = _budget(tmp_path)
    original_reserve = budget.reserve_attempt
    original_usage = budget.record_usage
    row = _row()
    row["persona"] = "Maria beskrives som gift i teksten. " + (
        "Dette er en syntetisk person med hverdagsbeskrivelser. " * 8
    )
    changed_facts: dict[str, dict[str, object]] = {
        "marital_status": {"old": "single", "new": "married"}
    }
    checkpoint_path = tmp_path / "review.json"

    def reserve(request_id: str, request: dict[str, JSONValue]) -> Decimal:
        events.append("reservation")
        return original_reserve(request_id, request)

    def record_usage(
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
    monkeypatch.setattr(budget, "record_usage", record_usage)

    result = run_proxy_review(
        row=row,
        candidate_row=_candidate_row(changed_facts, row=row),
        changed_facts=changed_facts,
        prompt=_PROMPT,
        config=_config(),
        budget=budget,
        checkpoint_path=checkpoint_path,
        transport=_transport(
            json.dumps(
                {
                    "disposition": "unchanged_consistent",
                    "patches": [],
                    "unchanged_evidence": [
                        {
                            "field": "marital_status",
                            "kind": "new_value_present",
                            "quote": "gift",
                        }
                    ],
                    "manual_review_reason": None,
                }
            ),
            requests,
        ),
    )

    assert result.disposition == "needs_manual_review"
    assert result.manual_review_reason == "insufficient_evidence"
    assert result.patches == ()
    assert result.unchanged_evidence == ()
    assert events == ["reservation", "usage"]
    assert len(requests) == 1

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["disposition"] == "needs_manual_review"
    assert checkpoint["manual_review_reason"] == "insufficient_evidence"
    assert checkpoint["patches"] == []
    assert checkpoint["unchanged_evidence"] == []
    assert "completion" not in checkpoint
    assert "gift" not in json.dumps(checkpoint, ensure_ascii=False)

    def must_not_send(_: httpx.Request) -> httpx.Response:
        raise AssertionError("a manual checkpoint resume must not make a request")

    resumed = run_proxy_review(
        row=row,
        candidate_row=_candidate_row(changed_facts, row=row),
        changed_facts=changed_facts,
        prompt=_PROMPT,
        config=_config(),
        budget=budget,
        checkpoint_path=checkpoint_path,
        transport=httpx.MockTransport(must_not_send),
    )

    assert resumed == result
    assert events == ["reservation", "usage"]
    assert len(requests) == 1


def test_records_usage_before_semantic_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Record token usage for a successful HTTP response before abstaining."""
    events: list[str] = []
    budget = _budget(tmp_path)
    original_usage = budget.record_usage
    original_validate = review_runner.validate_prose_review

    def record_usage(
        request_id: str, *, input_tokens: int, output_tokens: int, response_sha256: str
    ) -> None:
        events.append("usage")
        original_usage(
            request_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            response_sha256=response_sha256,
        )

    def validate_with_event(
        *,
        original_text: str,
        changed_facts: Mapping[str, Mapping[str, object]],
        response: str | Mapping[str, object],
        verified_context: Mapping[str, object] | None = None,
    ) -> object:
        events.append("validate")
        return original_validate(
            original_text=original_text,
            changed_facts=changed_facts,
            response=response,
            verified_context=verified_context,
        )

    monkeypatch.setattr(budget, "record_usage", record_usage)
    monkeypatch.setattr(review_runner, "validate_prose_review", validate_with_event)

    result = _run(tmp_path, _transport("not json", []), budget=budget)

    assert result.disposition == "needs_manual_review"
    assert result.manual_review_reason == "insufficient_evidence"
    assert events == ["usage", "validate", "validate"]
    assert (tmp_path / "review.json").exists()


@pytest.mark.parametrize("marital_status", ["single", "married_or_separated", None])
def test_rejects_null_detail_context_without_allowed_status(
    tmp_path: Path, marital_status: object
) -> None:
    """Fail closed when the target marital category is not concrete and allowed."""
    row, candidate_row, changed_facts = _null_detail_review_inputs("divorced")
    row["marital_status"] = marital_status
    candidate_row["marital_status"] = marital_status
    requests: list[httpx.Request] = []

    with pytest.raises(ProxyReviewError, match="concrete marital_status"):
        run_proxy_review(
            row=row,
            candidate_row=candidate_row,
            changed_facts=changed_facts,
            prompt=_PROMPT,
            config=_config(),
            budget=_budget(tmp_path),
            checkpoint_path=tmp_path / "review.json",
            transport=_transport("{}", requests),
        )

    assert not requests
    assert not (tmp_path / "review.json").exists()


def test_rejects_old_null_detail_checkpoint_without_payload_hash(
    tmp_path: Path,
) -> None:
    """Do not reuse legacy checkpoints for requests that now include context."""
    row, candidate_row, changed_facts = _null_detail_review_inputs("divorced")
    run_proxy_review(
        row=row,
        candidate_row=candidate_row,
        changed_facts=changed_facts,
        prompt=_PROMPT,
        config=_config(),
        budget=_budget(tmp_path),
        checkpoint_path=tmp_path / "review.json",
        transport=_transport(
            json.dumps(
                {
                    "disposition": "needs_manual_review",
                    "patches": [],
                    "unchanged_evidence": [],
                    "manual_review_reason": "ambiguous",
                }
            ),
            [],
        ),
    )
    checkpoint_path = tmp_path / "review.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint.pop("payload_sha256")
    unsigned = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    checkpoint_path.chmod(0o600)

    with pytest.raises(ProxyReviewError, match="malformed"):
        run_proxy_review(
            row=row,
            candidate_row=candidate_row,
            changed_facts=changed_facts,
            prompt=_PROMPT,
            config=_config(),
            budget=_budget(tmp_path),
            checkpoint_path=checkpoint_path,
            transport=httpx.MockTransport(lambda _: httpx.Response(500)),
        )


def test_rejects_sensitive_identity_terms_before_network(tmp_path: Path) -> None:
    """Reuse the patch runner's outbound sensitive-identity guard."""
    requests: list[httpx.Request] = []
    with pytest.raises(ProxyReviewError):
        _run(
            tmp_path,
            _transport("{}", requests),
            changed_facts={
                "skills_and_expertise": {
                    "old": ["planlægning"] * 3,
                    "new": ["seksuel orientering"],
                }
            },
        )
    assert not requests


def test_rejects_stale_checkpoint_inputs(tmp_path: Path) -> None:
    """Refuse to resume a decision bound to different verified inputs."""
    _run(
        tmp_path,
        _transport(
            json.dumps(
                {
                    "disposition": "needs_manual_review",
                    "patches": [],
                    "unchanged_evidence": [],
                    "manual_review_reason": "ambiguous",
                }
            ),
            [],
        ),
    )

    with pytest.raises(ProxyReviewError, match="changed"):
        run_proxy_review(
            row=_row(),
            candidate_row=_candidate_row({"age": {"old": 41, "new": 42}}),
            changed_facts={"age": {"old": 41, "new": 42}},
            prompt=_PROMPT,
            config=_config(),
            budget=_budget(tmp_path),
            checkpoint_path=tmp_path / "review.json",
            transport=httpx.MockTransport(lambda _: httpx.Response(500)),
        )


def test_rejects_stale_null_detail_context_checkpoint(tmp_path: Path) -> None:
    """Bind resumed null-detail decisions to the current contextual payload."""
    row, candidate_row, changed_facts = _null_detail_review_inputs("divorced")
    run_proxy_review(
        row=row,
        candidate_row=candidate_row,
        changed_facts=changed_facts,
        prompt=_PROMPT,
        config=_config(),
        budget=_budget(tmp_path),
        checkpoint_path=tmp_path / "review.json",
        transport=_transport(
            json.dumps(
                {
                    "disposition": "needs_manual_review",
                    "patches": [],
                    "unchanged_evidence": [],
                    "manual_review_reason": "ambiguous",
                }
            ),
            [],
        ),
    )
    checkpoint_path = tmp_path / "review.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["payload_sha256"] = "b" * 64
    unsigned = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    checkpoint_path.chmod(0o600)

    with pytest.raises(ProxyReviewError, match="changed"):
        run_proxy_review(
            row=row,
            candidate_row=candidate_row,
            changed_facts=changed_facts,
            prompt=_PROMPT,
            config=_config(),
            budget=_budget(tmp_path),
            checkpoint_path=checkpoint_path,
            transport=httpx.MockTransport(lambda _: httpx.Response(500)),
        )


def test_reserves_before_network_and_omits_private_row_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reserve durably before I/O while sending only the minimal review payload."""
    events: list[str] = []
    requests: list[httpx.Request] = []
    budget = _budget(tmp_path)
    original_reserve = budget.reserve_attempt

    def reserve(request_id: str, request: dict[str, JSONValue]) -> Decimal:
        events.append("reserved")
        return original_reserve(request_id, request)

    monkeypatch.setattr(budget, "reserve_attempt", reserve)
    run_proxy_review(
        row=_row(),
        candidate_row=_candidate_row(
            {"marital_status": {"old": "single", "new": "married"}}
        ),
        changed_facts={"marital_status": {"old": "single", "new": "married"}},
        prompt=_PROMPT,
        config=_config(),
        budget=budget,
        checkpoint_path=tmp_path / "review.json",
        transport=_transport(
            json.dumps(
                {
                    "disposition": "needs_manual_review",
                    "patches": [],
                    "unchanged_evidence": [],
                    "manual_review_reason": "ambiguous",
                }
            ),
            requests,
            events,
        ),
    )

    assert events[:2] == ["reserved", "network"]
    assert "private-sex" not in requests[0].content.decode()
    assert "candidate_row" not in requests[0].content.decode()


def test_revalidates_checkpoint_patch_snippets(tmp_path: Path) -> None:
    """A valid checksum cannot make forged patch evidence acceptable."""
    _run(
        tmp_path,
        _transport(
            json.dumps(
                {
                    "disposition": "patched",
                    "patches": [
                        {"old_excerpt": "Før ændring", "new_excerpt": "Efter ændring"}
                    ],
                    "unchanged_evidence": [],
                    "manual_review_reason": None,
                }
            ),
            [],
        ),
    )
    checkpoint_path = tmp_path / "review.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["patches"] = [
        {"old_excerpt": "not present", "new_excerpt": "Efter ændring"}
    ]
    unsigned = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    checkpoint["checkpoint_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    checkpoint_path.chmod(0o600)

    with pytest.raises(ProxyReviewError):
        _run(tmp_path, httpx.MockTransport(lambda _: httpx.Response(500)))


def test_wrong_null_detail_quote_abstains_manual(tmp_path: Path) -> None:
    """Store a bounded manual decision when contextual quote validation fails."""
    row, candidate_row, changed_facts = _null_detail_review_inputs("divorced")
    persona = "Hun er separeret. " + "Dette er en syntetisk person. " * 12
    row["persona"] = persona
    candidate_row["persona"] = persona

    result = run_proxy_review(
        row=row,
        candidate_row=candidate_row,
        changed_facts=changed_facts,
        prompt=_PROMPT,
        config=_config(),
        budget=_budget(tmp_path),
        checkpoint_path=tmp_path / "review.json",
        transport=_transport(
            json.dumps(
                {
                    "disposition": "unchanged_consistent",
                    "patches": [],
                    "unchanged_evidence": [
                        {
                            "field": "legal_status_detail",
                            "kind": "new_value_present",
                            "quote": "Hun er separeret",
                        }
                    ],
                    "manual_review_reason": None,
                }
            ),
            [],
        ),
    )

    assert result.disposition == "needs_manual_review"
    assert result.manual_review_reason == "insufficient_evidence"
    checkpoint = json.loads((tmp_path / "review.json").read_text(encoding="utf-8"))
    assert checkpoint["disposition"] == "needs_manual_review"
