"""Offline tests for the private local proxy patch verifier."""

from __future__ import annotations

import hashlib
import json
import stat
import threading
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

import danish_personas.generation.proxy_budget as proxy_budget
import danish_personas.generation.proxy_patch_verifier as patch_verifier
from danish_personas.generation.models import GenerationConfig
from danish_personas.generation.proxy_budget import JSONValue, ProxyBudget
from danish_personas.generation.proxy_patch_verifier import (
    ProxyPatchVerificationError,
    run_proxy_patch_verification,
)
from danish_personas.release.prose_patch_verification import ProsePatchSecondReview

_PROMPT = "Kontrollér en minimal rettelse."
_FIRST_SHA = "a" * 64
_OLD_EXCERPT = "ugift"
_NEW_EXCERPT = "gift"


def test_accepts_and_resumes_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persist an accept verdict and revalidate it without another request."""
    requests: list[httpx.Request] = []
    result = run_proxy_patch_verification(
        row=_row(),
        candidate_row=_candidate_row(),
        changed_facts=_facts(),
        proposed_text=_proposed_text(),
        patches=_patches(),
        first_checkpoint_sha256=_FIRST_SHA,
        prompt=_PROMPT,
        config=_config(),
        budget=_budget(tmp_path=tmp_path, monkeypatch=monkeypatch),
        checkpoint_path=tmp_path / "checkpoints" / "verify.json",
        transport=_transport(_review("accept"), requests),
    )

    assert result.accepted is True
    checkpoint = tmp_path / "checkpoints" / "verify.json"
    assert checkpoint.stat().st_mode & 0o777 == 0o600
    assert checkpoint.parent.stat().st_mode & 0o777 == 0o700

    resumed = run_proxy_patch_verification(
        row=_row(),
        candidate_row=_candidate_row(),
        changed_facts=_facts(),
        proposed_text=_proposed_text(),
        patches=_patches(),
        first_checkpoint_sha256=_FIRST_SHA,
        prompt=_PROMPT,
        config=_config(),
        budget=_budget(tmp_path=tmp_path, monkeypatch=monkeypatch),
        checkpoint_path=checkpoint,
        transport=_failing_transport(),
    )

    assert resumed == result
    assert len(requests) == 1


def _budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProxyBudget:
    monkeypatch.setattr(proxy_budget, "USER_BUDGET_PATH", tmp_path / "budget.jsonl")
    monkeypatch.setattr(
        proxy_budget, "USER_UNCAPPED_BUDGET_PATH", tmp_path / "uncapped.jsonl"
    )
    monkeypatch.setattr(
        proxy_budget,
        "USER_PATCH_VERIFICATION_BUDGET_PATH",
        tmp_path / "patch-verification.jsonl",
    )
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
    ProxyBudget(
        ledger_path=tmp_path / "ignored.jsonl",
        registry_path=registry,
        campaign="synthetic-patch-verification-test",
        source_hash="b" * 64,
        prompt_hash=hashlib.sha256(_PROMPT.encode()).hexdigest(),
        schema_hash=hashlib.sha256(_canonical_schema()).hexdigest(),
        cap_usd=Decimal("1"),
    )
    return ProxyBudget(
        ledger_path=tmp_path / "ignored.jsonl",
        registry_path=registry,
        campaign="synthetic-patch-verification-test",
        source_hash="b" * 64,
        prompt_hash=hashlib.sha256(_PROMPT.encode()).hexdigest(),
        schema_hash=hashlib.sha256(_canonical_schema()).hexdigest(),
        cap_usd=Decimal("1"),
        uncapped=True,
        uncapped_purpose="patch_verification",
    )


def _canonical_schema() -> bytes:
    return json.dumps(
        ProsePatchSecondReview.provider_json_schema(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _candidate_row() -> dict[str, object]:
    candidate = dict(_row())
    candidate["marital_status"] = "married"
    return candidate


def _row() -> dict[str, object]:
    return {
        "persona_id": "private-id",
        "record_id": "private-id",
        "identity_sidecar": {"origin_code": "private-origin-code"},
        "origin_country": "private English origin",
        "origin_country_da": "private Danish origin",
        "sexual_orientation": "private-orientation",
        "persona": "Anna er ugift. " + "Dette er en syntetisk person. " * 12,
        "marital_status": "single",
        "age": 41,
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
        "prompt": Path("config/persona-verify-da.md"),
        "origin_label_contract": Path("config/folk2-ieland-labels-da.yaml"),
    }
    values.update(overrides)
    return GenerationConfig.model_validate(values)


def _facts() -> dict[str, dict[str, object]]:
    return {"marital_status": {"old": "single", "new": "married"}}


def _failing_transport() -> httpx.MockTransport:
    def respond(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("transport should not be called")

    return httpx.MockTransport(respond)


def _patches() -> list[dict[str, str]]:
    return [{"old_excerpt": _OLD_EXCERPT, "new_excerpt": _NEW_EXCERPT}]


def _proposed_text() -> str:
    return str(_row()["persona"]).replace(_OLD_EXCERPT, _NEW_EXCERPT, 1)


def _review(verdict: str, facts: dict[str, dict[str, object]] | None = None) -> str:
    if verdict == "accept":
        reasons: list[str] = []
        evidence = [
            {
                "field": field,
                "status": "corrected",
                "original_quote": _OLD_EXCERPT,
                "proposed_quote": _NEW_EXCERPT,
            }
            for field in (facts or _facts())
        ]
    elif verdict == "reject":
        reasons = ["fact_mismatch"]
        evidence = []
    else:
        reasons = ["ambiguity"]
        evidence = []
    return json.dumps(
        {"verdict": verdict, "reasons": reasons, "fact_evidence": evidence},
        ensure_ascii=False,
    )


def _transport(
    response_content: str, seen: list[httpx.Request], events: list[str] | None = None
) -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if events is not None:
            events.append("network")
        assert "max_tokens" not in json.loads(request.content)
        return _completion_response(content=response_content)

    return httpx.MockTransport(respond)


def _completion_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "response-1",
            "model": "gpt-6-luna",
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        },
    )


def test_adds_verified_null_detail_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Include only a verified broad marital category for null fine detail."""
    row = _row()
    row["marital_status"] = "divorced"
    row["legal_status_detail"] = "separated"
    facts: dict[str, dict[str, object]] = {
        "legal_status_detail": {"old": "separated", "new": None}
    }
    candidate = dict(row)
    candidate["legal_status_detail"] = None
    requests: list[httpx.Request] = []

    run_proxy_patch_verification(
        row=row,
        candidate_row=candidate,
        changed_facts=facts,
        proposed_text=_proposed_text(),
        patches=_patches(),
        first_checkpoint_sha256=_FIRST_SHA,
        prompt=_PROMPT,
        config=_config(),
        budget=_budget(tmp_path=tmp_path, monkeypatch=monkeypatch),
        checkpoint_path=tmp_path / "null-detail.json",
        transport=_transport(_review("reject", facts=facts), requests),
    )

    body = requests[0].content.decode()
    assert "target_marital_category_da" in body
    assert "skilt" in body
    assert "separated" in body


def test_failed_http_retry_gets_unique_reservations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reserve each retried attempt with a fresh durable request ID."""
    budget = _budget(tmp_path=tmp_path, monkeypatch=monkeypatch)
    request_ids: list[str] = []
    events: list[str] = []
    original_reserve = budget.reserve_attempt

    def reserve(request_id: str, request: dict[str, JSONValue]) -> Decimal:
        request_ids.append(request_id)
        events.append("reserved")
        return original_reserve(request_id, request)

    monkeypatch.setattr(budget, "reserve_attempt", reserve)
    run_proxy_patch_verification(
        row=_row(),
        candidate_row=_candidate_row(),
        changed_facts=_facts(),
        proposed_text=_proposed_text(),
        patches=_patches(),
        first_checkpoint_sha256=_FIRST_SHA,
        prompt=_PROMPT,
        config=_config(maximum_http_attempts=2),
        budget=budget,
        checkpoint_path=tmp_path / "retry.json",
        transport=_retry_transport(events),
    )

    assert events == ["reserved", "network", "reserved", "network"]
    assert len(request_ids) == 2
    assert len(set(request_ids)) == 2


def _retry_transport(events: list[str]) -> httpx.MockTransport:
    calls = 0

    def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        events.append("network")
        if calls == 1:
            return httpx.Response(500, json={"error": "retry"})
        return _completion_response(content=_review("reject"))

    return httpx.MockTransport(respond)


def test_invalid_completion_is_accounted_then_checkpointed_manual(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Record usage for schema-invalid content before local abstention."""
    budget = _budget(tmp_path=tmp_path, monkeypatch=monkeypatch)
    events: list[str] = []
    original_record_usage = budget.record_usage

    def record_usage(
        request_id: str, *, input_tokens: int, output_tokens: int, response_sha256: str
    ) -> None:
        events.append("usage")
        original_record_usage(
            request_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            response_sha256=response_sha256,
        )

    monkeypatch.setattr(budget, "record_usage", record_usage)
    result = run_proxy_patch_verification(
        row=_row(),
        candidate_row=_candidate_row(),
        changed_facts=_facts(),
        proposed_text=_proposed_text(),
        patches=_patches(),
        first_checkpoint_sha256=_FIRST_SHA,
        prompt=_PROMPT,
        config=_config(),
        budget=budget,
        checkpoint_path=tmp_path / "invalid.json",
        transport=_transport("not-json", [], events),
    )

    assert events == ["network", "usage"]
    assert result.accepted is False
    assert result.review_verdict == "needs_manual_review"
    assert result.reasons == ["invalid_quote_evidence"]
    checkpoint_text = (tmp_path / "invalid.json").read_text(encoding="utf-8")
    assert "not-json" not in checkpoint_text


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
        self: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False
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
            patch_verifier._prepare_checkpoint_parent(parent=parent)
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


def test_prepare_checkpoint_parent_rejects_public_prefix_mode(tmp_path: Path) -> None:
    """Reject an existing hash-prefix directory with group/world permissions."""
    parent = tmp_path / "checkpoints" / "ab"
    parent.mkdir(mode=0o700, parents=True)
    parent.chmod(0o755)

    with pytest.raises(ProxyPatchVerificationError, match="mode 0700"):
        patch_verifier._prepare_checkpoint_parent(parent=parent)


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

    with pytest.raises(ProxyPatchVerificationError, match="real directory"):
        patch_verifier._prepare_checkpoint_parent(parent=parent)


def test_privacy_guard_omits_private_fields_and_reserves_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Send only the bounded verifier payload, never row identifiers."""
    budget = _budget(tmp_path=tmp_path, monkeypatch=monkeypatch)
    requests: list[httpx.Request] = []
    events: list[str] = []
    original_reserve = budget.reserve_attempt

    def reserve(request_id: str, request: dict[str, JSONValue]) -> Decimal:
        events.append("reserved")
        return original_reserve(request_id, request)

    monkeypatch.setattr(budget, "reserve_attempt", reserve)
    run_proxy_patch_verification(
        row=_row(),
        candidate_row=_candidate_row(),
        changed_facts=_facts(),
        proposed_text=_proposed_text(),
        patches=_patches(),
        first_checkpoint_sha256=_FIRST_SHA,
        prompt=_PROMPT,
        config=_config(),
        budget=budget,
        checkpoint_path=tmp_path / "privacy.json",
        transport=_transport(_review("reject"), requests, events),
    )

    body = requests[0].content.decode()
    assert events[:2] == ["reserved", "network"]
    assert "private-id" not in body
    assert "origin_country" not in body
    assert "identity_sidecar" not in body
    assert "sexual_orientation" not in body
    assert "candidate_row" not in body
    assert "original_persona" in body
    assert "proposed_persona" in body


@pytest.mark.parametrize(
    ("verdict", "expected_reason"),
    [("reject", "fact_mismatch"), ("needs_manual_review", "ambiguity")],
)
def test_reject_and_abstain_checkpoint_privately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verdict: str, expected_reason: str
) -> None:
    """Store every provider verdict class for deterministic resume."""
    result = run_proxy_patch_verification(
        row=_row(),
        candidate_row=_candidate_row(),
        changed_facts=_facts(),
        proposed_text=_proposed_text(),
        patches=_patches(),
        first_checkpoint_sha256=_FIRST_SHA,
        prompt=_PROMPT,
        config=_config(),
        budget=_budget(tmp_path=tmp_path, monkeypatch=monkeypatch),
        checkpoint_path=tmp_path / f"{verdict}.json",
        transport=_transport(_review(verdict), []),
    )

    assert result.accepted is False
    assert result.review_verdict == verdict
    assert result.reasons == [expected_reason]
    assert (tmp_path / f"{verdict}.json").stat().st_mode & 0o777 == 0o600


def test_rejects_sensitive_text_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scan proposed prose and excerpts, not only structured fact values."""
    with pytest.raises(ProxyPatchVerificationError, match="sensitive-identity"):
        run_proxy_patch_verification(
            row=_row(),
            candidate_row=_candidate_row(),
            changed_facts=_facts(),
            proposed_text=_proposed_text() + " Seksuel orientering.",
            patches=_patches(),
            first_checkpoint_sha256=_FIRST_SHA,
            prompt=_PROMPT,
            config=_config(),
            budget=_budget(tmp_path=tmp_path, monkeypatch=monkeypatch),
            checkpoint_path=tmp_path / "sensitive.json",
            transport=_failing_transport(),
        )


def test_rejects_stale_checkpoint_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse a checkpoint bound to a different proposed text."""
    checkpoint = tmp_path / "stale.json"
    run_proxy_patch_verification(
        row=_row(),
        candidate_row=_candidate_row(),
        changed_facts=_facts(),
        proposed_text=_proposed_text(),
        patches=_patches(),
        first_checkpoint_sha256=_FIRST_SHA,
        prompt=_PROMPT,
        config=_config(),
        budget=_budget(tmp_path=tmp_path, monkeypatch=monkeypatch),
        checkpoint_path=checkpoint,
        transport=_transport(_review("accept"), []),
    )

    with pytest.raises(ProxyPatchVerificationError, match="changed"):
        run_proxy_patch_verification(
            row=_row(),
            candidate_row=_candidate_row(),
            changed_facts=_facts(),
            proposed_text=_proposed_text().replace("gift", "nygift", 1),
            patches=[{"old_excerpt": _OLD_EXCERPT, "new_excerpt": "nygift"}],
            first_checkpoint_sha256=_FIRST_SHA,
            prompt=_PROMPT,
            config=_config(),
            budget=_budget(tmp_path=tmp_path, monkeypatch=monkeypatch),
            checkpoint_path=checkpoint,
            transport=_failing_transport(),
        )
