"""Offline contracts for the private v4 persona prose review CLI."""

from __future__ import annotations

import json
import threading
import time
import typing as t
from pathlib import Path

import httpx
import polars as pl
import pytest

from danish_personas.generation.models import GenerationConfig
from danish_personas.generation.prose_review import ProseReviewResult
from danish_personas.io import sha256_file
from scripts import review_persona_prose as review

PERSONA = (
    "Dette er en syntetisk persona med neutral, hverdagsnær tekst. "
    "Personen beskrives uden navne, adresser eller andre direkte identifikatorer. "
    "Teksten bruges kun som testgrundlag for en privat gennemgang af faktuelle "
    "ændringer i attributter og er derfor bevidst generisk og ufarlig. "
    "Den fortsætter med almindelige detaljer om dagligdag, arbejde og fritid."
)


def test_429_retry_after_then_success_uses_header_delay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retry-After controls the bounded delay for transient rate limits."""
    paths = _write_inputs(tmp_path)
    calls = 0
    sleeps: list[float] = []

    class FakeBudget:
        def __init__(self, **_kwargs: object) -> None:
            pass

    def rate_limited_runner(**_kwargs: object) -> ProseReviewResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            request = httpx.Request("POST", "https://example.test")
            response = httpx.Response(
                429, headers={"Retry-After": "7"}, request=request
            )
            raise httpx.HTTPStatusError(
                "rate limited", request=request, response=response
            )
        return _result("unchanged_consistent")

    monkeypatch.setattr(review, "ProxyBudget", FakeBudget)
    monkeypatch.setattr(review.time, "sleep", sleeps.append)
    summary = review.run_review_campaign(
        paths=paths,
        execute=True,
        max_rows=None,
        workers=1,
        expected_original_sha256=sha256_file(paths.original),
        expected_candidate_sha256=sha256_file(paths.candidate),
        review_runner=rate_limited_runner,
    )

    assert calls == 2
    assert sleeps == [7.0]
    assert summary["attempted"] == 1
    assert summary["transient_retries"] == 1
    assert summary["processed"] == 1
    assert summary["pending"] == 0


def _result(
    disposition: t.Literal["patched", "unchanged_consistent", "needs_manual_review"],
) -> ProseReviewResult:
    return ProseReviewResult(
        disposition=disposition,
        original_text=PERSONA,
        proposed_text=PERSONA,
        changed_fraction=0.0,
        patches=(),
        unchanged_evidence=(),
        manual_review_reason=None,
        unchanged_consistent_note=None,
    )


def _write_inputs(
    tmp_path: Path,
    *,
    include_second_reviewable: bool = False,
    extra_reviewable_count: int = 0,
) -> review.ReviewPaths:
    original_rows: list[dict[str, object]] = [
        {
            "persona_id": "age-row",
            "persona": PERSONA,
            "age": 41,
            "job_title": "analytiker",
            "detailed_status_code": "A",
            "sexual_orientation": "not collected",
        },
        {
            "persona_id": "code-only-row",
            "persona": PERSONA,
            "age": 50,
            "job_title": "lærer",
            "detailed_status_code": "B",
            "sexual_orientation": "not collected",
        },
        {
            "persona_id": "privacy-row",
            "persona": PERSONA,
            "age": 31,
            "job_title": "kok",
            "detailed_status_code": "C",
            "sexual_orientation": "not collected",
        },
        {
            "persona_id": "ignored-row",
            "persona": PERSONA,
            "age": 60,
            "job_title": "mekaniker",
            "detailed_status_code": "D",
            "sexual_orientation": "not collected",
        },
    ]
    candidate_rows = [dict(row) for row in original_rows]
    candidate_rows[0]["age"] = 42
    candidate_rows[1]["detailed_status_code"] = "B2"
    candidate_rows[2]["age"] = 32
    candidate_rows[2]["sexual_orientation"] = "private"
    if include_second_reviewable:
        original_rows.append(
            {
                "persona_id": "job-row",
                "persona": PERSONA,
                "age": 44,
                "job_title": "analytiker",
                "detailed_status_code": "E",
                "sexual_orientation": "not collected",
            }
        )
        row = dict(original_rows[-1])
        row["job_title"] = "rådgiver"
        candidate_rows.append(row)
    for index in range(extra_reviewable_count):
        original_rows.append(
            {
                "persona_id": f"extra-review-row-{index:02d}",
                "persona": PERSONA,
                "age": 35 + index,
                "job_title": "analytiker",
                "detailed_status_code": "F",
                "sexual_orientation": "not collected",
            }
        )
        row = dict(original_rows[-1])
        row["age"] = 36 + index
        candidate_rows.append(row)

    original = tmp_path / "original.parquet"
    candidate = tmp_path / "candidate.parquet"
    pl.DataFrame(original_rows).write_parquet(original)
    pl.DataFrame(candidate_rows).write_parquet(candidate)
    triage = tmp_path / "triage.json"
    personas = {
        "age-row": {"classification": review.TRIAGE_CLASSIFICATION},
        "code-only-row": {"classification": review.TRIAGE_CLASSIFICATION},
        "privacy-row": {"classification": review.TRIAGE_CLASSIFICATION},
        "ignored-row": {"classification": "ok"},
    }
    if include_second_reviewable:
        personas["job-row"] = {"classification": review.TRIAGE_CLASSIFICATION}
    for index in range(extra_reviewable_count):
        personas[f"extra-review-row-{index:02d}"] = {
            "classification": review.TRIAGE_CLASSIFICATION
        }
    triage.write_text(
        json.dumps({"personas": personas, "counts": {}}, ensure_ascii=False),
        encoding="utf-8",
    )
    prompt = tmp_path / "persona-review-da.md"
    prompt.write_text(
        "Gennemgå kun de tilladte faktuelle ændringer.\n", encoding="utf-8"
    )
    registry = tmp_path / "models-store.json"
    registry.write_text("{}\n", encoding="utf-8")
    return review.ReviewPaths(
        original=original,
        candidate=candidate,
        triage=triage,
        prompt=prompt,
        output_dir=tmp_path / "persona-review-v4",
        registry=registry,
    )


def test_500_retry_then_success_counts_one_logical_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Transient server failures are retried within the same logical row."""
    paths = _write_inputs(tmp_path)
    calls = 0
    sleeps: list[float] = []

    class FakeBudget:
        def __init__(self, **_kwargs: object) -> None:
            pass

    def flaky_runner(**_kwargs: object) -> ProseReviewResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            request = httpx.Request("POST", "https://example.test")
            response = httpx.Response(500, request=request)
            raise httpx.HTTPStatusError(
                "server error", request=request, response=response
            )
        return _result("patched")

    monkeypatch.setattr(review, "ProxyBudget", FakeBudget)
    monkeypatch.setattr(review.time, "sleep", sleeps.append)
    summary = review.run_review_campaign(
        paths=paths,
        execute=True,
        max_rows=None,
        workers=1,
        expected_original_sha256=sha256_file(paths.original),
        expected_candidate_sha256=sha256_file(paths.candidate),
        review_runner=flaky_runner,
    )

    status = json.loads((paths.output_dir / "status.json").read_text())
    assert calls == 2
    assert sleeps == [1.0]
    assert summary["attempted"] == 1
    assert status["attempted"] == 1
    assert status["transient_retries"] == 1
    assert status["failed"] == 0
    assert status["processed"] == 1
    assert status["pending"] == 0


def test_delayed_runner_never_exceeds_worker_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bounded scheduling keeps delayed provider work within the worker limit."""
    paths = _write_inputs(
        tmp_path, include_second_reviewable=True, extra_reviewable_count=4
    )
    active = 0
    max_active = 0
    calls = 0
    lock = threading.Lock()

    class FakeBudget:
        def __init__(self, **_kwargs: object) -> None:
            pass

    def delayed_runner(**_kwargs: object) -> ProseReviewResult:
        nonlocal active, max_active, calls
        with lock:
            active += 1
            max_active = max(max_active, active)
            calls += 1
        time.sleep(0.01)
        with lock:
            active -= 1
        return _result("unchanged_consistent")

    monkeypatch.setattr(review, "ProxyBudget", FakeBudget)
    summary = review.run_review_campaign(
        paths=paths,
        execute=True,
        max_rows=None,
        workers=2,
        expected_original_sha256=sha256_file(paths.original),
        expected_candidate_sha256=sha256_file(paths.candidate),
        review_runner=delayed_runner,
    )

    assert calls == 6
    assert max_active <= 2
    assert summary["processed"] == 6
    assert summary["pending"] == 0


def test_dry_run_selects_triage_rows_from_real_allowed_differences(
    tmp_path: Path,
) -> None:
    """Dry-run labels no-change and privacy rows without provider or output I/O."""
    paths = _write_inputs(tmp_path)
    called = False

    def fake_runner(**_kwargs: object) -> ProseReviewResult:
        nonlocal called
        called = True
        return _result("patched")

    summary = review.run_review_campaign(
        paths=paths,
        execute=False,
        max_rows=None,
        workers=2,
        expected_original_sha256=sha256_file(paths.original),
        expected_candidate_sha256=sha256_file(paths.candidate),
        review_runner=fake_runner,
    )

    assert called is False
    assert summary["dry_run"] is True
    assert summary["total_triage_selected"] == 3
    assert summary["reviewable"] == 1
    assert summary["would_process"] == 1
    assert summary["no_changed_fact"] == 1
    assert summary["privacy_skipped"] == 1
    assert not paths.output_dir.exists()


def test_repeated_500_stops_without_invoking_later_pending_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exhausted transient retries stop without queueing further rows."""
    paths = _write_inputs(
        tmp_path, include_second_reviewable=True, extra_reviewable_count=2
    )
    calls = 0
    sleeps: list[float] = []

    class FakeBudget:
        def __init__(self, **_kwargs: object) -> None:
            pass

    def repeatedly_failing_runner(**_kwargs: object) -> ProseReviewResult:
        nonlocal calls
        calls += 1
        request = httpx.Request("POST", "https://example.test")
        response = httpx.Response(500, request=request)
        raise httpx.HTTPStatusError("server error", request=request, response=response)

    monkeypatch.setattr(review, "ProxyBudget", FakeBudget)
    monkeypatch.setattr(review.time, "sleep", sleeps.append)
    with pytest.raises(review.PersonaProseReviewError, match="stopped"):
        review.run_review_campaign(
            paths=paths,
            execute=True,
            max_rows=None,
            workers=1,
            expected_original_sha256=sha256_file(paths.original),
            expected_candidate_sha256=sha256_file(paths.candidate),
            review_runner=repeatedly_failing_runner,
        )

    status = json.loads((paths.output_dir / "status.json").read_text())
    assert calls == 5
    assert sleeps == [1.0, 2.0, 4.0, 8.0]
    assert status["failed"] == 1
    assert status["attempted"] == 1
    assert status["transient_retries"] == 4
    assert status["processed"] == 0
    assert status["pending"] == 4
    assert status["processed_persona_hashes"] == []


def test_run_uses_uncapped_budget_and_resumes_without_duplicate_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pilot prefixes use the same manifest and skip processed hashes on resume."""
    paths = _write_inputs(tmp_path, include_second_reviewable=True)
    calls: list[dict[str, dict[str, object]]] = []
    budget_kwargs: list[dict[str, object]] = []

    class FakeBudget:
        def __init__(self, **kwargs: object) -> None:
            budget_kwargs.append(kwargs)

    def fake_runner(
        *,
        row: dict[str, object],
        candidate_row: dict[str, object],
        changed_facts: dict[str, dict[str, object]],
        prompt: str,
        config: GenerationConfig,
        budget: object,
        checkpoint_path: Path,
        transport: httpx.BaseTransport,
    ) -> ProseReviewResult:
        del row, candidate_row, prompt, budget, transport
        calls.append(changed_facts)
        assert config.max_tokens is None
        assert config.maximum_http_attempts == 1
        assert checkpoint_path.parent.parent == paths.output_dir / "checkpoints"
        return _result("patched" if len(calls) == 1 else "unchanged_consistent")

    monkeypatch.setattr(review, "ProxyBudget", FakeBudget)
    expected_original = sha256_file(paths.original)
    expected_candidate = sha256_file(paths.candidate)

    first = review.run_review_campaign(
        paths=paths,
        execute=True,
        max_rows=1,
        workers=1,
        expected_original_sha256=expected_original,
        expected_candidate_sha256=expected_candidate,
        review_runner=fake_runner,
    )
    manifest = json.loads((paths.output_dir / "manifest.json").read_text())
    second = review.run_review_campaign(
        paths=paths,
        execute=True,
        max_rows=None,
        workers=1,
        expected_original_sha256=expected_original,
        expected_candidate_sha256=expected_candidate,
        review_runner=fake_runner,
    )

    assert len(calls) == 2
    assert calls[0] == {"age": {"old": 41, "new": 42}}
    assert calls[1] == {"job_title": {"old": "analytiker", "new": "rådgiver"}}
    assert first["pending"] == 1
    assert second["pending"] == 0
    assert second["patched"] == 1
    assert second["unchanged_consistent"] == 1
    assert budget_kwargs[-1]["uncapped"] is True
    assert "max_rows" not in manifest
    assert (paths.output_dir.stat().st_mode & 0o777) == 0o700
    assert ((paths.output_dir / "status.json").stat().st_mode & 0o777) == 0o600

    paths.prompt.write_text("Prompten er ændret.\n", encoding="utf-8")
    with pytest.raises(review.PersonaProseReviewError, match="pins do not match"):
        review.run_review_campaign(
            paths=paths,
            execute=True,
            max_rows=None,
            workers=1,
            expected_original_sha256=expected_original,
            expected_candidate_sha256=expected_candidate,
            review_runner=fake_runner,
        )


def test_source_hash_validation_fails_closed(tmp_path: Path) -> None:
    """Changed original or v4 candidate sources are rejected before provider setup."""
    paths = _write_inputs(tmp_path)

    with pytest.raises(review.PersonaProseReviewError, match="original parquet"):
        review.run_review_campaign(
            paths=paths,
            execute=False,
            max_rows=None,
            workers=1,
            expected_original_sha256="0" * 64,
            expected_candidate_sha256=sha256_file(paths.candidate),
            review_runner=lambda **_kwargs: _result("patched"),
        )
    with pytest.raises(review.PersonaProseReviewError, match="candidate parquet"):
        review.run_review_campaign(
            paths=paths,
            execute=False,
            max_rows=None,
            workers=1,
            expected_original_sha256=sha256_file(paths.original),
            expected_candidate_sha256="0" * 64,
            review_runner=lambda **_kwargs: _result("patched"),
        )


def test_unsafe_error_stops_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Privacy, ledger, checkpoint, and protocol failures remain hard stops."""
    paths = _write_inputs(tmp_path)
    calls = 0
    sleeps: list[float] = []

    class FakeBudget:
        def __init__(self, **_kwargs: object) -> None:
            pass

    def unsafe_runner(**_kwargs: object) -> ProseReviewResult:
        nonlocal calls
        calls += 1
        raise review.ProxyReviewError("checkpoint state is unsafe")

    monkeypatch.setattr(review, "ProxyBudget", FakeBudget)
    monkeypatch.setattr(review.time, "sleep", sleeps.append)
    with pytest.raises(review.PersonaProseReviewError, match="stopped"):
        review.run_review_campaign(
            paths=paths,
            execute=True,
            max_rows=None,
            workers=1,
            expected_original_sha256=sha256_file(paths.original),
            expected_candidate_sha256=sha256_file(paths.candidate),
            review_runner=unsafe_runner,
        )

    status = json.loads((paths.output_dir / "status.json").read_text())
    assert calls == 1
    assert sleeps == []
    assert status["failed"] == 1
    assert status["attempted"] == 1
    assert status["transient_retries"] == 0
    assert status["processed"] == 0
    assert status["pending"] == 1
    assert status["processed_persona_hashes"] == []
