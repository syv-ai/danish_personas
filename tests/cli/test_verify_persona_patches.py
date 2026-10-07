"""Offline contracts for the private v4 patch-verification CLI."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import polars as pl
import pytest

from danish_personas.generation.models import GenerationConfig
from danish_personas.generation.prose_review import validate_prose_review
from danish_personas.generation.proxy_patch_runner import _validated_input
from danish_personas.generation.proxy_review_runner import (
    _payload_with_null_detail_context,
    _save_checkpoint,
)
from danish_personas.io import canonical_json, sha256_file, sha256_text
from danish_personas.release.prose_patch_verification import (
    ProsePatchVerificationResult,
)
from scripts import build_prose_review_v4_dashboard as dashboard
from scripts import verify_persona_patches as verify

PERSONA = (
    "Dette er en syntetisk persona med neutral og hverdagsnær tekst. "
    "Personen omtales som en rolig borger med alder 41 i en generisk "
    "beskrivelse uden navne, adresser eller andre direkte identifikatorer. "
    "Teksten bruges kun som testgrundlag for privat kontrol af faktuelle "
    "ændringer i attributter og arbejder som rådgiver. Den fortsætter med "
    "almindelige detaljer om dagligdag, fritid, vaner og lokale aktiviteter, så "
    "den er lang nok til den afgrænsede proxykontrakt uden at afsløre noget."
)


def test_dry_run_revalidates_first_pass_without_provider_or_outputs(
    tmp_path: Path,
) -> None:
    """Dry-run exposes available patched rows but does not call the verifier."""
    paths = _write_fixture(tmp_path)
    called = False

    def fake_runner(**_kwargs: object) -> ProsePatchVerificationResult:
        nonlocal called
        called = True
        return _verification_result(accepted=True)

    summary = verify.run_patch_verification_campaign(
        paths=paths, execute=False, max_rows=None, workers=2, verify_runner=fake_runner
    )

    assert called is False
    assert summary["dry_run"] is True
    assert summary["available"] == 1
    assert summary["would_process"] == 1
    assert summary["manual"] == 1
    assert summary["unchanged_consistent"] == 1
    assert summary["pending"] == 1
    assert summary["accepted_is_provisional"] is True
    assert "provisional" in str(summary["provisional_notice"])
    assert not paths.output_dir.exists()


def test_h90_dry_run_requires_h90_first_pass_manifest(tmp_path: Path) -> None:
    """The H90 verifier refuses v4 first-pass manifests and accepts H90 pins."""
    h90_paths = _write_fixture(
        tmp_path / "h90", budget_purpose=verify.H90_BUDGET_PURPOSE
    )
    v4_paths = _write_fixture(tmp_path / "v4")

    h90_summary = verify.run_patch_verification_campaign(
        paths=h90_paths,
        execute=False,
        max_rows=None,
        workers=1,
        verify_runner=_unused_runner,
    )

    assert h90_summary["dry_run"] is True
    assert h90_summary["campaign"] == verify.H90_CAMPAIGN
    assert h90_summary["available"] == 1
    with pytest.raises(dashboard.ProseReviewV4DashboardError, match="campaign"):
        verify.run_patch_verification_campaign(
            paths=verify.VerifyPaths(
                original=h90_paths.original,
                candidate=h90_paths.candidate,
                triage=h90_paths.triage,
                first_prompt=h90_paths.first_prompt,
                verify_prompt=h90_paths.verify_prompt,
                registry=h90_paths.registry,
                first_status=v4_paths.first_status,
                first_manifest=v4_paths.first_manifest,
                first_checkpoint_root=v4_paths.first_checkpoint_root,
                output_dir=h90_paths.output_dir,
                budget_purpose=verify.H90_BUDGET_PURPOSE,
            ),
            execute=False,
            max_rows=None,
            workers=1,
            verify_runner=_unused_runner,
        )


def test_h90_private_inputs_fail_closed_before_provider(tmp_path: Path) -> None:
    """H90 verification refuses shared source files before provider access."""
    paths = _write_fixture(tmp_path, budget_purpose=verify.H90_BUDGET_PURPOSE)
    paths.candidate.chmod(0o644)

    with pytest.raises(verify.PatchVerificationCampaignError, match="private"):
        verify.run_patch_verification_campaign(
            paths=paths,
            execute=False,
            max_rows=None,
            workers=1,
            verify_runner=_unused_runner,
        )


def _unused_runner(**_kwargs: object) -> ProsePatchVerificationResult:
    raise AssertionError("provider runner must not be called")


def _verification_result(*, accepted: bool) -> ProsePatchVerificationResult:
    return ProsePatchVerificationResult(
        accepted=accepted,
        review_verdict="accept" if accepted else "reject",
        reasons=[] if accepted else ["fact_mismatch"],
        changed_fraction=0.01,
        changed_characters=8,
        patch_count=1,
        original_checkpoint_sha256="a" * 64,
        provisional=True,
        requires_later_release_gate=True,
        fact_evidence=[],
    )


class FixturePaths(verify.VerifyPaths):
    """Typed alias for synthetic verification paths."""


def _write_fixture(
    tmp_path: Path,
    *,
    include_growth_row: bool = False,
    budget_purpose: verify.BudgetPurpose = verify.DEFAULT_BUDGET_PURPOSE,
) -> FixturePaths:
    root = tmp_path / "private"
    first_dir = root / "persona-review-v4"
    output_dir = root / "persona-verify-v4"
    if budget_purpose == verify.H90_BUDGET_PURPOSE:
        first_dir = root / "persona-review-h90-v5"
        output_dir = root / "persona-verify-h90-v5"
    checkpoints = first_dir / "checkpoints"
    checkpoints.mkdir(parents=True, mode=0o700)
    original = root / "original.parquet"
    candidate = root / "candidate.parquet"
    triage = root / "triage.json"
    first_prompt = root / "persona-review-da.md"
    verify_prompt = root / "persona-verify-da.md"
    registry = root / "models-store.json"

    rows = [
        _row(persona_id="patched-row", age=41, job_title="rådgiver"),
        _row(persona_id="manual-row", age=50, job_title="lærer"),
        _row(persona_id="unchanged-row", age=60, job_title="mekaniker"),
    ]
    candidates = [dict(row) for row in rows]
    candidates[0]["age"] = 42
    candidates[1]["age"] = 51
    candidates[2]["age"] = 61
    if include_growth_row:
        rows.append(_row(persona_id="growth-row", age=44, job_title="rådgiver"))
        candidate_row = dict(rows[-1])
        candidate_row["job_title"] = "analytiker"
        candidates.append(candidate_row)
    pl.DataFrame(rows).write_parquet(original)
    pl.DataFrame(candidates).write_parquet(candidate)
    if budget_purpose == verify.H90_BUDGET_PURPOSE:
        candidate.with_suffix(".report.json").write_text(
            json.dumps({"candidate_h90_v5_sha256": sha256_file(candidate)}) + "\n",
            encoding="utf-8",
        )
    triage.write_text('{"personas": {}, "counts": {}}\n', encoding="utf-8")
    if budget_purpose == verify.H90_BUDGET_PURPOSE:
        for private_path in (
            original,
            candidate,
            triage,
            candidate.with_suffix(".report.json"),
        ):
            private_path.chmod(0o600)
    first_prompt.write_text("Gennemgå kun tilladte ændringer.\n", encoding="utf-8")
    verify_prompt.write_text("Kontrollér patchen konservativt.\n", encoding="utf-8")
    registry.write_text('{"models": []}\n', encoding="utf-8")

    paths = FixturePaths(
        original=original,
        candidate=candidate,
        triage=triage,
        first_prompt=first_prompt,
        verify_prompt=verify_prompt,
        registry=registry,
        first_status=first_dir / "status.json",
        first_manifest=first_dir / "manifest.json",
        first_checkpoint_root=first_dir,
        output_dir=output_dir,
        budget_purpose=budget_purpose,
    )
    manifest = _first_manifest(paths=paths)
    _write_json(paths.first_manifest, manifest)
    processed_ids = ("patched-row", "manual-row", "unchanged-row")
    for persona_id, disposition in zip(
        processed_ids,
        ("patched", "needs_manual_review", "unchanged_consistent"),
        strict=True,
    ):
        _write_first_checkpoint(
            paths=paths,
            manifest=manifest,
            persona_id=persona_id,
            disposition=disposition,
        )
    _write_first_status(paths=paths, manifest=manifest, persona_ids=processed_ids)
    return paths


def _first_manifest(*, paths: FixturePaths) -> dict[str, dashboard.JSONValue]:
    inputs: dict[str, dashboard.JSONValue] = {
        "triage": sha256_file(paths.triage),
        "prompt": sha256_file(paths.first_prompt),
        "registry": sha256_file(paths.registry),
        "schema": sha256_text(
            canonical_json(dashboard.ProseReviewResponse.provider_json_schema())
        ),
    }
    campaign = dashboard.CAMPAIGN
    if paths.budget_purpose == verify.H90_BUDGET_PURPOSE:
        campaign = dashboard.H90_CAMPAIGN
        inputs.update(
            {
                "original_v4": sha256_file(paths.original),
                "candidate_h90_v5": sha256_file(paths.candidate),
                "h90_report": sha256_file(paths.candidate.with_suffix(".report.json")),
            }
        )
    else:
        inputs.update(
            {
                "original": sha256_file(paths.original),
                "candidate_v4": sha256_file(paths.candidate),
            }
        )
    return {
        "version": 1,
        "campaign": campaign,
        "inputs": inputs,
        "model": dashboard.MODEL,
        "base_url": dashboard.BASE_URL,
        "allowed_facts": sorted(dashboard._ALLOWED_FACTS),
    }


def _row(*, persona_id: str, age: int, job_title: str) -> dict[str, object]:
    return {
        "persona_id": persona_id,
        "persona": PERSONA,
        "age": age,
        "job_title": job_title,
        "sexual_orientation": "not collected",
    }


def _write_first_checkpoint(
    *,
    paths: FixturePaths,
    manifest: dict[str, dashboard.JSONValue],
    persona_id: str,
    disposition: str,
) -> None:
    original = _row_by_id(path=paths.original, persona_id=persona_id)
    candidate = _row_by_id(path=paths.candidate, persona_id=persona_id)
    changed_facts = dashboard._changed_facts_mapping(
        original=original, candidate=candidate
    )
    prompt = paths.first_prompt.read_text(encoding="utf-8")
    original_text, pre_context_payload = _validated_input(
        original, candidate, changed_facts, None, None, prompt
    )
    payload = _payload_with_null_detail_context(
        payload=pre_context_payload,
        row=original,
        candidate_row=candidate,
        changed_facts=changed_facts,
    )
    binding = dashboard._checkpoint_binding(
        original=original,
        candidate=candidate,
        changed_facts=changed_facts,
        original_text=original_text,
        prompt=prompt,
        manifest=manifest,
        payload=payload,
        campaign=str(manifest["campaign"]),
    )
    response = _first_response(disposition=disposition, changed_facts=changed_facts)
    result = validate_prose_review(
        original_text=original_text,
        changed_facts=changed_facts,
        response=response,
        verified_context=None,
    )
    persona_hash = sha256_text(persona_id)
    checkpoint_path = paths.first_checkpoint_root / "checkpoints" / persona_hash[:2]
    checkpoint_path.mkdir(parents=True, mode=0o700)
    _save_checkpoint(
        path=checkpoint_path / f"{persona_hash}.json", binding=binding, result=result
    )


def _first_response(
    *, disposition: str, changed_facts: dict[str, dict[str, object]]
) -> dict[str, object]:
    if disposition == "patched":
        if "job_title" in changed_facts:
            return {
                "disposition": "patched",
                "patches": [{"old_excerpt": "rådgiver", "new_excerpt": "analytiker"}],
                "unchanged_evidence": [],
                "manual_review_reason": None,
            }
        return {
            "disposition": "patched",
            "patches": [{"old_excerpt": "alder 41", "new_excerpt": "alder 42"}],
            "unchanged_evidence": [],
            "manual_review_reason": None,
        }
    if disposition == "needs_manual_review":
        return {
            "disposition": "needs_manual_review",
            "patches": [],
            "unchanged_evidence": [],
            "manual_review_reason": "ambiguous",
        }
    field = next(iter(changed_facts))
    return {
        "disposition": "unchanged_consistent",
        "patches": [],
        "unchanged_evidence": [
            {"field": field, "kind": "fact_not_stated", "quote": ""}
        ],
        "manual_review_reason": None,
    }


def _row_by_id(*, path: Path, persona_id: str) -> dict[str, dashboard.JSONValue]:
    for row in pl.read_parquet(path).to_dicts():
        if row["persona_id"] == persona_id:
            return row
    raise AssertionError(f"missing row {persona_id}")


def _write_first_status(
    *,
    paths: FixturePaths,
    manifest: dict[str, dashboard.JSONValue],
    persona_ids: tuple[str, ...],
) -> None:
    dispositions = {
        "patched-row": "patched",
        "manual-row": "needs_manual_review",
        "unchanged-row": "unchanged_consistent",
        "growth-row": "patched",
    }
    processed: list[dashboard.JSONValue] = [
        sha256_text(persona_id) for persona_id in persona_ids
    ]
    status: dict[str, dashboard.JSONValue] = {
        "version": 1,
        "manifest": manifest,
        "reviewable": len(persona_ids),
        "patched": sum(
            dispositions[persona_id] == "patched" for persona_id in persona_ids
        ),
        "unchanged_consistent": sum(
            dispositions[persona_id] == "unchanged_consistent"
            for persona_id in persona_ids
        ),
        "needs_manual_review": sum(
            dispositions[persona_id] == "needs_manual_review"
            for persona_id in persona_ids
        ),
        "failed": 0,
        "attempted": len(persona_ids),
        "processed": len(persona_ids),
        "pending": 0,
        "processed_persona_hashes": processed,
    }
    _write_json(paths.first_status, status)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    content = json.dumps(value, ensure_ascii=False, sort_keys=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)


def test_follow_historical_failed_attempts_allow_terminal_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Historical first-pass failures do not block recovered terminal rows."""
    paths = _write_fixture(tmp_path)
    status = json.loads(paths.first_status.read_text(encoding="utf-8"))
    status["failed"] = 2
    status["attempted"] = 5
    _write_json(paths.first_status, status)
    calls = 0
    budget_calls = 0

    class FakeBudget:
        def __init__(self, **_kwargs: object) -> None:
            nonlocal budget_calls
            budget_calls += 1

    def fake_runner(**_kwargs: object) -> ProsePatchVerificationResult:
        nonlocal calls
        calls += 1
        return _verification_result(accepted=True)

    monkeypatch.setattr(verify, "ProxyBudget", FakeBudget)

    summary = verify.follow_patch_verification_campaign(
        paths=paths, execute=True, max_rows=None, workers=1, verify_runner=fake_runner
    )

    assert calls == 1
    assert budget_calls == 1
    assert summary["processed"] == 1
    assert summary["pending"] == 0
    assert summary["first_pass_pending"] == 0
    assert summary["first_pass_patched"] == 1
    assert summary["follow_first_pass"] is True


def test_follow_manifest_change_fails_before_idle_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Follow mode revalidates first-pass pins after each sleep."""
    paths = _write_fixture(tmp_path)
    _set_first_status_progress(paths=paths, reviewable=4, pending=1)
    calls = 0
    now = 0.0

    class FakeBudget:
        def __init__(self, **_kwargs: object) -> None:
            pass

    def fake_runner(**_kwargs: object) -> ProsePatchVerificationResult:
        nonlocal calls
        calls += 1
        return _verification_result(accepted=True)

    def fake_sleep(seconds: float) -> None:
        nonlocal now
        now += seconds
        manifest = json.loads(paths.first_manifest.read_text(encoding="utf-8"))
        manifest["campaign"] = "changed"
        _write_json(paths.first_manifest, manifest)

    monkeypatch.setattr(verify, "ProxyBudget", FakeBudget)
    monkeypatch.setattr(verify.time, "monotonic", lambda: now)
    monkeypatch.setattr(verify.time, "sleep", fake_sleep)

    with pytest.raises(Exception, match="manifest"):
        verify.follow_patch_verification_campaign(
            paths=paths,
            execute=True,
            max_rows=None,
            workers=1,
            verify_runner=fake_runner,
            poll_seconds=300.0,
            stall_seconds=600.0,
        )

    assert calls == 1


def _set_first_status_progress(
    *, paths: FixturePaths, reviewable: int, pending: int, failed: int = 0
) -> None:
    status = json.loads(paths.first_status.read_text(encoding="utf-8"))
    status["reviewable"] = reviewable
    status["pending"] = pending
    status["failed"] = failed
    _write_json(paths.first_status, status)


def test_follow_polls_growth_until_first_pass_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Follow mode waits for growth and exits only after first-pass completion."""
    paths = _write_fixture(tmp_path, include_growth_row=True)
    _set_first_status_progress(paths=paths, reviewable=4, pending=1)
    calls: list[Path] = []
    budget_calls = 0
    sleeps: list[float] = []
    now = 0.0

    class FakeBudget:
        def __init__(self, **_kwargs: object) -> None:
            nonlocal budget_calls
            budget_calls += 1

    def fake_runner(**kwargs: object) -> ProsePatchVerificationResult:
        checkpoint_path = kwargs["checkpoint_path"]
        assert isinstance(checkpoint_path, Path)
        calls.append(checkpoint_path)
        return _verification_result(accepted=True)

    def fake_sleep(seconds: float) -> None:
        nonlocal now
        sleeps.append(seconds)
        now += seconds
        if len(sleeps) == 1:
            _append_growth_checkpoint(paths)

    monkeypatch.setattr(verify, "ProxyBudget", FakeBudget)
    monkeypatch.setattr(verify.time, "monotonic", lambda: now)
    monkeypatch.setattr(verify.time, "sleep", fake_sleep)

    summary = verify.follow_patch_verification_campaign(
        paths=paths,
        execute=True,
        max_rows=None,
        workers=1,
        verify_runner=fake_runner,
        poll_seconds=300.0,
        stall_seconds=5_400.0,
    )

    assert sleeps == [300.0]
    assert len(calls) == 2
    assert budget_calls == 2
    assert summary["processed"] == 2
    assert summary["pending"] == 0
    assert summary["first_pass_pending"] == 0
    assert summary["first_pass_patched"] == 2
    assert summary["follow_first_pass"] is True


def _append_growth_checkpoint(paths: FixturePaths) -> None:
    manifest = _first_manifest(paths=paths)
    _write_first_checkpoint(
        paths=paths, manifest=manifest, persona_id="growth-row", disposition="patched"
    )
    _write_first_status(
        paths=paths,
        manifest=manifest,
        persona_ids=("patched-row", "manual-row", "unchanged-row", "growth-row"),
    )


def test_follow_requires_run_and_consistent_first_pass(tmp_path: Path) -> None:
    """Follow mode never starts in dry-run or with inconsistent status counts."""
    paths = _write_fixture(tmp_path)

    def fake_runner(**_kwargs: object) -> ProsePatchVerificationResult:
        raise AssertionError("provider must not be called")

    with pytest.raises(verify.PatchVerificationCampaignError, match="requires --run"):
        verify.follow_patch_verification_campaign(
            paths=paths,
            execute=False,
            max_rows=None,
            workers=1,
            verify_runner=fake_runner,
        )

    _set_first_status_progress(paths=paths, reviewable=5, pending=0, failed=1)
    with pytest.raises(verify.PatchVerificationCampaignError, match="inconsistent"):
        verify.follow_patch_verification_campaign(
            paths=paths,
            execute=True,
            max_rows=None,
            workers=1,
            verify_runner=fake_runner,
        )

    assert not paths.output_dir.exists()


def test_follow_stall_timeout_avoids_idle_provider_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Follow mode aborts after bounded idle polling without provider calls."""
    paths = _write_fixture(tmp_path)
    _set_first_status_progress(paths=paths, reviewable=4, pending=1, failed=1)
    calls = 0
    budget_calls = 0
    sleeps: list[float] = []
    now = 0.0

    class FakeBudget:
        def __init__(self, **_kwargs: object) -> None:
            nonlocal budget_calls
            budget_calls += 1

    def fake_runner(**_kwargs: object) -> ProsePatchVerificationResult:
        nonlocal calls
        calls += 1
        return _verification_result(accepted=True)

    def fake_sleep(seconds: float) -> None:
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    monkeypatch.setattr(verify, "ProxyBudget", FakeBudget)
    monkeypatch.setattr(verify.time, "monotonic", lambda: now)
    monkeypatch.setattr(verify.time, "sleep", fake_sleep)

    with pytest.raises(verify.PatchVerificationCampaignError, match="not advanced"):
        verify.follow_patch_verification_campaign(
            paths=paths,
            execute=True,
            max_rows=None,
            workers=1,
            verify_runner=fake_runner,
            poll_seconds=300.0,
            stall_seconds=600.0,
        )

    assert sleeps == [300.0, 300.0]
    assert calls == 1
    assert budget_calls == 1


def test_run_processes_only_patched_and_resumes_as_first_pass_grows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The verification manifest omits dynamic first-pass checkpoint counts."""
    paths = _write_fixture(tmp_path, include_growth_row=True)
    calls: list[dict[str, object]] = []
    budget_kwargs: list[dict[str, object]] = []

    class FakeBudget:
        def __init__(self, **kwargs: object) -> None:
            budget_kwargs.append(kwargs)

    def fake_runner(
        *,
        row: dict[str, object],
        candidate_row: dict[str, object],
        changed_facts: dict[str, dict[str, object]],
        proposed_text: str,
        patches: list[dict[str, str]],
        first_checkpoint_sha256: str,
        prompt: str,
        config: GenerationConfig,
        budget: object,
        checkpoint_path: Path,
        transport: httpx.BaseTransport,
    ) -> ProsePatchVerificationResult:
        del row, candidate_row, prompt, budget, transport
        calls.append(
            {
                "changed_facts": changed_facts,
                "proposed_text": proposed_text,
                "patches": patches,
                "first_checkpoint_sha256": first_checkpoint_sha256,
                "checkpoint_path": checkpoint_path,
            }
        )
        assert config.model == verify.MODEL
        assert config.maximum_http_attempts == 1
        assert checkpoint_path.parent.parent == paths.output_dir / "checkpoints"
        return _verification_result(accepted=True)

    monkeypatch.setattr(verify, "ProxyBudget", FakeBudget)

    first = verify.run_patch_verification_campaign(
        paths=paths, execute=True, max_rows=None, workers=1, verify_runner=fake_runner
    )
    first_manifest = json.loads((paths.output_dir / "manifest.json").read_text())

    _append_growth_checkpoint(paths)
    second = verify.run_patch_verification_campaign(
        paths=paths, execute=True, max_rows=None, workers=1, verify_runner=fake_runner
    )
    second_manifest = json.loads((paths.output_dir / "manifest.json").read_text())

    assert first["available"] == 1
    assert first["processed"] == 1
    assert second["available"] == 2
    assert second["processed"] == 2
    assert second["accepted"] == 2
    assert len(calls) == 2
    assert calls[0]["changed_facts"] == {"age": {"old": 41, "new": 42}}
    assert calls[1]["changed_facts"] == {
        "job_title": {"old": "rådgiver", "new": "analytiker"}
    }
    assert budget_kwargs[-1]["uncapped"] is True
    assert budget_kwargs[-1]["uncapped_purpose"] == "patch_verification"
    assert first_manifest == second_manifest
    assert "available" not in first_manifest
    assert "checkpoint" not in json.dumps(first_manifest)
    assert (paths.output_dir.stat().st_mode & 0o777) == 0o700
    assert ((paths.output_dir / "status.json").stat().st_mode & 0o777) == 0o600


def test_h90_run_uses_dedicated_purpose_without_v4_contamination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H90 verification writes separate status and ledger bindings only."""
    v4_paths = _write_fixture(tmp_path / "v4")
    h90_paths = _write_fixture(
        tmp_path / "h90", budget_purpose=verify.H90_BUDGET_PURPOSE
    )
    budget_kwargs: list[dict[str, object]] = []
    checkpoint_paths: list[Path] = []

    class FakeBudget:
        def __init__(self, **kwargs: object) -> None:
            budget_kwargs.append(kwargs)

    def fake_runner(
        *,
        row: dict[str, object],
        candidate_row: dict[str, object],
        changed_facts: dict[str, dict[str, object]],
        proposed_text: str,
        patches: list[dict[str, str]],
        first_checkpoint_sha256: str,
        prompt: str,
        config: GenerationConfig,
        budget: object,
        checkpoint_path: Path,
        transport: httpx.BaseTransport,
    ) -> ProsePatchVerificationResult:
        del (
            row,
            candidate_row,
            changed_facts,
            proposed_text,
            patches,
            first_checkpoint_sha256,
            prompt,
            config,
            budget,
            transport,
        )
        checkpoint_paths.append(checkpoint_path)
        return _verification_result(accepted=True)

    monkeypatch.setattr(verify, "ProxyBudget", FakeBudget)

    summary = verify.run_patch_verification_campaign(
        paths=h90_paths,
        execute=True,
        max_rows=None,
        workers=1,
        verify_runner=fake_runner,
    )

    manifest = json.loads((h90_paths.output_dir / "manifest.json").read_text())
    assert summary["campaign"] == verify.H90_CAMPAIGN
    assert summary["processed"] == 1
    assert manifest["campaign"] == verify.H90_CAMPAIGN
    assert manifest["first_pass_campaign"] == verify.H90_FIRST_PASS_CAMPAIGN
    assert manifest["budget_purpose"] == verify.H90_BUDGET_PURPOSE
    assert "candidate_h90_v5" in manifest["inputs"]
    assert "candidate_v4" not in manifest["inputs"]
    assert budget_kwargs[-1]["uncapped_purpose"] == "h90_v5_verification"
    assert checkpoint_paths[0].is_relative_to(h90_paths.output_dir / "checkpoints")
    assert not v4_paths.output_dir.exists()


def test_stale_first_pass_checkpoint_fails_before_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First-pass digest and binding are checked before any second-pass call."""
    paths = _write_fixture(tmp_path)
    _tamper_first_checkpoint(paths=paths, persona_id="patched-row")
    called = False

    class FakeBudget:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("budget must not be created before revalidation")

    def fake_runner(**_kwargs: object) -> ProsePatchVerificationResult:
        nonlocal called
        called = True
        return _verification_result(accepted=True)

    monkeypatch.setattr(verify, "ProxyBudget", FakeBudget)

    with pytest.raises(Exception, match="checksum|digest|binding|decision"):
        verify.run_patch_verification_campaign(
            paths=paths,
            execute=True,
            max_rows=None,
            workers=1,
            verify_runner=fake_runner,
        )

    assert called is False
    assert not paths.output_dir.exists()


def _tamper_first_checkpoint(*, paths: FixturePaths, persona_id: str) -> None:
    persona_hash = sha256_text(persona_id)
    path = paths.first_checkpoint_root / "checkpoints" / persona_hash[:2]
    path = path / f"{persona_hash}.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["changed_fraction"] = 0.25
    path.write_text(
        json.dumps(document, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    path.chmod(0o600)


def test_transient_500_retries_without_completing_failed_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Transient provider failures remain pending until a retry succeeds."""
    paths = _write_fixture(tmp_path)
    attempts = 0

    class FakeBudget:
        def __init__(self, **_kwargs: object) -> None:
            pass

    def fake_runner(**_kwargs: object) -> ProsePatchVerificationResult:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            request = httpx.Request("POST", "https://proxy.invalid")
            response = httpx.Response(500, request=request)
            raise httpx.HTTPStatusError(
                "server failed", request=request, response=response
            )
        return _verification_result(accepted=False)

    monkeypatch.setattr(verify, "ProxyBudget", FakeBudget)
    monkeypatch.setattr(verify.time, "sleep", lambda _seconds: None)

    summary = verify.run_patch_verification_campaign(
        paths=paths, execute=True, max_rows=None, workers=1, verify_runner=fake_runner
    )

    assert attempts == 2
    assert summary["rejected"] == 1
    assert summary["accepted"] == 0
    assert summary["transient_retries"] == 1
    assert summary["pending"] == 0
