"""Offline CLI contracts for merged provisional v5 previews."""

from __future__ import annotations

import json
import typing as t
from decimal import Decimal
from pathlib import Path

import httpx
import polars as pl
import pytest

from danish_personas.generation.proxy_budget import ProxyBudget
from danish_personas.generation.proxy_patch_verifier import run_proxy_patch_verification
from danish_personas.io import canonical_json, sha256_file, sha256_text
from danish_personas.release import provisional_v5_preview as preview
from scripts import build_provisional_prose_candidate as v4_candidate
from scripts import verify_persona_patches as verify
from tests.cli.test_verify_persona_patches import (
    FixturePaths,
    _first_manifest,
    _write_first_checkpoint,
    _write_first_status,
    _write_fixture,
    _write_json,
)


def test_build_dry_run_and_write_preserve_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The builder verifies both campaigns offline before writing new files."""
    monkeypatch.setattr(verify, "ProxyBudget", _FakeBudget)
    paths = _write_complete_fixture(tmp_path)
    output = tmp_path / "private-output" / "merged-v5-PROVISIONAL.parquet"

    dry_run = _build(paths=paths, output=output, write_output=False)

    assert dry_run["dry_run"] is True
    assert dry_run["publication_allowed"] is False
    assert dry_run["v4_excluded_overlap_count"] == 1
    assert dry_run["h90_accepted_count"] == 1
    assert not output.exists()

    written = _build(paths=paths, output=output, write_output=True)
    frame = pl.read_parquet(output)
    report = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))

    assert written["written"] is True
    assert output.parent.stat().st_mode & 0o777 == 0o700
    assert output.stat().st_mode & 0o777 == 0o600
    assert output.with_suffix(".json").stat().st_mode & 0o777 == 0o600
    assert report["not_release_ready"] is True
    assert "alder 42" not in _persona(frame=frame, persona_id="patched-row")
    assert "analytiker" in _persona(frame=frame, persona_id="patched-row")


def _persona(*, frame: pl.DataFrame, persona_id: str) -> str:
    value = frame.filter(pl.col("persona_id") == persona_id)["persona"].item()
    assert isinstance(value, str)
    return value


class _AllPaths(t.NamedTuple):
    v4: FixturePaths
    h90: FixturePaths
    v4_preview: Path
    h90_second_status: Path
    h90_second_manifest: Path
    h90_second_checkpoint_root: Path


def _build(
    *, paths: _AllPaths, output: Path, write_output: bool
) -> dict[str, preview.JSONValue]:
    return preview.build_provisional_v5_preview(
        original=paths.v4.original,
        v4_structured=paths.v4.candidate,
        v5_h90=paths.h90.candidate,
        v4_prose_preview=paths.v4_preview,
        v4_triage=paths.v4.triage,
        v4_first_prompt=paths.v4.first_prompt,
        verify_prompt=paths.v4.verify_prompt,
        registry=paths.v4.registry,
        v4_first_status=paths.v4.first_status,
        v4_first_manifest=paths.v4.first_manifest,
        v4_first_checkpoint_root=paths.v4.first_checkpoint_root,
        v4_second_status=paths.v4.output_dir / "status.json",
        v4_second_manifest=paths.v4.output_dir / "manifest.json",
        v4_second_checkpoint_root=paths.v4.output_dir,
        h90_triage=paths.h90.triage,
        h90_first_prompt=paths.h90.first_prompt,
        h90_first_status=paths.h90.first_status,
        h90_first_manifest=paths.h90.first_manifest,
        h90_first_checkpoint_root=paths.h90.first_checkpoint_root,
        h90_second_status=paths.h90_second_status,
        h90_second_manifest=paths.h90_second_manifest,
        h90_second_checkpoint_root=paths.h90_second_checkpoint_root,
        output=output,
        write_output=write_output,
        expected_row_count=3,
        expected_h90_changed_count=3,
        expected_v4_accepted_count=1,
        expected_h90_accepted_count=1,
        h90_changed_fields=frozenset({"job_title"}),
    )


def _write_complete_fixture(tmp_path: Path) -> _AllPaths:
    v4_paths = _write_fixture(tmp_path / "v4")
    _write_v4_second_campaign(paths=v4_paths, accepted=True)
    v4_preview = tmp_path / "v4-preview" / "prose-candidate-v4.parquet"
    v4_candidate.build_provisional_prose_candidate(
        original=v4_paths.original,
        v4_structured=v4_paths.candidate,
        triage=v4_paths.triage,
        first_prompt=v4_paths.first_prompt,
        verify_prompt=v4_paths.verify_prompt,
        registry=v4_paths.registry,
        first_status=v4_paths.first_status,
        first_manifest=v4_paths.first_manifest,
        first_checkpoint_root=v4_paths.first_checkpoint_root,
        second_status=v4_paths.output_dir / "status.json",
        second_manifest=v4_paths.output_dir / "manifest.json",
        second_checkpoint_root=v4_paths.output_dir,
        output=v4_preview,
        publication_v1=tmp_path / "unused-publication-v1.parquet",
        write_output=True,
    )
    for private_path in (v4_paths.candidate,):
        private_path.chmod(0o600)
    h90_paths = _write_h90_first_campaign(tmp_path=tmp_path, v4_paths=v4_paths)
    _write_h90_second_campaign(paths=h90_paths)
    return _AllPaths(
        v4=v4_paths,
        h90=h90_paths,
        v4_preview=v4_preview,
        h90_second_status=h90_paths.output_dir / "status.json",
        h90_second_manifest=h90_paths.output_dir / "manifest.json",
        h90_second_checkpoint_root=h90_paths.output_dir,
    )


def _write_h90_first_campaign(
    *, tmp_path: Path, v4_paths: FixturePaths
) -> FixturePaths:
    root = tmp_path / "h90" / "private"
    first_dir = root / "persona-review-h90-v5"
    output_dir = root / "persona-verify-h90-v5"
    checkpoints = first_dir / "checkpoints"
    checkpoints.mkdir(parents=True, mode=0o700)
    candidate = root / "candidate-h90.parquet"
    rows = pl.read_parquet(v4_paths.candidate).to_dicts()
    changed_rows = []
    for row in rows:
        changed = dict(row)
        changed["job_title"] = "analytiker"
        changed_rows.append(changed)
    pl.DataFrame(changed_rows).write_parquet(candidate)
    candidate.chmod(0o600)
    h90_report = {
        "candidate_h90_v5_sha256": sha256_file(candidate),
        "changed_persona_id_sha256": [
            sha256_text(str(row["persona_id"])) for row in rows
        ],
        "source_share": 0.1,
        "source_h90_count": 1,
        "source_age20plus_count": 10,
        "changed_rows": len(rows),
    }
    _write_json(candidate.with_suffix(".report.json"), h90_report)
    triage = root / "triage.json"
    first_prompt = root / "persona-review-h90-da.md"
    verify_prompt = v4_paths.verify_prompt
    registry = v4_paths.registry
    triage.write_text('{"personas": {}, "counts": {}}\n', encoding="utf-8")
    triage.chmod(0o600)
    first_prompt.write_text("Gennemgå H90 ændringer.\n", encoding="utf-8")
    paths = FixturePaths(
        original=v4_paths.candidate,
        candidate=candidate,
        triage=triage,
        first_prompt=first_prompt,
        verify_prompt=verify_prompt,
        registry=registry,
        first_status=first_dir / "status.json",
        first_manifest=first_dir / "manifest.json",
        first_checkpoint_root=first_dir,
        output_dir=output_dir,
        budget_purpose=verify.H90_BUDGET_PURPOSE,
    )
    manifest = _first_manifest(paths=paths)
    _write_json(paths.first_manifest, manifest)
    for persona_id, disposition in zip(
        ("patched-row", "manual-row", "unchanged-row"),
        ("patched", "needs_manual_review", "unchanged_consistent"),
        strict=True,
    ):
        _write_first_checkpoint(
            paths=paths,
            manifest=manifest,
            persona_id=persona_id,
            disposition=disposition,
        )
    _write_first_status(
        paths=paths,
        manifest=manifest,
        persona_ids=("patched-row", "manual-row", "unchanged-row"),
    )
    return paths


def _write_h90_second_campaign(*, paths: FixturePaths) -> None:
    _write_second_campaign(paths=paths, accepted=True, h90=True)


def _write_second_campaign(*, paths: FixturePaths, accepted: bool, h90: bool) -> None:
    loaded = verify.load_first_pass(paths=paths)
    manifest = verify._verification_manifest(paths=paths, loaded=loaded)
    _write_json(paths.output_dir / "manifest.json", manifest)
    budget = t.cast(
        ProxyBudget,
        _FakeBudget(
            registry_path=paths.registry,
            campaign=verify.H90_CAMPAIGN if h90 else verify.CAMPAIGN,
            source_hash=sha256_text(canonical_json(manifest)),
            prompt_hash=sha256_text(loaded.verify_prompt),
            schema_hash=t.cast(dict[str, str], manifest["inputs"])["schema"],
            uncapped=True,
            uncapped_purpose=verify.EDUCATION_VERIFICATION_PURPOSE
            if h90
            else verify.PATCH_VERIFICATION_PURPOSE,
        ),
    )
    config = verify._generation_config(prompt_path=paths.verify_prompt)
    row = loaded.rows[0]
    checkpoint_path = verify._checkpoint_path(
        output_dir=paths.output_dir, persona_hash=row.persona_hash
    )
    run_proxy_patch_verification(
        row=row.original_row,
        candidate_row=row.candidate_row,
        changed_facts=row.changed_facts,
        proposed_text=row.proposed_text,
        patches=row.patches,
        first_checkpoint_sha256=row.first_checkpoint_sha256,
        prompt=loaded.verify_prompt,
        config=config,
        budget=budget,
        checkpoint_path=checkpoint_path,
        transport=_transport(_review(row=row, accepted=accepted)),
    )
    status: dict[str, object] = {
        "version": verify.STATUS_VERSION,
        "manifest": manifest,
        "available": len(loaded.rows),
        "accepted": 1 if accepted else 0,
        "rejected": 0 if accepted else 1,
        "manual": loaded.manual,
        "unchanged_consistent": loaded.unchanged_consistent,
        "failed": 0,
        "attempted": 1,
        "transient_retries": 0,
        "processed": len(loaded.rows) if h90 else 1,
        "pending": 0 if h90 else len(loaded.rows) - 1,
        "processed_persona_hashes": [row.persona_hash],
        "accepted_is_provisional": True,
        "provisional_notice": verify.PROVISIONAL_NOTICE,
    }
    if h90:
        status["processed_persona_hashes"] = [row.persona_hash]
        status["processed"] = 1
        status["available"] = 1
        status["manual"] = 1
        status["unchanged_consistent"] = 1
    _write_json(paths.output_dir / "status.json", status)


class _FakeBudget:
    """Minimal patch-verification budget with stable binding pins."""

    overhead = 0

    def __init__(self, **kwargs: object) -> None:
        self.uncapped = bool(kwargs["uncapped"])
        self.uncapped_purpose = str(kwargs["uncapped_purpose"])
        self.pins: dict[str, object] = {
            "type": "header",
            "model": verify.MODEL,
            "base_url": verify.BASE_URL,
            "max_tokens": 128_000,
            "input_usd_per_million": "0.1",
            "output_usd_per_million": "0.5",
            "campaign": kwargs["campaign"],
            "source_hash": kwargs["source_hash"],
            "prompt_hash": kwargs["prompt_hash"],
            "schema_hash": kwargs["schema_hash"],
            "uncapped": True,
            "uncapped_purpose": kwargs["uncapped_purpose"],
        }

    def record_usage(
        self,
        request_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
        response_sha256: str,
    ) -> None:
        assert request_id
        assert input_tokens >= 0
        assert output_tokens >= 0
        assert len(response_sha256) == 64

    def reserve_attempt(self, request_id: str, request: dict[str, object]) -> Decimal:
        assert request_id
        assert request
        return Decimal("0")


def _review(*, row: verify.VerifyRow, accepted: bool) -> str:
    if not accepted:
        return json.dumps(
            {"verdict": "reject", "reasons": ["fact_mismatch"], "fact_evidence": []},
            ensure_ascii=False,
        )
    patch = row.patches[0]
    evidence = [
        {
            "field": field,
            "status": "corrected",
            "original_quote": patch["old_excerpt"],
            "proposed_quote": patch["new_excerpt"],
        }
        for field in row.changed_facts
    ]
    return json.dumps(
        {"verdict": "accept", "reasons": [], "fact_evidence": evidence},
        ensure_ascii=False,
    )


def _transport(response_content: str) -> httpx.MockTransport:
    def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "response-1",
                "model": verify.MODEL,
                "choices": [{"message": {"content": response_content}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )

    return httpx.MockTransport(respond)


def _write_v4_second_campaign(*, paths: FixturePaths, accepted: bool) -> None:
    _write_second_campaign(paths=paths, accepted=accepted, h90=False)


def test_refuses_overwrite_and_publication_like_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unsafe write targets fail before any provider or checkpoint work."""
    monkeypatch.setattr(verify, "ProxyBudget", _FakeBudget)
    paths = _write_complete_fixture(tmp_path)
    existing = tmp_path / "safe" / "merged-v5-PROVISIONAL.parquet"
    existing.parent.mkdir()
    existing.write_text("exists", encoding="utf-8")

    with pytest.raises(preview.ProvisionalV5PreviewError, match="overwrite"):
        _build(paths=paths, output=existing, write_output=True)
    with pytest.raises(preview.ProvisionalV5PreviewError, match="publication-like"):
        _build(
            paths=paths, output=tmp_path / "publication-v1.parquet", write_output=True
        )


def test_stale_h90_checkpoint_fails_closed_before_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changed checkpoint bindings are rejected rather than regenerated."""
    monkeypatch.setattr(verify, "ProxyBudget", _FakeBudget)
    paths = _write_complete_fixture(tmp_path)
    status = json.loads(paths.h90_second_status.read_text(encoding="utf-8"))
    persona_hash = status["processed_persona_hashes"][0]
    checkpoint = verify._checkpoint_path(
        output_dir=paths.h90_second_checkpoint_root, persona_hash=persona_hash
    )
    document = json.loads(checkpoint.read_text(encoding="utf-8"))
    document["accepted"] = False
    checkpoint.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    checkpoint.chmod(0o600)

    output = tmp_path / "private-output" / "merged-v5-PROVISIONAL.parquet"
    with pytest.raises(
        preview.ProvisionalV5PreviewError, match="checkpoint validation failed"
    ) as exc_info:
        _build(paths=paths, output=output, write_output=False)

    assert isinstance(
        exc_info.value.__cause__, v4_candidate.ProvisionalProseCandidateError
    )
    assert "patched-row" not in str(exc_info.value)
    assert not output.exists()
