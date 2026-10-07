"""Offline contracts for provisional prose-candidate assembly."""

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
from danish_personas.io import canonical_json, sha256_text
from scripts import build_provisional_prose_candidate as candidate
from scripts import verify_persona_patches as verify
from tests.cli.test_verify_persona_patches import FixturePaths, _write_fixture


class _FakeBudget:
    """Minimal patch-verification budget with stable binding pins."""

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

    def reserve_attempt(self, request_id: str, request: dict[str, object]) -> Decimal:
        assert request_id
        assert request
        return Decimal("0")

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


class _CheckpointBudget(_FakeBudget):
    """Budget shape used only to forge genuine fixture checkpoints."""

    overhead = 0


def test_dry_run_and_write_revalidate_accepted_checkpoint_without_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only independently accepted rows are written into the preview candidate."""
    paths = _write_fixture(tmp_path)
    output = tmp_path / "candidate" / "prose-candidate-v5-PROVISIONAL.parquet"
    publication_v1 = tmp_path / "candidate" / "publication-v1.parquet"
    monkeypatch.setattr(verify, "ProxyBudget", _FakeBudget)
    _write_second_campaign(paths=paths, accepted=True)

    dry_run = _build(
        paths=paths, output=output, publication_v1=publication_v1, write_output=False
    )

    assert dry_run["dry_run"] is True
    assert dry_run["accepted_patch_count"] == 1
    assert dry_run["manual_count"] == 1
    assert dry_run["pending_count"] == 0
    assert dry_run["release_ready"] is False
    assert dry_run["not_release_ready"] is True
    assert dry_run["first_pass_counts"] == {
        "needs_manual_review": 1,
        "patched": 1,
        "unchanged_consistent": 1,
    }
    assert not output.exists()

    written = _build(
        paths=paths, output=output, publication_v1=publication_v1, write_output=True
    )
    report = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    preview = pl.read_parquet(output)

    assert written["written"] is True
    assert report["accepted_patch_count"] == 1
    assert "not release-ready" in report["provisional_notice"]
    assert output.parent.stat().st_mode & 0o777 == 0o700
    assert output.stat().st_mode & 0o777 == 0o600
    assert output.with_suffix(".json").stat().st_mode & 0o777 == 0o600
    assert (
        "alder 42"
        in preview.filter(pl.col("persona_id") == "patched-row")["persona"].item()
    )
    assert (
        "alder 42"
        not in preview.filter(pl.col("persona_id") == "manual-row")["persona"].item()
    )


def test_duplicate_second_status_and_stale_checkpoint_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Status IDs and second-pass checkpoint digests must remain immutable."""
    monkeypatch.setattr(verify, "ProxyBudget", _FakeBudget)
    duplicate_paths = _write_fixture(tmp_path / "duplicate")
    _write_second_campaign(paths=duplicate_paths, accepted=True)
    status = json.loads(duplicate_paths.output_dir.joinpath("status.json").read_text())
    status["processed_persona_hashes"].append(status["processed_persona_hashes"][0])
    _write_json(duplicate_paths.output_dir / "status.json", status)

    with pytest.raises(Exception, match="processed hashes|IDs"):
        _build(
            paths=duplicate_paths,
            output=tmp_path / "duplicate.parquet",
            publication_v1=tmp_path / "publication-v1.parquet",
            write_output=False,
        )

    stale_paths = _write_fixture(tmp_path / "stale")
    _write_second_campaign(paths=stale_paths, accepted=True)
    checkpoint = next(stale_paths.output_dir.glob("checkpoints/*/*.json"))
    document = json.loads(checkpoint.read_text(encoding="utf-8"))
    document["changed_fraction"] = 0.25
    _write_json(checkpoint, document)

    with pytest.raises(Exception, match="checksum"):
        _build(
            paths=stale_paths,
            output=tmp_path / "stale.parquet",
            publication_v1=tmp_path / "publication-v1.parquet",
            write_output=False,
        )


def test_write_refuses_existing_candidate_and_publication_guard(tmp_path: Path) -> None:
    """The write path refuses to overwrite candidates or publication-v1 files."""
    output = tmp_path / "prose-candidate-v5-PROVISIONAL.parquet"
    publication_v1 = tmp_path / "publication-v1.parquet"
    output.write_bytes(b"existing")

    with pytest.raises(candidate.ProvisionalProseCandidateError, match="overwrite"):
        _build_missing_sources(
            output=output, publication_v1=publication_v1, write_output=True
        )

    output.unlink()
    publication_v1.write_bytes(b"existing")
    with pytest.raises(candidate.ProvisionalProseCandidateError, match="overwrite"):
        _build_missing_sources(
            output=output, publication_v1=publication_v1, write_output=True
        )


def _build(
    *, paths: FixturePaths, output: Path, publication_v1: Path, write_output: bool
) -> dict[str, candidate.JSONValue]:
    return candidate.build_provisional_prose_candidate(
        original=paths.original,
        v4_structured=paths.candidate,
        triage=paths.triage,
        first_prompt=paths.first_prompt,
        verify_prompt=paths.verify_prompt,
        registry=paths.registry,
        first_status=paths.first_status,
        first_manifest=paths.first_manifest,
        first_checkpoint_root=paths.first_checkpoint_root,
        second_status=paths.output_dir / "status.json",
        second_manifest=paths.output_dir / "manifest.json",
        second_checkpoint_root=paths.output_dir,
        output=output,
        publication_v1=publication_v1,
        write_output=write_output,
    )


def _build_missing_sources(
    *, output: Path, publication_v1: Path, write_output: bool
) -> None:
    root = output.parent
    candidate.build_provisional_prose_candidate(
        original=root / "missing-original.parquet",
        v4_structured=root / "missing-v4.parquet",
        triage=root / "missing-triage.json",
        first_prompt=root / "missing-first.md",
        verify_prompt=root / "missing-verify.md",
        registry=root / "missing-registry.json",
        first_status=root / "missing-first-status.json",
        first_manifest=root / "missing-first-manifest.json",
        first_checkpoint_root=root / "missing-first-root",
        second_status=root / "missing-second-status.json",
        second_manifest=root / "missing-second-manifest.json",
        second_checkpoint_root=root / "missing-second-root",
        output=output,
        publication_v1=publication_v1,
        write_output=write_output,
    )


def _write_second_campaign(*, paths: FixturePaths, accepted: bool) -> None:
    loaded = verify.load_first_pass(paths=paths)
    manifest = verify._verification_manifest(paths=paths, loaded=loaded)
    _write_json(paths.output_dir / "manifest.json", manifest)
    budget = t.cast(
        ProxyBudget,
        _checkpoint_budget(paths=paths, prompt=loaded.verify_prompt, manifest=manifest),
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
        "processed": 1,
        "pending": len(loaded.rows) - 1,
        "processed_persona_hashes": [row.persona_hash],
        "accepted_is_provisional": True,
        "provisional_notice": verify.PROVISIONAL_NOTICE,
    }
    _write_json(paths.output_dir / "status.json", status)


def _checkpoint_budget(
    *, paths: FixturePaths, prompt: str, manifest: dict[str, candidate.JSONValue]
) -> _CheckpointBudget:
    inputs = manifest["inputs"]
    if not isinstance(inputs, dict) or not isinstance(inputs.get("schema"), str):
        raise AssertionError("fixture manifest schema is malformed")
    return _CheckpointBudget(
        registry_path=paths.registry,
        campaign=verify.CAMPAIGN,
        source_hash=sha256_text(canonical_json(manifest)),
        prompt_hash=sha256_text(prompt),
        schema_hash=inputs["schema"],
        uncapped=True,
        uncapped_purpose=verify.PATCH_VERIFICATION_PURPOSE,
    )


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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    content = json.dumps(value, ensure_ascii=False, sort_keys=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)
