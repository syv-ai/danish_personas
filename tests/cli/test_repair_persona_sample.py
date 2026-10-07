"""Offline contracts for the focused persona sample repair CLI."""

from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path

import httpx
import polars as pl
import pytest

from danish_personas.generation.models import GenerationConfig
from danish_personas.generation.proxy_patch_runner import ProxyPatchProposal
from danish_personas.io import sha256_file
from scripts import repair_persona_sample as repair


def _unused_runner(
    _row: dict[str, object],
    _candidate_row: dict[str, object],
    _changed_facts: dict[str, dict[str, object]],
    _gender: str | None,
    _partner_gender: str | None,
    _prompt: str,
    _config: GenerationConfig,
    _budget: object,
    _checkpoint_path: Path,
    _transport: httpx.BaseTransport,
) -> ProxyPatchProposal:
    raise AssertionError("dry-run must not call the proxy patch runner")


def test_baseline_and_cap_fail_closed(tmp_path: Path) -> None:
    """Changed baseline files or caps beyond 10 USD are refused before requests."""
    paths = _write_inputs(tmp_path)

    with pytest.raises(repair.RepairSampleError, match="SHA-256"):
        repair.run_repair_campaign(
            paths=paths,
            max_attempts=2,
            dry_run=True,
            cost_cap_usd=Decimal("10"),
            expected_original_sha256="0" * 64,
            patch_runner=_unused_runner,
        )
    with pytest.raises(repair.RepairSampleError, match="at most 10 USD"):
        repair.run_repair_campaign(
            paths=paths,
            max_attempts=2,
            dry_run=True,
            cost_cap_usd=Decimal("10.01"),
            expected_original_sha256=sha256_file(paths.original),
            patch_runner=_unused_runner,
        )


def _write_inputs(tmp_path: Path) -> repair.RepairPaths:
    original_rows = [
        _row("legal", legal_status_detail=None),
        _row("job", job_function="Pædagog", job_function_code="A"),
        _row("unresolved", marital_status="single"),
        _row("identity", gender="man"),
        _row("forbidden", sexual_orientation="private"),
        _row("unsupported", origin_country_code="DK"),
        _row("persona", education_level="short"),
    ]
    candidate_rows = [
        _row("legal", legal_status_detail="married"),
        _row("job", job_function="Lærer", job_function_code="B"),
        _row("unresolved", marital_status="married"),
        _row("identity", gender="woman"),
        _row("forbidden", sexual_orientation="changed"),
        _row("unsupported", origin_country_code="NO"),
        _row("persona", education_level="long", persona=_persona("changed")),
    ]
    identity_rows = [
        {
            "persona_id": row["persona_id"],
            "gender": "woman" if row["persona_id"] == "legal" else "man",
            "partner_gender": "man" if row["persona_id"] == "legal" else "woman",
            "sexual_orientation": "private",
            "transgender": False,
            "variation_in_sex_characteristics": False,
        }
        for row in original_rows
    ]
    paths = repair.RepairPaths(
        original=tmp_path / "train-00000-of-00001.parquet",
        candidate=tmp_path / "attribute-candidate-v2.parquet",
        identity_sidecar=tmp_path / "paired-identity-v2.parquet",
        triage=tmp_path / "prose-triage-v1.json",
        report=tmp_path / "attribute-candidate-v2.report.json",
        prompt=tmp_path / "persona-patch-da.md",
        output_dir=tmp_path / "private-output",
        registry=tmp_path / "models-store.json",
    )
    pl.DataFrame(original_rows).write_parquet(paths.original)
    pl.DataFrame(candidate_rows).write_parquet(paths.candidate)
    pl.DataFrame(identity_rows).write_parquet(paths.identity_sidecar)
    paths.triage.write_text(
        json.dumps(
            {
                "personas": {
                    "legal": _triage(["legal_status_detail"]),
                    "job": _triage(["job_function", "job_function_code"]),
                    "unresolved": _triage(["marital_status"]),
                    "identity": _triage(["gender"]),
                    "forbidden": _triage(["sexual_orientation"]),
                    "unsupported": _triage(["origin_country_code"]),
                    "persona": _triage(["education_level"]),
                },
                "counts": {"needs_prose_review_or_regeneration": 7},
            }
        ),
        encoding="utf-8",
    )
    paths.report.write_text(
        json.dumps(
            {
                "changed": {},
                "prose_regeneration": {},
                "unresolved": {"unresolved": ["x"]},
            }
        ),
        encoding="utf-8",
    )
    paths.prompt.write_text("Ret kun små modsætninger.\n", encoding="utf-8")
    paths.registry.write_text("{}\n", encoding="utf-8")
    os.chmod(tmp_path, 0o700)
    return paths


def _persona(seed: str) -> str:
    base = (
        f"Persona {seed} bor i en dansk kommune og har en stabil hverdag med "
        "arbejde, familie, fritidsinteresser og praktiske rutiner. "
    )
    return (base * 5)[:420]


def _row(persona_id: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "persona_id": persona_id,
        "persona": _persona(persona_id),
        "legal_status_detail": None,
        "marital_status": "married_or_separated",
        "education_level": "medium",
        "job_function": "Pædagog",
        "job_function_code": "A",
        "origin_country_code": "DK",
        "gender": "woman",
        "sexual_orientation": "private",
    }
    row.update(overrides)
    return row


def _triage(fields: list[str]) -> dict[str, object]:
    return {
        "classification": "needs_prose_review_or_regeneration",
        "changed_fields": fields,
        "reasons": fields,
    }


def test_cost_preflight_and_pinned_generation_config_are_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-dry runs initialise the fixed budget before making patch requests."""
    paths = _write_inputs(tmp_path)
    events: list[str] = []
    checkpoints: list[Path] = []

    class FakeBudget:
        def __init__(self, **kwargs: object) -> None:
            events.append("budget")
            assert kwargs["registry_path"] == paths.registry
            assert kwargs["campaign"] == repair.CAMPAIGN
            assert kwargs["cap_usd"] == Decimal("10")
            assert kwargs["prompt_hash"] == repair.sha256_text(
                paths.prompt.read_text(encoding="utf-8")
            )

    def fake_runner(
        row: dict[str, object],
        candidate_row: dict[str, object],
        changed_facts: dict[str, dict[str, object]],
        gender: str | None,
        partner_gender: str | None,
        prompt: str,
        config: GenerationConfig,
        budget: object,
        checkpoint_path: Path,
        transport: httpx.BaseTransport,
    ) -> ProxyPatchProposal:
        events.append("runner")
        assert isinstance(budget, FakeBudget)
        assert isinstance(transport, httpx.HTTPTransport)
        assert prompt == paths.prompt.read_text(encoding="utf-8")
        assert config.base_url == repair.BASE_URL
        assert config.model == repair.MODEL
        assert config.max_tokens is None
        assert config.reasoning_effort == "none"
        assert config.maximum_http_attempts == 1
        assert gender is None
        assert partner_gender is None
        assert row["persona"] == candidate_row["persona"]
        assert set(changed_facts) <= repair._ALLOWED_FACTS
        checkpoints.append(checkpoint_path)
        return ProxyPatchProposal(str(row["persona"]), 0.0, (), checkpoint_path)

    monkeypatch.setattr(repair, "ProxyBudget", FakeBudget)

    summary = repair.run_repair_campaign(
        paths=paths,
        max_attempts=2,
        dry_run=False,
        cost_cap_usd=Decimal("10"),
        expected_original_sha256=sha256_file(paths.original),
        patch_runner=fake_runner,
    )

    assert events == ["budget", "runner", "runner"]
    assert summary["processed"] == 2
    assert summary["proposed"] == 2
    assert summary["attempted"] == 2
    assert all(
        "legal" not in str(path) and "job" not in str(path) for path in checkpoints
    )


def test_default_batch_config_and_dry_run_counts(tmp_path: Path) -> None:
    """The 100-row limit must not exceed the per-client limit of one request."""
    paths = _write_inputs(tmp_path)
    config = repair._generation_config(prompt_path=paths.prompt)
    assert config.maximum_total_requests == 1
    selected = repair.select_eligible_repairs(
        inputs=repair.load_repair_inputs(paths=paths), max_attempts=100
    )
    summary = repair._dry_run_summary(
        selected=selected * 4,
        manifest=repair.build_manifest(
            paths=paths,
            max_attempts=100,
            cost_cap_usd=Decimal("10"),
            expected_original_sha256=sha256_file(paths.original),
        ),
    )
    assert summary["selected_count"] == 8
    preview = summary["selected"]
    assert isinstance(preview, list)
    assert len(preview) == repair.DRY_RUN_LIMIT


def test_dry_run_selects_safe_rows_without_private_identity_columns(
    tmp_path: Path,
) -> None:
    """Dry-run emits only hashed IDs and excludes unresolved and forbidden rows."""
    paths = _write_inputs(tmp_path)
    loaded = repair.load_repair_inputs(paths=paths)

    selected = repair.select_eligible_repairs(inputs=loaded, max_attempts=10)
    summary = repair.run_repair_campaign(
        paths=paths,
        max_attempts=10,
        dry_run=True,
        cost_cap_usd=Decimal("10"),
        expected_original_sha256=sha256_file(paths.original),
        patch_runner=_unused_runner,
    )

    assert [row.persona_id for row in selected] == ["legal", "job"]
    assert all("sexual_orientation" not in row for row in loaded.identity_rows.values())
    assert summary["dry_run"] is True
    assert summary["selected_count"] == 2
    assert summary["selected"] == [
        {
            "persona_sha256": repair.sha256_text("legal"),
            "changed_fields": ["legal_status_detail"],
        },
        {
            "persona_sha256": repair.sha256_text("job"),
            "changed_fields": ["job_function"],
        },
    ]
    assert not (paths.output_dir / "status.json").exists()


def test_resume_keeps_private_status_and_skips_processed_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume checks the manifest and stores processed IDs only in private status."""
    paths = _write_inputs(tmp_path)
    calls: list[str] = []

    class FakeBudget:
        def __init__(self, **_kwargs: object) -> None:
            pass

    def fake_runner(
        row: dict[str, object],
        _candidate_row: dict[str, object],
        _changed_facts: dict[str, dict[str, object]],
        _gender: str | None,
        _partner_gender: str | None,
        _prompt: str,
        _config: GenerationConfig,
        _budget: object,
        checkpoint_path: Path,
        _transport: httpx.BaseTransport,
    ) -> ProxyPatchProposal:
        persona_id = str(row["persona_id"])
        calls.append(persona_id)
        return ProxyPatchProposal(str(row["persona"]), 0.0, (), checkpoint_path)

    monkeypatch.setattr(repair, "ProxyBudget", FakeBudget)
    expected_sha = sha256_file(paths.original)
    repair.run_repair_campaign(
        paths=paths,
        max_attempts=2,
        dry_run=False,
        cost_cap_usd=Decimal("10"),
        expected_original_sha256=expected_sha,
        patch_runner=fake_runner,
    )
    repair.run_repair_campaign(
        paths=paths,
        max_attempts=2,
        dry_run=False,
        cost_cap_usd=Decimal("10"),
        expected_original_sha256=expected_sha,
        patch_runner=fake_runner,
    )

    status_path = paths.output_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert calls == ["legal", "job"]
    assert status["processed_persona_ids"] == ["legal", "job"]
    assert status["processed"] == 2
    assert status["skipped"] == 0
    assert (paths.output_dir.stat().st_mode & 0o777) == 0o700
    assert (status_path.stat().st_mode & 0o777) == 0o600

    paths.prompt.write_text("Retningslinjen er ændret.\n", encoding="utf-8")
    with pytest.raises(repair.RepairSampleError, match="pins do not match"):
        repair.run_repair_campaign(
            paths=paths,
            max_attempts=2,
            dry_run=False,
            cost_cap_usd=Decimal("10"),
            expected_original_sha256=expected_sha,
            patch_runner=fake_runner,
        )
