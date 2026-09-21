"""Focused tests for the machine-readable persona scripts."""

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from click.testing import CliRunner

from danish_personas.io import sha256_file, write_json
from danish_personas.models import FROZEN_SAMPLE_SCHEMA_VERSION, FrozenSampleManifest
from danish_personas.release import upload as upload_service
from scripts import build_dataset, generate_persona
from tests.generation.manifest_helpers import origin_contract_fields


def test_build_dataset_prints_merged_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dataset command keeps progress separate from its clean path output."""
    pilot_dir = tmp_path / "pilot"
    pilot_dir.mkdir()
    output_path = pilot_dir / "generated-personas.parquet"
    output_path.write_bytes(b"parquet")
    monkeypatch.setattr(build_dataset, "run_pilot", lambda **_: pilot_dir)
    monkeypatch.setattr(
        build_dataset,
        "validate_persona_pilot",
        lambda **_: SimpleNamespace(passed=True),
    )
    arguments = [
        "--input",
        str(tmp_path / "sample.parquet"),
        "--rows",
        "1",
        "--request-limit",
        "2",
        "--input-price-per-million",
        "0",
        "--output-price-per-million",
        "0",
    ]

    result = CliRunner().invoke(build_dataset.main, arguments)

    assert result.exit_code == 0, result.output
    assert result.stdout == f"{output_path}\n"
    assert "Starting persona dataset build" in result.stderr


def test_build_dataset_requires_release_inputs_before_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An upload request cannot bypass the offline release boundary."""
    called = False

    def fail_run(**_: object) -> Path:
        nonlocal called
        called = True
        raise AssertionError("generation must not start")

    monkeypatch.setattr(build_dataset, "run_pilot", fail_run)
    result = CliRunner().invoke(
        build_dataset.main,
        [
            "--rows",
            "1",
            "--request-limit",
            "2",
            "--input-price-per-million",
            "0",
            "--output-price-per-million",
            "0",
            "--hf-repo",
            "org/dataset",
        ],
    )

    assert result.exit_code != 0
    assert "--attestation" in result.output
    assert not called


def test_generate_persona_emits_only_validated_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The single-persona command keeps diagnostics off stdout."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pl.DataFrame({"persona": ["Dette er en dansk syntetisk persona."]}).write_parquet(
        run_dir / "generated-personas.parquet"
    )
    calls: dict[str, object] = {}

    def generate(**kwargs: object) -> Path:
        calls.update(kwargs)
        return run_dir

    monkeypatch.setattr(
        generate_persona,
        "prepare_standard_sample",
        lambda: (tmp_path / "sample.parquet", tmp_path / "sample.manifest.json"),
    )
    monkeypatch.setattr(generate_persona, "_sample_offset", lambda **_: 1)
    monkeypatch.setattr(generate_persona, "generate_personas", generate)
    monkeypatch.setattr(
        generate_persona,
        "validate_persona_run",
        lambda **_: SimpleNamespace(passed=True),
    )

    result = CliRunner().invoke(generate_persona.main)

    assert result.exit_code == 0, result.output
    assert result.stdout == "Dette er en dansk syntetisk persona.\n"
    assert calls["offset"] == 1
    assert calls["sample_manifest_path"] == tmp_path / "sample.manifest.json"
    assert "Loading and validating persona inputs" in result.stderr
    assert "Dette er en dansk syntetisk persona." not in result.stderr


def test_sample_offset_selects_a_valid_frozen_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The local sampler bounds its choice by the validated sample manifest."""
    sample_path = tmp_path / "text-development-seeds.parquet"
    pl.DataFrame({"persona_id": ["persona-1", "persona-2"]}).write_parquet(sample_path)
    manifest = FrozenSampleManifest(
        sample_schema_version=FROZEN_SAMPLE_SCHEMA_VERSION,
        source_run_id="upstream-run",
        rows=2,
        strata=[],
        method="test",
        data_file=sample_path.name,
        sha256=sha256_file(sample_path),
        llm_calls=0,
        **origin_contract_fields(),
    )
    manifest_path = tmp_path / "text-development-seeds.manifest.json"
    write_json(path=manifest_path, payload=manifest)
    bounds: list[int] = []

    def randbelow(bound: int) -> int:
        bounds.append(bound)
        return bound - 1

    monkeypatch.setattr(generate_persona.secrets, "randbelow", randbelow)

    assert (
        generate_persona._sample_offset(
            input_path=sample_path, sample_manifest_path=manifest_path
        )
        == 1
    )
    assert bounds == [2]


def test_upload_release_delegates_without_accepting_a_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The focused uploader passes only the verified folder and repository ID."""
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    calls: list[dict[str, object]] = []

    class FakeApi:
        def upload_folder(self, **kwargs: object) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(upload_service, "HfApi", FakeApi)
    upload_service.upload_release(repo_id="org/dataset", release_dir=release_dir)

    assert calls == [
        {
            "folder_path": str(release_dir),
            "repo_id": "org/dataset",
            "repo_type": "dataset",
            "create_pr": True,
        }
    ]
