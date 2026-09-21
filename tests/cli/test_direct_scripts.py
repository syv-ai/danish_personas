"""Focused tests for the machine-readable persona scripts."""

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from click.testing import CliRunner

from danish_personas.release import upload as upload_service
from scripts import build_dataset, generate_persona


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
        "--rows",
        "1",
        "--maximum-total-requests",
        "2",
        "--input-price-per-million",
        "0",
        "--output-price-per-million",
        "0",
    ]

    result = CliRunner().invoke(build_dataset.main, arguments)

    assert result.exit_code == 0, result.output
    assert result.stdout == f"{output_path}\n"


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
            "--maximum-total-requests",
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
    monkeypatch.setattr(generate_persona, "generate_personas", lambda **_: run_dir)
    monkeypatch.setattr(
        generate_persona,
        "validate_persona_run",
        lambda **_: SimpleNamespace(passed=True),
    )

    result = CliRunner().invoke(generate_persona.main)

    assert result.exit_code == 0, result.output
    assert result.stdout == "Dette er en dansk syntetisk persona.\n"


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
