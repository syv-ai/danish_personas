"""Focused tests for the machine-readable persona scripts."""

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from pydantic import ValidationError

from danish_personas.generation.config import load_generation_config
from danish_personas.io import sha256_file, write_json
from danish_personas.models import FROZEN_SAMPLE_SCHEMA_VERSION, FrozenSampleManifest
from danish_personas.release import upload as upload_service
from scripts import build_dataset, generate_persona
from tests.generation.manifest_helpers import origin_contract_fields


def test_build_dataset_prints_merged_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The dataset command keeps progress separate from its clean path output."""
    pilot_dir = tmp_path / "pilot"
    pilot_dir.mkdir()
    output_path = pilot_dir / "generated-personas.parquet"
    output_path.write_bytes(b"parquet")
    calls: dict[str, object] = {}

    def run(**kwargs: object) -> Path:
        calls.update(kwargs)
        return pilot_dir

    monkeypatch.setattr(build_dataset, "run_pilot", run)
    monkeypatch.setattr(
        build_dataset,
        "validate_persona_pilot",
        lambda **_: SimpleNamespace(passed=True),
    )
    config = _config(
        overrides=[
            f"build_dataset.input={tmp_path / 'sample.parquet'}",
            f"build_dataset.output_dir={tmp_path / 'output'}",
            "build_dataset.rows=1",
            "build_dataset.request_limit=2",
            "build_dataset.input_price_per_million=0",
            "build_dataset.output_price_per_million=0",
            "llm.model=overridden-model",
        ]
    )

    build_dataset.main.__wrapped__(config)

    captured = capsys.readouterr()
    assert captured.out == f"{output_path}\n"
    assert "Starting persona dataset build" in captured.err
    config_path = calls["config_path"]
    assert isinstance(config_path, Path)
    assert load_generation_config(config_path).model == "overridden-model"


def _config(*, overrides: list[str]) -> DictConfig:
    """Compose the public Hydra configuration with test overrides.

    Returns:
        The composed test configuration.
    """
    with initialize_config_dir(
        version_base=None, config_dir=str(Path("config").resolve())
    ):
        return compose(config_name="config", overrides=overrides)


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
    config = _config(
        overrides=[
            "build_dataset.rows=1",
            "build_dataset.request_limit=2",
            "build_dataset.input_price_per_million=0",
            "build_dataset.output_price_per_million=0",
            "build_dataset.hf_repo=org/dataset",
        ]
    )

    with pytest.raises(ValidationError, match="build_dataset.attestation"):
        build_dataset._run(config=config)

    assert not called


def test_generate_persona_emits_only_validated_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
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
    config = _config(
        overrides=[
            f"generate_persona.output_dir={tmp_path / 'output'}",
            "llm.model=overridden-model",
        ]
    )

    generate_persona.main.__wrapped__(config)

    captured = capsys.readouterr()
    assert captured.out == "Dette er en dansk syntetisk persona.\n"
    assert calls["offset"] == 1
    assert calls["sample_manifest_path"] == tmp_path / "sample.manifest.json"
    assert "Loading and validating persona inputs" in captured.err
    assert "Dette er en dansk syntetisk persona." not in captured.err
    config_path = calls["config_path"]
    assert isinstance(config_path, Path)
    assert load_generation_config(config_path).model == "overridden-model"


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
