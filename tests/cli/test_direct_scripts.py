"""Focused tests for the machine-readable persona scripts."""

from pathlib import Path

import polars as pl
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from danish_personas.checksum import ChecksumValidationPolicy
from danish_personas.generation.config import load_generation_config
from scripts import build_dataset, generate_persona


def test_build_dataset_default_request_limit_is_30() -> None:
    """The public dataset command has a bounded default request budget."""
    config = _config(overrides=["build_dataset.rows=1"])

    assert config.build_dataset.request_limit == 30


def _config(*, overrides: list[str]) -> DictConfig:
    """Compose the public Hydra configuration with test overrides.

    Returns:
        The composed test configuration.
    """
    with initialize_config_dir(
        version_base=None, config_dir=str(Path("config").resolve())
    ):
        return compose(config_name="config", overrides=overrides)


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
    config = _config(
        overrides=[
            f"build_dataset.input={tmp_path / 'sample.parquet'}",
            f"build_dataset.output_dir={tmp_path / 'output'}",
            "build_dataset.rows=1",
            "build_dataset.request_limit=2",
            "llm.model=overridden-model",
        ]
    )

    build_dataset.main.__wrapped__(config)

    captured = capsys.readouterr()
    assert captured.out == f"{output_path}\n"
    assert "Starting persona dataset build" in captured.err
    assert calls["input_price_per_million"] == 0.0
    assert calls["output_price_per_million"] == 0.0
    config_path = calls["config_path"]
    assert isinstance(config_path, Path)
    assert load_generation_config(config_path).model == "overridden-model"


def test_build_dataset_uploads_generated_parquet_directly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publication needs only a repository ID and generated Parquet file."""
    pilot_dir = tmp_path / "pilot"
    pilot_dir.mkdir()
    output_path = pilot_dir / "generated-personas.parquet"
    output_path.write_bytes(b"parquet")
    uploads: list[tuple[str, Path]] = []
    monkeypatch.setattr(build_dataset, "run_pilot", lambda **_: pilot_dir)
    monkeypatch.setattr(
        build_dataset,
        "upload_dataset",
        lambda *, repo_id, parquet_path: uploads.append((repo_id, parquet_path)),
    )
    config = _config(
        overrides=[
            f"build_dataset.input={tmp_path / 'sample.parquet'}",
            f"build_dataset.output_dir={tmp_path / 'output'}",
            "build_dataset.rows=1",
            "build_dataset.hf_repo=owner/dataset",
        ]
    )

    build_dataset.main.__wrapped__(config)

    assert uploads == [("owner/dataset", output_path)]


def test_generate_persona_emits_only_schema_valid_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The single-persona command keeps diagnostics off stdout."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pl.DataFrame({"persona": ["Dette er en syntetisk persona."]}).write_parquet(
        run_dir / "generated-personas.parquet"
    )
    calls: dict[str, object] = {}

    def generate(**kwargs: object) -> Path:
        calls.update(kwargs)
        return run_dir

    monkeypatch.setattr(
        generate_persona,
        "prepare_standard_sample",
        lambda **_: (tmp_path / "sample.parquet", tmp_path / "sample.manifest.json"),
    )
    monkeypatch.setattr(generate_persona, "_sample_offset", lambda **_: 1)
    monkeypatch.setattr(generate_persona, "generate_personas", generate)
    config = _config(
        overrides=[
            f"generate_persona.output_dir={tmp_path / 'output'}",
            "llm.model=overridden-model",
        ]
    )

    generate_persona.main.__wrapped__(config)

    captured = capsys.readouterr()
    assert captured.out == "Dette er en syntetisk persona.\n"
    assert calls["offset"] == 1
    assert calls["sample_manifest_path"] == tmp_path / "sample.manifest.json"
    assert calls["checksum_policy"] is ChecksumValidationPolicy.IGNORE
    assert "content_validation_policy" not in calls
