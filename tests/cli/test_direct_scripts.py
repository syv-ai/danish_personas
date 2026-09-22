"""Focused tests for the machine-readable persona scripts."""

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from pydantic import ValidationError

from danish_personas.checksum import ChecksumValidationPolicy
from danish_personas.generation.config import load_generation_config
from danish_personas.generation.policy import ContentValidationPolicy
from danish_personas.io import sha256_file, write_json
from danish_personas.models import FROZEN_SAMPLE_SCHEMA_VERSION, FrozenSampleManifest
from danish_personas.release import upload as upload_service
from danish_personas.release.models import ReleasePolicy, ReviewAttestation
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


def test_build_dataset_default_request_limit_is_30() -> None:
    """The public dataset command has a bounded default request budget."""
    config = _config(overrides=[])

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


@pytest.mark.parametrize("malformed_name", ["policy", "attestation"])
def test_build_dataset_release_evidence_fails_before_all_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, malformed_name: str
) -> None:
    """Malformed release evidence cannot create snapshots or start the pipeline."""
    evidence = _release_evidence(tmp_path)
    evidence[malformed_name].write_text(
        "enabled: [" if malformed_name == "policy" else "{}", encoding="utf-8"
    )
    calls: list[str] = []

    def fail(name: str) -> object:
        def operation(**_: object) -> object:
            calls.append(name)
            raise AssertionError(f"{name} must not start")

        return operation

    monkeypatch.setattr(build_dataset, "prepare_standard_sample", fail("prepare"))
    monkeypatch.setattr(
        build_dataset, "persist_effective_generation_config", fail("snapshot")
    )
    monkeypatch.setattr(build_dataset, "run_pilot", fail("generation"))
    monkeypatch.setattr(build_dataset, "package_release", fail("package"))
    monkeypatch.setattr(build_dataset, "upload_release", fail("upload"))
    config = _config(
        overrides=[
            "build_dataset.rows=1",
            "build_dataset.request_limit=2",
            "build_dataset.hf_repo=org/dataset",
            *[f"build_dataset.{name}={path}" for name, path in evidence.items()],
            f"build_dataset.output_dir={tmp_path / 'output'}",
        ]
    )

    with pytest.raises((ValidationError, yaml.YAMLError)):
        build_dataset._run(config=config)

    assert calls == []
    assert not (tmp_path / "output").exists()


def _release_evidence(tmp_path: Path) -> dict[str, Path]:
    """Write a minimal, internally consistent release evidence set.

    Returns:
        Paths to the policy, attestation, dataset card, and licence.
    """
    licence = tmp_path / "licence.txt"
    licence.write_bytes(b"CC-BY-4.0\n")
    policy = tmp_path / "policy.yaml"
    policy_model = ReleasePolicy(
        version=1,
        enabled=True,
        approved_models=("gpt-5.6-sol",),
        dataset_licence="CC-BY-4.0",
        licence_file_sha256=sha256_file(licence),
    )
    policy.write_text(
        yaml.safe_dump(policy_model.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    attestation = tmp_path / "attestation.json"
    attestation_model = ReviewAttestation(
        version=1,
        release_approved=True,
        blinded=True,
        reviewer_id="reviewer-1",
        protocol="blinded-human-review",
        protocol_version="1",
        reviewed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        pilot_id="a" * 16,
        output_sha256="b" * 64,
        population_rows=10_000,
        reviewed_persona_ids=("persona-1",),
    )
    attestation.write_text(
        json.dumps(attestation_model.model_dump(mode="json")), encoding="utf-8"
    )
    card = tmp_path / "card.md"
    card.write_text("# Synthetic dataset\n", encoding="utf-8")
    return {
        "attestation": attestation,
        "policy": policy,
        "dataset_card": card,
        "licence": licence,
    }


def test_build_dataset_release_evidence_rejects_bad_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A licence checksum mismatch fails before deterministic preparation."""
    evidence = _release_evidence(tmp_path)
    policy_payload = yaml.safe_load(evidence["policy"].read_text(encoding="utf-8"))
    policy_payload["licence_file_sha256"] = "0" * 64
    evidence["policy"].write_text(
        yaml.safe_dump(policy_payload, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setattr(
        build_dataset,
        "prepare_standard_sample",
        lambda: (_ for _ in ()).throw(AssertionError("preparation must not start")),
    )
    config = _config(
        overrides=[
            "build_dataset.rows=1",
            "build_dataset.request_limit=2",
            "build_dataset.hf_repo=org/dataset",
            *[f"build_dataset.{name}={path}" for name, path in evidence.items()],
        ]
    )

    with pytest.raises(ValueError, match="does not match policy"):
        build_dataset._run(config=config)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("dataset_card", "Dataset card must be non-empty"),
        ("licence", "Licence must be non-empty"),
    ],
)
def test_build_dataset_release_evidence_rejects_empty_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, message: str
) -> None:
    """Empty publication content fails before deterministic preparation."""
    evidence = _release_evidence(tmp_path)
    evidence[field].write_text(" \n", encoding="utf-8")
    monkeypatch.setattr(
        build_dataset,
        "prepare_standard_sample",
        lambda: (_ for _ in ()).throw(AssertionError("preparation must not start")),
    )
    config = _config(
        overrides=[
            "build_dataset.rows=1",
            "build_dataset.request_limit=2",
            "build_dataset.hf_repo=org/dataset",
            *[f"build_dataset.{name}={path}" for name, path in evidence.items()],
        ]
    )

    with pytest.raises(ValueError, match=message):
        build_dataset._run(config=config)


@pytest.mark.parametrize("path_kind", ["missing", "directory"])
def test_build_dataset_release_paths_must_be_regular_files(
    tmp_path: Path, path_kind: str
) -> None:
    """Publication paths must point to existing regular files."""
    path = tmp_path / "missing.json"
    if path_kind == "directory":
        path.mkdir()
    config = _config(
        overrides=[
            "build_dataset.hf_repo=org/dataset",
            f"build_dataset.attestation={path}",
        ]
    )

    with pytest.raises(ValidationError, match="attestation"):
        build_dataset._run(config=config)


def test_build_dataset_requires_release_inputs_before_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An upload request cannot bypass the offline release boundary."""
    calls: list[str] = []

    def fail(name: str) -> object:
        def operation(**_: object) -> object:
            calls.append(name)
            raise AssertionError(f"{name} must not start")

        return operation

    monkeypatch.setattr(build_dataset, "prepare_standard_sample", fail("prepare"))
    monkeypatch.setattr(
        build_dataset, "persist_effective_generation_config", fail("snapshot")
    )
    monkeypatch.setattr(build_dataset, "run_pilot", fail("generation"))
    monkeypatch.setattr(build_dataset, "package_release", fail("package"))
    monkeypatch.setattr(build_dataset, "upload_release", fail("upload"))
    config = _config(
        overrides=[
            "build_dataset.rows=1",
            "build_dataset.request_limit=2",
            "build_dataset.hf_repo=org/dataset",
        ]
    )

    with pytest.raises(ValidationError, match="build_dataset.attestation"):
        build_dataset._run(config=config)

    assert calls == []


def test_generate_persona_emits_only_schema_valid_text(
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

    with initialize_config_dir(
        version_base=None, config_dir=str(Path("config").resolve())
    ):
        generate_persona.main.__wrapped__(config)

    captured = capsys.readouterr()
    assert captured.out == "Dette er en dansk syntetisk persona.\n"
    assert calls["offset"] == 1
    assert calls["sample_manifest_path"] == tmp_path / "sample.manifest.json"
    assert calls["checksum_policy"] is ChecksumValidationPolicy.IGNORE
    assert calls["content_validation_policy"] is ContentValidationPolicy.SCHEMA_ONLY
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
