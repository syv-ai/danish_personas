"""Generation provenance and replay-binding tests."""

import json
from pathlib import Path

import pytest
import yaml
from generation_test_helpers import MockGenerationClient, write_generation_inputs

from danish_personas.generation.pipeline import generate_personas
from danish_personas.generation.report import validate_persona_run
from danish_personas.io import write_json


def test_persona_validation_binds_custom_mapping_during_generation_and_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Custom title mappings govern initial validation and checkpoint replay."""
    paths = write_generation_inputs(root=tmp_path)
    custom_title = "specialtilpasset forretningsrådgiver"
    mapping_data = yaml.safe_load(
        (Path(__file__).parents[2] / "config/job-function-titles.yaml").read_text(
            encoding="utf-8"
        )
    )
    mapping_data["job_functions"]["24"]["titles"] = [custom_title]
    mapping_path = tmp_path / "custom-job-function-titles.yaml"
    mapping_path.write_text(
        yaml.safe_dump(mapping_data, allow_unicode=True), encoding="utf-8"
    )
    config_data = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    config_data["job_title_mapping"] = str(mapping_path)
    paths["config"].write_text(
        yaml.safe_dump(config_data, allow_unicode=True), encoding="utf-8"
    )

    class CustomMappingClient(MockGenerationClient):
        job_title = custom_title

    mismatch_config = tmp_path / "default-job-title-config.yaml"
    mismatch_data = dict(config_data)
    mismatch_data.pop("job_title_mapping")
    mismatch_config.write_text(
        yaml.safe_dump(mismatch_data, allow_unicode=True), encoding="utf-8"
    )
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", CustomMappingClient
    )
    with pytest.raises(ValueError, match="allowlisted"):
        generate_personas(
            input_path=paths["sample"],
            sample_manifest_path=paths["sample_manifest"],
            config_path=mismatch_config,
            output_dir=tmp_path / "mismatched-output",
            rows=1,
            live=True,
        )

    run_dir = generate_personas(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "outputs",
        rows=1,
        live=True,
    )
    assert validate_persona_run(run_dir=run_dir).passed

    default_config_data = dict(config_data)
    default_config_data.pop("job_title_mapping")
    paths["config"].write_text(
        yaml.safe_dump(default_config_data, allow_unicode=True), encoding="utf-8"
    )
    assert not validate_persona_run(run_dir=run_dir).passed

    paths["config"].write_text(
        yaml.safe_dump(config_data, allow_unicode=True), encoding="utf-8"
    )
    mapping_data["job_functions"]["24"]["titles"] = ["anden forretningsrådgiver"]
    mapping_path.write_text(
        yaml.safe_dump(mapping_data, allow_unicode=True), encoding="utf-8"
    )
    assert not validate_persona_run(run_dir=run_dir).passed


def test_persona_validation_rejects_mapping_binding_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay validation rejects forged checkpoint and manifest mapping fields."""
    paths = write_generation_inputs(root=tmp_path)
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", MockGenerationClient
    )
    run_dir = generate_personas(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "outputs",
        rows=1,
        live=True,
    )
    checkpoint_path = next((run_dir / "checkpoints").glob("*.json"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    original_checkpoint = checkpoint_path.read_bytes()
    checkpoint["job_title_mapping_content"]["version"] = 999
    write_json(path=checkpoint_path, payload=checkpoint)
    assert not validate_persona_run(run_dir=run_dir).passed

    checkpoint_path.write_bytes(original_checkpoint)
    manifest_path = run_dir / "generation-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["job_title_mapping_sha256"] = "f" * 64
    write_json(path=manifest_path, payload=manifest)
    assert not validate_persona_run(run_dir=run_dir).passed


def test_persona_validation_rejects_origin_binding_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay rejects changed origin content, checksums, and path substitution."""
    paths = write_generation_inputs(root=tmp_path)
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", MockGenerationClient
    )
    run_dir = generate_personas(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "outputs",
        rows=1,
        live=True,
    )
    checkpoint_path = next((run_dir / "checkpoints").glob("*.json"))
    original_checkpoint = checkpoint_path.read_bytes()
    checkpoint = json.loads(original_checkpoint)
    checkpoint["origin_label_contract_content"]["labels_da"]["5100"] = "Libanon"
    write_json(path=checkpoint_path, payload=checkpoint)
    assert not validate_persona_run(run_dir=run_dir).passed

    checkpoint_path.write_bytes(original_checkpoint)
    manifest_path = run_dir / "generation-manifest.json"
    original_manifest = manifest_path.read_bytes()
    manifest = json.loads(original_manifest)
    manifest["origin_label_contract_sha256"] = "f" * 64
    write_json(path=manifest_path, payload=manifest)
    assert not validate_persona_run(run_dir=run_dir).passed

    manifest_path.write_bytes(original_manifest)
    alternate = tmp_path / "same-origin-contract.yaml"
    alternate.write_bytes(Path("config/folk2-ieland-labels-da.yaml").read_bytes())
    manifest = json.loads(original_manifest)
    manifest["origin_label_contract_file"] = str(alternate)
    write_json(path=manifest_path, payload=manifest)
    assert not validate_persona_run(run_dir=run_dir).passed


@pytest.mark.parametrize("artifact", ["checkpoint", "manifest"])
def test_persona_validation_rejects_stale_validator_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str
) -> None:
    """Checkpoint and run manifests must use the current validator contract."""
    paths = write_generation_inputs(root=tmp_path)
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", MockGenerationClient
    )
    run_dir = generate_personas(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "outputs",
        rows=1,
        live=True,
    )
    path = (
        next((run_dir / "checkpoints").glob("*.json"))
        if artifact == "checkpoint"
        else run_dir / "generation-manifest.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["validator_version"] = "persona-safety-v14"
    write_json(path=path, payload=payload)

    assert not validate_persona_run(run_dir=run_dir).passed


def test_persona_validation_reports_missing_manifest(tmp_path: Path) -> None:
    """A missing required manifest produces a failed report rather than an error."""
    run_dir = tmp_path / "missing-manifest"
    run_dir.mkdir()

    report = validate_persona_run(run_dir=run_dir)

    assert not report.passed
    assert report.subject_id == run_dir.name
