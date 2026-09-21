"""Pilot identity, merging, and validation-integrity tests."""

import json
from pathlib import Path

import polars as pl
import pytest
from click.testing import CliRunner
from generation_test_helpers import MockGenerationClient, write_generation_inputs

from danish_personas.generation.pilot import run_pilot
from danish_personas.generation.report import validate_persona_pilot
from danish_personas.io import sha256_file, write_json
from scripts.build_dataset import main as pilot_main


def test_pilot_identity_changes_when_prompt_context_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing an effective prompt creates a new pilot artefact directory."""
    paths = write_generation_inputs(root=tmp_path)
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", MockGenerationClient
    )
    MockGenerationClient.requests = 0
    arguments = [
        "--input",
        str(paths["sample"]),
        "--sample-manifest",
        str(paths["sample_manifest"]),
        "--config",
        str(paths["config"]),
        "--output-dir",
        str(tmp_path / "pilot"),
        "--rows",
        "1",
        "--batch-size",
        "1",
        "--concurrency",
        "1",
        "--maximum-total-requests",
        "5",
        "--input-price-per-million",
        "0.3",
        "--output-price-per-million",
        "1.2",
    ]
    first = CliRunner().invoke(pilot_main, arguments)
    assert first.exit_code == 0, first.output
    paths["personas_prompt"].write_text("En ændret dansk prompt", encoding="utf-8")
    second = CliRunner().invoke(pilot_main, arguments)
    assert second.exit_code == 0, second.output

    pilot_dirs = list((tmp_path / "pilot").iterdir())
    assert len(pilot_dirs) == 2
    manifests = [
        json.loads((pilot_dir / "pilot-manifest.json").read_text(encoding="utf-8"))
        for pilot_dir in pilot_dirs
    ]
    assert manifests[0]["pilot_id"] != manifests[1]["pilot_id"]
    assert len({manifest["generation_context_sha256"] for manifest in manifests}) == 2


def test_pilot_identity_supports_a_batch_larger_than_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pilot identity retains the requested batch size without a new manifest field."""
    paths = write_generation_inputs(root=tmp_path)
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", MockGenerationClient
    )
    MockGenerationClient.requests = 0

    pilot_dir = run_pilot(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "pilot",
        rows=1,
        batch_size=2,
        concurrency=1,
        delay_between_batches=0.0,
        maximum_total_requests=5,
        input_price_per_million=0.3,
        output_price_per_million=1.2,
    )

    assert validate_persona_pilot(pilot_dir=pilot_dir).passed


def test_pilot_merges_validated_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pilot runner merges each bounded invocation exactly once."""
    paths = write_generation_inputs(root=tmp_path)
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", MockGenerationClient
    )
    MockGenerationClient.requests = 0
    result = CliRunner().invoke(
        pilot_main,
        [
            "--input",
            str(paths["sample"]),
            "--sample-manifest",
            str(paths["sample_manifest"]),
            "--config",
            str(paths["config"]),
            "--output-dir",
            str(tmp_path / "pilot"),
            "--rows",
            "2",
            "--batch-size",
            "1",
            "--concurrency",
            "1",
            "--maximum-total-requests",
            "10",
            "--input-price-per-million",
            "0.3",
            "--output-price-per-million",
            "1.2",
        ],
    )
    assert result.exit_code == 0, result.output
    output_path = next((tmp_path / "pilot").glob("*/generated-personas.parquet"))
    output = pl.read_parquet(output_path)
    assert output.get_column("persona_id").to_list() == ["persona-1", "persona-2"]
    assert MockGenerationClient.requests == 2


def test_pilot_revalidates_shards_without_rewriting_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh shard validation rejects stale reports and preserves their bytes."""
    paths = write_generation_inputs(root=tmp_path)
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", MockGenerationClient
    )
    MockGenerationClient.requests = 0
    result = CliRunner().invoke(
        pilot_main,
        [
            "--input",
            str(paths["sample"]),
            "--sample-manifest",
            str(paths["sample_manifest"]),
            "--config",
            str(paths["config"]),
            "--output-dir",
            str(tmp_path / "pilot"),
            "--rows",
            "1",
            "--batch-size",
            "1",
            "--concurrency",
            "1",
            "--maximum-total-requests",
            "10",
            "--input-price-per-million",
            "0.3",
            "--output-price-per-million",
            "1.2",
        ],
    )
    assert result.exit_code == 0, result.output
    pilot_dir = next((tmp_path / "pilot").iterdir())
    pilot_manifest_path = pilot_dir / "pilot-manifest.json"
    pilot_manifest = json.loads(pilot_manifest_path.read_text(encoding="utf-8"))
    reference = pilot_manifest["batch_runs"][0]
    report_path = pilot_dir / reference["validation_report_file"]
    report_bytes = report_path.read_bytes()
    report_sha256 = sha256_file(report_path)
    checkpoint_path = next((pilot_dir / "batches").glob("*/checkpoints/*.json"))
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint = json.loads(checkpoint_bytes)
    checkpoint["generation_context_sha256"] = "f" * 64
    checkpoint_path.write_text(json.dumps(checkpoint))

    assert not validate_persona_pilot(pilot_dir=pilot_dir).passed
    assert report_path.read_bytes() == report_bytes
    assert sha256_file(report_path) == report_sha256


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_manifest",
        "malformed_manifest",
        "accounting",
        "llm_flag",
        "pilot_id",
        "rows",
        "batch_reference",
        "config_path",
        "batch_upstream",
        "stored_report",
    ],
)
def test_pilot_validation_rejects_independent_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    """Each pilot identity, aggregate, and artefact tamper starts from validity."""
    paths = write_generation_inputs(root=tmp_path)
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", MockGenerationClient
    )
    MockGenerationClient.requests = 0
    pilot_dir = run_pilot(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "pilot",
        rows=1,
        batch_size=1,
        concurrency=1,
        delay_between_batches=0.0,
        maximum_total_requests=5,
        input_price_per_million=0.3,
        output_price_per_million=1.2,
    )
    assert validate_persona_pilot(pilot_dir=pilot_dir).passed
    manifest_path = pilot_dir / "pilot-manifest.json"

    if tamper == "missing_manifest":
        manifest_path.unlink()
    elif tamper == "malformed_manifest":
        manifest_path.write_text("{")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if tamper == "accounting":
            manifest["requests"] += 1
        elif tamper == "llm_flag":
            manifest["llm_generation"] = False
        elif tamper == "pilot_id":
            manifest["pilot_id"] = "forged-pilot-id"
        elif tamper == "rows":
            manifest["rows"] += 1
        elif tamper == "batch_reference":
            manifest["batch_runs"][0]["rows"] += 1
        elif tamper == "config_path":
            alternate_config = tmp_path / "same-generation.yaml"
            alternate_config.write_bytes(paths["config"].read_bytes())
            manifest["generation_config_file"] = str(alternate_config)
        elif tamper == "batch_upstream":
            reference = manifest["batch_runs"][0]
            batch_manifest_path = pilot_dir / reference["manifest_file"]
            batch_manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
            batch_manifest["upstream_run_id"] = "different-upstream"
            write_json(path=batch_manifest_path, payload=batch_manifest)
            reference["manifest_sha256"] = sha256_file(batch_manifest_path)
        else:
            reference = manifest["batch_runs"][0]
            report_path = pilot_dir / reference["validation_report_file"]
            stored_report = json.loads(report_path.read_text(encoding="utf-8"))
            stored_report["kind"] = "demographics"
            write_json(path=report_path, payload=stored_report)
            reference["validation_report_sha256"] = sha256_file(report_path)
        write_json(path=manifest_path, payload=manifest)

    assert not validate_persona_pilot(pilot_dir=pilot_dir).passed


def test_pilot_validation_rejects_mapping_binding_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pilot validation rejects forged references and stored report bindings."""
    paths = write_generation_inputs(root=tmp_path)
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", MockGenerationClient
    )
    pilot_dir = run_pilot(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "pilot",
        rows=1,
        batch_size=1,
        concurrency=1,
        delay_between_batches=0.0,
        maximum_total_requests=5,
        input_price_per_million=0.3,
        output_price_per_million=1.2,
    )
    manifest_path = pilot_dir / "pilot-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["batch_runs"][0]["job_title_mapping_version"] = 999
    write_json(path=manifest_path, payload=manifest)
    assert not validate_persona_pilot(pilot_dir=pilot_dir).passed

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = manifest["batch_runs"][0]
    report_path = pilot_dir / reference["validation_report_file"]
    stored_report = json.loads(report_path.read_text(encoding="utf-8"))
    stored_report["job_title_mapping_version"] = 999
    write_json(path=report_path, payload=stored_report)
    reference["validation_report_sha256"] = sha256_file(report_path)
    write_json(path=manifest_path, payload=manifest)
    assert not validate_persona_pilot(pilot_dir=pilot_dir).passed


def test_pilot_validation_uses_repository_root_from_another_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh pilot validation resolves relative evidence outside process cwd."""
    paths = write_generation_inputs(root=tmp_path)
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", MockGenerationClient
    )
    MockGenerationClient.requests = 0
    pilot_dir = run_pilot(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "pilot",
        rows=1,
        batch_size=1,
        concurrency=1,
        delay_between_batches=0.0,
        maximum_total_requests=5,
        input_price_per_million=0.3,
        output_price_per_million=1.2,
    )
    manifest_path = pilot_dir / "pilot-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative_input = paths["sample"].relative_to(tmp_path)
    relative_sample_manifest = paths["sample_manifest"].relative_to(tmp_path)
    relative_config = paths["config"].relative_to(tmp_path)
    manifest["input_file"] = str(relative_input)
    manifest["sample_manifest_file"] = str(relative_sample_manifest)
    manifest["generation_config_file"] = str(relative_config)
    for reference in manifest["batch_runs"]:
        shard_path = pilot_dir / reference["manifest_file"]
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        shard["input_file"] = str(relative_input)
        shard["sample_manifest_file"] = str(relative_sample_manifest)
        shard["generation_config_file"] = str(relative_config)
        write_json(path=shard_path, payload=shard)
        reference["manifest_sha256"] = sha256_file(shard_path)
    write_json(path=manifest_path, payload=manifest)
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "config").mkdir(parents=True)
    (elsewhere / "config" / "folk2-ieland-labels-da.yaml").write_bytes(
        (tmp_path / "config" / "folk2-ieland-labels-da.yaml").read_bytes()
    )
    monkeypatch.chdir(elsewhere)
    assert validate_persona_pilot(pilot_dir=pilot_dir, repository_root=tmp_path).passed
