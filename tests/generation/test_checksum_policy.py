"""Scoped checksum-policy regression tests."""

from pathlib import Path

import polars as pl
import pytest
from generation_test_helpers import write_generation_inputs

from danish_personas.checksum import ChecksumValidationPolicy
from danish_personas.generation.pipeline import validate_upstream_sample
from danish_personas.io import write_json
from danish_personas.models import FrozenSampleManifest, RunManifest, ValidationReport
from danish_personas.workflows import _valid_existing_sample
from scripts import generate_persona


def test_checksum_only_upstream_mismatches_are_scoped_to_ignore_policy(
    tmp_path: Path,
) -> None:
    """The CLI policy tolerates stored sample and upstream file hashes only."""
    paths = write_generation_inputs(root=tmp_path)
    sample_manifest = FrozenSampleManifest.model_validate_json(
        paths["sample_manifest"].read_text(encoding="utf-8")
    )
    write_json(
        path=paths["sample_manifest"],
        payload=sample_manifest.model_copy(update={"sha256": "0" * 64}),
    )
    run_manifest_path = paths["sample"].parent / "run-manifest.json"
    run_manifest = RunManifest.model_validate_json(
        run_manifest_path.read_text(encoding="utf-8")
    )
    write_json(
        path=run_manifest_path,
        payload=run_manifest.model_copy(update={"data_sha256": "1" * 64}),
    )

    with pytest.raises(ValueError, match="checksum"):
        validate_upstream_sample(
            input_path=paths["sample"], sample_manifest_path=paths["sample_manifest"]
        )
    assert (
        validate_upstream_sample(
            input_path=paths["sample"],
            sample_manifest_path=paths["sample_manifest"],
            checksum_policy=ChecksumValidationPolicy.IGNORE,
        ).run_id
        == "upstream-run"
    )


def test_checksum_tolerant_offset_selection_keeps_filename_and_row_count_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Random offset selection ignores only the persisted sample checksum."""
    paths = write_generation_inputs(root=tmp_path)
    sample_manifest_path = paths["sample_manifest"]
    sample_manifest = FrozenSampleManifest.model_validate_json(
        sample_manifest_path.read_text(encoding="utf-8")
    )
    write_json(
        path=sample_manifest_path,
        payload=sample_manifest.model_copy(update={"sha256": "0" * 64}),
    )
    monkeypatch.setattr(generate_persona.secrets, "randbelow", lambda bound: 0)

    with pytest.raises(ValueError, match="checksum"):
        generate_persona._sample_offset(
            input_path=paths["sample"], sample_manifest_path=sample_manifest_path
        )
    assert (
        generate_persona._sample_offset(
            input_path=paths["sample"],
            sample_manifest_path=sample_manifest_path,
            checksum_policy=ChecksumValidationPolicy.IGNORE,
        )
        == 0
    )
    assert not _valid_existing_sample(
        sample_path=paths["sample"],
        manifest_path=sample_manifest_path,
        source_run_id="upstream-run",
        rows=2,
    )
    assert _valid_existing_sample(
        sample_path=paths["sample"],
        manifest_path=sample_manifest_path,
        source_run_id="upstream-run",
        rows=2,
        checksum_policy=ChecksumValidationPolicy.IGNORE,
    )


def test_checksum_tolerant_upstream_validation_keeps_membership_strict(
    tmp_path: Path,
) -> None:
    """Ignoring hashes does not permit a row absent from the upstream run."""
    paths = write_generation_inputs(root=tmp_path)
    sample = pl.read_parquet(paths["sample"]).with_columns(
        pl.when(pl.col("persona_id") == "persona-1")
        .then(pl.lit("Aarhus"))
        .otherwise(pl.col("municipality"))
        .alias("municipality")
    )
    sample.write_parquet(paths["sample"])

    with pytest.raises(ValueError, match="absent from validated Phase-2 data"):
        validate_upstream_sample(
            input_path=paths["sample"],
            sample_manifest_path=paths["sample_manifest"],
            checksum_policy=ChecksumValidationPolicy.IGNORE,
        )


def test_origin_hash_only_mismatch_is_tolerated_without_content_relaxation(
    tmp_path: Path,
) -> None:
    """The scoped policy allows a shared hash drift but still checks content."""
    paths = write_generation_inputs(root=tmp_path)
    bad_hash = "f" * 64
    sample_manifest_path = paths["sample_manifest"]
    sample_manifest = FrozenSampleManifest.model_validate_json(
        sample_manifest_path.read_text(encoding="utf-8")
    )
    write_json(
        path=sample_manifest_path,
        payload=sample_manifest.model_copy(
            update={"origin_labels_contract_sha256": bad_hash}
        ),
    )
    run_manifest_path = paths["sample"].parent / "run-manifest.json"
    run_manifest = RunManifest.model_validate_json(
        run_manifest_path.read_text(encoding="utf-8")
    )
    write_json(
        path=run_manifest_path,
        payload=run_manifest.model_copy(
            update={"origin_labels_contract_sha256": bad_hash}
        ),
    )
    report_path = paths["sample"].parent / "validation-report.json"
    report = ValidationReport.model_validate_json(
        report_path.read_text(encoding="utf-8")
    )
    write_json(
        path=report_path,
        payload=report.model_copy(update={"origin_labels_contract_sha256": bad_hash}),
    )

    assert (
        validate_upstream_sample(
            input_path=paths["sample"],
            sample_manifest_path=sample_manifest_path,
            checksum_policy=ChecksumValidationPolicy.IGNORE,
        ).run_id
        == "upstream-run"
    )

    write_json(
        path=sample_manifest_path,
        payload=sample_manifest.model_copy(
            update={
                "origin_labels_contract_sha256": bad_hash,
                "origin_labels_contract_content": "not the reviewed contract",
            }
        ),
    )
    with pytest.raises(ValueError):
        validate_upstream_sample(
            input_path=paths["sample"],
            sample_manifest_path=sample_manifest_path,
            checksum_policy=ChecksumValidationPolicy.IGNORE,
        )
