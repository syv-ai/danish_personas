"""Validation report for completed LLM persona smoke runs."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from pydantic import ValidationError

from ..io import canonical_json, sha256_file, sha256_text, write_json
from ..models import MetricResult, ValidationReport
from .models import (
    GeneratedAttributes,
    GenerationManifest,
    PersonaCheckpoint,
    PersonaDescriptions,
)
from .pipeline import validate_upstream_sample
from .validation import VALIDATOR_VERSION, parse_attributes, parse_descriptions


def validate_persona_run(run_dir: Path) -> ValidationReport:
    """Validate output integrity, safety, and upstream preservation.

    Args:
        run_dir:
            Completed persona generation run.

    Returns:
        Machine-readable validation report.
    """
    manifest = GenerationManifest.model_validate_json(
        (run_dir / "generation-manifest.json").read_text()
    )
    output_path = run_dir / manifest.output_file
    output = pl.read_parquet(output_path)
    upstream = (
        pl.read_parquet(manifest.input_file).sort("persona_id").head(manifest.rows)
    )
    upstream_columns = upstream.columns
    ids = output.get_column("persona_id").to_list()
    try:
        validated_upstream = validate_upstream_sample(
            input_path=manifest.input_file,
            sample_manifest_path=manifest.sample_manifest_file,
        )
        provenance_passed = validated_upstream.run_id == manifest.upstream_run_id
    except OSError, ValueError, ValidationError:
        provenance_passed = False
    checks: list[MetricResult] = [
        _metric(
            name="output_checksum",
            passed=sha256_file(output_path) == manifest.output_sha256,
        ),
        _metric(name="row_count", passed=output.height == manifest.rows),
        _metric(name="upstream_provenance", passed=provenance_passed),
        _metric(
            name="ordered_persona_ids",
            passed=sha256_text(canonical_json(ids))
            == manifest.ordered_persona_ids_sha256,
        ),
        _metric(
            name="upstream_preservation",
            passed=output.select(upstream_columns).equals(upstream),
        ),
    ]
    validation_errors = 0
    checkpoint_errors = 0
    for row in output.iter_rows(named=True):
        try:
            attributes = GeneratedAttributes.model_validate(
                {name: row[name] for name in GeneratedAttributes.model_fields}
            )
            descriptions = PersonaDescriptions.model_validate(
                {name: row[name] for name in PersonaDescriptions.model_fields}
            )
            parse_attributes(attributes.model_dump_json())
            parse_descriptions(descriptions.model_dump_json())
        except ValidationError, ValueError:
            validation_errors += 1
        checkpoint_path = run_dir / "checkpoints" / f"{row['persona_id']}.json"
        try:
            checkpoint = PersonaCheckpoint.model_validate_json(
                checkpoint_path.read_text()
            )
            checkpoint_input = {name: row[name] for name in upstream_columns}
            if (
                checkpoint.input_sha256 != sha256_text(canonical_json(checkpoint_input))
                or checkpoint.generation_context_sha256
                != manifest.generation_context_sha256
                or checkpoint.validator_version != VALIDATOR_VERSION
                or checkpoint.validator_version != manifest.validator_version
                or any(
                    response.model != manifest.model
                    for response in checkpoint.responses
                )
            ):
                checkpoint_errors += 1
        except OSError, ValidationError:
            checkpoint_errors += 1
    checks.append(
        MetricResult(
            name="generated_content_errors",
            passed=validation_errors == 0,
            value=validation_errors,
            threshold=0,
            details=(
                "All generated fields satisfy schema, Danish, safety, and "
                "duplication gates."
            ),
        )
    )
    checks.append(
        MetricResult(
            name="checkpoint_provenance_errors",
            passed=checkpoint_errors == 0,
            value=checkpoint_errors,
            threshold=0,
            details="Checkpoints match their input, model, prompts, and validator.",
        )
    )
    report = ValidationReport(
        kind="personas",
        passed=all(metric.passed for metric in checks),
        created_at=datetime.now(tz=UTC).isoformat(),
        subject_id=manifest.run_id,
        metrics=checks,
    )
    write_json(path=run_dir / "validation-report.json", payload=report)
    return report


def _metric(name: str, passed: bool) -> MetricResult:
    return MetricResult(
        name=name,
        passed=passed,
        value="pass" if passed else "fail",
        threshold="pass",
        details="Mandatory persona generation integrity gate.",
    )
