"""Validation report for completed LLM persona smoke runs."""

import math
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
    PilotManifest,
)
from .pipeline import models_match, validate_upstream_sample
from .validation import VALIDATOR_VERSION, parse_attributes, parse_descriptions


def validate_persona_pilot(pilot_dir: Path) -> ValidationReport:
    """Validate a merged pilot and every checksummed shard.

    Args:
        pilot_dir:
            Completed pilot directory.

    Returns:
        Machine-readable validation report.
    """
    manifest = PilotManifest.model_validate_json(
        (pilot_dir / "pilot-manifest.json").read_text()
    )
    output_path = pilot_dir / manifest.output_file
    output = pl.read_parquet(output_path)
    expected = (
        pl.read_parquet(manifest.input_file).sort("persona_id").head(manifest.rows)
    )
    batch_outputs: list[pl.DataFrame] = []
    batch_manifests: list[GenerationManifest] = []
    batch_errors = 0
    offsets: list[int] = []
    for reference in manifest.batch_runs:
        manifest_path = pilot_dir / reference.manifest_file
        report_path = pilot_dir / reference.validation_report_file
        try:
            batch_manifest = GenerationManifest.model_validate_json(
                manifest_path.read_text()
            )
            batch_report = ValidationReport.model_validate_json(report_path.read_text())
            if (
                sha256_file(manifest_path) != reference.manifest_sha256
                or sha256_file(report_path) != reference.validation_report_sha256
                or batch_manifest.run_id != reference.run_id
                or batch_manifest.offset != reference.offset
                or batch_manifest.rows != reference.rows
                or batch_manifest.model != manifest.model
                or batch_manifest.base_url != manifest.base_url
                or batch_manifest.generation_context_sha256
                != manifest.generation_context_sha256
                or batch_manifest.requests > manifest.maximum_shard_requests
                or not batch_report.passed
                or batch_report.subject_id != batch_manifest.run_id
            ):
                batch_errors += 1
                continue
            batch_output_path = manifest_path.parent / batch_manifest.output_file
            if sha256_file(batch_output_path) != batch_manifest.output_sha256:
                batch_errors += 1
                continue
            batch_outputs.append(pl.read_parquet(batch_output_path))
            batch_manifests.append(batch_manifest)
            offsets.append(reference.offset)
        except OSError, ValidationError, ValueError:
            batch_errors += 1
    expected_offsets = list(range(0, manifest.rows, manifest.batch_size))
    merged_batches = (
        pl.concat(batch_outputs).sort("persona_id") if batch_outputs else pl.DataFrame()
    )
    content_errors = _count_content_errors(output=output)
    try:
        upstream = validate_upstream_sample(
            input_path=manifest.input_file,
            sample_manifest_path=manifest.sample_manifest_file,
        )
        provenance_passed = (
            sha256_file(manifest.input_file) == manifest.input_sha256
            and sha256_file(manifest.sample_manifest_file)
            == manifest.sample_manifest_sha256
            and sha256_file(manifest.generation_config_file)
            == manifest.generation_config_sha256
            and upstream.run_id == manifest.upstream_run_id
        )
    except OSError, ValidationError, ValueError:
        provenance_passed = False
    checks = [
        _metric(
            name="output_checksum",
            passed=sha256_file(output_path) == manifest.output_sha256,
        ),
        _metric(name="row_count", passed=output.height == manifest.rows),
        _metric(name="upstream_provenance", passed=provenance_passed),
        _metric(
            name="upstream_preservation",
            passed=output.select(expected.columns).equals(expected),
        ),
        _metric(
            name="batch_provenance",
            passed=(
                batch_errors == 0
                and len(manifest.batch_runs) == manifest.batches
                and sorted(offsets) == expected_offsets
                and merged_batches.equals(output)
            ),
            value=batch_errors,
        ),
        _metric(
            name="aggregate_provenance",
            passed=_pilot_aggregates_match(
                manifest=manifest, batch_manifests=batch_manifests
            ),
        ),
        _metric(
            name="request_budget",
            passed=manifest.requests <= manifest.maximum_total_requests,
            value=manifest.requests,
            threshold=manifest.maximum_total_requests,
        ),
        _metric(
            name="generated_content_errors",
            passed=content_errors == 0,
            value=content_errors,
            threshold=0,
        ),
    ]
    report = ValidationReport(
        kind="persona_pilot",
        passed=all(metric.passed for metric in checks),
        created_at=datetime.now(tz=UTC).isoformat(),
        subject_id=manifest.pilot_id,
        metrics=checks,
    )
    write_json(path=pilot_dir / "pilot-validation-report.json", payload=report)
    return report


def _count_content_errors(output: pl.DataFrame) -> int:
    errors = 0
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
            errors += 1
    return errors


def _metric(
    name: str,
    passed: bool,
    value: float | int | str | None = None,
    threshold: float | int | str | None = None,
) -> MetricResult:
    return MetricResult(
        name=name,
        passed=passed,
        value=value if value is not None else ("pass" if passed else "fail"),
        threshold=threshold if threshold is not None else "pass",
        details="Mandatory persona generation integrity gate.",
    )


def _pilot_aggregates_match(
    manifest: PilotManifest, batch_manifests: list[GenerationManifest]
) -> bool:
    if not batch_manifests:
        return False
    prompt_tokens = sum(item.prompt_tokens for item in batch_manifests)
    completion_tokens = sum(item.completion_tokens for item in batch_manifests)
    list_price_cost = (
        prompt_tokens * manifest.input_price_per_million_usd
        + completion_tokens * manifest.output_price_per_million_usd
    ) / 1_000_000
    provider_costs = [item.estimated_cost_usd for item in batch_manifests]
    provider_cost = (
        sum(cost for cost in provider_costs if cost is not None)
        if all(cost is not None for cost in provider_costs)
        else None
    )
    providers = sorted(
        {provider for item in batch_manifests for provider in item.inference_providers}
    )
    return (
        manifest.rows == sum(item.rows for item in batch_manifests)
        and manifest.batches == len(batch_manifests)
        and manifest.requests == sum(item.requests for item in batch_manifests)
        and manifest.retries == sum(item.retries for item in batch_manifests)
        and manifest.prompt_tokens == prompt_tokens
        and manifest.completion_tokens == completion_tokens
        and manifest.total_tokens == sum(item.total_tokens for item in batch_manifests)
        and math.isclose(
            manifest.list_price_estimated_cost_usd, list_price_cost, rel_tol=1e-12
        )
        and manifest.provider_estimated_cost_usd == provider_cost
        and manifest.inference_providers == providers
        and all(
            item.upstream_run_id == manifest.upstream_run_id
            and item.input_file == manifest.input_file
            and item.input_sha256 == manifest.input_sha256
            and item.sample_manifest_file == manifest.sample_manifest_file
            and item.generation_config_file == manifest.generation_config_file
            and item.generation_config_sha256 == manifest.generation_config_sha256
            and item.generation_context_sha256 == manifest.generation_context_sha256
            and item.validator_version == manifest.validator_version
            and item.attributes_prompt_sha256 == manifest.attributes_prompt_sha256
            and item.personas_prompt_sha256 == manifest.personas_prompt_sha256
            and item.model == manifest.model
            and item.base_url == manifest.base_url
            for item in batch_manifests
        )
    )


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
        pl.read_parquet(manifest.input_file)
        .sort("persona_id")
        .slice(manifest.offset, manifest.rows)
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
                    not models_match(configured=manifest.model, returned=response.model)
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
