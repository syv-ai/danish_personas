"""Validation report for completed LLM persona smoke runs."""

import math
import os
import typing as t
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ..io import canonical_json, load_yaml_model, sha256_file, sha256_text, write_json
from ..models import MetricResult, ValidationReport
from .identity import generation_run_id, persona_pilot_id
from .job_titles import (
    DEFAULT_JOB_TITLE_MAPPING_PATH,
    JobFunctionTitleMapping,
    job_title_mapping_sha256,
    load_job_title_mapping,
)
from .models import (
    GeneratedAttributes,
    GenerationConfig,
    GenerationManifest,
    PersonaCheckpoint,
    PersonaDescriptions,
    PilotManifest,
    RequestLedger,
)
from .pipeline import generation_context_sha256, models_match, validate_upstream_sample
from .validation import VALIDATOR_VERSION, parse_attributes, parse_descriptions


def validate_persona_pilot(
    pilot_dir: Path, repository_root: Path | None = None
) -> ValidationReport:
    """Validate a merged pilot and freshly validate every shard.

    Args:
        pilot_dir:
            Completed pilot directory.
        repository_root (optional):
            Repository root for manifest input and configuration paths.

    Returns:
        Machine-readable validation report, including for malformed artefacts.
    """
    try:
        report = _build_persona_pilot_report(
            pilot_dir=pilot_dir, repository_root=repository_root
        )
    except (OSError, UnicodeError, ValueError, pl.exceptions.PolarsError) as error:
        report = _failed_report(
            kind="persona_pilot", subject_id=pilot_dir.name, error=error
        )
    write_json(path=pilot_dir / "pilot-validation-report.json", payload=report)
    return report


def _build_persona_pilot_report(
    pilot_dir: Path, repository_root: Path | None = None
) -> ValidationReport:
    """Build a merged-pilot report and freshly validate every shard.

    Args:
        pilot_dir:
            Completed pilot directory.
        repository_root (optional):
            Repository root for manifest input and configuration paths.

    Returns:
        Machine-readable validation report.
    """
    manifest = PilotManifest.model_validate_json(
        (pilot_dir / "pilot-manifest.json").read_text(encoding="utf-8")
    )
    try:
        output = pl.read_parquet(pilot_dir / manifest.output_file)
    except OSError, UnicodeError, ValueError, pl.exceptions.PolarsError:
        output = pl.DataFrame()
    try:
        expected = (
            pl.read_parquet(_repository_path(repository_root, manifest.input_file))
            .sort("persona_id")
            .head(manifest.rows)
        )
    except OSError, UnicodeError, ValueError, pl.exceptions.PolarsError:
        expected = pl.DataFrame()
    batch_outputs: list[pl.DataFrame] = []
    batch_manifests: list[GenerationManifest] = []
    batch_errors = 0
    offsets: list[int] = []
    for reference in manifest.batch_runs:
        manifest_path = pilot_dir / reference.manifest_file
        report_path = pilot_dir / reference.validation_report_file
        try:
            batch_manifest = GenerationManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            batch_report = ValidationReport.model_validate_json(
                report_path.read_text(encoding="utf-8")
            )
            fresh_report = _build_persona_run_report(
                manifest_path.parent, repository_root=repository_root
            )
            if (
                not _checksum_matches(manifest_path, reference.manifest_sha256)
                or not _checksum_matches(
                    report_path, reference.validation_report_sha256
                )
                or batch_manifest.run_id != reference.run_id
                or batch_manifest.offset != reference.offset
                or batch_manifest.rows != reference.rows
                or batch_manifest.model != manifest.model
                or batch_manifest.base_url != manifest.base_url
                or batch_manifest.generation_context_sha256
                != manifest.generation_context_sha256
                or (
                    reference.job_title_mapping_file
                    != batch_manifest.job_title_mapping_file
                    or reference.job_title_mapping_sha256
                    != batch_manifest.job_title_mapping_sha256
                    or reference.job_title_mapping_version
                    != batch_manifest.job_title_mapping_version
                    or reference.job_title_mapping_content
                    != batch_manifest.job_title_mapping_content
                )
                or (
                    batch_report.job_title_mapping_file
                    != batch_manifest.job_title_mapping_file
                    or batch_report.job_title_mapping_sha256
                    != batch_manifest.job_title_mapping_sha256
                    or batch_report.job_title_mapping_version
                    != batch_manifest.job_title_mapping_version
                    or batch_report.job_title_mapping_content
                    != (
                        batch_manifest.job_title_mapping_content.model_dump(mode="json")
                        if batch_manifest.job_title_mapping_content is not None
                        else None
                    )
                )
                or batch_manifest.job_title_mapping_file
                != manifest.job_title_mapping_file
                or batch_manifest.job_title_mapping_sha256
                != manifest.job_title_mapping_sha256
                or batch_manifest.job_title_mapping_version
                != manifest.job_title_mapping_version
                or batch_manifest.job_title_mapping_content
                != manifest.job_title_mapping_content
                or batch_manifest.requests > manifest.maximum_shard_requests
                or batch_report.kind != "personas"
                or not batch_report.passed
                or batch_report.subject_id != batch_manifest.run_id
                or not fresh_report.passed
            ):
                batch_errors += 1
                continue
            batch_output_path = manifest_path.parent / batch_manifest.output_file
            batch_outputs.append(pl.read_parquet(batch_output_path))
            batch_manifests.append(batch_manifest)
            offsets.append(reference.offset)
        except OSError, UnicodeError, ValueError, pl.exceptions.PolarsError:
            batch_errors += 1
    expected_ranges = [
        (offset, min(manifest.batch_size, manifest.rows - offset))
        for offset in range(0, manifest.rows, manifest.batch_size)
    ]
    expected_offsets = [offset for offset, _ in expected_ranges]
    references_match = [
        (reference.offset, reference.rows) for reference in manifest.batch_runs
    ] == expected_ranges
    try:
        merged_batches = (
            pl.concat(batch_outputs).sort("persona_id")
            if batch_outputs
            else pl.DataFrame()
        )
    except OSError, UnicodeError, ValueError, pl.exceptions.PolarsError:
        merged_batches = pl.DataFrame()
    config_path = _repository_path(repository_root, manifest.generation_config_file)
    mapping_binding = _load_mapping_binding(
        config_path=config_path, repository_root=repository_root
    )
    content_errors = _count_content_errors(
        output=output, mapping_binding=mapping_binding
    )
    provenance_passed = _pilot_provenance_matches(
        pilot_dir=pilot_dir, manifest=manifest, repository_root=repository_root
    )
    checks = [
        _metric(
            name="output_checksum",
            passed=_checksum_matches(
                pilot_dir / manifest.output_file, manifest.output_sha256
            ),
        ),
        _metric(name="row_count", passed=output.height == manifest.rows),
        _metric(name="upstream_provenance", passed=provenance_passed),
        _metric(
            name="upstream_preservation",
            passed=_frames_equal(output=output, expected=expected),
        ),
        _metric(
            name="batch_provenance",
            passed=(
                batch_errors == 0
                and len(manifest.batch_runs) == manifest.batches
                and references_match
                and offsets == expected_offsets
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
        job_title_mapping_file=manifest.job_title_mapping_file,
        job_title_mapping_sha256=manifest.job_title_mapping_sha256,
        job_title_mapping_version=manifest.job_title_mapping_version,
        job_title_mapping_content=(
            manifest.job_title_mapping_content.model_dump(mode="json")
            if manifest.job_title_mapping_content is not None
            else None
        ),
    )
    return report


def _build_persona_run_report(
    run_dir: Path, repository_root: Path | None = None
) -> ValidationReport:
    """Build a persona-run report without writing it to disk.

    Returns:
        Validation report for the run.
    """
    manifest = GenerationManifest.model_validate_json(
        (run_dir / "generation-manifest.json").read_text(encoding="utf-8")
    )
    output_path = run_dir / manifest.output_file
    try:
        output = pl.read_parquet(output_path)
    except OSError, UnicodeError, ValueError, pl.exceptions.PolarsError:
        output = pl.DataFrame()
    try:
        upstream = (
            pl.read_parquet(_repository_path(repository_root, manifest.input_file))
            .sort("persona_id")
            .slice(manifest.offset, manifest.rows)
        )
    except OSError, UnicodeError, ValueError, pl.exceptions.PolarsError:
        upstream = pl.DataFrame()
    try:
        ids = output.get_column("persona_id").to_list()
    except OSError, UnicodeError, ValueError, pl.exceptions.PolarsError:
        ids = []
    provenance_passed = _persona_provenance_matches(
        manifest=manifest, repository_root=repository_root
    )
    checks: list[MetricResult] = [
        _metric(
            name="output_checksum",
            passed=_checksum_matches(output_path, manifest.output_sha256),
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
            passed=_frames_equal(output=output, expected=upstream),
        ),
    ]
    config_path = (
        _repository_path(repository_root, manifest.generation_config_file)
        if manifest.generation_config_file is not None
        else None
    )
    mapping_binding = _load_mapping_binding(
        config_path=config_path, repository_root=repository_root
    )
    validation_errors = _count_content_errors(
        output=output, mapping_binding=mapping_binding
    )
    checkpoint_errors = _count_checkpoint_errors(
        run_dir=run_dir,
        output=output,
        upstream_columns=upstream.columns,
        manifest=manifest,
        config_path=config_path,
        mapping_binding=mapping_binding,
        repository_root=repository_root,
    )
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
    return ValidationReport(
        kind="personas",
        passed=all(metric.passed for metric in checks),
        created_at=datetime.now(tz=UTC).isoformat(),
        subject_id=manifest.run_id,
        metrics=checks,
        job_title_mapping_file=manifest.job_title_mapping_file,
        job_title_mapping_sha256=manifest.job_title_mapping_sha256,
        job_title_mapping_version=manifest.job_title_mapping_version,
        job_title_mapping_content=(
            manifest.job_title_mapping_content.model_dump(mode="json")
            if manifest.job_title_mapping_content is not None
            else None
        ),
    )


def _checksum_matches(path: Path | None, expected: str) -> bool:
    """Return whether a file exists and has the expected checksum."""
    try:
        return path is not None and sha256_file(path) == expected
    except OSError, UnicodeError, ValueError, pl.exceptions.PolarsError:
        return False


def _count_checkpoint_errors(
    *,
    run_dir: Path,
    output: pl.DataFrame,
    upstream_columns: list[str],
    manifest: GenerationManifest,
    config_path: Path | None,
    mapping_binding: tuple[Path, JobFunctionTitleMapping, str] | None,
    repository_root: Path | None,
) -> int:
    """Count checkpoint, response-sequence, ledger, and accounting errors.

    Returns:
        Number of checkpoint integrity errors.
    """
    errors = 0
    output_ids: set[str] = set()
    checkpoints: list[PersonaCheckpoint] = []
    required_columns = {
        "persona_id",
        *upstream_columns,
        *GeneratedAttributes.model_fields,
        *PersonaDescriptions.model_fields,
    }
    if not required_columns.issubset(output.columns):
        return 1
    for row in output.iter_rows(named=True):
        try:
            persona_id = str(row["persona_id"])
            output_ids.add(persona_id)
            checkpoint = PersonaCheckpoint.model_validate_json(
                (run_dir / "checkpoints" / f"{persona_id}.json").read_text(
                    encoding="utf-8"
                )
            )
            checkpoints.append(checkpoint)
            checkpoint_input = {name: row[name] for name in upstream_columns}
            checkpoint_values = {
                **checkpoint.attributes.model_dump(),
                **checkpoint.descriptions.model_dump(),
            }
            output_values = {name: row[name] for name in checkpoint_values}
            replay_valid, stage_attempts = _responses_match_checkpoint(
                checkpoint=checkpoint, demographic=row, mapping_binding=mapping_binding
            )
            if (
                checkpoint.persona_id != persona_id
                or checkpoint.input_sha256
                != sha256_text(canonical_json(checkpoint_input))
                or checkpoint.generation_context_sha256
                != manifest.generation_context_sha256
                or checkpoint.validator_version != VALIDATOR_VERSION
                or checkpoint.validator_version != manifest.validator_version
                or mapping_binding is None
                or not _mapping_binding_matches(
                    path=checkpoint.job_title_mapping_file,
                    sha256=checkpoint.job_title_mapping_sha256,
                    version=checkpoint.job_title_mapping_version,
                    content=checkpoint.job_title_mapping_content,
                    expected_path=mapping_binding[0],
                    expected_mapping=mapping_binding[1],
                    expected_sha256=mapping_binding[2],
                    repository_root=repository_root,
                )
                or checkpoint_values != output_values
                or checkpoint.attempts != len(checkpoint.responses)
                or checkpoint.http_requests
                < sum(response.request_attempts for response in checkpoint.responses)
                or any(
                    not models_match(configured=manifest.model, returned=response.model)
                    or response.total_tokens
                    != response.prompt_tokens + response.completion_tokens
                    for response in checkpoint.responses
                )
                or not replay_valid
                or not _stage_attempts_within_config(
                    config_path=config_path, stage_attempts=stage_attempts
                )
            ):
                errors += 1
        except OSError, UnicodeError, ValueError, pl.exceptions.PolarsError:
            errors += 1
    try:
        checkpoint_paths = list((run_dir / "checkpoints").glob("*.json"))
        checkpoint_ids = {
            path.stem for path in checkpoint_paths if ".attributes" not in path.stem
        }
        errors += len(checkpoint_ids - output_ids)
        errors += sum(
            path.name.endswith(".attributes.json") for path in checkpoint_paths
        )
    except OSError, UnicodeError, ValueError, pl.exceptions.PolarsError:
        errors += 1
    if not _checkpoint_accounting_matches(
        run_dir=run_dir,
        manifest=manifest,
        checkpoints=checkpoints,
        config_path=config_path,
    ):
        errors += 1
    return errors


def _checkpoint_accounting_matches(
    *,
    run_dir: Path,
    manifest: GenerationManifest,
    checkpoints: list[PersonaCheckpoint],
    config_path: Path | None,
) -> bool:
    """Bind checkpoint usage and the request ledger to the generation manifest.

    Returns:
        Whether all request and response accounting matches exactly.
    """
    if config_path is None or len(checkpoints) != manifest.rows:
        return False
    try:
        config = load_yaml_model(path=config_path, model=GenerationConfig)
        ledger = RequestLedger.model_validate_json(
            (run_dir / "request-ledger.json").read_text(encoding="utf-8")
        )
    except OSError, UnicodeError, ValueError, pl.exceptions.PolarsError:
        return False
    responses = [
        response for checkpoint in checkpoints for response in checkpoint.responses
    ]
    costs = [response.estimated_cost_usd for response in responses]
    estimated_cost = (
        sum(cost for cost in costs if cost is not None)
        if costs and all(cost is not None for cost in costs)
        else None
    )
    providers = sorted(
        {
            response.inference_provider
            for response in responses
            if response.inference_provider is not None
        }
    )
    http_requests = sum(checkpoint.http_requests for checkpoint in checkpoints)
    return (
        manifest.requests == http_requests
        and ledger.attempts == http_requests
        and ledger.generation_context_sha256 == manifest.generation_context_sha256
        and ledger.maximum_attempts == config.maximum_total_requests
        and ledger.attempts <= ledger.maximum_attempts
        and manifest.requests >= manifest.rows * 2
        and manifest.retries == manifest.requests - manifest.rows * 2
        and manifest.prompt_tokens
        == sum(response.prompt_tokens for response in responses)
        and manifest.completion_tokens
        == sum(response.completion_tokens for response in responses)
        and manifest.total_tokens
        == sum(response.total_tokens for response in responses)
        and manifest.estimated_cost_usd == estimated_cost
        and manifest.inference_providers == providers
    )


def _mapping_binding_matches(
    *,
    path: Path | None,
    sha256: str | None,
    version: int | None,
    content: JobFunctionTitleMapping | None,
    expected_path: Path,
    repository_root: Path | None = None,
    expected_mapping: JobFunctionTitleMapping,
    expected_sha256: str,
) -> bool:
    """Check every persisted title-mapping binding against the effective input.

    Returns:
        Whether all persisted mapping fields match the effective allowlist.
    """
    return (
        path is not None
        and _repository_path(repository_root, path) == expected_path
        and sha256 == expected_sha256
        and version == expected_mapping.version
        and content == expected_mapping
    )


def _repository_path(root: Path | None, value: Path | None) -> Path:
    if value is None:
        raise ValueError("A repository path is required")
    base = Path.cwd() if root is None else root
    candidate = value if value.is_absolute() else base / value
    return Path(os.path.abspath(os.path.normpath(candidate)))


def _responses_match_checkpoint(
    *,
    checkpoint: PersonaCheckpoint,
    demographic: dict[str, object],
    mapping_binding: tuple[Path, JobFunctionTitleMapping, str] | None,
) -> tuple[bool, tuple[int, int]]:
    """Replay the two response stages and bind accepted content to the checkpoint.

    Returns:
        Whether the sequence is valid and the number of responses for each stage.
    """
    if mapping_binding is None:
        return False, (0, 0)
    parsers = (
        lambda content: parse_attributes(
            content, demographic, job_title_mapping=mapping_binding[1]
        ),
        lambda content: parse_descriptions(content, demographic, checkpoint.attributes),
    )
    expected = (checkpoint.attributes, checkpoint.descriptions)
    response_index = 0
    stage_attempts: list[int] = []
    for parser, expected_value in zip(parsers, expected, strict=True):
        attempts = 0
        accepted = False
        while response_index < len(checkpoint.responses):
            response = checkpoint.responses[response_index]
            response_index += 1
            attempts += 1
            try:
                parsed = parser(response.content)
            except ValueError:
                continue
            if parsed != expected_value:
                return False, (0, 0)
            accepted = True
            break
        if not accepted:
            return False, (0, 0)
        stage_attempts.append(attempts)
    return response_index == len(checkpoint.responses), (
        stage_attempts[0],
        stage_attempts[1],
    )


def _stage_attempts_within_config(
    *, config_path: Path | None, stage_attempts: tuple[int, int]
) -> bool:
    """Check response attempts against the persisted generation configuration.

    Returns:
        Whether each stage stayed within its validation-attempt limit.
    """
    if config_path is None:
        return False
    try:
        config = load_yaml_model(path=config_path, model=GenerationConfig)
    except OSError, UnicodeError, ValueError, pl.exceptions.PolarsError:
        return False
    return all(
        1 <= attempts <= config.maximum_validation_attempts
        for attempts in stage_attempts
    )


def _count_content_errors(
    *,
    output: pl.DataFrame,
    mapping_binding: tuple[Path, JobFunctionTitleMapping, str] | None,
) -> int:
    errors = 0
    required_columns = {
        *GeneratedAttributes.model_fields,
        *PersonaDescriptions.model_fields,
    }
    if mapping_binding is None:
        return max(1, output.height)
    if not required_columns.issubset(output.columns):
        return max(1, output.height)
    for row in output.iter_rows(named=True):
        try:
            attributes = GeneratedAttributes.model_validate(
                {name: row[name] for name in GeneratedAttributes.model_fields}
            )
            descriptions = PersonaDescriptions.model_validate(
                {name: row[name] for name in PersonaDescriptions.model_fields}
            )
            parse_attributes(
                attributes.model_dump_json(), row, job_title_mapping=mapping_binding[1]
            )
            parse_descriptions(descriptions.model_dump_json(), row, attributes)
        except OSError, UnicodeError, ValueError, pl.exceptions.PolarsError:
            errors += 1
    return errors


def _frames_equal(*, output: pl.DataFrame, expected: pl.DataFrame) -> bool:
    """Compare frames without allowing a malformed schema to raise.

    Returns:
        Whether the output preserves the expected frame.
    """
    try:
        return output.select(expected.columns).equals(expected)
    except OSError, UnicodeError, ValueError, pl.exceptions.PolarsError:
        return False


def _load_mapping_binding(
    *, config_path: Path | None, repository_root: Path | None
) -> tuple[Path, JobFunctionTitleMapping, str] | None:
    """Load the validated title mapping selected by a generation config.

    Returns:
        The effective mapping binding, or ``None`` when its config or mapping is
        unavailable or invalid.
    """
    if config_path is None:
        return None
    try:
        config = load_yaml_model(path=config_path, model=GenerationConfig)
        return _effective_mapping(config=config, repository_root=repository_root)
    except OSError, UnicodeError, ValueError, pl.exceptions.PolarsError:
        return None


def _effective_mapping(
    config: GenerationConfig, repository_root: Path | None
) -> tuple[Path, JobFunctionTitleMapping, str]:
    """Load the exact title mapping selected by the effective configuration.

    Returns:
        Effective mapping path, parsed allowlist, and file checksum.
    """
    path = _repository_path(
        repository_root, config.job_title_mapping or DEFAULT_JOB_TITLE_MAPPING_PATH
    )
    mapping = load_job_title_mapping(path=path)
    return path, mapping, job_title_mapping_sha256(path=path)


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


def _persona_provenance_matches(
    manifest: GenerationManifest, repository_root: Path | None = None
) -> bool:
    """Recompute the run's identity, range, and generation-context bindings.

    Returns:
        Whether all derivable provenance bindings match.
    """
    try:
        input_path = _repository_path(repository_root, manifest.input_file)
        sample_manifest_path = _repository_path(
            repository_root, manifest.sample_manifest_file
        )
        validated_upstream = validate_upstream_sample(
            input_path=input_path, sample_manifest_path=sample_manifest_path
        )
        if manifest.generation_config_file is None:
            return False
        config_path = _repository_path(repository_root, manifest.generation_config_file)
        config = load_yaml_model(path=config_path, model=GenerationConfig)
        mapping_path, mapping, mapping_sha256 = _effective_mapping(
            config=config, repository_root=repository_root
        )
        attributes_prompt = _repository_path(
            repository_root, config.attributes_prompt
        ).read_text(encoding="utf-8")
        personas_prompt = _repository_path(
            repository_root, config.personas_prompt
        ).read_text(encoding="utf-8")
        sample = pl.read_parquet(input_path).sort("persona_id")
        if manifest.offset + manifest.rows > sample.height:
            return False
        selected_ids = (
            sample.slice(manifest.offset, manifest.rows)
            .get_column("persona_id")
            .to_list()
        )
        ordered_ids_sha256 = sha256_text(canonical_json(selected_ids))
        context_sha256 = generation_context_sha256(
            config=config,
            attributes_prompt=attributes_prompt,
            personas_prompt=personas_prompt,
            job_title_mapping=mapping,
            job_title_mapping_sha256=mapping_sha256,
        )
        input_sha256 = sha256_file(input_path)
        return (
            manifest.llm_generation
            and manifest.validator_version == VALIDATOR_VERSION
            and config.llm_generation_enabled
            and input_sha256 == manifest.input_sha256
            and _checksum_matches(config_path, manifest.generation_config_sha256)
            and validated_upstream.run_id == manifest.upstream_run_id
            and config.model == manifest.model
            and config.base_url == manifest.base_url
            and manifest.rows <= config.maximum_smoke_rows
            and manifest.requests <= config.maximum_total_requests
            and ordered_ids_sha256 == manifest.ordered_persona_ids_sha256
            and sha256_text(attributes_prompt) == manifest.attributes_prompt_sha256
            and sha256_text(personas_prompt) == manifest.personas_prompt_sha256
            and context_sha256 == manifest.generation_context_sha256
            and _mapping_binding_matches(
                path=manifest.job_title_mapping_file,
                sha256=manifest.job_title_mapping_sha256,
                version=manifest.job_title_mapping_version,
                content=manifest.job_title_mapping_content,
                expected_path=mapping_path,
                expected_mapping=mapping,
                expected_sha256=mapping_sha256,
                repository_root=repository_root,
            )
            and manifest.run_id
            == generation_run_id(
                input_sha256=input_sha256,
                generation_context_sha256=context_sha256,
                ordered_persona_ids_sha256=ordered_ids_sha256,
            )
        )
    except OSError, UnicodeError, ValueError, pl.exceptions.PolarsError:
        return False


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
        manifest.llm_generation
        and manifest.validator_version == VALIDATOR_VERSION
        and manifest.rows == sum(item.rows for item in batch_manifests)
        and manifest.batches == len(batch_manifests)
        and manifest.batches == len(manifest.batch_runs)
        and manifest.requests == sum(item.requests for item in batch_manifests)
        and manifest.retries == manifest.requests - manifest.rows * 2
        and manifest.retries == sum(item.retries for item in batch_manifests)
        and manifest.prompt_tokens == prompt_tokens
        and manifest.completion_tokens == completion_tokens
        and manifest.total_tokens == sum(item.total_tokens for item in batch_manifests)
        and manifest.list_price_estimated_cost_usd == list_price_cost
        and manifest.provider_estimated_cost_usd == provider_cost
        and manifest.inference_providers == providers
        and all(
            item.llm_generation
            and item.upstream_run_id == manifest.upstream_run_id
            and item.input_file == manifest.input_file
            and item.input_sha256 == manifest.input_sha256
            and item.sample_manifest_file == manifest.sample_manifest_file
            and item.generation_config_file == manifest.generation_config_file
            and item.generation_config_sha256 == manifest.generation_config_sha256
            and item.generation_context_sha256 == manifest.generation_context_sha256
            and item.validator_version == manifest.validator_version
            and item.job_title_mapping_file == manifest.job_title_mapping_file
            and item.job_title_mapping_sha256 == manifest.job_title_mapping_sha256
            and item.job_title_mapping_version == manifest.job_title_mapping_version
            and item.job_title_mapping_content == manifest.job_title_mapping_content
            and item.attributes_prompt_sha256 == manifest.attributes_prompt_sha256
            and item.personas_prompt_sha256 == manifest.personas_prompt_sha256
            and item.model == manifest.model
            and item.base_url == manifest.base_url
            for item in batch_manifests
        )
    )


def _pilot_provenance_matches(
    *, pilot_dir: Path, manifest: PilotManifest, repository_root: Path | None = None
) -> bool:
    """Recompute the pilot identity and all derivable provenance bindings.

    Returns:
        Whether all pilot provenance and identity invariants hold.
    """
    try:
        input_path = _repository_path(repository_root, manifest.input_file)
        sample_manifest_path = _repository_path(
            repository_root, manifest.sample_manifest_file
        )
        config_path = _repository_path(repository_root, manifest.generation_config_file)
        upstream = validate_upstream_sample(
            input_path=input_path, sample_manifest_path=sample_manifest_path
        )
        config = load_yaml_model(path=config_path, model=GenerationConfig)
        mapping_path, mapping, mapping_sha256 = _effective_mapping(
            config=config, repository_root=repository_root
        )
        attributes_prompt = _repository_path(
            repository_root, config.attributes_prompt
        ).read_text(encoding="utf-8")
        personas_prompt = _repository_path(
            repository_root, config.personas_prompt
        ).read_text(encoding="utf-8")
        input_sha256 = sha256_file(input_path)
        config_sha256 = sha256_file(config_path)
        context_sha256 = generation_context_sha256(
            config=config,
            attributes_prompt=attributes_prompt,
            personas_prompt=personas_prompt,
            job_title_mapping=mapping,
            job_title_mapping_sha256=mapping_sha256,
        )
        sample_rows = pl.read_parquet(input_path).height
        expected_batches = math.ceil(manifest.rows / manifest.batch_size)
        expected_pilot_ids = {
            persona_pilot_id(
                input_sha256=input_sha256,
                generation_config_sha256=config_sha256,
                generation_context_sha256=context_sha256,
                rows=manifest.rows,
                batch_size=identity_batch_size,
            )
            for identity_batch_size in range(
                manifest.batch_size, config.maximum_smoke_rows + 1
            )
            if list(range(0, manifest.rows, identity_batch_size))
            == [reference.offset for reference in manifest.batch_runs]
        }
        return (
            manifest.llm_generation
            and manifest.validator_version == VALIDATOR_VERSION
            and config.llm_generation_enabled
            and manifest.pilot_id in expected_pilot_ids
            and pilot_dir.name == manifest.pilot_id
            and manifest.rows <= sample_rows
            and manifest.batches == expected_batches
            and len(manifest.batch_runs) == expected_batches
            and manifest.batch_size
            == max(reference.rows for reference in manifest.batch_runs)
            and manifest.batch_size <= config.maximum_smoke_rows
            and manifest.maximum_shard_requests == config.maximum_total_requests
            and expected_batches * manifest.maximum_shard_requests
            <= manifest.maximum_total_requests
            and input_sha256 == manifest.input_sha256
            and _checksum_matches(sample_manifest_path, manifest.sample_manifest_sha256)
            and config_sha256 == manifest.generation_config_sha256
            and upstream.run_id == manifest.upstream_run_id
            and config.model == manifest.model
            and config.base_url == manifest.base_url
            and sha256_text(attributes_prompt) == manifest.attributes_prompt_sha256
            and sha256_text(personas_prompt) == manifest.personas_prompt_sha256
            and context_sha256 == manifest.generation_context_sha256
            and _mapping_binding_matches(
                path=manifest.job_title_mapping_file,
                sha256=manifest.job_title_mapping_sha256,
                version=manifest.job_title_mapping_version,
                content=manifest.job_title_mapping_content,
                expected_path=mapping_path,
                expected_mapping=mapping,
                expected_sha256=mapping_sha256,
                repository_root=repository_root,
            )
        )
    except OSError, UnicodeError, ValueError, pl.exceptions.PolarsError:
        return False


def _failed_report(
    *,
    kind: t.Literal["personas", "persona_pilot"],
    subject_id: str,
    error: BaseException,
) -> ValidationReport:
    """Build a deterministic failed report for an unreadable required artefact.

    Returns:
        Failed validation report describing the artefact error type.
    """
    metric = MetricResult(
        name="artefact_integrity",
        passed=False,
        value=1,
        threshold=0,
        details=f"Required artefact is missing or malformed ({type(error).__name__}).",
    )
    return ValidationReport(
        kind=kind,
        passed=False,
        created_at=datetime.now(tz=UTC).isoformat(),
        subject_id=subject_id,
        metrics=[metric],
    )


def validate_persona_run(
    run_dir: Path, repository_root: Path | None = None
) -> ValidationReport:
    """Validate output integrity, safety, and upstream preservation.

    Args:
        run_dir:
            Completed persona generation run.
        repository_root (optional):
            Repository root for manifest input and configuration paths.

    Returns:
        Machine-readable validation report, including for malformed artefacts.
    """
    try:
        report = _build_persona_run_report(
            run_dir=run_dir, repository_root=repository_root
        )
    except (OSError, UnicodeError, ValueError, pl.exceptions.PolarsError) as error:
        report = _failed_report(kind="personas", subject_id=run_dir.name, error=error)
    write_json(path=run_dir / "validation-report.json", payload=report)
    return report
