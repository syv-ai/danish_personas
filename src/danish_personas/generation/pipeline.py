"""Resumable two-stage persona generation orchestration."""

import collections.abc as c
import logging
import os
import typing as t
from pathlib import Path

import polars as pl
from pydantic import BaseModel

from ..io import canonical_json, load_yaml_model, sha256_file, sha256_text, write_json
from ..ladders import MOST_SPECIFIC_RESOLUTION
from ..models import RunManifest, ValidationReport
from .client import OpenAIClient, RequestBudgetExceeded
from .models import (
    AttributeCheckpoint,
    FrozenSampleManifest,
    GeneratedAttributes,
    GenerationConfig,
    GenerationManifest,
    LLMResponse,
    PersonaCheckpoint,
    PersonaDescriptions,
    RequestLedger,
)
from .validation import VALIDATOR_VERSION, parse_attributes, parse_descriptions

LOGGER = logging.getLogger(__name__)
# Sampler back-off provenance, withheld from prompts: it records how a value was
# obtained, not anything about the person.
AUDIT_FIELDS = frozenset(MOST_SPECIFIC_RESOLUTION)
GeneratedModel = t.TypeVar("GeneratedModel", bound=BaseModel)


def generate_personas(
    input_path: Path,
    sample_manifest_path: Path,
    config_path: Path,
    output_dir: Path,
    rows: int,
    live: bool,
    offset: int = 0,
) -> Path:
    """Generate structured attributes and six persona descriptions.

    Args:
        input_path:
            Frozen Phase-2 development sample.
        sample_manifest_path:
            Checksum manifest for the frozen sample.
        config_path:
            Local generation configuration.
        output_dir:
            Root directory for generation runs.
        rows:
            Number of records, capped at five per invocation.
        live:
            Whether network calls are explicitly authorised.
        offset:
            Zero-based position within the ordered frozen sample.

    Returns:
        Planned or completed generation run directory.

    Raises:
        ValueError:
            If the requested range is outside the frozen sample.
    """
    config = load_yaml_model(path=config_path, model=GenerationConfig)
    _validate_guards(config=config, rows=rows, live=live)
    upstream_run = validate_upstream_sample(
        input_path=input_path, sample_manifest_path=sample_manifest_path
    )
    sample = pl.read_parquet(input_path).sort("persona_id")
    if offset < 0 or offset + rows > sample.height:
        message = "Requested row range exceeds the frozen sample"
        raise ValueError(message)
    frame = sample.slice(offset, rows)
    selected_ids = frame.get_column("persona_id").to_list()
    ordered_ids_sha = sha256_text(canonical_json(selected_ids))
    attributes_prompt = config.attributes_prompt.read_text(encoding="utf-8")
    personas_prompt = config.personas_prompt.read_text(encoding="utf-8")
    generation_context_sha = sha256_text(
        canonical_json(
            {
                "config": config.model_dump(mode="json"),
                "attributes_prompt_sha256": sha256_text(attributes_prompt),
                "personas_prompt_sha256": sha256_text(personas_prompt),
                "attributes_schema": GeneratedAttributes.model_json_schema(),
                "personas_schema": PersonaDescriptions.model_json_schema(),
                "validator_version": VALIDATOR_VERSION,
                "withheld_fields": sorted(AUDIT_FIELDS),
            }
        )
    )
    run_id = sha256_text(
        ":".join([sha256_file(input_path), generation_context_sha, ordered_ids_sha])
    )[:16]
    run_dir = output_dir / run_id
    if not live:
        LOGGER.info(
            "Dry run %s: %s rows, %s planned requests, model=%s",
            run_id,
            rows,
            rows * 2,
            config.model,
        )
        return run_dir

    api_key = os.environ.get(config.api_key_env) if config.api_key_env else None
    ledger_path = run_dir / "request-ledger.json"
    ledger = _load_request_ledger(
        run_dir=run_dir,
        generation_context_sha=generation_context_sha,
        maximum_attempts=config.maximum_total_requests,
    )

    def record_request(attempts: int) -> None:
        nonlocal ledger
        if attempts > ledger.maximum_attempts:
            message = "Generation HTTP request budget is exhausted"
            raise RequestBudgetExceeded(message)
        ledger = ledger.model_copy(update={"attempts": attempts})
        write_json(path=ledger_path, payload=ledger)

    client = OpenAIClient(
        config=config,
        api_key=api_key,
        initial_requests_made=ledger.attempts,
        record_request=record_request,
    )
    checkpoints: list[PersonaCheckpoint] = []
    try:
        for row in frame.iter_rows(named=True):
            checkpoints.append(
                _generate_one(
                    row=t.cast(dict[str, object], row),
                    run_dir=run_dir,
                    config=config,
                    client=client,
                    attributes_prompt=attributes_prompt,
                    personas_prompt=personas_prompt,
                    generation_context_sha=generation_context_sha,
                )
            )
    finally:
        client.close()
    output_path = _write_output(frame=frame, checkpoints=checkpoints, run_dir=run_dir)
    responses = [response for item in checkpoints for response in item.responses]
    manifest = GenerationManifest(
        run_id=run_id,
        upstream_run_id=upstream_run.run_id,
        input_file=input_path,
        sample_manifest_file=sample_manifest_path,
        input_sha256=sha256_file(input_path),
        ordered_persona_ids_sha256=ordered_ids_sha,
        generation_config_file=config_path,
        generation_config_sha256=sha256_file(config_path),
        generation_context_sha256=generation_context_sha,
        validator_version=VALIDATOR_VERSION,
        attributes_prompt_sha256=sha256_text(attributes_prompt),
        personas_prompt_sha256=sha256_text(personas_prompt),
        model=config.model or "",
        base_url=config.base_url or "",
        rows=rows,
        offset=offset,
        requests=ledger.attempts,
        retries=max(0, ledger.attempts - rows * 2),
        prompt_tokens=sum(response.prompt_tokens for response in responses),
        completion_tokens=sum(response.completion_tokens for response in responses),
        total_tokens=sum(response.total_tokens for response in responses),
        estimated_cost_usd=_sum_estimated_cost(responses=responses),
        inference_providers=sorted(
            {
                response.inference_provider
                for response in responses
                if response.inference_provider
            }
        ),
        output_file=Path(output_path.name),
        output_sha256=sha256_file(output_path),
        llm_generation=True,
    )
    write_json(path=run_dir / "generation-manifest.json", payload=manifest)
    LOGGER.info(
        "Completed persona smoke run %s with %s requests", run_id, manifest.requests
    )
    return run_dir


def _generate_one(
    row: dict[str, object],
    run_dir: Path,
    config: GenerationConfig,
    client: OpenAIClient,
    attributes_prompt: str,
    personas_prompt: str,
    generation_context_sha: str,
) -> PersonaCheckpoint:
    persona_id = str(row["persona_id"])
    input_sha = sha256_text(canonical_json(row))
    prompt_row = {key: value for key, value in row.items() if key not in AUDIT_FIELDS}
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / f"{persona_id}.json"
    attribute_path = checkpoint_dir / f"{persona_id}.attributes.json"
    if checkpoint_path.exists():
        checkpoint = PersonaCheckpoint.model_validate_json(
            checkpoint_path.read_text(encoding="utf-8")
        )
        _validate_checkpoint(
            checkpoint=checkpoint,
            input_sha=input_sha,
            generation_context_sha=generation_context_sha,
            model=config.model or "",
        )
        parse_descriptions(checkpoint.descriptions.model_dump_json())
        return checkpoint

    if attribute_path.exists():
        attribute_checkpoint = AttributeCheckpoint.model_validate_json(
            attribute_path.read_text(encoding="utf-8")
        )
        _validate_checkpoint(
            checkpoint=attribute_checkpoint,
            input_sha=input_sha,
            generation_context_sha=generation_context_sha,
            model=config.model or "",
        )
        attributes = attribute_checkpoint.attributes
        responses = list(attribute_checkpoint.responses)
        http_requests = attribute_checkpoint.http_requests
    else:
        responses = []
        request_start = client.requests_made
        attributes = _complete_validated(
            client=client,
            prompt=attributes_prompt,
            payload={"demographics_and_personality": prompt_row},
            schema_name="generated_attributes",
            schema=t.cast(dict[str, object], GeneratedAttributes.model_json_schema()),
            parser=parse_attributes,
            maximum_attempts=config.maximum_validation_attempts,
            responses=responses,
        )
        http_requests = client.requests_made - request_start
        attribute_checkpoint = AttributeCheckpoint(
            persona_id=persona_id,
            input_sha256=input_sha,
            generation_context_sha256=generation_context_sha,
            validator_version=VALIDATOR_VERSION,
            attributes=attributes,
            responses=responses,
            http_requests=http_requests,
        )
        write_json(path=attribute_path, payload=attribute_checkpoint)

    request_start = client.requests_made
    try:
        descriptions = _complete_validated(
            client=client,
            prompt=personas_prompt,
            payload={
                "demographics_and_personality": prompt_row,
                "generated_attributes": attributes.model_dump(mode="json"),
            },
            schema_name="persona_descriptions",
            schema=t.cast(dict[str, object], PersonaDescriptions.model_json_schema()),
            parser=parse_descriptions,
            maximum_attempts=config.maximum_validation_attempts,
            responses=responses,
        )
    except Exception:
        attribute_checkpoint = AttributeCheckpoint(
            persona_id=persona_id,
            input_sha256=input_sha,
            generation_context_sha256=generation_context_sha,
            validator_version=VALIDATOR_VERSION,
            attributes=attributes,
            responses=responses,
            http_requests=http_requests + client.requests_made - request_start,
        )
        write_json(path=attribute_path, payload=attribute_checkpoint)
        raise
    http_requests += client.requests_made - request_start
    checkpoint = PersonaCheckpoint(
        persona_id=persona_id,
        input_sha256=input_sha,
        generation_context_sha256=generation_context_sha,
        validator_version=VALIDATOR_VERSION,
        attributes=attributes,
        descriptions=descriptions,
        responses=responses,
        attempts=len(responses),
        http_requests=http_requests,
    )
    write_json(path=checkpoint_path, payload=checkpoint)
    attribute_path.unlink(missing_ok=True)
    return checkpoint


def _complete_validated(
    client: OpenAIClient,
    prompt: str,
    payload: dict[str, object],
    schema_name: str,
    schema: dict[str, object],
    parser: c.Callable[[str], GeneratedModel],
    maximum_attempts: int,
    responses: list[LLMResponse],
) -> GeneratedModel:
    current_payload = payload
    last_error: ValueError | None = None
    for _ in range(maximum_attempts):
        response = client.complete(
            system_prompt=prompt,
            user_payload=current_payload,
            schema_name=schema_name,
            json_schema=schema,
        )
        responses.append(response)
        try:
            return parser(response.content)
        except ValueError as error:
            last_error = error
            responses[-1] = response.model_copy(update={"content": ""})
            current_payload = {
                **payload,
                "validation_feedback": str(error),
                "instruction": "Ret JSON-svaret uden at ændre de faste input.",
            }
    if last_error is None:
        message = "Generation exhausted attempts without validation feedback"
        raise RuntimeError(message)
    raise last_error


def _validate_checkpoint(
    checkpoint: AttributeCheckpoint | PersonaCheckpoint,
    input_sha: str,
    generation_context_sha: str,
    model: str,
) -> None:
    if checkpoint.input_sha256 != input_sha:
        message = f"Stale checkpoint input for {checkpoint.persona_id}"
        raise ValueError(message)
    if checkpoint.generation_context_sha256 != generation_context_sha:
        message = f"Stale generation context for {checkpoint.persona_id}"
        raise ValueError(message)
    if checkpoint.validator_version != VALIDATOR_VERSION:
        message = f"Stale validator context for {checkpoint.persona_id}"
        raise ValueError(message)
    parse_attributes(checkpoint.attributes.model_dump_json())
    if any(
        not models_match(configured=model, returned=response.model)
        for response in checkpoint.responses
    ):
        message = f"Checkpoint model mismatch for {checkpoint.persona_id}"
        raise ValueError(message)


def models_match(configured: str, returned: str) -> bool:
    """Match a returned model to an optional HF provider-qualified model ID.

    Args:
        configured:
            Requested model, optionally suffixed with an HF provider.
        returned:
            Model identifier returned by the provider.

    Returns:
        Whether both identifiers refer to the same underlying model.
    """
    return (
        configured.partition(":")[0].casefold() == returned.partition(":")[0].casefold()
    )


def _load_request_ledger(
    run_dir: Path, generation_context_sha: str, maximum_attempts: int
) -> RequestLedger:
    ledger_path = run_dir / "request-ledger.json"
    if ledger_path.exists():
        ledger = RequestLedger.model_validate_json(
            ledger_path.read_text(encoding="utf-8")
        )
        if (
            ledger.generation_context_sha256 != generation_context_sha
            or ledger.maximum_attempts != maximum_attempts
        ):
            message = "Stale request ledger does not match generation context"
            raise ValueError(message)
    else:
        checkpoints: dict[str, AttributeCheckpoint | PersonaCheckpoint] = {}
        checkpoint_dir = run_dir / "checkpoints"
        for checkpoint_path in checkpoint_dir.glob("*.json"):
            if checkpoint_path.name.endswith(".attributes.json"):
                continue
            checkpoint = PersonaCheckpoint.model_validate_json(
                checkpoint_path.read_text(encoding="utf-8")
            )
            checkpoints[checkpoint.persona_id] = checkpoint
        for checkpoint_path in checkpoint_dir.glob("*.attributes.json"):
            checkpoint = AttributeCheckpoint.model_validate_json(
                checkpoint_path.read_text(encoding="utf-8")
            )
            checkpoints.setdefault(checkpoint.persona_id, checkpoint)
        ledger = RequestLedger(
            generation_context_sha256=generation_context_sha,
            attempts=sum(item.http_requests for item in checkpoints.values()),
            maximum_attempts=maximum_attempts,
        )
        write_json(path=ledger_path, payload=ledger)
    if ledger.attempts > maximum_attempts:
        message = "Persisted HTTP requests exceed the generation budget"
        raise RequestBudgetExceeded(message)
    return ledger


def _sum_estimated_cost(responses: list[LLMResponse]) -> float | None:
    costs = [response.estimated_cost_usd for response in responses]
    if not costs or any(cost is None for cost in costs):
        return None
    return sum(t.cast(list[float], costs))


def _validate_guards(config: GenerationConfig, rows: int, live: bool) -> None:
    if rows < 1 or rows > config.maximum_smoke_rows:
        message = f"Rows must be between 1 and {config.maximum_smoke_rows}"
        raise ValueError(message)
    if live and not config.llm_generation_enabled:
        message = "LLM generation is disabled in the selected configuration"
        raise ValueError(message)
    if rows * 2 > config.maximum_total_requests:
        message = "Planned stages exceed the configured HTTP request budget"
        raise ValueError(message)
    if live and (not config.base_url or not config.model):
        message = "Live generation requires base_url and model"
        raise ValueError(message)


def _write_output(
    frame: pl.DataFrame, checkpoints: list[PersonaCheckpoint], run_dir: Path
) -> Path:
    generated_rows: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        generated_rows.append(
            {
                "persona_id": checkpoint.persona_id,
                **checkpoint.attributes.model_dump(mode="json"),
                **checkpoint.descriptions.model_dump(mode="json"),
            }
        )
    generated = pl.DataFrame(generated_rows)
    output = frame.join(generated, on="persona_id", how="left", validate="1:1")
    output_path = run_dir / "generated-personas.parquet"
    run_dir.mkdir(parents=True, exist_ok=True)
    output.write_parquet(output_path, compression="zstd")
    return output_path


def validate_upstream_sample(
    input_path: Path, sample_manifest_path: Path
) -> RunManifest:
    """Prove a frozen sample is an unchanged subset of a validated run.

    Args:
        input_path:
            Frozen sample Parquet file.
        sample_manifest_path:
            Frozen sample manifest.

    Returns:
        Validated upstream demographic run manifest.

    Raises:
        ValueError:
            If any checksum, provenance, schema, order, or membership check fails.
    """
    sample_manifest = FrozenSampleManifest.model_validate_json(
        sample_manifest_path.read_text(encoding="utf-8")
    )
    if sample_manifest.sha256 != sha256_file(input_path):
        message = "Frozen sample checksum does not match its manifest"
        raise ValueError(message)
    if sample_manifest.data_file != Path(input_path.name):
        message = "Frozen sample filename does not match its manifest"
        raise ValueError(message)
    sample = pl.read_parquet(input_path)
    if sample.height != sample_manifest.rows:
        message = "Frozen sample row count does not match its manifest"
        raise ValueError(message)
    if sample.n_unique("persona_id") != sample.height or not sample.equals(
        sample.sort("persona_id")
    ):
        message = "Frozen sample identifiers must be unique and ordered"
        raise ValueError(message)

    run_dir = input_path.parent
    upstream = RunManifest.model_validate_json(
        (run_dir / "run-manifest.json").read_text(encoding="utf-8")
    )
    report = ValidationReport.model_validate_json(
        (run_dir / "validation-report.json").read_text(encoding="utf-8")
    )
    upstream_path = run_dir / upstream.data_file
    if sha256_file(upstream_path) != upstream.data_sha256:
        message = "Validated Phase-2 data checksum does not match its manifest"
        raise ValueError(message)
    if (
        not report.passed
        or report.kind != "demographics"
        or report.subject_id != upstream.run_id
        or upstream.llm_calls != 0
    ):
        message = "Phase-2 validation report is missing, stale, or failed"
        raise ValueError(message)
    if (
        sample_manifest.source_run_id != upstream.run_id
        or sample_manifest.llm_calls != 0
    ):
        message = "Frozen sample belongs to a different upstream run"
        raise ValueError(message)
    upstream_frame = pl.read_parquet(upstream_path)
    if sample.columns != upstream_frame.columns:
        message = "Frozen sample schema differs from the validated Phase-2 data"
        raise ValueError(message)
    unmatched = sample.join(
        upstream_frame, on=sample.columns, how="anti", nulls_equal=True
    )
    if unmatched.height:
        message = "Frozen sample contains rows absent from validated Phase-2 data"
        raise ValueError(message)
    return upstream
