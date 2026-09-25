"""Resumable, bounded persona-pilot orchestration service."""

import collections.abc as c
import concurrent.futures as futures
import logging
import threading
import time
import typing as t
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from ..io import canonical_json, sha256_file, sha256_text, write_json
from .identity import persona_pilot_id
from .models import (
    GenerationManifest,
    PilotBatchReference,
    PilotCostLedger,
    PilotCostReservation,
    PilotManifest,
)
from .pipeline import GenerationContext, generate_personas, prepare_generation_context

LOGGER = logging.getLogger(__name__)


class _PilotCostBudget:
    """Atomically persist conservative shard reservations and settlements."""

    def __init__(
        self,
        *,
        path: Path,
        input_sha256: str,
        generation_config_sha256: str,
        model: str,
        input_price_per_million: float,
        output_price_per_million: float,
        maximum_campaign_cost_usd: float,
        maximum_shard_cost_usd: float,
    ) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._identity = {
            "input_sha256": input_sha256,
            "generation_config_sha256": generation_config_sha256,
            "model": model,
            "input_price_per_million_usd": input_price_per_million,
            "output_price_per_million_usd": output_price_per_million,
            "maximum_campaign_cost_usd": maximum_campaign_cost_usd,
            "maximum_shard_cost_usd": maximum_shard_cost_usd,
        }
        with self._lock:
            self._load()

    @property
    def identity_sha256(self) -> str:
        """Checksum binding shard request ledgers to this budget."""
        return sha256_text(canonical_json(self._identity))

    def reserve(self, *, offset: int) -> None:
        """Reserve one shard before it is submitted to the executor.

        Raises:
            ValueError:
                If the shard is already settled or the campaign cap is exhausted.
        """
        with self._lock:
            ledger = self._load()
            existing = next(
                (item for item in ledger.reservations if item.offset == offset), None
            )
            if existing is not None:
                if existing.status == "settled":
                    raise ValueError(f"Shard offset {offset} is already settled")
                return
            reserved = sum(
                item.maximum_cost_usd
                for item in ledger.reservations
                if item.status == "reserved"
            )
            committed = sum(
                item.settled_cost_usd or 0.0
                for item in ledger.reservations
                if item.status == "settled"
            )
            maximum = t.cast(float, self._identity["maximum_shard_cost_usd"])
            campaign_cap = t.cast(float, self._identity["maximum_campaign_cost_usd"])
            if committed + reserved + maximum > campaign_cap:
                raise ValueError("Campaign cost cap would be exceeded")
            ledger.reservations.append(
                PilotCostReservation(
                    offset=offset, maximum_cost_usd=maximum, status="reserved"
                )
            )
            self._write(ledger)

    def measured_cost(self, manifest: GenerationManifest) -> float:
        """Return measured list-price cost for one completed shard."""
        input_price = t.cast(float, self._identity["input_price_per_million_usd"])
        output_price = t.cast(float, self._identity["output_price_per_million_usd"])
        return (
            manifest.prompt_tokens * input_price
            + manifest.completion_tokens * output_price
        ) / 1_000_000

    def settle(self, *, offset: int, measured_cost_usd: float) -> None:
        """Settle a successful shard to measured list-price token cost.

        Raises:
            ValueError:
                If the reservation is missing or the measured cost exceeds it.
        """
        with self._lock:
            ledger = self._load()
            reservation = next(
                (item for item in ledger.reservations if item.offset == offset), None
            )
            if reservation is None:
                raise ValueError(f"Missing cost reservation for shard offset {offset}")
            if measured_cost_usd > reservation.maximum_cost_usd + 1e-12:
                raise ValueError("Measured shard cost exceeds its reservation")
            if reservation.status == "settled":
                if reservation.settled_cost_usd != measured_cost_usd:
                    raise ValueError("Settled shard cost was changed")
                return
            reservation.status = "settled"
            reservation.settled_cost_usd = measured_cost_usd
            self._write(ledger)

    def _load(self) -> PilotCostLedger:
        if self._path.exists():
            ledger = PilotCostLedger.model_validate_json(
                self._path.read_text(encoding="utf-8")
            )
            if (
                ledger.model_dump(mode="json", exclude={"reservations"})
                != self._identity
            ):
                raise ValueError(
                    "Pilot cost ledger identity does not match the campaign"
                )
            offsets = [item.offset for item in ledger.reservations]
            if len(offsets) != len(set(offsets)):
                raise ValueError("Pilot cost ledger contains duplicate shard offsets")
            return ledger
        ledger = PilotCostLedger(
            input_sha256=t.cast(str, self._identity["input_sha256"]),
            generation_config_sha256=t.cast(
                str, self._identity["generation_config_sha256"]
            ),
            model=t.cast(str, self._identity["model"]),
            input_price_per_million_usd=t.cast(
                float, self._identity["input_price_per_million_usd"]
            ),
            output_price_per_million_usd=t.cast(
                float, self._identity["output_price_per_million_usd"]
            ),
            maximum_campaign_cost_usd=t.cast(
                float, self._identity["maximum_campaign_cost_usd"]
            ),
            maximum_shard_cost_usd=t.cast(
                float, self._identity["maximum_shard_cost_usd"]
            ),
            reservations=[],
        )
        self._write(ledger)
        return ledger

    def _write(self, ledger: PilotCostLedger) -> None:
        write_json(path=self._path, payload=ledger)


def run_pilot(
    *,
    input_path: Path,
    sample_manifest_path: Path,
    config_path: Path,
    output_dir: Path,
    rows: int,
    batch_size: int,
    concurrency: int,
    delay_between_batches: float,
    maximum_total_requests: int | None,
    input_price_per_million: float = 0.0,
    output_price_per_million: float = 0.0,
    maximum_campaign_cost_usd: float | None = None,
    maximum_shard_cost_usd: float = 1.0,
    progress_callback: c.Callable[[int], None] | None = None,
) -> Path:
    """Generate and merge a schema-parsed, resumable persona dataset.

    Each invocation remains bounded to five records by the generation pipeline.

    Args:
        input_path:
            Frozen Phase-2 development sample.
        sample_manifest_path:
            Checksum manifest for the frozen sample.
        config_path:
            Local generation configuration.
        output_dir:
            Root directory for pilot runs.
        rows:
            Number of records to generate.
        batch_size:
            Maximum records in each generation shard.
        concurrency:
            Maximum number of shards generated concurrently.
        delay_between_batches:
            Seconds to wait before scheduling another wave of shards.
        maximum_total_requests:
            Global HTTP request budget for the pilot, or ``None`` for unlimited
            requests.
        input_price_per_million:
            Input-token price used for cost accounting.
        output_price_per_million:
            Output-token price used for cost accounting.
        maximum_campaign_cost_usd (optional):
            Positive pilot-wide list-price cap. Defaults to no cap.
        maximum_shard_cost_usd:
            Conservative maximum list-price cost reserved for every shard.
        progress_callback (optional):
            Callback invoked with the number of rows in each completed shard.

    Returns:
        Completed pilot directory.

    Raises:
        ValueError:
            If provenance, limits, generation, or merging fails.
    """
    _validate_pilot_arguments(
        rows=rows,
        batch_size=batch_size,
        concurrency=concurrency,
        delay_between_batches=delay_between_batches,
        maximum_total_requests=maximum_total_requests,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        maximum_campaign_cost_usd=maximum_campaign_cost_usd,
        maximum_shard_cost_usd=maximum_shard_cost_usd,
    )
    context = prepare_generation_context(
        input_path=input_path,
        sample_manifest_path=sample_manifest_path,
        config_path=config_path,
    )
    config = context.config
    sample = context.sample
    if rows > sample.height:
        message = "Requested pilot exceeds the frozen sample"
        raise ValueError(message)
    if batch_size > config.maximum_rows_per_shard:
        message = "Pilot batch size exceeds the per-invocation row limit"
        raise ValueError(message)
    offsets = list(range(0, rows, batch_size))
    generation_context_sha = context.generation_context_sha256
    if maximum_total_requests is not None and config.maximum_total_requests is not None:
        worst_case_requests = len(offsets) * config.maximum_total_requests
        if worst_case_requests > maximum_total_requests:
            message = (
                f"Worst-case pilot requests ({worst_case_requests}) exceed the pilot "
                f"limit ({maximum_total_requests})"
            )
            raise ValueError(message)
    pilot_id = persona_pilot_id(
        input_sha256=context.input_sha256,
        generation_config_sha256=context.generation_config_sha256,
        generation_context_sha256=generation_context_sha,
        rows=rows,
        batch_size=batch_size,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        maximum_campaign_cost_usd=maximum_campaign_cost_usd,
        maximum_shard_cost_usd=maximum_shard_cost_usd,
    )
    pilot_dir = output_dir / pilot_id
    budget = (
        _PilotCostBudget(
            path=pilot_dir / "pilot-cost-ledger.json",
            input_sha256=context.input_sha256,
            generation_config_sha256=context.generation_config_sha256,
            model=config.model or "",
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
            maximum_campaign_cost_usd=maximum_campaign_cost_usd,
            maximum_shard_cost_usd=maximum_shard_cost_usd,
        )
        if maximum_campaign_cost_usd is not None
        else None
    )
    batch_root = pilot_dir / "batches"
    run_dirs = _run_batches(
        offsets=offsets,
        rows=rows,
        batch_size=batch_size,
        concurrency=concurrency,
        delay_between_batches=delay_between_batches,
        input_path=input_path,
        sample_manifest_path=sample_manifest_path,
        config_path=config_path,
        output_dir=batch_root,
        expected_generation_context_sha256=generation_context_sha,
        maximum_total_requests=maximum_total_requests,
        context=context,
        budget=budget,
        pilot_identity_sha256=(
            budget.identity_sha256
            if budget is not None
            else sha256_text(canonical_json({"pilot_id": pilot_id}))
        ),
        progress_callback=progress_callback,
    )
    manifests = [
        GenerationManifest.model_validate_json(
            (run_dir / "generation-manifest.json").read_text(encoding="utf-8")
        )
        for run_dir in run_dirs
    ]
    requests = sum(manifest.requests for manifest in manifests)
    if maximum_total_requests is not None and requests > maximum_total_requests:
        message = "Completed pilot exceeds its global request limit"
        raise ValueError(message)
    return _merge_pilot(
        pilot_dir=pilot_dir,
        run_dirs=run_dirs,
        manifests=manifests,
        expected=sample.head(rows),
        input_path=input_path,
        sample_manifest_path=sample_manifest_path,
        config_path=config_path,
        maximum_total_requests=maximum_total_requests,
        maximum_shard_requests=config.maximum_total_requests,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        maximum_campaign_cost_usd=maximum_campaign_cost_usd,
        maximum_shard_cost_usd=maximum_shard_cost_usd,
    )


def _merge_pilot(
    *,
    pilot_dir: Path,
    run_dirs: list[Path],
    manifests: list[GenerationManifest],
    expected: pl.DataFrame,
    input_path: Path,
    sample_manifest_path: Path,
    config_path: Path,
    maximum_total_requests: int | None,
    maximum_shard_requests: int | None,
    input_price_per_million: float,
    output_price_per_million: float,
    maximum_campaign_cost_usd: float | None,
    maximum_shard_cost_usd: float | None,
) -> Path:
    output = pl.concat(
        [
            pl.read_parquet(run_dir / manifest.output_file)
            for run_dir, manifest in zip(run_dirs, manifests, strict=True)
        ],
        how="vertical_relaxed",
    ).sort("persona_id")
    expected_ids = expected.get_column("persona_id").to_list()
    if (
        output.height != expected.height
        or output.n_unique("persona_id") != output.height
        or output.get_column("persona_id").to_list() != expected_ids
        or not output.select(expected.columns).equals(expected)
    ):
        message = "Merged pilot does not preserve the requested frozen records"
        raise ValueError(message)
    pilot_dir.mkdir(parents=True, exist_ok=True)
    output_path = pilot_dir / "generated-personas.parquet"
    output.write_parquet(output_path, compression="zstd")
    shard_bindings = {
        (
            manifest.generation_context_sha256,
            manifest.origin_label_contract_file,
            manifest.origin_label_contract_sha256,
            manifest.origin_label_contract_version,
            manifest.origin_label_contract_content.model_dump_json(),
        )
        for manifest in manifests
    }
    if len(shard_bindings) != 1:
        message = "Pilot shards have inconsistent generation contexts"
        raise ValueError(message)
    prompt_tokens = sum(manifest.prompt_tokens for manifest in manifests)
    completion_tokens = sum(manifest.completion_tokens for manifest in manifests)
    list_price_cost = (
        prompt_tokens * input_price_per_million
        + completion_tokens * output_price_per_million
    ) / 1_000_000
    provider_costs = [manifest.estimated_cost_usd for manifest in manifests]
    provider_cost = (
        sum(cost for cost in provider_costs if cost is not None)
        if all(cost is not None for cost in provider_costs)
        else None
    )
    batch_runs = [
        PilotBatchReference(
            offset=manifest.offset,
            rows=manifest.rows,
            run_id=manifest.run_id,
            manifest_file=(run_dir / "generation-manifest.json").relative_to(pilot_dir),
            manifest_sha256=sha256_file(run_dir / "generation-manifest.json"),
            job_title_mapping_file=manifest.job_title_mapping_file,
            job_title_mapping_sha256=manifest.job_title_mapping_sha256,
            job_title_mapping_version=manifest.job_title_mapping_version,
            job_title_mapping_content=manifest.job_title_mapping_content,
            origin_label_contract_file=manifest.origin_label_contract_file,
            origin_label_contract_sha256=manifest.origin_label_contract_sha256,
            origin_label_contract_version=manifest.origin_label_contract_version,
            origin_label_contract_content=manifest.origin_label_contract_content,
        )
        for run_dir, manifest in zip(run_dirs, manifests, strict=True)
    ]
    first = manifests[0]
    pilot_manifest = PilotManifest(
        pilot_id=pilot_dir.name,
        created_at=datetime.now(tz=UTC).isoformat(),
        model=first.model,
        base_url=first.base_url,
        upstream_run_id=first.upstream_run_id,
        input_file=input_path,
        input_sha256=sha256_file(input_path),
        sample_manifest_file=sample_manifest_path,
        sample_manifest_sha256=sha256_file(sample_manifest_path),
        generation_config_file=config_path,
        generation_config_sha256=sha256_file(config_path),
        generation_context_sha256=first.generation_context_sha256,
        job_title_mapping_file=first.job_title_mapping_file,
        job_title_mapping_sha256=first.job_title_mapping_sha256,
        job_title_mapping_version=first.job_title_mapping_version,
        job_title_mapping_content=first.job_title_mapping_content,
        origin_label_contract_file=first.origin_label_contract_file,
        origin_label_contract_sha256=first.origin_label_contract_sha256,
        origin_label_contract_version=first.origin_label_contract_version,
        origin_label_contract_content=first.origin_label_contract_content,
        prompt_sha256=first.prompt_sha256,
        rows=output.height,
        batch_size=max(manifest.rows for manifest in manifests),
        batches=len(manifests),
        batch_runs=batch_runs,
        maximum_total_requests=maximum_total_requests,
        maximum_shard_requests=maximum_shard_requests,
        maximum_campaign_cost_usd=maximum_campaign_cost_usd,
        maximum_shard_cost_usd=maximum_shard_cost_usd,
        requests=sum(manifest.requests for manifest in manifests),
        retries=sum(manifest.retries for manifest in manifests),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=sum(manifest.total_tokens for manifest in manifests),
        input_price_per_million_usd=input_price_per_million,
        output_price_per_million_usd=output_price_per_million,
        list_price_estimated_cost_usd=list_price_cost,
        provider_estimated_cost_usd=provider_cost,
        inference_providers=sorted(
            {
                provider
                for manifest in manifests
                for provider in manifest.inference_providers
            }
        ),
        output_file=Path(output_path.name),
        output_sha256=sha256_file(output_path),
        llm_generation=True,
    )
    write_json(path=pilot_dir / "pilot-manifest.json", payload=pilot_manifest)
    return pilot_dir


def _run_batches(
    *,
    offsets: list[int],
    rows: int,
    batch_size: int,
    concurrency: int,
    delay_between_batches: float,
    input_path: Path,
    sample_manifest_path: Path,
    config_path: Path,
    output_dir: Path,
    expected_generation_context_sha256: str,
    maximum_total_requests: int | None,
    context: GenerationContext,
    budget: _PilotCostBudget | None,
    pilot_identity_sha256: str | None,
    progress_callback: c.Callable[[int], None] | None,
) -> list[Path]:
    run_dirs = _find_completed_batches(
        offsets=offsets,
        rows=rows,
        batch_size=batch_size,
        output_dir=output_dir,
        expected_generation_context_sha256=expected_generation_context_sha256,
        budget=budget,
        progress_callback=progress_callback,
    )
    remaining = iter(offset for offset in offsets if offset not in run_dirs)
    if len(run_dirs) == len(offsets):
        return [run_dirs[offset] for offset in offsets]
    executor = futures.ThreadPoolExecutor(max_workers=concurrency)
    pending: dict[futures.Future[Path], int] = {}

    def submit(offset: int) -> None:
        if budget is not None:
            budget.reserve(offset=offset)
        pending[
            _submit_batch(
                executor=executor,
                offset=offset,
                rows=rows,
                batch_size=batch_size,
                input_path=input_path,
                sample_manifest_path=sample_manifest_path,
                config_path=config_path,
                output_dir=output_dir,
                context=context,
                pilot_identity_sha256=pilot_identity_sha256,
            )
        ] = offset

    try:
        for _ in range(min(concurrency, len(offsets) - len(run_dirs))):
            next_offset = next(remaining, None)
            if next_offset is not None:
                submit(next_offset)
        while pending:
            completed, _ = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
            completed_count = _process_completed_batches(
                completed=completed,
                pending=pending,
                submit=submit,
                run_dirs=run_dirs,
                rows=rows,
                batch_size=batch_size,
                maximum_total_requests=maximum_total_requests,
                budget=budget,
                progress_callback=progress_callback,
            )
            if delay_between_batches and not pending:
                time.sleep(delay_between_batches)
            for _ in range(completed_count):
                next_offset = next(remaining, None)
                if next_offset is not None:
                    submit(next_offset)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return [run_dirs[offset] for offset in offsets]


def _find_completed_batches(
    *,
    offsets: list[int],
    rows: int,
    batch_size: int,
    output_dir: Path,
    expected_generation_context_sha256: str,
    budget: _PilotCostBudget | None,
    progress_callback: c.Callable[[int], None] | None,
) -> dict[int, Path]:
    """Find completed shards left by an earlier pilot attempt.

    Returns:
        Completed run directories keyed by batch offset.

    Raises:
        ValueError:
            If a completed shard conflicts with the requested pilot.
    """
    if not output_dir.exists():
        return {}
    expected_offsets = set(offsets)
    run_dirs: dict[int, Path] = {}
    for run_dir in sorted(output_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "generation-manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _validated_completed_batch(
            run_dir=run_dir,
            manifest_path=manifest_path,
            expected_offsets=expected_offsets,
            rows=rows,
            batch_size=batch_size,
            expected_generation_context_sha256=expected_generation_context_sha256,
        )
        if manifest.offset in run_dirs:
            message = f"Duplicate completed pilot batch offset: {manifest.offset}"
            raise ValueError(message)
        run_dirs[manifest.offset] = run_dir
        if budget is not None:
            budget.settle(
                offset=manifest.offset, measured_cost_usd=budget.measured_cost(manifest)
            )
        _report_progress(
            callback=progress_callback, rows=min(batch_size, rows - manifest.offset)
        )
    return run_dirs


def _report_progress(*, callback: c.Callable[[int], None] | None, rows: int) -> None:
    """Report rows after a shard has passed validation."""
    if callback is not None:
        callback(rows)


def _validated_completed_batch(
    *,
    run_dir: Path,
    manifest_path: Path,
    expected_offsets: set[int],
    rows: int,
    batch_size: int,
    expected_generation_context_sha256: str,
) -> GenerationManifest:
    """Load and validate one completed pilot batch.

    Returns:
        The validated generation manifest.

    Raises:
        ValueError:
            If the manifest or its output is invalid.
    """
    try:
        manifest = GenerationManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValueError) as error:
        message = f"Invalid completed pilot batch manifest: {manifest_path}"
        raise ValueError(message) from error
    expected_rows = (
        min(batch_size, rows - manifest.offset)
        if manifest.offset in expected_offsets
        else None
    )
    if (
        manifest.run_id != run_dir.name
        or manifest.offset not in expected_offsets
        or manifest.rows != expected_rows
        or manifest.generation_context_sha256 != expected_generation_context_sha256
    ):
        message = f"Conflicting completed pilot batch manifest: {manifest_path}"
        raise ValueError(message)
    if manifest.output_file.is_absolute():
        message = f"Pilot batch output must be relative: {manifest_path}"
        raise ValueError(message)
    output_path = (run_dir / manifest.output_file).resolve()
    try:
        output_path.relative_to(run_dir.resolve())
        output_checksum = sha256_file(output_path)
    except (OSError, ValueError) as error:
        message = f"Invalid completed pilot batch output: {manifest_path}"
        raise ValueError(message) from error
    if output_checksum != manifest.output_sha256:
        message = f"Pilot batch output checksum mismatch: {manifest_path}"
        raise ValueError(message)
    return manifest


def _process_completed_batches(
    *,
    completed: set[futures.Future[Path]],
    pending: dict[futures.Future[Path], int],
    submit: c.Callable[[int], None],
    run_dirs: dict[int, Path],
    rows: int,
    batch_size: int,
    maximum_total_requests: int | None,
    budget: _PilotCostBudget | None,
    progress_callback: c.Callable[[int], None] | None,
) -> int:
    """Collect completed pilot batches.

    Returns:
        The number of newly completed batches.
    """
    completed_count = 0
    for completed_future in completed:
        offset = pending.pop(completed_future)
        try:
            run_dir = completed_future.result()
        except Exception:
            if maximum_total_requests is None:
                submit(offset)
                continue
            raise
        run_dirs[offset] = run_dir
        if budget is not None:
            manifest = GenerationManifest.model_validate_json(
                (run_dir / "generation-manifest.json").read_text(encoding="utf-8")
            )
            budget.settle(
                offset=offset, measured_cost_usd=budget.measured_cost(manifest)
            )
        completed_count += 1
        _report_progress(
            callback=progress_callback, rows=min(batch_size, rows - offset)
        )
    return completed_count


def _submit_batch(
    *,
    executor: futures.ThreadPoolExecutor,
    offset: int,
    rows: int,
    batch_size: int,
    input_path: Path,
    sample_manifest_path: Path,
    config_path: Path,
    output_dir: Path,
    context: GenerationContext,
    pilot_identity_sha256: str | None,
) -> futures.Future[Path]:
    return executor.submit(
        generate_personas,
        input_path=input_path,
        sample_manifest_path=sample_manifest_path,
        config_path=config_path,
        output_dir=output_dir,
        rows=min(batch_size, rows - offset),
        offset=offset,
        context=context,
        pilot_identity_sha256=pilot_identity_sha256,
    )


def _validate_pilot_arguments(
    *,
    rows: int,
    batch_size: int,
    concurrency: int,
    delay_between_batches: float,
    maximum_total_requests: int | None,
    input_price_per_million: float,
    output_price_per_million: float,
    maximum_campaign_cost_usd: float | None,
    maximum_shard_cost_usd: float,
) -> None:
    if rows < 1:
        raise ValueError("Pilot rows must be at least 1")
    if batch_size < 1 or batch_size > 5:
        raise ValueError("Pilot batch size must be between 1 and 5")
    if not 1 <= concurrency <= 8:
        raise ValueError("Pilot concurrency must be between 1 and 8")
    if delay_between_batches < 0:
        raise ValueError("Pilot batch delay must not be negative")
    if maximum_total_requests is not None and maximum_total_requests < 1:
        raise ValueError("Pilot request budget must be at least 1")
    if input_price_per_million < 0 or output_price_per_million < 0:
        raise ValueError("Pilot token prices must not be negative")
    if maximum_campaign_cost_usd is not None and maximum_campaign_cost_usd <= 0:
        raise ValueError("Pilot campaign cost cap must be positive")
    if maximum_shard_cost_usd <= 0:
        raise ValueError("Pilot shard cost reservation must be positive")
    if (
        maximum_campaign_cost_usd is not None
        and maximum_shard_cost_usd > maximum_campaign_cost_usd
    ):
        raise ValueError("Pilot shard reservation exceeds the campaign cost cap")
