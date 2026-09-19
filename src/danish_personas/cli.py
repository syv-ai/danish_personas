"""Unified command-line interface for Danish persona services."""

import collections.abc as c
import logging
import typing as t
from pathlib import Path

import click

from .generation.pilot import run_pilot
from .generation.pipeline import generate_personas
from .generation.report import validate_persona_pilot, validate_persona_run
from .io import load_yaml_model
from .models import SamplingConfig, SourceLock, SourcesConfig, ValidationReport
from .release.cli import release
from .sampling.freeze import freeze_sample
from .sampling.generator import generate_records
from .sources.acquisition import fetch_sources, resolve_sources
from .sources.archive import (
    DEFAULT_ARCHIVE,
    RAW_DIRECTORY,
    build_raw_archive,
    restore_raw_sources,
)
from .sources.prepare import prepare_bundle
from .validation.checks import validate_demographics, validate_sources

LOGGER = logging.getLogger(__name__)
DEFAULT_RAW_PARENT = Path("data")
DEFAULT_RAW_DIR = DEFAULT_RAW_PARENT / RAW_DIRECTORY
DEFAULT_SAMPLING = Path("config/sampling.yaml")
DEFAULT_VALIDATION = Path("config/validation.yaml")
DEFAULT_CATEGORIES = Path("config/categories.yaml")
DEFAULT_LOCK = Path("config/sources.lock.yaml")
DEFAULT_SAMPLE_FILENAME = "text-development-seeds.parquet"


@click.group()
def cli() -> None:
    """Build, validate, and generate Danish persona artefacts."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@cli.group()
def demographics() -> None:
    """Generate deterministic demographic records."""


@demographics.command(name="run")
@click.option("--bundle", "bundle_dir", type=click.Path(path_type=Path), required=True)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_SAMPLING,
    show_default=True,
)
@click.option("--rows", type=click.IntRange(min=1), required=True)
@click.option("--seed", type=int, required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
def demographics_run(
    bundle_dir: Path, config_path: Path, rows: int, seed: int, output_dir: Path
) -> None:
    """Generate one reproducible Phase 2 run.

    Raises:
        click.ClickException: If generation fails.
    """
    try:
        run_dir = generate_records(
            bundle_dir=bundle_dir,
            sampling_config_path=config_path,
            output_dir=output_dir,
            rows=rows,
            seed=seed,
        )
    except Exception as error:
        raise click.ClickException(str(error)) from error
    _report_path(run_dir, "Generated run")


def _report_path(path: Path, message: str) -> None:
    LOGGER.info("%s: %s", message, path)
    click.echo(str(path))


def _require_pass(passed: bool, message: str) -> None:
    if not passed:
        raise click.ClickException(message)


def _require_network(network: bool) -> None:
    if not network:
        raise click.ClickException(
            "This command can access the network; pass --network explicitly"
        )


@cli.group(name="personas")
def personas() -> None:
    """Generate guarded persona shards and pilots."""


@personas.command(name="pilot")
@click.option("--input", "input_path", type=click.Path(path_type=Path), required=True)
@click.option("--sample-manifest", type=click.Path(path_type=Path), required=True)
@click.option("--config", "config_path", type=click.Path(path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--rows", type=click.IntRange(min=1), required=True)
@click.option("--batch-size", type=click.IntRange(min=1, max=5), default=5)
@click.option("--concurrency", type=click.IntRange(min=1, max=8), default=4)
@click.option("--delay-between-batches", type=click.FloatRange(min=0), default=0.0)
@click.option("--maximum-total-requests", type=click.IntRange(min=1), required=True)
@click.option("--input-price-per-million", type=click.FloatRange(min=0), required=True)
@click.option("--output-price-per-million", type=click.FloatRange(min=0), required=True)
@click.option(
    "--live", is_flag=True, help="Explicitly authorise all pilot model requests."
)
def personas_pilot(
    input_path: Path,
    sample_manifest: Path,
    config_path: Path,
    output_dir: Path,
    rows: int,
    batch_size: int,
    concurrency: int,
    delay_between_batches: float,
    maximum_total_requests: int,
    input_price_per_million: float,
    output_price_per_million: float,
    live: bool,
) -> None:
    """Generate a bounded, resumable persona pilot.

    Raises:
        click.ClickException: If approval or generation fails.
    """
    if not live:
        raise click.ClickException("Pilot generation requires explicit --live approval")
    try:
        pilot_dir = run_pilot(
            input_path=input_path,
            sample_manifest_path=sample_manifest,
            config_path=config_path,
            output_dir=output_dir,
            rows=rows,
            batch_size=batch_size,
            concurrency=concurrency,
            delay_between_batches=delay_between_batches,
            maximum_total_requests=maximum_total_requests,
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
        )
    except Exception as error:
        raise click.ClickException(str(error)) from error
    _report_path(pilot_dir, "Persona pilot")


@personas.command(name="shard")
@click.option("--input", "input_path", type=click.Path(path_type=Path), required=True)
@click.option("--sample-manifest", type=click.Path(path_type=Path), required=True)
@click.option("--config", "config_path", type=click.Path(path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--rows", type=click.IntRange(min=1, max=5), required=True)
@click.option("--offset", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--live", is_flag=True, help="Explicitly authorise model requests.")
def personas_shard(
    input_path: Path,
    sample_manifest: Path,
    config_path: Path,
    output_dir: Path,
    rows: int,
    offset: int,
    live: bool,
) -> None:
    """Plan or execute one guarded persona-generation shard.

    Raises:
        click.ClickException: If generation or its guards fail.
    """
    try:
        run_dir = generate_personas(
            input_path=input_path,
            sample_manifest_path=sample_manifest,
            config_path=config_path,
            output_dir=output_dir,
            rows=rows,
            offset=offset,
            live=live,
        )
    except Exception as error:
        raise click.ClickException(str(error)) from error
    _report_path(run_dir, "Persona shard")


@cli.group(name="sample")
def sample() -> None:
    """Create deterministic Phase 3 development samples."""


@sample.command(name="freeze")
@click.option("--run", "run_dir", type=click.Path(path_type=Path), required=True)
@click.option("--rows", type=click.IntRange(min=1), default=1000, show_default=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def sample_freeze(run_dir: Path, rows: int, output: Path) -> None:
    """Freeze a round-robin sample from a validated deterministic run.

    Raises:
        click.ClickException: If sample freezing fails.
    """
    try:
        sample_path = freeze_sample(run_dir=run_dir, rows=rows, output=output)
    except Exception as error:
        raise click.ClickException(str(error)) from error
    _report_path(sample_path, "Frozen sample")


@cli.group()
def sources() -> None:
    """Restore, pack, prepare, resolve, and fetch source data."""


@sources.command()
@click.option("--lock", "lock_path", type=click.Path(path_type=Path), required=True)
@click.option("--raw-dir", type=click.Path(path_type=Path), required=True)
@click.option(
    "--network",
    is_flag=True,
    help="Explicitly authorise network access for source fetching.",
)
def fetch(lock_path: Path, raw_dir: Path, network: bool) -> None:
    """Fetch locked source snapshots without overwriting valid snapshots.

    Raises:
        click.ClickException: If approval or fetching fails.
    """
    _require_network(network)
    try:
        lock = load_yaml_model(path=lock_path, model=SourceLock)
        result = fetch_sources(lock=lock, raw_dir=raw_dir)
    except Exception as error:
        raise click.ClickException(str(error)) from error
    _report_path(
        raw_dir,
        f"Fetched {len(result.table_manifests)} tables and "
        f"{len(result.classification_manifests)} classifications",
    )


@sources.command(name="pack")
@click.option(
    "--raw-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_RAW_DIR,
    show_default=True,
)
@click.option(
    "--archive",
    "archive_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_ARCHIVE,
    show_default=True,
)
def pack(raw_dir: Path, archive_path: Path) -> None:
    """Pack raw snapshots into a reproducible archive.

    Raises:
        click.ClickException: If packing fails.
    """
    try:
        count = build_raw_archive(raw_dir=raw_dir, archive_path=archive_path)
    except Exception as error:
        raise click.ClickException(str(error)) from error
    _report_path(archive_path, f"Packed {count} files")


@sources.command()
@click.option("--lock", "lock_path", type=click.Path(path_type=Path), required=True)
@click.option(
    "--categories", "categories_path", type=click.Path(path_type=Path), required=True
)
@click.option("--raw-dir", type=click.Path(path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
def prepare(
    lock_path: Path, categories_path: Path, raw_dir: Path, output_dir: Path
) -> None:
    """Prepare an offline demographic source bundle.

    Raises:
        click.ClickException: If preparation fails.
    """
    try:
        bundle = prepare_bundle(
            lock_path=lock_path,
            categories_path=categories_path,
            raw_dir=raw_dir,
            output_dir=output_dir,
        )
    except Exception as error:
        raise click.ClickException(str(error)) from error
    _report_path(bundle, "Prepared bundle")


@sources.command()
@click.option("--config", "config_path", type=click.Path(path_type=Path), required=True)
@click.option("--lock", "lock_path", type=click.Path(path_type=Path), required=True)
@click.option(
    "--network",
    is_flag=True,
    help="Explicitly authorise network access for source resolution.",
)
def resolve(config_path: Path, lock_path: Path, network: bool) -> None:
    """Resolve dynamic selectors into an explicit source lock.

    Raises:
        click.ClickException: If approval or resolution fails.
    """
    _require_network(network)
    try:
        config = load_yaml_model(path=config_path, model=SourcesConfig)
        lock = resolve_sources(config=config, lock_path=lock_path)
    except Exception as error:
        raise click.ClickException(str(error)) from error
    _report_path(lock_path, f"Resolved {len(lock.sources)} tables")


@sources.command()
@click.option(
    "--archive",
    "archive_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_ARCHIVE,
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data"),
    show_default=True,
)
@click.option(
    "--force",
    is_flag=True,
    help="Replace an existing raw snapshot directory after staging succeeds.",
)
def restore(archive_path: Path, output_dir: Path, force: bool) -> None:
    """Restore immutable raw snapshots from the archive.

    Raises:
        click.ClickException: If restoration fails.
    """
    try:
        count = restore_raw_sources(
            archive_path=archive_path, output_dir=output_dir, force=force
        )
    except Exception as error:
        raise click.ClickException(str(error)) from error
    _report_path(output_dir / RAW_DIRECTORY, f"Restored {count} files")


@cli.group(name="validate")
def validate() -> None:
    """Run mandatory artefact validation gates."""


@validate.command(name="demographics")
@click.option("--run", "run_dir", type=click.Path(path_type=Path), required=True)
@click.option("--bundle", "bundle_dir", type=click.Path(path_type=Path), required=True)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_VALIDATION,
    show_default=True,
)
@click.option(
    "--categories",
    "categories_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_CATEGORIES,
    show_default=True,
)
def validate_demographics_command(
    run_dir: Path, bundle_dir: Path, config_path: Path, categories_path: Path
) -> None:
    """Validate a deterministic demographic and OCEAN run."""
    report = _call_report(
        validate_demographics,
        run_dir=run_dir,
        bundle_dir=bundle_dir,
        validation_config_path=config_path,
        categories_path=categories_path,
    )
    _require_pass(report.passed, "Demographic validation failed")


def _call_report(function: object, **kwargs: object) -> ValidationReport:
    callable_function = t.cast(c.Callable[..., ValidationReport], function)
    try:
        return callable_function(**kwargs)
    except Exception as error:
        raise click.ClickException(str(error)) from error


@validate.command(name="personas")
@click.option("--run", "run_dir", type=click.Path(path_type=Path), required=True)
def validate_personas_command(run_dir: Path) -> None:
    """Validate a persona generation shard."""
    report = _call_report(validate_persona_run, run_dir=run_dir)
    _require_pass(report.passed, "Persona validation failed")


@validate.command(name="pilot")
@click.option(
    "--pilot", "--run", "pilot_dir", type=click.Path(path_type=Path), required=True
)
def validate_pilot_command(pilot_dir: Path) -> None:
    """Validate a merged persona pilot and every shard."""
    report = _call_report(validate_persona_pilot, pilot_dir=pilot_dir)
    _require_pass(report.passed, "Pilot validation failed")


@validate.command(name="sources")
@click.option("--bundle", "bundle_dir", type=click.Path(path_type=Path), required=True)
def validate_sources_command(bundle_dir: Path) -> None:
    """Validate a prepared source bundle."""
    report = _call_report(validate_sources, bundle_dir=bundle_dir)
    _require_pass(report.passed, "Source validation failed")


@cli.group(name="workflow")
def workflow() -> None:
    """Run complete, offline deterministic workflows."""


@workflow.command(name="deterministic")
@click.option("--target", type=click.Choice(["smoke", "statistical"]), required=True)
@click.option(
    "--archive",
    "archive_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_ARCHIVE,
    show_default=True,
)
@click.option(
    "--lock",
    "lock_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_LOCK,
    show_default=True,
)
@click.option(
    "--categories",
    "categories_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_CATEGORIES,
    show_default=True,
)
@click.option(
    "--sampling",
    "sampling_config_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_SAMPLING,
    show_default=True,
)
@click.option(
    "--validation",
    "validation_config_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_VALIDATION,
    show_default=True,
)
@click.option(
    "--raw-parent",
    type=click.Path(path_type=Path),
    default=DEFAULT_RAW_PARENT,
    show_default=True,
    help="Parent directory for the fixed restored raw snapshot directory.",
)
@click.option(
    "--processed-dir",
    type=click.Path(path_type=Path),
    default=Path("data/processed"),
    show_default=True,
)
@click.option(
    "--smoke-run-dir",
    "--smoke-run",
    "--smoke-output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/runs/smoke"),
    show_default=True,
)
@click.option(
    "--statistical-run-dir",
    "--statistical-run",
    "--statistical-output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/runs/statistical"),
    show_default=True,
)
@click.option(
    "--sample",
    "--sample-path",
    "--sample-output",
    "sample_path",
    type=click.Path(path_type=Path),
    help=(
        "Frozen sample path. Defaults to text-development-seeds.parquet inside the "
        "returned statistical run directory."
    ),
)
@click.option(
    "--sample-rows", type=click.IntRange(min=1), default=1000, show_default=True
)
@click.option("--skip-restore", is_flag=True)
@click.option("--force-restore", is_flag=True)
def deterministic(
    target: str,
    archive_path: Path,
    lock_path: Path,
    categories_path: Path,
    sampling_config_path: Path,
    validation_config_path: Path,
    raw_parent: Path,
    processed_dir: Path,
    smoke_run_dir: Path,
    statistical_run_dir: Path,
    sample_path: Path | None,
    sample_rows: int,
    skip_restore: bool,
    force_restore: bool,
) -> None:
    """Restore, prepare, validate, and generate deterministic artefacts.

    Raises:
        click.ClickException: If a stage fails or a validation gate rejects output.
    """
    if force_restore and skip_restore:
        raise click.ClickException("--force-restore cannot be used with --skip-restore")
    raw_dir = raw_parent / RAW_DIRECTORY
    try:
        if not skip_restore:
            restored = restore_raw_sources(
                archive_path=archive_path, output_dir=raw_parent, force=force_restore
            )
            _report_path(raw_dir, f"Restored {restored} files")
        bundle_dir = prepare_bundle(
            lock_path=lock_path,
            categories_path=categories_path,
            raw_dir=raw_dir,
            output_dir=processed_dir,
        )
        _report_path(bundle_dir, "Prepared bundle")
        _require_pass(
            validate_sources(bundle_dir=bundle_dir).passed, "Source validation failed"
        )
        sampling = load_yaml_model(path=sampling_config_path, model=SamplingConfig)
        smoke_dir = generate_records(
            bundle_dir=bundle_dir,
            sampling_config_path=sampling_config_path,
            output_dir=smoke_run_dir,
            rows=sampling.smoke_rows,
            seed=sampling.seed,
        )
        _require_pass(
            validate_demographics(
                run_dir=smoke_dir,
                bundle_dir=bundle_dir,
                validation_config_path=validation_config_path,
                categories_path=categories_path,
            ).passed,
            "Smoke demographic validation failed",
        )
        _report_path(smoke_dir, "Generated and validated smoke run")
        if target == "statistical":
            statistical_dir = generate_records(
                bundle_dir=bundle_dir,
                sampling_config_path=sampling_config_path,
                output_dir=statistical_run_dir,
                rows=sampling.statistical_rows,
                seed=sampling.seed,
            )
            _require_pass(
                validate_demographics(
                    run_dir=statistical_dir,
                    bundle_dir=bundle_dir,
                    validation_config_path=validation_config_path,
                    categories_path=categories_path,
                ).passed,
                "Statistical demographic validation failed",
            )
            _report_path(statistical_dir, "Generated and validated statistical run")
            frozen = freeze_sample(
                run_dir=statistical_dir,
                rows=sample_rows,
                output=(
                    sample_path
                    if sample_path is not None
                    else statistical_dir / DEFAULT_SAMPLE_FILENAME
                ),
            )
            _report_path(frozen, "Frozen sample")
    except click.ClickException:
        raise
    except Exception as error:
        raise click.ClickException(str(error)) from error


cli.add_command(release)
main = cli


if __name__ == "__main__":
    main()
