"""Importable orchestration for the standard deterministic input workflow."""

import logging
from pathlib import Path

from .checksum import ChecksumValidationPolicy
from .io import load_yaml_model, sha256_file
from .models import FrozenSampleManifest, RunManifest, SamplingConfig
from .origin_labels import DEFAULT_ORIGIN_LABEL_CONTRACT_PATH
from .sampling.freeze import (
    ORIGIN_MARGINAL_METHOD,
    ORIGIN_STRATUM,
    STRATA,
    freeze_sample,
)
from .sampling.generator import generate_records
from .sources.archive import DEFAULT_ARCHIVE, RAW_DIRECTORY, restore_raw_sources
from .sources.prepare import prepare_bundle
from .validation.checks import validate_demographics, validate_sources

LOGGER = logging.getLogger(__name__)
DEFAULT_LOCK_PATH = Path("config/sources.lock.yaml")
DEFAULT_CATEGORIES_PATH = Path("config/categories.yaml")
DEFAULT_SAMPLING_CONFIG_PATH = Path("config/sampling.yaml")
DEFAULT_VALIDATION_CONFIG_PATH = Path("config/validation.yaml")
DEFAULT_RAW_PARENT = Path("data")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_SMOKE_RUN_DIR = Path("data/runs/smoke")
DEFAULT_STATISTICAL_RUN_DIR = Path("data/runs/statistical")
DEFAULT_SAMPLE_FILENAME = "text-development-seeds.parquet"
DEFAULT_SAMPLE_ROWS = 1000


def prepare_standard_sample(
    *,
    archive_path: Path = DEFAULT_ARCHIVE,
    lock_path: Path = DEFAULT_LOCK_PATH,
    categories_path: Path = DEFAULT_CATEGORIES_PATH,
    sampling_config_path: Path = DEFAULT_SAMPLING_CONFIG_PATH,
    validation_config_path: Path = DEFAULT_VALIDATION_CONFIG_PATH,
    raw_parent: Path = DEFAULT_RAW_PARENT,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    smoke_run_dir: Path = DEFAULT_SMOKE_RUN_DIR,
    statistical_run_dir: Path = DEFAULT_STATISTICAL_RUN_DIR,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    sample_path: Path | None = None,
    origin_labels_contract_path: Path = DEFAULT_ORIGIN_LABEL_CONTRACT_PATH,
    checksum_policy: ChecksumValidationPolicy = ChecksumValidationPolicy.STRICT,
) -> tuple[Path, Path]:
    """Prepare and validate the standard frozen sample for persona generation.

    Existing raw snapshots, prepared bundles, deterministic runs, and valid samples are
    reused by their underlying content-addressed services. A missing raw snapshot tree
    is restored from the committed archive before source preparation.

    Args:
        archive_path:
            Committed raw-source archive.
        lock_path:
            Resolved source lock.
        categories_path:
            Canonical category mappings.
        sampling_config_path:
            Deterministic sampling configuration.
        validation_config_path:
            Deterministic validation thresholds.
        raw_parent:
            Parent directory for restored raw snapshots.
        processed_dir:
            Destination for prepared source bundles.
        smoke_run_dir:
            Destination for the smoke demographic run.
        statistical_run_dir:
            Destination for the statistical demographic run.
        sample_rows:
            Number of rows in the frozen development sample.
        sample_path (optional):
            Sample destination. Defaults to the statistical run directory.
        origin_labels_contract_path:
            Reviewed Danish origin-label contract.
        checksum_policy:
            Whether persisted checksum comparisons are strict. Defaults to strict.

    Returns:
        The frozen sample path and its adjacent manifest path.

    Raises:
        ValueError:
            If any validation gate fails or an existing sample is inconsistent.
    """
    raw_dir = raw_parent / RAW_DIRECTORY
    if not raw_dir.exists():
        LOGGER.info("Restoring committed raw source archive")
        restore_raw_sources(archive_path=archive_path, output_dir=raw_parent)

    bundle_dir = prepare_bundle(
        lock_path=lock_path,
        categories_path=categories_path,
        raw_dir=raw_dir,
        output_dir=processed_dir,
        origin_labels_contract_path=origin_labels_contract_path,
        checksum_policy=checksum_policy,
    )
    _require_pass(
        validate_sources(bundle_dir=bundle_dir, checksum_policy=checksum_policy).passed,
        "Source validation failed",
    )

    sampling = load_yaml_model(path=sampling_config_path, model=SamplingConfig)
    smoke_dir = generate_records(
        bundle_dir=bundle_dir,
        sampling_config_path=sampling_config_path,
        output_dir=smoke_run_dir,
        rows=sampling.smoke_rows,
        seed=sampling.seed,
        checksum_policy=checksum_policy,
    )
    _require_pass(
        validate_demographics(
            run_dir=smoke_dir,
            bundle_dir=bundle_dir,
            validation_config_path=validation_config_path,
            categories_path=categories_path,
            checksum_policy=checksum_policy,
        ).passed,
        "Smoke demographic validation failed",
    )

    statistical_dir = generate_records(
        bundle_dir=bundle_dir,
        sampling_config_path=sampling_config_path,
        output_dir=statistical_run_dir,
        rows=sampling.statistical_rows,
        seed=sampling.seed,
        checksum_policy=checksum_policy,
    )
    _require_pass(
        validate_demographics(
            run_dir=statistical_dir,
            bundle_dir=bundle_dir,
            validation_config_path=validation_config_path,
            categories_path=categories_path,
            checksum_policy=checksum_policy,
        ).passed,
        "Statistical demographic validation failed",
    )

    output = sample_path or statistical_dir / DEFAULT_SAMPLE_FILENAME
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        if not _valid_existing_sample(
            sample_path=output,
            manifest_path=manifest_path,
            source_run_id=_run_id(statistical_dir, checksum_policy=checksum_policy),
            rows=sample_rows,
            checksum_policy=checksum_policy,
        ):
            raise ValueError("Existing frozen sample failed checksum validation")
        LOGGER.info("Reusing frozen sample %s", output)
    else:
        freeze_sample(
            run_dir=statistical_dir,
            rows=sample_rows,
            output=output,
            checksum_policy=checksum_policy,
        )
    return output, manifest_path


def _require_pass(passed: bool, message: str) -> None:
    if not passed:
        raise ValueError(message)


def _run_id(
    run_dir: Path,
    checksum_policy: ChecksumValidationPolicy = ChecksumValidationPolicy.STRICT,
) -> str:
    """Read the content-addressed identifier from a run manifest.

    Args:
        run_dir:
            Deterministic run directory.
        checksum_policy:
            Whether persisted digest bindings must match. Defaults to strict.

    Returns:
        The deterministic run identifier.
    """
    manifest = RunManifest.model_validate_json(
        (run_dir / "run-manifest.json").read_text(encoding="utf-8"),
        context={"checksum_policy": checksum_policy},
    )
    return manifest.run_id


def _valid_existing_sample(
    *,
    sample_path: Path,
    manifest_path: Path,
    source_run_id: str,
    rows: int,
    checksum_policy: ChecksumValidationPolicy = ChecksumValidationPolicy.STRICT,
) -> bool:
    if not sample_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = FrozenSampleManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8"),
            context={"checksum_policy": checksum_policy},
        )
    except OSError, ValueError:
        return False
    return (
        manifest.data_file == Path(sample_path.name)
        and manifest.source_run_id == source_run_id
        and manifest.rows == rows
        and manifest.mode == "population_proportional"
        and manifest.strata == [ORIGIN_STRATUM, *STRATA]
        and manifest.method.startswith(f"{ORIGIN_MARGINAL_METHOD}, then ")
        and (
            not checksum_policy.validates_checksums
            or sha256_file(sample_path) == manifest.sha256
        )
    )
