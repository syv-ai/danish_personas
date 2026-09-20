"""Deterministic Phase-3 sample freezing service."""

import json
import logging
import os
import stat
import tempfile
from pathlib import Path

import polars as pl

from ..io import sha256_file
from ..models import (
    FROZEN_SAMPLE_SCHEMA_VERSION,
    SAMPLER_SCHEMA_VERSION,
    FrozenSampleManifest,
    RunManifest,
)

LOGGER = logging.getLogger(__name__)


def freeze_sample(*, run_dir: Path, rows: int, output: Path) -> Path:
    """Select and persist a deterministic stratified development sample.

    Args:
        run_dir:
            Validated deterministic demographic run directory.
        rows:
            Number of records to select.
        output:
            Destination Parquet path for the frozen sample.

    Returns:
        Path to the written frozen sample.

    Raises:
        SampleSizeError:
            If the requested sample is larger than the source run.
        ValueError:
            If rows is less than one or either output path is not a direct,
            non-linked child of the validated run directory.
    """
    if rows < 1:
        raise ValueError("Requested sample must contain at least one row")
    canonical_run_dir = _canonical_run_directory(run_dir=run_dir)
    output_path, manifest_path = _validate_destinations(
        run_dir=canonical_run_dir, output=output
    )
    manifest = RunManifest.model_validate_json(
        (canonical_run_dir / "run-manifest.json").read_text(encoding="utf-8")
    )
    if manifest.sampler_schema_version != SAMPLER_SCHEMA_VERSION:
        raise ValueError("Cannot freeze a legacy demographic run")
    frame = pl.read_parquet(canonical_run_dir / manifest.data_file)
    if rows > frame.height:
        raise SampleSizeError("Requested sample exceeds the run row count")
    groups = frame.sort("persona_id").partition_by(
        ["municipality_code", "education_level", "labour_market_status"],
        maintain_order=True,
    )
    selected: list[pl.DataFrame] = []
    depth = 0
    while len(selected) < rows:
        added = False
        for group in groups:
            if depth < group.height:
                selected.append(group.slice(depth, 1))
                added = True
                if len(selected) == rows:
                    break
        if not added:
            break
        depth += 1
    sample = pl.concat(selected).sort("persona_id")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=canonical_run_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        sample.write_parquet(temporary, compression="zstd")
        _validate_destinations(run_dir=canonical_run_dir, output=output_path)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    sample_manifest = FrozenSampleManifest(
        sample_schema_version=FROZEN_SAMPLE_SCHEMA_VERSION,
        source_run_id=manifest.run_id,
        rows=sample.height,
        strata=["municipality_code", "education_level", "labour_market_status"],
        method="deterministic round-robin within sorted strata",
        data_file=output_path.name,
        sha256=sha256_file(output_path),
        llm_calls=0,
        origin_labels_contract_path=manifest.origin_labels_contract_path,
        origin_labels_contract_version=manifest.origin_labels_contract_version,
        origin_labels_contract_sha256=manifest.origin_labels_contract_sha256,
        origin_labels_contract_content=manifest.origin_labels_contract_content,
    )
    _write_manifest(
        path=manifest_path,
        run_dir=canonical_run_dir,
        payload=sample_manifest.model_dump(mode="json"),
    )
    LOGGER.info("Frozen %s development records at %s", sample.height, output_path)
    return output


class SampleSizeError(ValueError):
    """Raised when a requested sample is larger than its source run."""


def _canonical_run_directory(*, run_dir: Path) -> Path:
    """Resolve and validate the source run directory.

    Args:
        run_dir:
            Source run directory.

    Returns:
        Canonical source run directory.

    Raises:
        ValueError:
            If the source run cannot be resolved or is not a directory.
    """
    try:
        canonical = run_dir.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("Validated source run directory cannot be resolved") from error
    if not canonical.is_dir():
        raise ValueError("Validated source run directory must be a directory")
    return canonical


def _validate_destinations(*, run_dir: Path, output: Path) -> tuple[Path, Path]:
    """Validate both direct output paths before either can be written.

    Args:
        run_dir:
            Canonical source run directory.
        output:
            Requested sample output path.

    Returns:
        Canonical lexical paths for the sample and adjacent manifest.

    Raises:
        ValueError:
            If either destination is outside the run or linked.
    """
    if ".." in output.parts:
        message = "Frozen sample output must not contain traversal components"
        raise ValueError(message)
    output_path = _lexical_absolute(path=output)
    manifest_path = output_path.with_suffix(".manifest.json")
    _validate_destination(run_dir=run_dir, path=output_path, label="sample output")
    _validate_destination(run_dir=run_dir, path=manifest_path, label="sample manifest")
    return output_path, manifest_path


def _lexical_absolute(*, path: Path) -> Path:
    """Make an absolute path without resolving symlink targets.

    Args:
        path:
            Path to normalise lexically.

    Returns:
        Absolute, lexically normalised path.
    """
    return Path(os.path.abspath(os.fspath(path)))


def _validate_destination(*, run_dir: Path, path: Path, label: str) -> None:
    """Reject aliases, links, and non-files at a direct output destination.

    Args:
        run_dir:
            Canonical source run directory.
        path:
            Destination path to inspect.
        label:
            Human-readable destination name for errors.

    Raises:
        ValueError:
            If the destination is not a direct, non-linked file path.
    """
    if ".." in path.parts or path.parent != run_dir:
        message = (
            f"Frozen {label} must remain inside its validated run directory "
            "as a direct path"
        )
        raise ValueError(message)
    current = Path(path.anchor)
    try:
        for component in path.parts[1:]:
            current /= component
            if current.is_symlink():
                raise ValueError(f"Frozen {label} must not contain a symlink")
        entry = path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise ValueError(f"Frozen {label} cannot be inspected") from error
    if path.is_symlink() or not stat.S_ISREG(entry.st_mode):
        raise ValueError(f"Frozen {label} must be a regular non-linked file")


def _write_manifest(*, path: Path, run_dir: Path, payload: dict[str, object]) -> None:
    """Atomically write a manifest through a private temporary file."""
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=run_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        _validate_destination(run_dir=run_dir, path=path, label="sample manifest")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
