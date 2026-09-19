"""Deterministic Phase-3 sample freezing service."""

import logging
from pathlib import Path

import polars as pl

from ..io import sha256_file, write_json
from ..models import FROZEN_SAMPLE_SCHEMA_VERSION, RunManifest

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
            If rows is less than one or output leaves the validated run directory.
    """
    if rows < 1:
        raise ValueError("Requested sample must contain at least one row")
    if output.parent.resolve() != run_dir.resolve():
        message = "Frozen sample must remain inside its validated run directory"
        raise ValueError(message)
    manifest = RunManifest.model_validate_json(
        (run_dir / "run-manifest.json").read_text(encoding="utf-8")
    )
    frame = pl.read_parquet(run_dir / manifest.data_file)
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
    output.parent.mkdir(parents=True, exist_ok=True)
    sample.write_parquet(output, compression="zstd")
    sample_manifest: dict[str, object] = {
        "sample_schema_version": FROZEN_SAMPLE_SCHEMA_VERSION,
        "source_run_id": manifest.run_id,
        "rows": sample.height,
        "strata": ["municipality_code", "education_level", "labour_market_status"],
        "method": "deterministic round-robin within sorted strata",
        "data_file": output.name,
        "sha256": sha256_file(output),
        "llm_calls": 0,
    }
    write_json(path=output.with_suffix(".manifest.json"), payload=sample_manifest)
    LOGGER.info("Frozen %s development records at %s", sample.height, output)
    return output


class SampleSizeError(ValueError):
    """Raised when a requested sample is larger than its source run."""
