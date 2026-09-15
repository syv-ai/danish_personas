"""Deterministic stratified sampling of a validated demographic run."""

import logging
from pathlib import Path

import polars as pl

from ..io import sha256_file, write_json
from ..models import RunManifest

LOGGER = logging.getLogger(__name__)

STRATA = ["region_code", "education_level", "labour_market_status"]


def freeze_sample(run_dir: Path, rows: int, output: Path) -> Path:
    """Select a round-robin sample across demographic strata.

    Args:
        run_dir:
            Validated demographic run.
        rows:
            Maximum number of rows to freeze, capped at the size of the run.
        output:
            Destination Parquet file; its manifest is written beside it.

    Returns:
        The frozen sample file.
    """
    manifest = RunManifest.model_validate_json(
        (run_dir / "run-manifest.json").read_text(encoding="utf-8")
    )
    frame = pl.read_parquet(run_dir / manifest.data_file)
    rows = min(rows, frame.height)
    groups = frame.sort("persona_id").partition_by(STRATA, maintain_order=True)
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
    write_json(
        path=output.with_suffix(".manifest.json"),
        payload={
            "source_run_id": manifest.run_id,
            "rows": sample.height,
            "strata": STRATA,
            "method": "deterministic round-robin within sorted strata",
            "data_file": output.name,
            "sha256": sha256_file(output),
            "llm_calls": 0,
        },
    )
    LOGGER.info("Frozen %s development records at %s", sample.height, output)
    return output
