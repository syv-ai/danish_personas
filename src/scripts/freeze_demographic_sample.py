"""Freeze a deterministic stratified development sample for Phase 3."""

import logging
from pathlib import Path

import click
import polars as pl

from danish_personas.io import sha256_file, write_json
from danish_personas.models import RunManifest


@click.command()
@click.option("--run", "run_dir", type=click.Path(path_type=Path), required=True)
@click.option("--rows", type=click.IntRange(min=1), default=1000, show_default=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def main(run_dir: Path, rows: int, output: Path) -> None:
    """Select a round-robin sample across demographic strata.

    Raises:
        click.ClickException:
            If the requested sample is larger than the source run.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = RunManifest.model_validate_json(
        (run_dir / "run-manifest.json").read_text(encoding="utf-8")
    )
    frame = pl.read_parquet(run_dir / manifest.data_file)
    if rows > frame.height:
        raise click.ClickException("Requested sample exceeds the run row count")
    groups = frame.sort("persona_id").partition_by(
        ["region_code", "education_level", "labour_market_status"], maintain_order=True
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
        "source_run_id": manifest.run_id,
        "rows": sample.height,
        "strata": ["region_code", "education_level", "labour_market_status"],
        "method": "deterministic round-robin within sorted strata",
        "data_file": output.name,
        "sha256": sha256_file(output),
        "llm_calls": 0,
    }
    write_json(path=output.with_suffix(".manifest.json"), payload=sample_manifest)
    logging.info("Frozen %s development records at %s", sample.height, output)


if __name__ == "__main__":
    main()
