"""Freeze a deterministic stratified development sample for Phase 3."""

import logging
from pathlib import Path

import click

from danish_personas.sampling.freeze import freeze_sample


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
    try:
        freeze_sample(run_dir=run_dir, rows=rows, output=output)
    except ValueError as error:
        raise click.ClickException(str(error)) from error


if __name__ == "__main__":
    main()
