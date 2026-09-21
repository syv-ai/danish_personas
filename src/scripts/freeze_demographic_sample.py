"""Freeze a deterministic stratified development sample for Phase 3."""

import logging
from pathlib import Path

import click

from danish_personas.cli_logging import configure_cli_logging
from danish_personas.sampling.freeze import SampleSizeError, freeze_sample


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
    configure_cli_logging()
    logging.info("Freezing a %s-row demographic development sample", rows)
    try:
        freeze_sample(run_dir=run_dir, rows=rows, output=output)
    except SampleSizeError as error:
        raise click.ClickException(str(error)) from error
    logging.info("Completed demographic sample freeze")


if __name__ == "__main__":
    main()
