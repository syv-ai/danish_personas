"""Run and merge a resumable pilot from guarded persona-generation shards."""

import logging
from pathlib import Path

import click

from danish_personas.generation.pilot import run_pilot

LOGGER = logging.getLogger(__name__)


@click.command()
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
def main(
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
    """Generate a validated pilot without weakening per-invocation safety limits.

    Raises:
        click.ClickException:
            If live approval, provenance, generation, or validation fails.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
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
    LOGGER.info("Completed persona pilot: %s", pilot_dir)


if __name__ == "__main__":
    main()
