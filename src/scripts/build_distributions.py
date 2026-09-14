"""Prepare an offline demographic sampling bundle."""

import logging
from pathlib import Path

import click

from danish_personas.sources.prepare import prepare_bundle


@click.command()
@click.option("--lock", "lock_path", type=click.Path(path_type=Path), required=True)
@click.option(
    "--categories", "categories_path", type=click.Path(path_type=Path), required=True
)
@click.option("--raw-dir", type=click.Path(path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
def main(
    lock_path: Path, categories_path: Path, raw_dir: Path, output_dir: Path
) -> None:
    """Normalise downloaded aggregates into prepared distributions."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    bundle_dir = prepare_bundle(
        lock_path=lock_path,
        categories_path=categories_path,
        raw_dir=raw_dir,
        output_dir=output_dir,
    )
    logging.info("Prepared bundle: %s", bundle_dir)


if __name__ == "__main__":
    main()
