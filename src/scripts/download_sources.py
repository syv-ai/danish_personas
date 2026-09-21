"""Resolve and fetch immutable Statistics Denmark source snapshots."""

import logging
from pathlib import Path

import click

from danish_personas.cli_logging import configure_cli_logging
from danish_personas.io import load_yaml_model
from danish_personas.models import SourceLock, SourcesConfig
from danish_personas.sources.acquisition import fetch_sources, resolve_sources
from danish_personas.sources.exceptions import SourceAcquisitionError


@click.group()
def main() -> None:
    """Manage official aggregate source snapshots."""
    configure_cli_logging()


@main.command()
@click.option("--lock", "lock_path", type=click.Path(path_type=Path), required=True)
@click.option("--raw-dir", type=click.Path(path_type=Path), required=True)
def fetch(lock_path: Path, raw_dir: Path) -> None:
    """Fetch every locked source without overwriting snapshots.

    Raises:
        click.ClickException:
            If source acquisition fails.
    """
    logging.info("Loading source lock before fetching snapshots")
    lock = load_yaml_model(path=lock_path, model=SourceLock)
    try:
        result = fetch_sources(lock=lock, raw_dir=raw_dir)
    except SourceAcquisitionError as error:
        raise click.ClickException(str(error)) from error
    logging.info(
        "Verified %s table and %s classification snapshots",
        len(result.table_manifests),
        len(result.classification_manifests),
    )


@main.command()
@click.option("--config", "config_path", type=click.Path(path_type=Path), required=True)
@click.option("--lock", "lock_path", type=click.Path(path_type=Path), required=True)
def resolve(config_path: Path, lock_path: Path) -> None:
    """Resolve dynamic table selectors into explicit source queries.

    Raises:
        click.ClickException:
            If source resolution fails.
    """
    logging.info("Loading source configuration before resolving selectors")
    config = load_yaml_model(path=config_path, model=SourcesConfig)
    try:
        lock = resolve_sources(config=config, lock_path=lock_path)
    except SourceAcquisitionError as error:
        raise click.ClickException(str(error)) from error
    logging.info(
        "Resolved %s source tables and %s classifications to %s",
        len(lock.sources),
        len(lock.classifications),
        lock_path,
    )


if __name__ == "__main__":
    main()
