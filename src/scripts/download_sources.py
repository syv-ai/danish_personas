"""Resolve and fetch immutable Statistics Denmark source snapshots."""

import logging
from pathlib import Path

import click

from danish_personas.io import load_yaml_model
from danish_personas.models import SourceLock, SourcesConfig
from danish_personas.sources.statbank import fetch_sources, resolve_sources


@click.group()
def main() -> None:
    """Manage official aggregate source snapshots."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@main.command()
@click.option("--lock", "lock_path", type=click.Path(path_type=Path), required=True)
@click.option("--raw-dir", type=click.Path(path_type=Path), required=True)
def fetch(lock_path: Path, raw_dir: Path) -> None:
    """Fetch every locked source without overwriting snapshots."""
    lock = load_yaml_model(path=lock_path, model=SourceLock)
    manifests = fetch_sources(lock=lock, raw_dir=raw_dir)
    logging.info("Verified %s immutable source snapshots", len(manifests))


@main.command()
@click.option("--config", "config_path", type=click.Path(path_type=Path), required=True)
@click.option("--lock", "lock_path", type=click.Path(path_type=Path), required=True)
def resolve(config_path: Path, lock_path: Path) -> None:
    """Resolve dynamic table selectors into explicit source queries."""
    config = load_yaml_model(path=config_path, model=SourcesConfig)
    lock = resolve_sources(config=config, lock_path=lock_path)
    logging.info("Resolved %s source tables to %s", len(lock.sources), lock_path)


if __name__ == "__main__":
    main()
