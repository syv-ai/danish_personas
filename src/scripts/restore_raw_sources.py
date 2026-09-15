"""Restore the committed Statistics Denmark source snapshots."""

import logging
from pathlib import Path

import click

from danish_personas.sources.restore import (
    DEFAULT_ARCHIVE,
    RAW_DIRECTORY,
    restore_snapshots,
)

__all__ = ["DEFAULT_ARCHIVE", "RAW_DIRECTORY", "main"]


@click.command()
@click.option(
    "--archive",
    "archive_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_ARCHIVE,
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data"),
    show_default=True,
)
@click.option(
    "--force",
    is_flag=True,
    help="Replace an existing raw snapshot directory after staging succeeds.",
)
def main(archive_path: Path, output_dir: Path, force: bool) -> None:
    """Restore immutable raw snapshots from the committed archive.

    Raises:
        click.ClickException:
            If the archive is missing, unsafe, corrupt, or the target already exists.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        restore_snapshots(archive_path=archive_path, output_dir=output_dir, force=force)
    except ValueError as error:
        raise click.ClickException(str(error)) from error


if __name__ == "__main__":
    main()
