"""Pack the immutable Statistics Denmark source snapshots reproducibly."""

import logging
from pathlib import Path

import click

from danish_personas.sources.archive import (
    DEFAULT_ARCHIVE,
    RAW_DIRECTORY,
    build_raw_archive,
)
from danish_personas.sources.exceptions import SourceArchiveError


@click.command()
@click.option(
    "--raw-dir",
    type=click.Path(path_type=Path),
    default=Path("data") / RAW_DIRECTORY,
    show_default=True,
)
@click.option(
    "--archive",
    "archive_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_ARCHIVE,
    show_default=True,
)
def main(raw_dir: Path, archive_path: Path) -> None:
    """Pack raw snapshots into a byte-stable archive.

    Args:
        raw_dir:
            Directory holding the immutable raw snapshots.
        archive_path:
            Destination archive.

    Raises:
        click.ClickException:
            If the snapshot directory is missing or contains no regular files.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        packed_count = build_raw_archive(raw_dir=raw_dir, archive_path=archive_path)
    except SourceArchiveError as error:
        raise click.ClickException(str(error)) from error
    logging.info("Packed %s files into %s", packed_count, archive_path)


if __name__ == "__main__":
    main()
