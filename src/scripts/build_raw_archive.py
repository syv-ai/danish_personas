"""Pack the immutable Statistics Denmark source snapshots reproducibly."""

import logging
from pathlib import Path

import click

from danish_personas.cli_logging import configure_cli_logging
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

    Members are sorted by archive path and given fixed ownership, mode, and
    timestamp, and the compression level is pinned, so repacking unchanged
    snapshots reproduces identical bytes.

    Args:
        raw_dir:
            Optional directory holding the immutable raw snapshots. Defaults to
            ``data/<RAW_DIRECTORY>``.
        archive_path:
            Optional destination archive. Defaults to ``DEFAULT_ARCHIVE``.

    Raises:
        click.ClickException:
            If the snapshot directory is missing or contains no regular files.
    """
    configure_cli_logging()
    logging.info("Packing immutable source snapshots")
    try:
        packed_count = build_raw_archive(raw_dir=raw_dir, archive_path=archive_path)
    except SourceArchiveError as error:
        raise click.ClickException(str(error)) from error
    logging.info("Packed %s source files into %s", packed_count, archive_path)


if __name__ == "__main__":
    main()
