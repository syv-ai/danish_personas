"""Restore the committed Statistics Denmark source snapshots."""

import logging
from pathlib import Path

import click

from danish_personas.cli_logging import configure_cli_logging
from danish_personas.sources import archive as _archive
from danish_personas.sources.archive import DEFAULT_ARCHIVE, restore_raw_sources
from danish_personas.sources.exceptions import SourceArchiveError

RAW_DIRECTORY = _archive.RAW_DIRECTORY


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

    Args:
        archive_path:
            Path to the committed Zstandard-compressed tar archive.
        output_dir:
            Parent directory in which to restore the raw snapshot directory.
        force:
            Whether to replace an existing snapshot directory.

    Raises:
        click.ClickException:
            If the archive is missing, unsafe, corrupt, or the target already exists.
    """
    configure_cli_logging()
    logging.info("Restoring immutable source snapshots")
    try:
        restored_count = restore_raw_sources(
            archive_path=archive_path, output_dir=output_dir, force=force
        )
    except SourceArchiveError as error:
        raise click.ClickException(str(error)) from error
    logging.info("Restored %s immutable source files", restored_count)


if __name__ == "__main__":
    main()
