"""Pack the immutable Statistics Denmark source snapshots reproducibly."""

import logging
import tarfile
from compression.zstd import CompressionParameter
from pathlib import Path

import click

from scripts.restore_raw_sources import RAW_DIRECTORY

DEFAULT_ARCHIVE = Path("data") / f"{RAW_DIRECTORY}.tar.zst"
FILE_MODE = 0o644
# Pinned so the committed archive does not change size with library defaults.
COMPRESSION_OPTIONS: dict[int, int] = {CompressionParameter.compression_level: 19}


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
            Directory holding the immutable raw snapshots.
        archive_path:
            Destination archive.

    Raises:
        click.ClickException:
            If the snapshot directory is missing or contains no regular files.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not raw_dir.is_dir():
        message = f"Raw snapshot directory does not exist: {raw_dir}"
        raise click.ClickException(message)
    paths = sorted(path for path in raw_dir.rglob("*") if path.is_file())
    if not paths:
        message = f"Raw snapshot directory contains no files: {raw_dir}"
        raise click.ClickException(message)

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(
        name=archive_path, mode="w:zst", options=COMPRESSION_OPTIONS
    ) as archive:
        for path in paths:
            member = _stable_member(
                name=f"{RAW_DIRECTORY}/{path.relative_to(raw_dir).as_posix()}",
                size=path.stat().st_size,
            )
            with path.open("rb") as file:
                archive.addfile(member, fileobj=file)
    logging.info("Packed %s files into %s", len(paths), archive_path)


def _stable_member(name: str, size: int) -> tarfile.TarInfo:
    member = tarfile.TarInfo(name=name)
    member.size = size
    member.mode = FILE_MODE
    member.type = tarfile.REGTYPE
    member.mtime = 0
    member.uid = 0
    member.gid = 0
    member.uname = ""
    member.gname = ""
    return member


if __name__ == "__main__":
    main()
