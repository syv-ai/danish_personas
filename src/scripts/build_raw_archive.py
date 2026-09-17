"""Pack the immutable Statistics Denmark source snapshots reproducibly."""

import logging
import stat
import tarfile
from compression.zstd import CompressionParameter
from pathlib import Path

import click

from scripts.restore_raw_sources import DEFAULT_ARCHIVE, RAW_DIRECTORY

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
            Optional directory holding the immutable raw snapshots. Defaults to
            ``data/<RAW_DIRECTORY>``.
        archive_path:
            Optional destination archive. Defaults to ``DEFAULT_ARCHIVE``.

    Raises:
        click.ClickException:
            If the snapshot directory is missing or contains no regular files.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _validate_directory(path=raw_dir, description="Raw snapshot directory")
    paths = _regular_files(raw_dir=raw_dir)
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


def _regular_files(raw_dir: Path) -> list[Path]:
    """Return regular files and reject unsafe entries in a raw snapshot tree.

    Args:
        raw_dir:
            Root directory to inspect.

    Returns:
        Paths to regular files, sorted by their relative archive paths.

    Raises:
        click.ClickException:
            If the tree contains a symlink or another non-regular, non-directory
            entry.
    """
    paths: list[Path] = []
    for path in sorted(raw_dir.rglob("*")):
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            message = f"Raw snapshot tree contains a non-regular entry: {path}"
            raise click.ClickException(message)
        paths.append(path)
    return paths


def _stable_member(name: str, size: int) -> tarfile.TarInfo:
    """Build a tar member whose metadata does not vary between runs.

    Ownership, mode, and timestamp are fixed so that repacking unchanged
    snapshots reproduces identical archive bytes.

    Args:
        name:
            Path recorded inside the archive.
        size:
            Member size in bytes.

    Returns:
        Member header ready to write.
    """
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


def _validate_directory(path: Path, description: str) -> None:
    """Require a real directory without following a symbolic link.

    Args:
        path:
            Directory to validate.
        description:
            Human-readable description used in the error message.

    Raises:
        click.ClickException:
            If ``path`` is missing or is not a directory entry.
    """
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        message = f"{description} does not exist: {path}"
        raise click.ClickException(message) from error
    if not stat.S_ISDIR(metadata.st_mode):
        message = f"{description} is not a directory: {path}"
        raise click.ClickException(message)


if __name__ == "__main__":
    main()
