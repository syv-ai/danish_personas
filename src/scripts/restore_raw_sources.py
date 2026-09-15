"""Restore the committed Statistics Denmark source snapshots."""

import logging
import shutil
import tarfile
import tempfile
from pathlib import Path

import click

RAW_DIRECTORY = "raw-hardened-20260914"
DEFAULT_ARCHIVE = Path("data") / f"{RAW_DIRECTORY}.tar.zst"


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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    target = output_dir / RAW_DIRECTORY
    if _path_exists(path=target) and not force:
        message = f"Raw snapshot directory already exists: {target}"
        raise click.ClickException(message)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=output_dir, prefix=".raw-restore-"
        ) as temporary_name:
            temporary_dir = Path(temporary_name)
            with tarfile.open(name=archive_path, mode="r:zst") as archive:
                members = _validated_members(archive=archive)
                archive.extractall(path=temporary_dir, members=members, filter="data")
            staged = temporary_dir / RAW_DIRECTORY
            _install_staged_directory(
                staged=staged, target=target, temporary_dir=temporary_dir, force=force
            )
    except (OSError, tarfile.TarError, ValueError) as error:
        raise click.ClickException(str(error)) from error
    logging.info("Restored %s immutable source files", len(members))


def _install_staged_directory(
    staged: Path, target: Path, temporary_dir: Path, force: bool
) -> None:
    if not staged.is_dir():
        message = f"Raw source archive does not contain {RAW_DIRECTORY}"
        raise ValueError(message)
    previous = temporary_dir / "previous"
    had_previous = _path_exists(path=target)
    if had_previous:
        if not force:
            message = f"Raw snapshot directory already exists: {target}"
            raise ValueError(message)
        target.replace(previous)
    try:
        staged.replace(target)
    except OSError:
        if had_previous and not _path_exists(path=target):
            previous.replace(target)
        raise
    if had_previous:
        if previous.is_symlink() or previous.is_file():
            previous.unlink()
        else:
            shutil.rmtree(previous)


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _validated_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    if not members:
        message = "Raw source archive is empty"
        raise ValueError(message)
    for member in members:
        path = Path(member.name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or path.parts[0] != RAW_DIRECTORY
            or not member.isfile()
        ):
            message = f"Unsafe raw source archive member: {member.name}"
            raise ValueError(message)
    return members


if __name__ == "__main__":
    main()
