"""Safe restoration of the committed raw source archive."""

import logging
import shutil
import tarfile
import tempfile
from pathlib import Path

LOGGER = logging.getLogger(__name__)

DEFAULT_ARCHIVE = Path("data") / "raw-hardened-20260914.tar.zst"
RAW_DIRECTORY = "raw-hardened-20260914"


def restore_snapshots(
    archive_path: Path = DEFAULT_ARCHIVE,
    output_dir: Path = Path("data"),
    force: bool = False,
) -> Path:
    """Restore immutable raw snapshots from the committed archive.

    Args:
        archive_path:
            Committed Zstandard-compressed tar archive.
        output_dir:
            Parent directory in which to restore the raw snapshot directory.
        force:
            Whether to replace an existing snapshot directory.

    Returns:
        The restored raw snapshot directory.

    Raises:
        ValueError:
            If the archive is missing, unsafe, corrupt, or the target already exists.
    """
    target = output_dir / RAW_DIRECTORY
    if _path_exists(path=target) and not force:
        message = f"Raw snapshot directory already exists: {target}"
        raise ValueError(message)
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
    except (OSError, tarfile.TarError) as error:
        raise ValueError(str(error)) from error
    LOGGER.info("Restored %s immutable source files", len(members))
    return target


def _install_staged_directory(
    staged: Path, target: Path, temporary_dir: Path, force: bool
) -> None:
    if not staged.is_dir():
        message = f"Raw source archive does not contain {RAW_DIRECTORY}"
        raise ValueError(message)
    previous = temporary_dir / "previous"
    had_previous = _path_exists(path=target)
    if had_previous:
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
    # Extraction also runs with filter="data", which rejects absolute paths, parent
    # traversal and special members. This adds the archive-specific requirement that
    # every member is a regular file inside the expected snapshot directory.
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
