"""Safe, reproducible services for the raw source archive."""

import shutil
import stat
import tarfile
import tempfile
from compression.zstd import CompressionParameter
from pathlib import Path

from .exceptions import SourceArchiveError

RAW_DIRECTORY = "raw-hardened-20260918"
DEFAULT_ARCHIVE = Path("data") / f"{RAW_DIRECTORY}.tar.zst"
FILE_MODE = 0o644
# Pinned so the committed archive does not change size with library defaults.
COMPRESSION_OPTIONS: dict[int, int] = {CompressionParameter.compression_level: 19}


def build_raw_archive(raw_dir: Path, archive_path: Path) -> int:
    """Pack raw snapshots into a byte-stable archive.

    Members are sorted by archive path and given fixed ownership, mode, and
    timestamp, and the compression level is pinned, so repacking unchanged
    snapshots reproduces identical bytes.

    Args:
        raw_dir:
            Directory holding the immutable raw snapshots.
        archive_path:
            Destination archive.

    Returns:
        Number of packed files.

    Raises:
        SourceArchiveError:
            If the snapshot directory is missing, unsafe, or contains no
            regular files.
    """
    try:
        _validate_directory(path=raw_dir, description="Raw snapshot directory")
        paths = _regular_files(raw_dir=raw_dir)
        if not paths:
            message = f"Raw snapshot directory contains no files: {raw_dir}"
            raise SourceArchiveError(message)

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
    except SourceArchiveError:
        raise
    except (OSError, tarfile.TarError, ValueError) as error:
        raise SourceArchiveError(str(error)) from error
    return len(paths)


def _regular_files(raw_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(raw_dir.rglob("*")):
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            message = f"Raw snapshot tree contains a non-regular entry: {path}"
            raise SourceArchiveError(message)
        paths.append(path)
    return paths


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


def _validate_directory(path: Path, description: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        message = f"{description} does not exist: {path}"
        raise SourceArchiveError(message) from error
    if not stat.S_ISDIR(metadata.st_mode):
        message = f"{description} is not a directory: {path}"
        raise SourceArchiveError(message)


def restore_raw_sources(
    archive_path: Path, output_dir: Path, force: bool = False
) -> int:
    """Restore immutable raw snapshots from a compressed archive.

    Extraction is performed in a temporary directory below ``output_dir`` and
    installed only after every archive member has passed the safety checks.

    Args:
        archive_path:
            Path to the committed Zstandard-compressed tar archive.
        output_dir:
            Parent directory in which to restore the raw snapshot directory.
        force (optional):
            Whether to replace an existing snapshot directory. Defaults to False.

    Returns:
        Number of restored archive members.

    Raises:
        SourceArchiveError:
            If the archive is missing, unsafe, corrupt, or the target already
            exists.
    """
    target = output_dir / RAW_DIRECTORY
    if _path_exists(path=target) and not force:
        message = f"Raw snapshot directory already exists: {target}"
        raise SourceArchiveError(message)

    try:
        with tarfile.open(name=archive_path, mode="r:zst") as archive:
            members = _validated_members(archive=archive)
            output_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                dir=output_dir, prefix=".raw-restore-"
            ) as temporary_name:
                temporary_dir = Path(temporary_name)
                archive.extractall(path=temporary_dir, members=members, filter="data")
                staged = temporary_dir / RAW_DIRECTORY
                _install_staged_directory(
                    staged=staged,
                    target=target,
                    temporary_dir=temporary_dir,
                    force=force,
                )
    except SourceArchiveError:
        raise
    except (OSError, tarfile.TarError, ValueError) as error:
        raise SourceArchiveError(str(error)) from error
    return len(members)


def _install_staged_directory(
    staged: Path, target: Path, temporary_dir: Path, force: bool
) -> None:
    if not staged.is_dir():
        message = f"Raw source archive does not contain {RAW_DIRECTORY}"
        raise SourceArchiveError(message)
    previous = temporary_dir / "previous"
    had_previous = _path_exists(path=target)
    if had_previous:
        if not force:
            message = f"Raw snapshot directory already exists: {target}"
            raise SourceArchiveError(message)
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
        raise SourceArchiveError(message)
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
            raise SourceArchiveError(message)
    return members
