"""Integrity and schema verification for prepared source bundles."""

import collections.abc as c
import ctypes
import ctypes.wintypes as wintypes
import hashlib
import io
import json
import ntpath
import os
import stat
import typing as t
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ..checksum import ChecksumValidationPolicy
from ..models import PREPARED_BUNDLE_SCHEMA_VERSION, BundleManifest
from ..origin_labels import ORIGIN_LABEL_CONTRACT_SHA256, load_origin_label_contract

REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "normalized/folk1a_base_unpooled.parquet": frozenset(
        {
            "municipality_code",
            "municipality",
            "region_code",
            "region",
            "sex",
            "age",
            "age_band",
            "marital_status",
            "count",
            "suppressed",
        }
    ),
    "normalized/folk_age_sampling.parquet": frozenset(
        {
            "municipality_code",
            "municipality",
            "region_code",
            "region",
            "age_band",
            "sex",
            "age",
            "count",
            "suppressed",
        }
    ),
    "normalized/folk_marital_sampling.parquet": frozenset(
        {
            "municipality_code",
            "municipality",
            "region_code",
            "region",
            "age_band",
            "sex",
            "marital_status",
            "count",
            "suppressed",
        }
    ),
    "normalized/ras209_joint_unpooled.parquet": frozenset(
        {
            "municipality_code",
            "municipality",
            "region_code",
            "region",
            "age_band",
            "sex",
            "education_source_code",
            "education_level",
            "labour_market_status",
            "count",
            "suppressed",
        }
    ),
    "normalized/ras209_sampling.parquet": frozenset(
        {
            "municipality_code",
            "municipality",
            "region_code",
            "region",
            "age_band",
            "sex",
            "education_source_code",
            "education_level",
            "labour_market_status",
            "count",
            "suppressed",
        }
    ),
    "normalized/ras202_detail_unpooled.parquet": frozenset(
        {
            "age_band",
            "age_key",
            "sex",
            "labour_market_status",
            "detailed_status_code",
            "detailed_status",
            "count",
            "suppressed",
        }
    ),
    "normalized/ras202_sampling.parquet": frozenset(
        {
            "age_band",
            "sex",
            "labour_market_status",
            "detailed_status_code",
            "detailed_status",
            "count",
            "suppressed",
        }
    ),
    "normalized/befolk3_holdout.parquet": frozenset(
        {
            "municipality_code",
            "municipality",
            "region_code",
            "region",
            "sex",
            "age",
            "count",
            "suppressed",
        }
    ),
    "normalized/ras210_holdout.parquet": frozenset(
        {
            "municipality_code",
            "municipality",
            "region_code",
            "region",
            "status_group_code",
            "age_key",
            "sex",
            "count",
            "suppressed",
        }
    ),
    "normalized/folk2_origin_country_marginal.parquet": frozenset(
        {"origin_country_code", "origin_country", "origin_country_da", "count"}
    ),
    "normalized/job_function_sex_marginal.parquet": frozenset(
        {"job_function_code", "job_function", "sex", "count"}
    ),
    "normalized/geography_hierarchy.parquet": frozenset(
        {
            "municipality_code",
            "municipality",
            "landsdel_code",
            "landsdel",
            "region_code",
            "region",
        }
    ),
}
SOURCE_REPORT = "source-preparation-report.json"
BUNDLE_MANIFEST = "bundle-manifest.json"

_WINDOWS_GENERIC_READ = 0x80000000
_WINDOWS_FILE_READ_ATTRIBUTES = 0x00000080
_WINDOWS_FILE_SHARE_READ = 0x00000001
_WINDOWS_OPEN_EXISTING = 3
_WINDOWS_FILE_ATTRIBUTE_DIRECTORY = 0x00000010
_WINDOWS_FILE_ATTRIBUTE_DEVICE = 0x00000040
_WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT = 0x00000400
_WINDOWS_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_WINDOWS_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000


def _regular_file_inventory(*, bundle_dir: Path) -> dict[str, Path]:
    """Return the exact inventory, retaining the historical test seam."""
    capture = _capture_inventory(bundle_dir=bundle_dir)
    try:
        return {
            relative_path: item.physical_path
            for relative_path, item in capture.files.items()
        }
    finally:
        _close_capture_handles(capture=capture)


def _canonical_relative_key(relative_path: str) -> str:
    """Return a safe POSIX key for a manifest or filesystem relative path.

    Raises:
        ValueError:
            If the path is empty, rooted, anchored to a Windows drive, or has
            an empty, dot, or parent component.
    """
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("Prepared bundle file paths must not be empty")
    if "\x00" in relative_path:
        raise ValueError("Prepared bundle file paths must not contain NUL bytes")
    if relative_path.startswith(("/", "\\")):
        raise ValueError(
            f"Prepared bundle file path is absolute or rooted: {relative_path}"
        )
    # A drive-relative path (for example, C:foo) is unsafe too: its anchor is
    # determined by the process's current Windows drive rather than the bundle.
    windows_path = ntpath.splitdrive(relative_path.replace("/", "\\"))[0]
    if windows_path:
        raise ValueError(
            f"Prepared bundle file path has a Windows drive or device anchor: "
            f"{relative_path}"
        )
    parts = relative_path.replace("\\", "/").split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(
            "Prepared bundle file paths must have non-empty, non-dot, non-parent "
            f"components: {relative_path}"
        )
    return "/".join(parts)


@dataclass(frozen=True)
class _FileIdentity:
    """Identity and immutable metadata captured for one regular file."""

    device: int
    inode: int
    size: int
    modified_ns: int
    links: int


@dataclass(frozen=True)
class _CapturedFile:
    """Bytes and identity captured from one safely opened file."""

    relative_path: str
    physical_path: Path
    content: bytes
    identity: _FileIdentity


@dataclass
class _BundleCapture:
    """The complete safe capture of a prepared bundle."""

    files: dict[str, _CapturedFile]
    directories: dict[str, _FileIdentity]
    windows_handles: list[int]


def _capture_inventory(*, bundle_dir: Path) -> _BundleCapture:
    """Capture every regular file and directory entry without following links.

    Returns:
        The complete stable byte capture.
    """
    if os.name == "nt":
        return _capture_windows_inventory(bundle_dir=bundle_dir)
    return _capture_posix_inventory(bundle_dir=bundle_dir)


def _capture_posix_inventory(*, bundle_dir: Path) -> _BundleCapture:
    root = Path(bundle_dir)
    root_stat = _posix_lstat_directory(path=root)
    capture = _BundleCapture(files={}, directories={}, windows_handles=[])
    pending: list[tuple[Path, str, _FileIdentity]] = [(root, "", root_stat)]
    try:
        while pending:
            directory, prefix, expected = pending.pop()
            _posix_check_directory(path=directory, expected=expected)
            directory_key = _canonical_relative_key(prefix) if prefix else ""
            capture.directories[directory_key] = expected
            try:
                entries = sorted(os.scandir(directory), key=lambda item: item.name)
            except OSError as error:
                raise ValueError(
                    f"Prepared bundle directory cannot be read: {directory}"
                ) from error
            for entry in entries:
                relative_path = f"{prefix}/{entry.name}" if prefix else entry.name
                try:
                    entry_stat = entry.stat(follow_symlinks=False)
                except OSError as error:
                    raise ValueError(
                        f"Prepared bundle entry cannot be inspected: {relative_path}"
                    ) from error
                if stat.S_ISLNK(entry_stat.st_mode):
                    raise ValueError(
                        f"Prepared bundle inventory contains a symlink: {relative_path}"
                    )
                identity = _posix_identity(entry_stat)
                if stat.S_ISDIR(entry_stat.st_mode):
                    pending.append((Path(entry.path), relative_path, identity))
                    continue
                if not stat.S_ISREG(entry_stat.st_mode):
                    raise ValueError(
                        "Prepared bundle inventory contains a non-regular file: "
                        f"{relative_path}"
                    )
                if entry_stat.st_nlink != 1:
                    raise ValueError(
                        "Prepared bundle inventory contains a hard link: "
                        f"{relative_path}"
                    )
                canonical = _canonical_relative_key(relative_path)
                _ensure_path_stays_in_bundle(bundle_dir=root, relative_path=canonical)
                captured = _capture_posix_file(
                    path=Path(entry.path), relative_path=canonical, expected=identity
                )
                _add_capture(capture=capture, item=captured)
            _posix_check_directory(path=directory, expected=expected)
        return capture
    except OSError as error:
        raise ValueError("Prepared bundle entry cannot be safely captured") from error


def _add_capture(*, capture: _BundleCapture, item: _CapturedFile) -> None:
    folded = item.relative_path.casefold()
    existing = next((name for name in capture.files if name.casefold() == folded), None)
    if existing is not None:
        raise ValueError(
            "Prepared bundle inventory contains case-colliding files: "
            f"{existing} and {item.relative_path}"
        )
    capture.files[item.relative_path] = item


def _capture_posix_file(
    *, path: Path, relative_path: str, expected: _FileIdentity
) -> _CapturedFile:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd: int | None = None
    try:
        fd = os.open(path, flags)
        before = _posix_identity(os.fstat(fd))
        if before != expected or before.links != 1:
            raise ValueError(
                f"Prepared bundle file changed during capture: {relative_path}"
            )
        content = _read_descriptor(fd=fd)
        after = _posix_identity(os.fstat(fd))
        if after != before:
            raise ValueError(
                f"Prepared bundle file changed during capture: {relative_path}"
            )
        path_identity = _posix_identity(os.lstat(path))
        if path_identity != before:
            raise ValueError(
                f"Prepared bundle file changed during capture: {relative_path}"
            )
        # This pathname reopen is identity-only.  The bytes consumed by the
        # verifier always come from the first descriptor.
        _reopen_posix_identity(path=path, expected=before, relative_path=relative_path)
        return _CapturedFile(
            relative_path=relative_path,
            physical_path=path,
            content=content,
            identity=before,
        )
    except OSError as error:
        raise ValueError(
            f"Prepared bundle file cannot be safely captured: {relative_path}"
        ) from error
    finally:
        if fd is not None:
            os.close(fd)


def _posix_identity(result: os.stat_result) -> _FileIdentity:
    return _FileIdentity(
        device=int(result.st_dev),
        inode=int(result.st_ino),
        size=int(result.st_size),
        modified_ns=int(result.st_mtime_ns),
        links=int(result.st_nlink),
    )


def _read_descriptor(*, fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _reopen_posix_identity(
    *, path: Path, expected: _FileIdentity, relative_path: str
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    fd: int | None = None
    try:
        fd = os.open(path, flags)
        reopened = _posix_identity(os.fstat(fd))
        if reopened != expected or _posix_identity(os.lstat(path)) != expected:
            raise ValueError(
                f"Prepared bundle file changed during capture: {relative_path}"
            )
    except OSError as error:
        raise ValueError(
            f"Prepared bundle file cannot be reopened safely: {relative_path}"
        ) from error
    finally:
        if fd is not None:
            os.close(fd)


def _ensure_path_stays_in_bundle(*, bundle_dir: Path, relative_path: str) -> None:
    """Reject a path whose resolution leaves the bundle root.

    Raises:
        ValueError:
            If the resolved path is outside the bundle root.
    """
    try:
        root = bundle_dir.resolve(strict=True)
        candidate = (bundle_dir / Path(*relative_path.split("/"))).resolve(strict=False)
        candidate.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(
            f"Prepared bundle file path escapes its bundle root: {relative_path}"
        ) from error


def _posix_check_directory(*, path: Path, expected: _FileIdentity) -> None:
    current = _posix_lstat_directory(path=path)
    if current != expected:
        raise ValueError(f"Prepared bundle directory changed during capture: {path}")


def _posix_lstat_directory(*, path: Path) -> _FileIdentity:
    try:
        result = os.lstat(path)
    except OSError as error:
        raise ValueError(f"Prepared bundle directory cannot be read: {path}") from error
    if stat.S_ISLNK(result.st_mode) or not stat.S_ISDIR(result.st_mode):
        raise ValueError(f"Prepared bundle root is not a real directory: {path}")
    return _posix_identity(result)


def _capture_windows_inventory(*, bundle_dir: Path) -> _BundleCapture:
    capture = _BundleCapture(files={}, directories={}, windows_handles=[])
    state = _WindowsCaptureState(capture=capture)
    try:
        root = Path(bundle_dir)
        root_info = state.open_directory(path=root)
        _require_windows_directory(info=root_info, path=root)
        pending: list[tuple[Path, str, _WindowsInformation]] = [(root, "", root_info)]
        while pending:
            directory, prefix, expected = pending.pop()
            current = state.inspect_directory(path=directory)
            if current != expected:
                raise ValueError(
                    f"Prepared bundle directory changed during capture: {directory}"
                )
            directory_key = _canonical_relative_key(prefix) if prefix else ""
            capture.directories[directory_key] = expected_to_file_identity(expected)
            try:
                entries = sorted(os.scandir(directory), key=lambda item: item.name)
            except OSError as error:
                raise ValueError(
                    f"Prepared bundle directory cannot be read: {directory}"
                ) from error
            for entry in entries:
                relative_path = f"{prefix}/{entry.name}" if prefix else entry.name
                hint_directory = entry.is_dir(follow_symlinks=False)
                info = state.inspect_path(
                    path=Path(entry.path), directory=hint_directory
                )
                _require_windows_entry(info=info, relative_path=relative_path)
                if info.attributes & _WINDOWS_FILE_ATTRIBUTE_DIRECTORY:
                    state.retain_directory(path=Path(entry.path), info=info)
                    pending.append((Path(entry.path), relative_path, info))
                    continue
                canonical = _canonical_relative_key(relative_path)
                _ensure_path_stays_in_bundle(bundle_dir=root, relative_path=canonical)
                captured = state.capture_file(
                    path=Path(entry.path), relative_path=canonical, expected=info
                )
                _add_capture(capture=capture, item=captured)
            current = state.inspect_directory(path=directory)
            if current != expected:
                raise ValueError(
                    f"Prepared bundle directory changed during capture: {directory}"
                )
        return capture
    except OSError as error:
        state.close()
        raise ValueError("Prepared bundle entry cannot be safely captured") from error
    except Exception:
        state.close()
        raise


def _windows_close_inventory_handle(handle: int) -> None:
    _windows_kernel32().CloseHandle(handle)


def _windows_kernel32() -> ctypes.CDLL:
    loader = getattr(ctypes, "WinDLL")
    kernel32 = loader("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.GetFileInformationByHandle.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_WindowsHandleInformation),
    ]
    kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
    kernel32.ReadFile.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    ]
    kernel32.ReadFile.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


class _WindowsHandleInformation(ctypes.Structure):
    """Layout of the Windows ``BY_HANDLE_FILE_INFORMATION`` structure."""

    _fields_ = [
        ("file_attributes", wintypes.DWORD),
        ("creation_time", wintypes.FILETIME),
        ("last_access_time", wintypes.FILETIME),
        ("last_write_time", wintypes.FILETIME),
        ("volume_serial_number", wintypes.DWORD),
        ("file_size_high", wintypes.DWORD),
        ("file_size_low", wintypes.DWORD),
        ("number_of_links", wintypes.DWORD),
        ("file_index_high", wintypes.DWORD),
        ("file_index_low", wintypes.DWORD),
    ]


@dataclass(frozen=True)
class _WindowsInformation:
    """Identity and safety properties captured from a Windows handle."""

    attributes: int
    links: int
    volume: int
    file_id: int
    size: int
    modified: tuple[int, int]


class _WindowsCaptureState:
    """Own parent directory handles until the complete capture has finished."""

    def __init__(self, *, capture: _BundleCapture) -> None:
        self.capture = capture
        self._directories: dict[str, int] = {}

    def capture_file(
        self, *, path: Path, relative_path: str, expected: _WindowsInformation
    ) -> _CapturedFile:
        handle = _windows_open_capture_handle(path=path, directory=False)
        try:
            before = _windows_information(handle=handle)
            _require_windows_entry(info=before, relative_path=relative_path)
            if before != expected:
                raise ValueError(
                    f"Prepared bundle file changed during capture: {relative_path}"
                )
            content = _windows_read(handle=handle, path=path)
            after = _windows_information(handle=handle)
            if after != before:
                raise ValueError(
                    f"Prepared bundle file changed during capture: {relative_path}"
                )
            reopened = _windows_open_capture_handle(path=path, directory=False)
            try:
                if _windows_information(handle=reopened) != before:
                    raise ValueError(
                        f"Prepared bundle file changed during capture: {relative_path}"
                    )
            finally:
                _windows_close_inventory_handle(reopened)
            return _CapturedFile(
                relative_path=relative_path,
                physical_path=path,
                content=content,
                identity=expected_to_file_identity(before),
            )
        finally:
            _windows_close_inventory_handle(handle)

    def close(self) -> None:
        while self.capture.windows_handles:
            _windows_close_inventory_handle(self.capture.windows_handles.pop())

    def inspect_directory(self, *, path: Path) -> _WindowsInformation:
        key = os.path.normcase(str(path))
        handle = self._directories.get(key)
        if handle is None:
            return self.inspect_path(path=path, directory=True)
        return _windows_information(handle=handle)

    def inspect_path(self, *, path: Path, directory: bool) -> _WindowsInformation:
        handle = _windows_open_capture_handle(path=path, directory=directory)
        try:
            return _windows_information(handle=handle)
        finally:
            _windows_close_inventory_handle(handle)

    def open_directory(self, *, path: Path) -> _WindowsInformation:
        key = os.path.normcase(str(path))
        handle = _windows_open_capture_handle(path=path, directory=True)
        self._directories[key] = handle
        self.capture.windows_handles.append(handle)
        return _windows_information(handle=handle)

    def retain_directory(self, *, path: Path, info: _WindowsInformation) -> None:
        key = os.path.normcase(str(path))
        if key not in self._directories:
            handle = _windows_open_capture_handle(path=path, directory=True)
            try:
                actual = _windows_information(handle=handle)
                if actual != info:
                    raise ValueError(
                        f"Prepared bundle directory changed during capture: {path}"
                    )
            except Exception:
                _windows_close_inventory_handle(handle)
                raise
            self._directories[key] = handle
            self.capture.windows_handles.append(handle)


def _require_windows_entry(*, info: _WindowsInformation, relative_path: str) -> None:
    if info.attributes & _WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError(
            "Prepared bundle inventory contains a symlink or reparse point: "
            f"{relative_path}"
        )
    if info.attributes & _WINDOWS_FILE_ATTRIBUTE_DIRECTORY:
        return
    if info.attributes & _WINDOWS_FILE_ATTRIBUTE_DEVICE:
        raise ValueError(
            f"Prepared bundle inventory contains a non-regular file: {relative_path}"
        )
    if info.links != 1:
        raise ValueError(
            f"Prepared bundle inventory contains a hard link: {relative_path}"
        )


def _windows_information(*, handle: int) -> _WindowsInformation:
    information = _WindowsHandleInformation()
    kernel32 = _windows_kernel32()
    if not kernel32.GetFileInformationByHandle(handle, ctypes.byref(information)):
        error = _windows_last_error()
        raise OSError(error, "GetFileInformationByHandle failed")
    file_size = (int(information.file_size_high) << 32) | int(information.file_size_low)
    file_id = (int(information.file_index_high) << 32) | int(information.file_index_low)
    return _WindowsInformation(
        attributes=int(information.file_attributes),
        links=int(information.number_of_links),
        volume=int(information.volume_serial_number),
        file_id=file_id,
        size=file_size,
        modified=(
            int(information.last_write_time.dwHighDateTime),
            int(information.last_write_time.dwLowDateTime),
        ),
    )


def _windows_last_error() -> int:
    getter = t.cast(c.Callable[[], int], getattr(ctypes, "get_last_error"))
    return getter()


def _windows_open_capture_handle(*, path: Path, directory: bool) -> int:
    kernel32 = _windows_kernel32()
    access = _WINDOWS_FILE_READ_ATTRIBUTES | (0x80000000 if not directory else 0)
    flags = _WINDOWS_FILE_FLAG_OPEN_REPARSE_POINT
    if directory:
        flags |= _WINDOWS_FILE_FLAG_BACKUP_SEMANTICS
    raw_handle = kernel32.CreateFileW(
        str(path),
        access,
        _WINDOWS_FILE_SHARE_READ,
        None,
        _WINDOWS_OPEN_EXISTING,
        flags,
        None,
    )
    handle = _windows_handle_value(raw_handle)
    invalid_handle = ctypes.c_void_p(-1).value
    if handle in (-1, invalid_handle):
        error = _windows_last_error()
        raise OSError(error, f"CreateFileW failed: {path}")
    return handle


def _windows_handle_value(handle: object) -> int:
    if isinstance(handle, int):
        return handle
    value = getattr(handle, "value", None)
    if isinstance(value, int):
        return value
    raise OSError("CreateFileW returned an invalid handle")


def _windows_read(*, handle: int, path: Path) -> bytes:
    chunks: list[bytes] = []
    while True:
        buffer = ctypes.create_string_buffer(1024 * 1024)
        count = wintypes.DWORD()
        if not _windows_kernel32().ReadFile(
            handle, buffer, len(buffer), ctypes.byref(count), None
        ):
            error = _windows_last_error()
            raise OSError(error, f"ReadFile failed: {path}")
        if not count.value:
            return b"".join(chunks)
        chunks.append(buffer.raw[: count.value])


def expected_to_file_identity(info: _WindowsInformation) -> _FileIdentity:
    """Convert Windows identity data to the common capture identity.

    Returns:
        The common file identity.
    """
    return _FileIdentity(
        device=info.volume,
        inode=info.file_id,
        size=info.size,
        modified_ns=(info.modified[0] << 32) | info.modified[1],
        links=info.links,
    )


def _require_windows_directory(*, info: _WindowsInformation, path: Path) -> None:
    if info.attributes & _WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT or not (
        info.attributes & _WINDOWS_FILE_ATTRIBUTE_DIRECTORY
    ):
        raise ValueError(f"Prepared bundle root is not a real directory: {path}")


def _close_capture_handles(*, capture: _BundleCapture) -> None:
    """Close native handles retained by a Windows capture."""
    while capture.windows_handles:
        _windows_close_inventory_handle(capture.windows_handles.pop())


def verify_prepared_bundle(
    *,
    bundle_dir: Path,
    checksum_policy: ChecksumValidationPolicy = ChecksumValidationPolicy.STRICT,
) -> BundleManifest:
    """Verify a prepared bundle before sampling or validation.

    Args:
        bundle_dir:
            Prepared bundle directory.
        checksum_policy:
            Whether recorded file digests must match current bytes. Defaults to
            strict validation; relaxed callers retain inventory and semantic checks.

    Returns:
        The verified bundle manifest.

    """
    manifest, _ = _verify_prepared_bundle_capture(
        bundle_dir=bundle_dir, checksum_policy=checksum_policy
    )
    return manifest


def _verify_prepared_bundle_capture(
    *,
    bundle_dir: Path,
    checksum_policy: ChecksumValidationPolicy = ChecksumValidationPolicy.STRICT,
) -> tuple[BundleManifest, "_BundleCapture"]:
    """Capture and verify a bundle without reopening its files for consumption.

    Returns:
        The parsed manifest and the stable byte capture.

    """
    capture = _capture_inventory(bundle_dir=bundle_dir)
    try:
        manifest = _validate_capture(
            capture=capture, bundle_dir=bundle_dir, checksum_policy=checksum_policy
        )
    except Exception:
        _close_capture_handles(capture=capture)
        raise
    _close_capture_handles(capture=capture)
    return manifest, capture


def _validate_capture(
    *,
    capture: "_BundleCapture",
    bundle_dir: Path,
    checksum_policy: ChecksumValidationPolicy = ChecksumValidationPolicy.STRICT,
) -> BundleManifest:
    """Validate the bytes and identities in a complete bundle capture.

    Returns:
        The strictly parsed bundle manifest.

    Raises:
        ValueError:
            If a capture fails an integrity gate.
    """
    manifest_capture = capture.files.get(BUNDLE_MANIFEST)
    if manifest_capture is None:
        raise ValueError("Prepared bundle manifest is missing")
    try:
        manifest = BundleManifest.model_validate_json(
            manifest_capture.content, context={"checksum_policy": checksum_policy}
        )
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid prepared bundle manifest") from error
    if manifest.prepared_bundle_schema_version != PREPARED_BUNDLE_SCHEMA_VERSION:
        message = (
            "Prepared bundle uses unsupported schema version "
            f"{manifest.prepared_bundle_schema_version}"
        )
        raise ValueError(message)
    _verify_origin_contract_binding(manifest=manifest, checksum_policy=checksum_policy)
    canonical_files = _canonical_manifest_files(manifest.files, bundle_dir=bundle_dir)
    expected_inventory = {*canonical_files, BUNDLE_MANIFEST}
    actual_inventory = set(capture.files)
    missing_inventory = sorted(expected_inventory - actual_inventory)
    extra_inventory = sorted(actual_inventory - expected_inventory)
    _check_directory_inventory(capture=capture, canonical_files=canonical_files)
    if missing_inventory or extra_inventory:
        raise ValueError(
            "Prepared bundle file inventory does not match its manifest: "
            f"missing={missing_inventory}, extra={extra_inventory}"
        )
    required_files = {
        *(_canonical_relative_key(path) for path in REQUIRED_COLUMNS),
        _canonical_relative_key(SOURCE_REPORT),
    }
    missing_files = sorted(required_files - set(canonical_files))
    if missing_files:
        raise ValueError(f"Prepared bundle manifest is missing files: {missing_files}")
    for relative_path, checksum in canonical_files.items():
        item = capture.files[relative_path]
        if (
            checksum_policy.validates_checksums
            and _sha256_bytes(item.content) != checksum
        ):
            raise ValueError(f"Prepared bundle verification failed: {relative_path}")
    _verify_bundle_tables(
        capture=capture, manifest=manifest, checksum_policy=checksum_policy
    )
    try:
        report = json.loads(capture.files[SOURCE_REPORT].content)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
        raise ValueError("Prepared bundle source report is malformed") from error
    if not isinstance(report, dict) or report.get("passed") is not True:
        raise ValueError("Prepared bundle source preparation did not pass")
    _revalidate_captures(capture=capture)
    return manifest


def _canonical_manifest_files(
    files: dict[str, str], *, bundle_dir: Path | None = None
) -> dict[str, str]:
    """Canonicalise manifest keys while rejecting ambiguous paths.

    ``bundle-manifest.json`` is deliberately excluded from the file map.  It is
    the only file in a prepared bundle that is not covered by that map.

    Args:
        files:
            Manifest file checksums keyed by relative path.
        bundle_dir (optional):
            Bundle root used for lexical-resolution checks.

    Returns:
        Canonical POSIX paths mapped to their original checksums.

    Raises:
        ValueError:
            If a path is unsafe, duplicated, or case-colliding.
    """
    canonical_files: dict[str, str] = {}
    casefolded_files: dict[str, str] = {}
    for relative_path, checksum in files.items():
        canonical_path = _canonical_relative_key(relative_path)
        if canonical_path == BUNDLE_MANIFEST:
            raise ValueError(
                "Prepared bundle manifest must exclude bundle-manifest.json from "
                "its file map"
            )
        if bundle_dir is not None:
            _ensure_path_stays_in_bundle(
                bundle_dir=bundle_dir, relative_path=canonical_path
            )
        if canonical_path in canonical_files:
            raise ValueError(
                "Prepared bundle manifest contains duplicate file keys: "
                f"{canonical_path}"
            )
        casefolded_path = canonical_path.casefold()
        if casefolded_path in casefolded_files:
            raise ValueError(
                "Prepared bundle manifest contains case-colliding file keys: "
                f"{casefolded_files[casefolded_path]} and {canonical_path}"
            )
        canonical_files[canonical_path] = checksum
        casefolded_files[casefolded_path] = canonical_path
    return canonical_files


def _check_directory_inventory(
    *, capture: _BundleCapture, canonical_files: dict[str, str]
) -> None:
    """Reject missing or extra directories in the captured layout.

    Raises:
        ValueError:
            If the directory inventory differs from the manifest layout.
    """
    expected = {Path(".")}
    for relative_path in canonical_files:
        expected.update(Path(relative_path).parents)
    expected_directories = {
        path.as_posix() for path in expected if path != Path(".")
    } | {""}
    actual_directories = set(capture.directories)
    missing = sorted(expected_directories - actual_directories)
    extra = sorted(actual_directories - expected_directories)
    if missing or extra:
        raise ValueError(
            "Prepared bundle file inventory does not match its manifest: "
            f"missing={missing}, extra={extra}"
        )


def _revalidate_captures(*, capture: _BundleCapture) -> None:
    """Detect POSIX replacement after all byte-based checks have completed."""
    if os.name == "nt":
        return
    for relative_path, item in capture.files.items():
        _reopen_posix_identity(
            path=item.physical_path, expected=item.identity, relative_path=relative_path
        )


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _verify_bundle_tables(
    *,
    capture: _BundleCapture,
    manifest: BundleManifest,
    checksum_policy: ChecksumValidationPolicy = ChecksumValidationPolicy.STRICT,
) -> None:
    """Verify schemas and semantic origin labels in a captured bundle."""
    _verify_schemas(capture=capture)
    _verify_origin_table(
        capture=capture, manifest=manifest, checksum_policy=checksum_policy
    )


def _verify_origin_table(
    *,
    capture: _BundleCapture,
    manifest: BundleManifest,
    checksum_policy: ChecksumValidationPolicy = ChecksumValidationPolicy.STRICT,
) -> None:
    """Verify the bound Danish labels and complete origin partition.

    Raises:
        ValueError: If the origin table is incomplete or mismatched.
    """
    try:
        frame = pl.read_parquet(
            io.BytesIO(
                capture.files[
                    "normalized/folk2_origin_country_marginal.parquet"
                ].content
            )
        )
        contract = load_origin_label_contract(
            path=_bound_contract_path(manifest=manifest),
            checksum_policy=checksum_policy,
        )
    except (KeyError, OSError, ValueError, pl.exceptions.PolarsError) as error:
        raise ValueError("Prepared FOLK2 origin table cannot be validated") from error
    if (
        frame.height != len(contract.labels)
        or frame.null_count().sum_horizontal().item()
    ):
        raise ValueError("Prepared FOLK2 origin table is incomplete")
    codes = frame.get_column("origin_country_code").to_list()
    danish = dict(
        zip(codes, frame.get_column("origin_country_da").to_list(), strict=True)
    )
    english = dict(
        zip(codes, frame.get_column("origin_country").to_list(), strict=True)
    )
    if (
        set(codes) != set(contract.labels_en)
        or english != contract.labels_en
        or danish != contract.labels_da
    ):
        raise ValueError(
            "Prepared FOLK2 English/Danish labels do not match the contract"
        )


def _bound_contract_path(*, manifest: BundleManifest) -> Path:
    """Resolve and validate the manifest's repository-relative contract path.

    Returns:
        The contract path rooted at the repository working directory.

    Raises:
        ValueError: If the path contains an alias or escapes the repository.
    """
    relative = Path(manifest.origin_labels_contract_path)
    if (
        relative.is_absolute()
        or relative.as_posix() != manifest.origin_labels_contract_path
        or ".." in relative.parts
        or "\\" in manifest.origin_labels_contract_path
    ):
        raise ValueError("Prepared bundle origin-label contract path is not canonical")
    if relative != Path("config/folk2-ieland-labels-da.yaml"):
        raise ValueError("Prepared bundle origin-label contract path is not canonical")
    return Path.cwd() / relative


def _verify_schemas(*, capture: _BundleCapture) -> None:
    expected_schema = dict(REQUIRED_COLUMNS)
    for relative_path, expected_columns in expected_schema.items():
        try:
            columns = frozenset(
                pl.read_parquet(
                    io.BytesIO(capture.files[relative_path].content)
                ).columns
            )
        except (KeyError, OSError, pl.exceptions.PolarsError) as error:
            raise ValueError(
                f"Prepared bundle schema cannot be read: {relative_path}"
            ) from error
        if columns != expected_columns:
            missing = sorted(expected_columns - columns)
            unexpected = sorted(columns - expected_columns)
            message = (
                f"Prepared bundle schema mismatch for {relative_path}; "
                f"missing={missing}, unexpected={unexpected}"
            )
            raise ValueError(message)


def _verify_origin_contract_binding(
    *,
    manifest: BundleManifest,
    checksum_policy: ChecksumValidationPolicy = ChecksumValidationPolicy.STRICT,
) -> None:
    """Verify the repository contract bound into a schema-6 bundle.

    Raises:
        ValueError: If contract identity, bytes, or source binding changed.
    """
    fields = (
        manifest.origin_labels_contract_path,
        manifest.origin_labels_contract_sha256,
        manifest.origin_labels_contract_content,
    )
    if not all(fields) or manifest.origin_labels_contract_version < 1:
        raise ValueError("Prepared bundle origin-label contract binding is incomplete")
    contract_path = _bound_contract_path(manifest=manifest)
    try:
        contract_bytes = contract_path.read_bytes()
    except OSError as error:
        raise ValueError("Bound origin-label contract is unavailable") from error
    contract_checksum = _sha256_bytes(contract_bytes)
    if (
        checksum_policy.validates_checksums
        and contract_checksum != manifest.origin_labels_contract_sha256
    ):
        raise ValueError("Bound origin-label contract checksum changed")
    if (
        checksum_policy.validates_checksums
        and contract_checksum != ORIGIN_LABEL_CONTRACT_SHA256
    ):
        raise ValueError("Bound origin-label contract is not the reviewed contract")
    if (
        checksum_policy.validates_checksums
        and manifest.origin_labels_contract_content.encode("utf-8") != contract_bytes
    ):
        raise ValueError("Bound origin-label contract content changed")
    try:
        contract = load_origin_label_contract(
            path=contract_path, checksum_policy=checksum_policy
        )
    except (OSError, ValueError) as error:
        raise ValueError("Bound origin-label contract is invalid") from error
    if contract.version != manifest.origin_labels_contract_version:
        raise ValueError("Bound origin-label contract version changed")
    _verify_origin_snapshot_binding(
        manifest=manifest,
        source_metadata_en_sha256=contract.source_metadata_en_sha256,
        source_metadata_da_sha256=contract.source_metadata_da_sha256,
        checksum_policy=checksum_policy,
    )


def _verify_origin_snapshot_binding(
    *,
    manifest: BundleManifest,
    source_metadata_en_sha256: str,
    source_metadata_da_sha256: str,
    checksum_policy: ChecksumValidationPolicy = ChecksumValidationPolicy.STRICT,
) -> None:
    """Verify the FOLK2 snapshot's Danish metadata checksum.

    Raises:
        ValueError: If the FOLK2 snapshot is missing or mismatched.
    """
    folk2 = [
        snapshot
        for snapshot in manifest.source_snapshots
        if snapshot.table_id == "FOLK2"
    ]
    if len(folk2) != 1:
        raise ValueError("Prepared bundle must contain one FOLK2 source snapshot")
    if (
        checksum_policy.validates_checksums
        and folk2[0].metadata_sha256 != source_metadata_en_sha256
    ):
        raise ValueError("FOLK2 English metadata is not bound to the contract")
    if (
        checksum_policy.validates_checksums
        and folk2[0].metadata_da_sha256 != source_metadata_da_sha256
    ):
        raise ValueError("FOLK2 Danish metadata is not bound to the contract")
