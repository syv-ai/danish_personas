"""Integrity and schema verification for prepared source bundles."""

import collections.abc as c
import ctypes
import ctypes.wintypes as wintypes
import json
import ntpath
import os
import stat
import typing as t
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ..io import verify_checksums
from ..models import PREPARED_BUNDLE_SCHEMA_VERSION, BundleManifest

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
        {"origin_country_code", "origin_country", "count"}
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


def verify_prepared_bundle(*, bundle_dir: Path) -> BundleManifest:
    """Verify a prepared bundle before sampling or validation.

    Args:
        bundle_dir:
            Prepared bundle directory.

    Returns:
        The verified bundle manifest.

    Raises:
        ValueError:
            If the bundle is legacy, incomplete, malformed, tampered, or did not pass
            source preparation.
    """
    inventory = _regular_file_inventory(bundle_dir=bundle_dir)
    manifest_path = bundle_dir / BUNDLE_MANIFEST
    try:
        manifest = BundleManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Invalid prepared bundle manifest: {manifest_path}"
        ) from error
    if manifest.prepared_bundle_schema_version != PREPARED_BUNDLE_SCHEMA_VERSION:
        message = (
            "Prepared bundle uses unsupported schema version "
            f"{manifest.prepared_bundle_schema_version}"
        )
        raise ValueError(message)
    canonical_files = _canonical_manifest_files(manifest.files, bundle_dir=bundle_dir)
    expected_inventory = {*canonical_files, BUNDLE_MANIFEST}
    actual_inventory = set(inventory)
    missing_inventory = sorted(expected_inventory - actual_inventory)
    extra_inventory = sorted(actual_inventory - expected_inventory)
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
    verify_checksums(
        base_dir=bundle_dir,
        expected=canonical_files,
        message="Prepared bundle verification failed",
    )
    _verify_schemas(bundle_dir=bundle_dir)
    try:
        report = json.loads((bundle_dir / SOURCE_REPORT).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Prepared bundle source report is malformed") from error
    if not isinstance(report, dict) or report.get("passed") is not True:
        raise ValueError("Prepared bundle source preparation did not pass")
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


def _regular_file_inventory(*, bundle_dir: Path) -> dict[str, Path]:
    """Return the exact, non-linked regular-file inventory of a bundle.

    Directory traversal uses no-follow semantics throughout.  POSIX uses
    ``lstat``; Windows uses native handles.  The manifest is included in this
    inventory for comparison, but is handled as the one explicit exception to
    the manifest file map by the caller.

    Raises:
        ValueError:
            If the root or an entry is unsafe or cannot be inspected.
    """
    if os.name == "nt":
        _require_windows_inventory_directory(path=bundle_dir)
    else:
        try:
            root_stat = os.lstat(bundle_dir)
        except OSError as error:
            message = f"Prepared bundle directory cannot be read: {bundle_dir}"
            raise ValueError(message) from error
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            message = f"Prepared bundle root is not a real directory: {bundle_dir}"
            raise ValueError(message)

    inventory: dict[str, Path] = {}
    casefolded_paths: dict[str, str] = {}
    pending: list[tuple[Path, str]] = [(bundle_dir, "")]
    while pending:
        directory, prefix = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise ValueError(
                f"Prepared bundle directory cannot be read: {directory}"
            ) from error
        for entry in entries:
            relative_path = f"{prefix}/{entry.name}" if prefix else entry.name
            inspected = _inspect_inventory_entry(
                entry=entry, relative_path=relative_path, bundle_dir=bundle_dir
            )
            if inspected is None:
                pending.append((Path(entry.path), relative_path))
                continue
            canonical_path, physical_path = inspected
            casefolded_path = canonical_path.casefold()
            if casefolded_path in casefolded_paths:
                raise ValueError(
                    "Prepared bundle inventory contains case-colliding files: "
                    f"{casefolded_paths[casefolded_path]} and {canonical_path}"
                )
            inventory[canonical_path] = physical_path
            casefolded_paths[casefolded_path] = canonical_path
    return inventory


def _inspect_inventory_entry(
    *, entry: os.DirEntry[str], relative_path: str, bundle_dir: Path
) -> tuple[str, Path] | None:
    """Inspect one directory entry without following links.

    Args:
        entry:
            Directory entry to inspect.
        relative_path:
            Path relative to the bundle root.
        bundle_dir:
            Bundle root used for lexical-resolution checks.

    Returns:
        ``None`` for a directory, otherwise its canonical key and physical path.

    Raises:
        ValueError:
            If the entry is a link, special file, or hard link.
    """
    if os.name == "nt":
        is_directory = _inspect_windows_inventory_entry(
            entry=entry, path=Path(entry.path), relative_path=relative_path
        )
        if is_directory:
            return None
    else:
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
        if stat.S_ISDIR(entry_stat.st_mode):
            return None
        if not stat.S_ISREG(entry_stat.st_mode):
            raise ValueError(
                f"Prepared bundle inventory contains a non-regular file: "
                f"{relative_path}"
            )
        if entry_stat.st_nlink != 1:
            raise ValueError(
                f"Prepared bundle inventory contains a hard link: {relative_path}"
            )
    canonical_path = _canonical_relative_key(relative_path)
    _ensure_path_stays_in_bundle(bundle_dir=bundle_dir, relative_path=canonical_path)
    return canonical_path, Path(entry.path)


def _inspect_windows_inventory_entry(
    *, entry: os.DirEntry[str], path: Path, relative_path: str
) -> bool:
    """Inspect a bundle entry through a no-follow Windows handle.

    Returns:
        Whether the entry is a directory.

    Raises:
        ValueError:
            If the entry cannot be inspected or is unsafe for a bundle.
    """
    try:
        directory_hint = entry.is_dir(follow_symlinks=False) or entry.is_symlink()
        information = _inspect_windows_path(path=path, directory=directory_hint)
    except OSError as error:
        raise ValueError(
            f"Prepared bundle entry cannot be inspected: {relative_path}"
        ) from error
    attributes = information.attributes
    if attributes & _WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError(
            "Prepared bundle inventory contains a symlink or reparse point: "
            f"{relative_path}"
        )
    if attributes & _WINDOWS_FILE_ATTRIBUTE_DIRECTORY:
        return True
    if attributes & _WINDOWS_FILE_ATTRIBUTE_DEVICE:
        raise ValueError(
            f"Prepared bundle inventory contains a non-regular file: {relative_path}"
        )
    if information.number_of_links != 1:
        raise ValueError(
            f"Prepared bundle inventory contains a hard link: {relative_path}"
        )
    return False


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
class _WindowsInventoryInformation:
    """File information captured from a no-follow Windows handle."""

    attributes: int
    number_of_links: int


def _inspect_windows_path(
    *, path: Path, directory: bool
) -> _WindowsInventoryInformation:
    """Read Windows file information while denying replacement sharing.

    Returns:
        The attributes and hard-link count captured from the open handle.

    Raises:
        OSError:
            If the path cannot be opened or queried.
    """
    handle: int | None = None
    try:
        handle = _windows_open_inventory_handle(path=path, directory=directory)
        kernel32 = _windows_kernel32()
        information = _WindowsHandleInformation()
        if not kernel32.GetFileInformationByHandle(handle, ctypes.byref(information)):
            error = _windows_last_error()
            raise OSError(error, f"GetFileInformationByHandle failed: {path}")
        return _WindowsInventoryInformation(
            attributes=int(information.file_attributes),
            number_of_links=int(information.number_of_links),
        )
    finally:
        if handle is not None:
            _windows_close_inventory_handle(handle)


def _windows_close_inventory_handle(handle: int) -> None:
    """Close a native inventory handle, preserving the original failure."""
    _windows_kernel32().CloseHandle(handle)


def _windows_kernel32() -> ctypes.CDLL:
    """Return the configured Windows kernel32 API handle."""
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
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


def _windows_last_error() -> int:
    """Return the last Windows API error code."""
    getter = t.cast(c.Callable[[], int], getattr(ctypes, "get_last_error"))
    return getter()


def _windows_open_inventory_handle(*, path: Path, directory: bool) -> int:
    """Open a path without following its final reparse point.

    Returns:
        The native handle.

    Raises:
        OSError:
            If the path cannot be opened.
    """
    kernel32 = _windows_kernel32()
    access = _WINDOWS_FILE_READ_ATTRIBUTES
    if not directory:
        access |= _WINDOWS_GENERIC_READ
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
    """Convert a ctypes Windows handle to an integer.

    Returns:
        The native handle value.

    Raises:
        OSError:
            If the API returned an invalid handle.
    """
    if isinstance(handle, int):
        return handle
    value = getattr(handle, "value", None)
    if isinstance(value, int):
        return value
    raise OSError("CreateFileW returned an invalid handle")


def _require_windows_inventory_directory(*, path: Path) -> None:
    """Require a real directory using a no-follow native Windows handle.

    Raises:
        ValueError:
            If the path is missing, a reparse point, or not a directory.
    """
    try:
        information = _inspect_windows_path(path=path, directory=True)
    except OSError as error:
        message = f"Prepared bundle directory cannot be read: {path}"
        raise ValueError(message) from error
    if (
        information.attributes & _WINDOWS_FILE_ATTRIBUTE_REPARSE_POINT
        or not information.attributes & _WINDOWS_FILE_ATTRIBUTE_DIRECTORY
    ):
        raise ValueError(f"Prepared bundle root is not a real directory: {path}")


def _verify_schemas(*, bundle_dir: Path) -> None:
    for relative_path, expected_columns in REQUIRED_COLUMNS.items():
        try:
            columns = frozenset(
                pl.scan_parquet(bundle_dir / relative_path).collect_schema().names()
            )
        except (OSError, pl.exceptions.PolarsError) as error:
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
