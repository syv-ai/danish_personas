"""Integrity and schema verification for prepared source bundles."""

import json
import ntpath
import os
import stat
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

    Directory traversal uses ``lstat`` semantics throughout.  The manifest is
    included in this inventory for comparison, but is handled as the one
    explicit exception to the manifest file map by the caller.

    Raises:
        ValueError:
            If the root or an entry is unsafe or cannot be inspected.
    """
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
            f"Prepared bundle inventory contains a non-regular file: {relative_path}"
        )
    if entry_stat.st_nlink != 1:
        raise ValueError(
            f"Prepared bundle inventory contains a hard link: {relative_path}"
        )
    canonical_path = _canonical_relative_key(relative_path)
    _ensure_path_stays_in_bundle(bundle_dir=bundle_dir, relative_path=canonical_path)
    return canonical_path, Path(entry.path)


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
