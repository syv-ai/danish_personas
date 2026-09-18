"""Regression tests for source-validation provenance gates."""

import os
from pathlib import Path, PureWindowsPath

import pytest

from danish_personas.io import sha256_file, sha256_text, write_json
from danish_personas.models import BundleManifest, SnapshotManifest
from danish_personas.sampling.generator import generate_records
from danish_personas.sources.bundle import (
    _regular_file_inventory,
    verify_prepared_bundle,
)
from danish_personas.sources.prepare import _verify_existing_bundle, verify_raw_snapshot
from danish_personas.validation.checks import validate_demographics, validate_sources
from tests.test_non_llm_pipeline import _write_bundle


def test_missing_prepared_bundle_schema_version_is_rejected_everywhere(
    tmp_path: Path,
) -> None:
    """Consumers reject a valid-looking bundle when its schema is omitted."""
    bundle_dir, sampling_path, validation_path, categories_path = _write_bundle(
        root=tmp_path
    )
    run_dir = generate_records(
        bundle_dir=bundle_dir,
        sampling_config_path=sampling_path,
        output_dir=tmp_path / "runs",
        rows=20,
        seed=42,
    )
    manifest_path = bundle_dir / "bundle-manifest.json"
    manifest = BundleManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    payload = manifest.model_dump(mode="json")
    del payload["prepared_bundle_schema_version"]
    write_json(path=manifest_path, payload=payload)

    verification_paths = (
        (
            "strict manifest model",
            lambda: BundleManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            ),
            "Field required",
        ),
        (
            "prepared bundle verifier",
            lambda: verify_prepared_bundle(bundle_dir=bundle_dir),
            "Invalid prepared bundle manifest",
        ),
        (
            "source preparation reuse",
            lambda: _verify_existing_bundle(
                bundle_dir=bundle_dir, manifest_path=manifest_path
            ),
            "Invalid prepared bundle manifest",
        ),
        (
            "source validation",
            lambda: validate_sources(bundle_dir=bundle_dir),
            "Invalid prepared bundle manifest",
        ),
        (
            "demographic generation",
            lambda: generate_records(
                bundle_dir=bundle_dir,
                sampling_config_path=sampling_path,
                output_dir=tmp_path / "other-runs",
                rows=20,
                seed=42,
            ),
            "Invalid prepared bundle manifest",
        ),
        (
            "demographic validation",
            lambda: validate_demographics(
                run_dir=run_dir,
                bundle_dir=bundle_dir,
                validation_config_path=validation_path,
                categories_path=categories_path,
            ),
            "Invalid prepared bundle manifest",
        ),
    )

    for _, verify, message in verification_paths:
        with pytest.raises(ValueError, match=message):
            verify()


def test_nested_pass_cannot_override_failed_source_report(tmp_path: Path) -> None:
    """Source validation requires the top-level report result to pass."""
    bundle_dir, _, _, _ = _write_bundle(root=tmp_path)
    source_report = bundle_dir / "source-preparation-report.json"
    write_json(
        path=source_report,
        payload={"passed": False, "tables": {"example": {"passed": True}}},
    )
    manifest_path = bundle_dir / "bundle-manifest.json"
    manifest = BundleManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    files = dict(manifest.files)
    files[source_report.name] = sha256_file(source_report)
    write_json(path=manifest_path, payload=manifest.model_copy(update={"files": files}))

    with pytest.raises(ValueError, match="source preparation did not pass"):
        validate_sources(bundle_dir=bundle_dir)


def test_prepared_bundle_inventory_accepts_normal_files(tmp_path: Path) -> None:
    """The inventory contract accepts ordinary files on every platform."""
    bundle_dir, _, _, _ = _write_bundle(root=tmp_path)

    inventory = _regular_file_inventory(bundle_dir=bundle_dir)

    assert inventory["source-preparation-report.json"] == (
        bundle_dir / "source-preparation-report.json"
    )


def test_prepared_bundle_manifest_excludes_itself(tmp_path: Path) -> None:
    """The manifest is the sole regular file excluded from its own file map."""
    bundle_dir, _, _, _ = _write_bundle(root=tmp_path)
    manifest_path = bundle_dir / "bundle-manifest.json"
    manifest = BundleManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    files = dict(manifest.files)
    files["bundle-manifest.json"] = sha256_file(manifest_path)
    write_json(path=manifest_path, payload=manifest.model_copy(update={"files": files}))

    with pytest.raises(ValueError, match="exclude"):
        verify_prepared_bundle(bundle_dir=bundle_dir)


def test_prepared_bundle_verifier_accepts_windows_manifest_keys(tmp_path: Path) -> None:
    """Windows relative keys are canonicalised before bundle verification."""
    bundle_dir, _, _, _ = _write_bundle(root=tmp_path)
    manifest_path = bundle_dir / "bundle-manifest.json"
    manifest = BundleManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    windows_files = {
        "\\".join(PureWindowsPath(relative_path).parts): checksum
        for relative_path, checksum in manifest.files.items()
    }
    write_json(
        path=manifest_path, payload=manifest.model_copy(update={"files": windows_files})
    )

    verified = verify_prepared_bundle(bundle_dir=bundle_dir)

    assert verified.files == windows_files


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "case", "mixed"])
def test_prepared_bundle_verifier_rejects_inventory_attacks(
    tmp_path: Path, kind: str
) -> None:
    """Symlinks, hard links, case collisions, and separator aliases fail."""
    bundle_dir, _, _, _ = _write_bundle(root=tmp_path)
    target = bundle_dir / "source-preparation-report.json"
    if kind == "symlink":
        target.unlink()
        try:
            target.symlink_to(tmp_path / "outside")
        except OSError:
            pytest.skip("symbolic links are unavailable")
    elif kind == "hardlink":
        replacement = bundle_dir / "hardlink-source.txt"
        replacement.write_text("hard link", encoding="utf-8")
        target.unlink()
        try:
            os.link(replacement, target)
        except OSError:
            pytest.skip("hard links are unavailable")
    elif kind == "case":
        case_path = bundle_dir / "SOURCE-PREPARATION-REPORT.JSON"
        case_path.write_text("collision", encoding="utf-8")
        if os.path.samefile(case_path, target):
            pytest.skip("case-insensitive filesystem cannot contain this collision")
    else:
        if os.name == "nt":
            pytest.skip("mixed POSIX and Windows separators need POSIX filenames")
        (bundle_dir / "normalized").joinpath("alias\\name").write_text(
            "alias", encoding="utf-8"
        )
        (bundle_dir / "normalized" / "alias").mkdir()
        (bundle_dir / "normalized" / "alias" / "name").write_text(
            "alias", encoding="utf-8"
        )

    with pytest.raises(
        ValueError, match="symlink|reparse point|hard link|case-colliding"
    ):
        verify_prepared_bundle(bundle_dir=bundle_dir)


def test_prepared_bundle_verifier_rejects_inventory_extras_and_missing_files(
    tmp_path: Path,
) -> None:
    """The manifest and regular-file inventory must match exactly."""
    bundle_dir, _, _, _ = _write_bundle(root=tmp_path)
    extra = bundle_dir / "extra.txt"
    extra.write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="extra"):
        verify_prepared_bundle(bundle_dir=bundle_dir)

    extra.unlink()
    missing = bundle_dir / "source-preparation-report.json"
    missing.unlink()
    with pytest.raises(ValueError, match="missing"):
        verify_prepared_bundle(bundle_dir=bundle_dir)


@pytest.mark.parametrize("alias_kind", ["separator", "case"])
def test_prepared_bundle_verifier_rejects_manifest_aliases(
    tmp_path: Path, alias_kind: str
) -> None:
    """Canonical duplicate and case-folded manifest keys are rejected."""
    bundle_dir, _, _, _ = _write_bundle(root=tmp_path)
    manifest_path = bundle_dir / "bundle-manifest.json"
    manifest = BundleManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    files = dict(manifest.files)
    path, checksum = next(iter(files.items()))
    alias = path.replace("/", "\\") if alias_kind == "separator" else path.swapcase()
    files[alias] = checksum
    write_json(path=manifest_path, payload=manifest.model_copy(update={"files": files}))

    with pytest.raises(ValueError, match="duplicate|colliding"):
        verify_prepared_bundle(bundle_dir=bundle_dir)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO is unsupported")
def test_prepared_bundle_verifier_rejects_nonregular_inventory_entry(
    tmp_path: Path,
) -> None:
    """Special files are not valid prepared bundle contents."""
    bundle_dir, _, _, _ = _write_bundle(root=tmp_path)
    os.mkfifo(bundle_dir / "special.fifo")

    with pytest.raises(ValueError, match="non-regular"):
        verify_prepared_bundle(bundle_dir=bundle_dir)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "",
        ".",
        "normalized/.",
        "normalized//file.parquet",
        "normalized/../file.parquet",
        "../escape.txt",
        "/absolute.txt",
        "\\rooted.txt",
        "C:relative.txt",
        "C:/absolute.txt",
        "\\\\server\\share\\file.txt",
        "\\\\?\\C:\\device.txt",
        "normal\\..\\escape.txt",
    ],
)
def test_prepared_bundle_verifier_rejects_unsafe_manifest_paths(
    tmp_path: Path, unsafe_path: str
) -> None:
    """Manifest keys cannot use ambiguous or escaping path syntax."""
    bundle_dir, _, _, _ = _write_bundle(root=tmp_path)
    manifest_path = bundle_dir / "bundle-manifest.json"
    manifest = BundleManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    files = dict(manifest.files)
    _, checksum = files.popitem()
    files[unsafe_path] = checksum
    write_json(path=manifest_path, payload=manifest.model_copy(update={"files": files}))

    with pytest.raises(ValueError, match="path|paths"):
        verify_prepared_bundle(bundle_dir=bundle_dir)


def test_raw_query_must_match_source_lock(tmp_path: Path) -> None:
    """A self-consistent manifest cannot bless a query changed after locking."""
    contents = {
        "metadata-en.json": "{}\n",
        "metadata-da.json": "{}\n",
        "query.json": "{}\n",
        "data.csv": "value\n1\n",
        "response-headers.json": "{}\n",
    }
    for name, content in contents.items():
        (tmp_path / name).write_text(content, encoding="utf-8", newline="\n")
    snapshot = SnapshotManifest(
        table_id="TEST",
        role="test_role",
        period="2024",
        metadata_sha256=sha256_file(tmp_path / "metadata-en.json"),
        metadata_da_sha256=sha256_file(tmp_path / "metadata-da.json"),
        query_sha256=sha256_file(tmp_path / "query.json"),
        data_sha256=sha256_file(tmp_path / "data.csv"),
        response_headers_sha256=sha256_file(tmp_path / "response-headers.json"),
        retrieved_at="2026-09-14T00:00:00+00:00",
        data_bytes=(tmp_path / "data.csv").stat().st_size,
    )
    with pytest.raises(ValueError, match="does not match source lock"):
        verify_raw_snapshot(
            snapshot_dir=tmp_path,
            snapshot=snapshot,
            table_id="TEST",
            role="test_role",
            period="2024",
            expected_query='{"table":"TEST"}\n',
        )
    assert snapshot.query_sha256 == sha256_text(contents["query.json"])


def test_recomputed_source_failure_cannot_be_masked_by_bound_report(
    tmp_path: Path,
) -> None:
    """A changed semantic result cannot overwrite an earlier passing report."""
    bundle_dir, _, _, _ = _write_bundle(root=tmp_path)
    assert validate_sources(bundle_dir=bundle_dir).passed

    source_report = bundle_dir / "source-preparation-report.json"
    write_json(
        path=source_report,
        payload={
            "passed": True,
            "origin_country_checks": {"positive_total": {"passed": False}},
        },
    )
    manifest_path = bundle_dir / "bundle-manifest.json"
    manifest = BundleManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    files = dict(manifest.files)
    files[source_report.name] = sha256_file(source_report)
    write_json(path=manifest_path, payload=manifest.model_copy(update={"files": files}))

    with pytest.raises(ValueError, match="differs from recomputed validation"):
        validate_sources(bundle_dir=bundle_dir)


@pytest.mark.skipif(os.name != "nt", reason="Windows native inventory contract")
@pytest.mark.parametrize("kind", ["normal", "symlink", "hardlink"])
def test_windows_native_inventory_contract(tmp_path: Path, kind: str) -> None:
    """Windows inventory uses native identity for files and links."""
    bundle_dir, _, _, _ = _write_bundle(root=tmp_path)
    target = bundle_dir / "source-preparation-report.json"
    if kind == "normal":
        inventory = _regular_file_inventory(bundle_dir=bundle_dir)
        assert inventory["source-preparation-report.json"] == target
        return
    if kind == "symlink":
        target.unlink()
        try:
            target.symlink_to(tmp_path / "outside")
        except OSError:
            pytest.skip("symbolic links are unavailable")
    else:
        replacement = bundle_dir / "hardlink-source.txt"
        replacement.write_text("hard link", encoding="utf-8")
        target.unlink()
        try:
            os.link(replacement, target)
        except OSError:
            pytest.skip("hard links are unavailable")

    with pytest.raises(ValueError, match="symlink|reparse point|hard link"):
        _regular_file_inventory(bundle_dir=bundle_dir)
