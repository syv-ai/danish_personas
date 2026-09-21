"""Prepared-source provenance and report validation tests."""

from pathlib import Path

import pytest

from danish_personas.io import sha256_file, sha256_text, write_json
from danish_personas.models import BundleManifest, SnapshotManifest
from danish_personas.sampling.generator import generate_records
from danish_personas.sources.bundle import verify_prepared_bundle
from danish_personas.sources.prepare import _verify_existing_bundle, verify_raw_snapshot
from danish_personas.validation.checks import validate_demographics, validate_sources
from tests.support.bundles import _write_bundle


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
