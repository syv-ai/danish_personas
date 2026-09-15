"""Regression tests for source-validation provenance gates."""

from pathlib import Path

import pytest

from danish_personas.io import sha256_file, sha256_text, write_json
from danish_personas.models import BundleManifest, SnapshotManifest
from danish_personas.sources.prepare import verify_raw_snapshot
from danish_personas.validation.checks import validate_sources


def test_nested_pass_cannot_override_failed_source_report(tmp_path: Path) -> None:
    """Source validation requires the top-level report result to pass."""
    source_report = tmp_path / "source-preparation-report.json"
    write_json(
        path=source_report,
        payload={"passed": False, "tables": {"example": {"passed": True}}},
    )
    manifest = BundleManifest(
        bundle_id="failed-source-fixture",
        created_at="2026-09-14T00:00:00+00:00",
        source_lock_sha256="0" * 64,
        categories_sha256="1" * 64,
        origin_regions_sha256="2" * 64,
        source_snapshots=[],
        files={source_report.name: sha256_file(source_report)},
        reference_periods={},
        assumptions=[],
    )
    write_json(path=tmp_path / "bundle-manifest.json", payload=manifest)
    report = validate_sources(bundle_dir=tmp_path)
    assert not report.passed


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
        (tmp_path / name).write_text(content)
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
