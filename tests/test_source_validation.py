"""Regression tests for source-validation provenance gates."""

from pathlib import Path

from danish_personas.io import sha256_file, write_json
from danish_personas.models import BundleManifest
from danish_personas.validation.checks import validate_sources


def test_nested_pass_cannot_override_failed_source_report(tmp_path: Path) -> None:
    """Source validation requires the top-level report result to pass."""
    source_report = tmp_path / "source-preparation-report.json"
    write_json(
        path=source_report,
        payload={
            "passed": False,
            "tables": {"example": {"passed": True}},
        },
    )
    manifest = BundleManifest(
        bundle_id="failed-source-fixture",
        created_at="2026-09-14T00:00:00+00:00",
        source_lock_sha256="0" * 64,
        categories_sha256="1" * 64,
        source_snapshots=[],
        files={
            source_report.name: sha256_file(source_report),
        },
        reference_periods={},
        assumptions=[],
    )
    write_json(path=tmp_path / "bundle-manifest.json", payload=manifest)
    report = validate_sources(bundle_dir=tmp_path)
    assert not report.passed
