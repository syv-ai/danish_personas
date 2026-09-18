"""Offline contracts for municipality-level source preparation."""

from pathlib import Path

import polars as pl
import pytest

from danish_personas.io import sha256_file, write_json
from danish_personas.models import BundleManifest
from danish_personas.sources.prepare import (
    _add_geography,
    _pool_ras209,
    _verify_existing_bundle,
)


def test_existing_legacy_bundle_is_rejected(tmp_path: Path) -> None:
    """A bundle manifest from before the municipality contract cannot be reused."""
    report = tmp_path / "source-preparation-report.json"
    report.write_text('{"passed": true}\n', encoding="utf-8")
    manifest = BundleManifest(
        bundle_id="legacy",
        created_at="2026-09-17T00:00:00+00:00",
        source_lock_sha256="0" * 64,
        categories_sha256="1" * 64,
        source_snapshots=[],
        classification_snapshots=[],
        files={report.name: sha256_file(report)},
        reference_periods={},
        assumptions=[],
    )
    write_json(path=tmp_path / "bundle-manifest.json", payload=manifest)

    with pytest.raises(ValueError, match="unsupported schema version"):
        _verify_existing_bundle(
            bundle_dir=tmp_path, manifest_path=tmp_path / "bundle-manifest.json"
        )


def test_geography_lookup_uses_official_names_and_parents() -> None:
    """Source labels cannot override the validated hierarchy lookup."""
    geography = pl.DataFrame(
        {
            "municipality_code": ["411"],
            "municipality": ["Christiansø"],
            "landsdel_code": ["04"],
            "landsdel": ["Landsdel Bornholm"],
            "region_code": ["084"],
            "region": ["Region Hovedstaden"],
        }
    )
    source = pl.DataFrame(
        {"municipality_code": ["411"], "count": [1], "suppressed": [False]}
    )

    result = _add_geography(frame=source, geography=geography)

    assert result.row(0, named=True) == {
        "municipality_code": "411",
        "count": 1,
        "suppressed": False,
        "municipality": "Christiansø",
        "region_code": "084",
        "region": "Region Hovedstaden",
    }


def test_ras209_pooling_keeps_municipalities_in_the_joint() -> None:
    """Pooling must not collapse two municipalities sharing a region."""
    frame = pl.DataFrame(
        {
            "municipality_code": ["101", "147"],
            "municipality": ["København", "Frederiksberg"],
            "region_code": ["084", "084"],
            "region": ["Region Hovedstaden", "Region Hovedstaden"],
            "age_band": ["20-24", "20-24"],
            "sex": ["male", "male"],
            "education_source_code": ["H10", "H10"],
            "education_level": ["primary", "primary"],
            "labour_market_status": ["employed", "employed"],
            "count": [100, 200],
            "suppressed": [False, False],
        }
    )

    pooled = _pool_ras209(
        frame=frame,
        education_pooling={
            "primary": "primary",
            "secondary_or_vocational": "secondary_or_vocational",
            "higher_education": "higher_education",
            "not_stated": "not_stated",
        },
        release_rows=100_000,
        minimum_source_count=50,
        minimum_expected_release_count=5,
    )

    assert sorted(pooled.get_column("municipality_code").to_list()) == ["101", "147"]
    assert pooled.get_column("count").sum() == 300
