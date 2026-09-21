"""Tests for the public offline release packager."""

from __future__ import annotations

import polars as pl
import pytest
from support import ReleaseCase, coherent_evidence

from danish_personas.release import packager
from danish_personas.release.models import ReleaseEvidence, ReleaseManifest
from danish_personas.release.packager import ReleasePackagingError


@pytest.mark.parametrize(
    "version", [True, 1.0, "1", b"1"], ids=["bool", "float", "string", "bytes"]
)
def test_release_versions_require_exact_integer_one(
    release_case: ReleaseCase, version: object
) -> None:
    """Release contracts reject values that Pydantic could coerce to one."""
    manifest = release_case.manifest
    manifest_payload = {
        "version": version,
        "release_id": "a" * 32,
        "created_at": "2026-09-17T00:00:00+00:00",
        "pilot_id": manifest.pilot_id,
        "model": manifest.model,
        "rows": 1,
        "git_head": "a" * 40,
        "origin_url": "https://example.invalid/repo.git",
        "uv_lock_sha256": "a" * 64,
        "evidence_sha256": "a" * 64,
        "artifacts": [
            {"path": "README.md", "role": "dataset-card", "sha256": "a" * 64, "size": 0}
        ],
    }
    evidence_payload = coherent_evidence(release_case).model_dump(mode="json")
    evidence_payload["version"] = version
    with pytest.raises(ValueError):
        ReleaseManifest.model_validate(manifest_payload)
    with pytest.raises(ValueError):
        ReleaseEvidence.model_validate(evidence_payload)


def test_scanner_accepts_only_nullable_v2_fields(release_case: ReleaseCase) -> None:
    """Only nullable v2 fields may use Polars Null dtype."""
    output = pl.read_parquet(release_case.output)
    safe = output.with_columns(
        pl.lit(None, dtype=pl.Null).alias("career_goals_and_ambitions"),
        pl.lit(None, dtype=pl.Null).alias("job_title"),
    )
    packager._scan_dataframe(safe)
    unsafe = output.with_columns(pl.lit(None, dtype=pl.Null).alias("cultural_context"))
    with pytest.raises(ReleasePackagingError, match="logical dtype"):
        packager._scan_dataframe(unsafe)
    with pytest.raises(ReleasePackagingError, match="secret"):
        packager._scan_dataframe(
            output.with_columns(pl.lit("local file /tmp/secret").alias("job_title"))
        )
    with pytest.raises(ReleasePackagingError, match="logical dtype"):
        packager._scan_dataframe(
            output.with_columns(
                pl.lit("legacy visual guidance").alias("visual_persona")
            )
        )
