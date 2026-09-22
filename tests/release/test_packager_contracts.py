"""Tests for the public offline release packager."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from support import ReleaseCase, coherent_evidence

from danish_personas.generation.config import load_generation_config
from danish_personas.generation.partner_target import same_sex_partner_target
from danish_personas.release import packager
from danish_personas.release.common import validate_persona_output_rows
from danish_personas.release.models import ReleaseEvidence, ReleaseManifest
from danish_personas.release.packager import ReleasePackagingError


@pytest.mark.parametrize(
    "construction",
    [
        "Partneren hedder Aarhus.",
        "Hendes datter ved navn Danmark.",
        "Aarhus er hendes kæreste.",
        "Aarhus er 35 år.",
        "Aarhus's partner bor i byen.",
        "Aarhus’s partner bor i byen.",
        "Aarhus´s partner bor i byen.",
    ],
)
def test_release_allows_required_place_and_origin_labels(
    release_case: ReleaseCase, construction: str
) -> None:
    """Release validation preserves required grounded labels in name patterns."""
    output = (
        pl.read_parquet(release_case.output)
        .head(1)
        .with_columns(
            pl.col("persona").str.replace(
                "Han er 35 år", f"Han er 35 år. {construction}"
            )
        )
    )

    validate_persona_output_rows(output)


def test_release_checks_partner_target_for_each_persona_id(
    release_case: ReleaseCase,
) -> None:
    """Release validation does not cache rows across target-dependent IDs."""
    config = load_generation_config(Path("config/config.yaml")).model_copy(
        update={"same_sex_partner_probability": 0.5}
    )
    candidate_ids = [f"cache-target-{index}" for index in range(1000)]
    first_id, second_id = next(
        (left, right)
        for left in candidate_ids
        for right in candidate_ids
        if left != right
        and not same_sex_partner_target(
            persona_id=left, probability=config.same_sex_partner_probability
        )
        and same_sex_partner_target(
            persona_id=right, probability=config.same_sex_partner_probability
        )
    )
    base = (
        pl.read_parquet(release_case.output)
        .head(1)
        .with_columns(
            pl.col("persona")
            .str.replace("single", "gift med sin partner")
            .str.replace("har aldrig været gift", "er gift"),
            pl.lit("married_or_separated").alias("marital_status"),
            pl.lit("married").alias("legal_status_detail"),
            pl.lit("partnered").alias("current_relationship_status"),
            pl.lit("female").alias("partner_gender"),
        )
    )
    output = pl.concat(
        [
            base.with_columns(pl.lit(first_id).alias("persona_id")),
            base.with_columns(pl.lit(second_id).alias("persona_id")),
        ],
        how="vertical",
    )

    with pytest.raises(ValueError, match="generation-v5"):
        validate_persona_output_rows(output, generation_config=config)


@pytest.mark.parametrize(
    "construction",
    [
        "Hun hedder Maja og bor i Aarhus.",
        "Hendes datter hedder Emma.",
        "Hendes datter hedder Emma og går til håndbold.",
        "Partneren hedder Lars.",
        "Partneren Maja bor ofte i byen.",
        "Partneren kaldes Lars.",
        "Hendes datter ved navn Emma.",
        "Partner ved navn Lars.",
        "Partneren Lars.",
        "Personaens navn er Maja.",
        "Jeg er Maja.",
        "Hun er Maja.",
        "Han er Maja.",
        "Personaen er Maja.",
        "Personen er Maja.Navnet er Maja.",
        "Maja er hendes kæreste.",
        "Majas partner bor i byen.",
        "Maja's partner bor i byen.",
        "Maja’s partner bor i byen.",
        "Aarhus-Lars er 35 år.",
        "Aarhus Jensen er 35 år.",
        "San Marino'Lars er 35 år.",
        "San Marino’Lars er 35 år.",
        "San Marino´Lars er 35 år.",
        "Aarhus-Lars's partner bor i byen.",
        "Aarhus-Lars’s partner bor i byen.",
        "Aarhus-Lars´s partner bor i byen.",
    ],
)
def test_release_replays_no_person_name_validation(
    release_case: ReleaseCase, construction: str
) -> None:
    """Release validation rejects every reviewed name construction."""
    output = (
        pl.read_parquet(release_case.output)
        .head(1)
        .with_columns(
            pl.col("persona").str.replace(
                "Han er 35 år", f"Han er 35 år. {construction}"
            )
        )
    )

    with pytest.raises(ValueError, match="generation-v5"):
        validate_persona_output_rows(output)


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
