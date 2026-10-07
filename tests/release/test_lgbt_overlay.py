"""Contracts for the restricted local LGBT+ overlay."""

import polars as pl
import pytest

from danish_personas.release.lgbt_overlay import (
    LgbtOverlayConfig,
    generate_lgbt_overlay,
)


def test_overlay_is_deterministic_and_separate_from_input() -> None:
    """Generation is repeatable and preserves the supplied source frame."""
    frame = pl.DataFrame(
        {
            "persona_id": ["p-1", "p-2"],
            "age": [25, 40],
            "sex": ["female", "male"],
            "partner_gender": ["male", "female"],
            "origin_country": ["DK", "DK"],
            "status": ["employed", "unemployed"],
            "provider_payload": ["must-not-copy", "must-not-copy"],
        }
    )
    before = frame.clone()
    config = LgbtOverlayConfig(seed=19, scenario="high")

    result, provenance = generate_lgbt_overlay(frame, config=config)
    repeated, repeated_provenance = generate_lgbt_overlay(frame, config=config)

    assert result.equals(repeated)
    assert provenance == repeated_provenance
    assert frame.equals(before)
    assert result.columns == [
        "persona_id",
        "sexual_orientation_minority_identity",
        "trans_or_nonbinary_identity",
        "variation_in_sex_characteristics",
    ]
    assert "provider_payload" not in result.columns
    assert "provider_payload" not in str(provenance)


def test_unsupported_ages_are_unknown_on_every_axis() -> None:
    """The estimates do not support children or adults aged 65 and over."""
    frame = pl.DataFrame(
        {"persona_id": ["young", "older", "missing"], "age": [15, 65, None]}
    )
    overlay, provenance = generate_lgbt_overlay(frame, config=LgbtOverlayConfig())

    for column in overlay.columns[1:]:
        assert overlay[column].to_list() == ["unknown", "unknown", "unknown"]
    assert provenance["supported_age_count"] == 0
    assert provenance["unknown_age_count"] == 3


def test_100k_marginals_follow_published_rates_without_subtypes() -> None:
    """Large synthetic marginals approximate broad source-supported ranges."""
    count = 100_000
    frame = pl.DataFrame(
        {
            "persona_id": [f"synthetic-{index}" for index in range(count)],
            "age": [35] * count,
        }
    )
    overlay, provenance = generate_lgbt_overlay(
        frame, config=LgbtOverlayConfig(seed=2026, scenario="high")
    )
    counts = provenance["marginal_counts"]
    assert isinstance(counts, dict)
    orientation = counts["sexual_orientation_minority_identity"]
    gender_identity = counts["trans_or_nonbinary_identity"]
    characteristics = counts["variation_in_sex_characteristics"]
    assert abs(orientation["minority"] / count - 0.065) < 0.002
    assert 0.003 < gender_identity["yes"] / count < 0.007
    gender_identity_total = (
        gender_identity["yes"] + gender_identity["uncertain"]
    ) / count
    assert 0.012 < gender_identity_total < 0.016
    assert 0.008 < characteristics["yes"] / count < 0.012
    assert (
        0.015 < (characteristics["yes"] + characteristics["uncertain"]) / count < 0.019
    )
    assert set(overlay["trans_or_nonbinary_identity"].unique().to_list()) <= {
        "yes",
        "no",
        "uncertain",
    }
    limitations = " ".join(provenance["limitations"])
    assert "joint distributions are unsupported" in limitations
    assert "8% overall" in limitations
    assert "relationships do not establish orientation" in limitations


def test_invalid_config_and_duplicate_ids_fail_closed() -> None:
    """Unsupported schema versions and ambiguous identifiers are rejected."""
    with pytest.raises(ValueError, match="Unsupported LGBT overlay version"):
        LgbtOverlayConfig(version=2)
    frame = pl.DataFrame({"persona_id": ["same", "same"], "age": [20, 30]})
    with pytest.raises(ValueError, match="unique"):
        generate_lgbt_overlay(frame, config=LgbtOverlayConfig())
