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
        "sexual_orientation_identity",
        "trans_or_nonbinary_identity",
        "variation_in_sex_characteristics",
    ]
    assert "provider_payload" not in result.columns
    assert "provider_payload" not in str(provenance)
    assert provenance["overlay_schema_version"] == 2


def test_unsupported_ages_and_sexes_are_unknown_on_orientation_axis() -> None:
    """Orientation is unsupported outside 18–64 and for unstratified sex values."""
    frame = pl.DataFrame(
        {
            "persona_id": ["young", "older", "missing", "unknown-sex"],
            "age": [17, 65, None, 30],
            "sex": ["male", "female", "male", "unknown"],
        }
    )
    overlay, provenance = generate_lgbt_overlay(frame, config=LgbtOverlayConfig())

    assert overlay["sexual_orientation_identity"].to_list() == [
        "unknown",
        "unknown",
        "unknown",
        "unknown",
    ]
    assert overlay["trans_or_nonbinary_identity"].to_list()[:3] == [
        "unknown",
        "unknown",
        "unknown",
    ]
    assert provenance["supported_age_range"] == [18, 64]
    assert provenance["supported_age_count"] == 0
    assert provenance["unknown_age_count"] == 4


def test_sex_specific_orientation_marginals_match_chart_rates() -> None:
    """Large stable samples reproduce each SHILD chart rate by sex."""
    count = 100_000
    frame = pl.DataFrame(
        {
            "persona_id": [f"synthetic-{index}" for index in range(count * 2)],
            "age": [35] * (count * 2),
            "sex": ["male"] * count + ["female"] * count,
        }
    )
    overlay, provenance = generate_lgbt_overlay(
        frame, config=LgbtOverlayConfig(seed=2026)
    )
    identities = overlay["sexual_orientation_identity"].to_list()
    expected_rates = {
        "male": {
            "homosexual": 0.024,
            "bisexual": 0.024,
            "asexual": 0.004,
            "other": 0.010,
        },
        "female": {
            "homosexual": 0.017,
            "bisexual": 0.032,
            "asexual": 0.007,
            "other": 0.009,
        },
    }
    for sex, expected in expected_rates.items():
        start = 0 if sex == "male" else count
        observed = identities[start : start + count]
        for category, rate in expected.items():
            assert abs(observed.count(category) / count - rate) < 0.002
        heterosexual_rate = 1 - sum(expected.values())
        assert abs(observed.count("heterosexual") / count - heterosexual_rate) < 0.002

    counts = provenance["marginal_counts"]["sexual_orientation_identity_by_sex"]
    assert counts["male"]["homosexual"] == identities[:count].count("homosexual")
    assert "SHILD 2020" in provenance["sources"]["sexual_orientation_identity"]
    assert provenance["sources"]["sample_size"] == 17929


def test_other_broad_axes_remain_available() -> None:
    """The orientation update does not remove the other two synthetic axes."""
    frame = pl.DataFrame(
        {
            "persona_id": [f"p-{index}" for index in range(1000)],
            "age": [35] * 1000,
            "sex": ["female"] * 1000,
        }
    )
    overlay, provenance = generate_lgbt_overlay(frame, config=LgbtOverlayConfig())
    counts = provenance["marginal_counts"]
    assert set(overlay["trans_or_nonbinary_identity"].unique().to_list()) <= {
        "yes",
        "no",
        "uncertain",
    }
    assert set(overlay["variation_in_sex_characteristics"].unique().to_list()) <= {
        "yes",
        "no",
        "uncertain",
    }
    limitations = " ".join(provenance["limitations"])
    assert "joint distributions are unsupported" in limitations
    assert counts["trans_or_nonbinary_identity"]


def test_invalid_config_and_duplicate_ids_fail_closed() -> None:
    """Unsupported schema versions and ambiguous identifiers are rejected."""
    with pytest.raises(ValueError, match="Unsupported LGBT overlay version"):
        LgbtOverlayConfig(version=1)
    frame = pl.DataFrame(
        {"persona_id": ["same", "same"], "age": [20, 30], "sex": ["male", "female"]}
    )
    with pytest.raises(ValueError, match="unique"):
        generate_lgbt_overlay(frame, config=LgbtOverlayConfig())
