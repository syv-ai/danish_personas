"""Contracts for the restricted local LGBT+ overlay."""

import polars as pl
import pytest

from danish_personas.release.lgbt_overlay import (
    LgbtOverlayConfig,
    generate_lgbt_overlay,
)


def test_invalid_config_and_duplicate_ids_fail_closed() -> None:
    """Unsupported schema versions and ambiguous identifiers are rejected."""
    with pytest.raises(ValueError, match="Unsupported LGBT overlay version"):
        LgbtOverlayConfig(version=1)
    frame = pl.DataFrame(
        {"persona_id": ["same", "same"], "age": [20, 30], "sex": ["male", "female"]}
    )
    with pytest.raises(ValueError, match="unique"):
        generate_lgbt_overlay(frame, config=LgbtOverlayConfig())


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
        "variation_in_sex_characteristics",
    ]
    assert "provider_payload" not in result.columns
    assert "provider_payload" not in str(provenance)
    assert provenance["overlay_schema_version"] == 3


def test_sex_characteristics_remain_separate() -> None:
    """Gender status is not sampled independently of the paired gender model."""
    frame = pl.DataFrame(
        {
            "persona_id": [f"p-{index}" for index in range(1000)],
            "age": [35] * 1000,
            "sex": ["female"] * 1000,
        }
    )
    overlay, provenance = generate_lgbt_overlay(frame, config=LgbtOverlayConfig())
    counts = _mapping(provenance["marginal_counts"])
    assert "trans_or_nonbinary_identity" not in overlay.columns
    assert set(overlay["variation_in_sex_characteristics"].unique().to_list()) <= {
        "yes",
        "no",
        "uncertain",
    }
    limitations = " ".join(_strings(provenance["limitations"]))
    assert "joint distributions are unsupported" in limitations
    assert counts["variation_in_sex_characteristics"]


def _mapping(value: object) -> dict[str, object]:
    """Narrow provenance objects to string-keyed mappings for assertions.

    Returns:
        The validated mapping.
    """
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _strings(value: object) -> list[str]:
    """Narrow provenance arrays to strings for assertions.

    Returns:
        The validated list of strings.
    """
    assert isinstance(value, list)
    assert all(isinstance(item, str) for item in value)
    return [item for item in value if isinstance(item, str)]


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

    marginal_counts = _mapping(provenance["marginal_counts"])
    counts = _mapping(marginal_counts["sexual_orientation_identity_by_sex"])
    male_counts = _mapping(counts["male"])
    assert male_counts["homosexual"] == identities[:count].count("homosexual")
    sources = _mapping(provenance["sources"])
    assert "SHILD 2020" in _string(sources["sexual_orientation_identity"])
    assert sources["sample_size"] == 17929


def _string(value: object) -> str:
    """Narrow a provenance value to a string for assertions.

    Returns:
        The validated string.
    """
    assert isinstance(value, str)
    return value


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
    assert provenance["supported_age_range"] == [18, 64]
    assert provenance["supported_age_count"] == 1
    assert provenance["unknown_age_count"] == 3
    assert provenance["orientation_supported_count"] == 0
