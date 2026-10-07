"""Local-only synthetic LGBT+ overlay, separate from canonical persona records.

This module produces orientation and sex-characteristics axes for restricted local use.
It must not be connected to generation-provider payloads, public releases, or uploads.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import polars as pl

SOURCE_URL = "https://datawrapper.dwcdn.net/3HTgD/5/data.csv"
OVERLAY_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class LgbtOverlayConfig:
    """Versioned settings for deterministic, local overlay generation.

    Attributes:
        version: Overlay semantics version. Only version 3 is supported.
        seed: Non-negative salt that allows reproducible independent scenarios.
        scenario: ``low`` applies published lower estimates; ``high`` represents
            published upper estimates with the excess recorded as uncertain.
    """

    version: int = OVERLAY_SCHEMA_VERSION
    seed: int = 0
    scenario: Literal["low", "high"] = "low"

    def __post_init__(self) -> None:
        """Reject unsupported or invalid overlay settings.

        Raises:
            ValueError:
                If the version, seed, or scenario is invalid.
        """
        if self.version != OVERLAY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported LGBT overlay version: {self.version}")
        if self.seed < 0:
            raise ValueError("Overlay seed must be non-negative")
        if self.scenario not in ("low", "high"):
            raise ValueError("Scenario must be 'low' or 'high'")


def generate_lgbt_overlay(
    frame: pl.DataFrame,
    *,
    config: LgbtOverlayConfig,
    persona_id_column: str = "persona_id",
    age_column: str = "age",
    sex_column: str = "sex",
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Create an ID-keyed overlay and aggregate-only provenance.

    The result is a new frame containing only the persona ID, orientation, and
    sex-characteristics variation; ``frame`` is never modified or returned.
    These axes use domain-separated SHA-256 hashes. Sexual orientation uses
    sex-specific SHILD 2020 rates for ages 18–64. Ages outside 18–64, including
    65+, are ``unknown`` on both axes. Gender and transgender status are modelled
    together in the restricted paired-identity overlay, not independently here.

    Args:
        frame: Canonical/local input frame with an ID and integer age column.
        config: Versioned scenario and deterministic seed.
        persona_id_column: Name of the unique identifier column.
        age_column: Name of the age column; age is used only for support eligibility.
        sex_column: Name of the sex column used for orientation-rate strata.

    Returns:
        A tuple of an ID-keyed overlay frame and provenance containing no row IDs.

    Raises:
        ValueError: If the version, required columns, IDs, or ID uniqueness is invalid.
    """
    required = {persona_id_column, age_column, sex_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing overlay input columns: {sorted(missing)}")

    rows = frame.select(persona_id_column, age_column, sex_column).iter_rows(named=True)
    identifiers: list[str] = []
    orientations: list[str] = []
    orientation_sexes: list[str] = []
    sex_characteristics: list[str] = []
    eligible = 0

    for row in rows:
        raw_id = row[persona_id_column]
        if raw_id is None or not str(raw_id):
            raise ValueError("Persona IDs must be non-empty")
        identifier = str(raw_id)
        identifiers.append(identifier)
        orientation_sexes.append(str(row[sex_column]))
        age = row[age_column]
        supported_age = isinstance(age, int) and 18 <= age <= 64
        if not supported_age:
            orientations.append("unknown")
            sex_characteristics.append("unknown")
            continue

        eligible += 1
        characteristics_draw = _draw(identifier, "sex_characteristics", config)
        orientations.append(
            _orientation_label(
                _draw(identifier, "orientation", config), row[sex_column]
            )
        )
        sex_characteristics.append(
            _range_label(
                characteristics_draw, low=0.01, high=0.017, scenario=config.scenario
            )
        )

    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Persona IDs must be unique")

    overlay = pl.DataFrame(
        {
            persona_id_column: identifiers,
            "sexual_orientation_identity": orientations,
            "variation_in_sex_characteristics": sex_characteristics,
        }
    )
    aggregate = {
        "overlay_schema_version": config.version,
        "scenario": config.scenario,
        "seed": config.seed,
        "row_count": len(identifiers),
        "supported_age_range": [18, 64],
        "supported_age_count": eligible,
        "unknown_age_count": len(identifiers) - eligible,
        "orientation_supported_count": sum(
            value != "unknown" for value in orientations
        ),
        "marginal_counts": {
            "sexual_orientation_identity": _counts(orientations),
            "sexual_orientation_identity_by_sex": {
                sex: _counts(
                    [
                        orientation
                        for orientation, row_sex in zip(
                            orientations, orientation_sexes, strict=True
                        )
                        if row_sex == sex
                    ]
                )
                for sex in ("male", "female")
            },
            "variation_in_sex_characteristics": _counts(sex_characteristics),
        },
        "sources": {
            "url": SOURCE_URL,
            "sample_size": 17929,
            "editorial_cutoff": "2025-07",
            "sexual_orientation_identity": (
                "SHILD 2020, n=17,929; published sex-specific chart"
            ),
            "sexual_orientation_identity_rates": {
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
                "remainder": "heterosexual",
            },
            "variation_in_sex_characteristics": (
                "Approximately 1% Danish self-report; 1.7% international definition"
            ),
        },
        "limitations": [
            "Synthetic marginal draws are not observed individual data.",
            "Axes are drawn independently; overlaps and joint distributions are "
            "unsupported and are not claimed.",
            "Orientation is sampled by source-sex survey stratum, not inferred "
            "from partner gender, origin, or status; relationships do not establish "
            "orientation.",
            "Sexual-orientation rates are conditional on the published male/female "
            "stratum; other sex values are unknown for this axis.",
            "No supported estimate is available outside ages 18–64; unsupported "
            "ages are unknown, not negative.",
            "The approximately 8% overall LGBT+ estimate is overlapping and is not "
            "used as a target or additive rate.",
            "Local restricted use only; never include in provider payloads or public "
            "releases without separate approval.",
        ],
    }
    return overlay, aggregate


def _orientation_label(draw: float, sex: object) -> str:
    """Sample a chart category from the sex-specific cumulative rates.

    Returns:
        The synthetic orientation identity, or ``unknown`` for an unstratified sex.
    """
    rates = {
        "male": (
            ("homosexual", 0.024),
            ("bisexual", 0.024),
            ("asexual", 0.004),
            ("other", 0.010),
        ),
        "female": (
            ("homosexual", 0.017),
            ("bisexual", 0.032),
            ("asexual", 0.007),
            ("other", 0.009),
        ),
    }
    if sex not in rates:
        return "unknown"
    cumulative = 0.0
    for label, rate in rates[sex]:
        cumulative += rate
        if draw < cumulative:
            return label
    return "heterosexual"


def _draw(identifier: str, axis: str, config: LgbtOverlayConfig) -> float:
    """Return a stable, axis-independent uniform draw for one persona."""
    material = f"lgbt-overlay:{config.version}:{config.seed}:{axis}:{identifier}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") / 2**64


def _range_label(
    draw: float, *, low: float, high: float, scenario: Literal["low", "high"]
) -> str:
    """Encode lower estimates as definite and upper-only as uncertain.

    Returns:
        A broad synthetic scenario label.
    """
    if draw < low:
        return "yes"
    if scenario == "high" and draw < high:
        return "uncertain"
    return "no"


def _counts(values: list[str]) -> dict[str, int]:
    """Count broad labels without retaining row-level information.

    Returns:
        Counts of labels present in the input.
    """
    labels = (
        "yes",
        "no",
        "uncertain",
        "unknown",
        "heterosexual",
        "homosexual",
        "bisexual",
        "asexual",
        "other",
    )
    return {label: values.count(label) for label in labels if label in values}
