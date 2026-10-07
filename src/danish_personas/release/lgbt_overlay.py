"""Local-only synthetic LGBT+ overlay, separate from canonical persona records.

This module produces broad synthetic identity axes for restricted local exploration.
It must not be connected to generation-provider payloads, public releases, or uploads.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import polars as pl

SOURCE_URL = "https://menneskeret.dk/lgbt-barometer/lgbt-hvad-hvor-mange"
OVERLAY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LgbtOverlayConfig:
    """Versioned settings for deterministic, local overlay generation.

    Attributes:
        version: Overlay semantics version. Only version 1 is supported.
        seed: Non-negative salt that allows reproducible independent scenarios.
        scenario: ``low`` applies published lower estimates; ``high`` represents
            published upper estimates with the excess recorded as uncertain.
    """

    version: int = OVERLAY_SCHEMA_VERSION
    seed: int = 0
    scenario: Literal["low", "high"] = "low"

    def __post_init__(self) -> None:
        """Reject unsupported or invalid overlay settings."""
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
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Create an ID-keyed overlay and aggregate-only provenance.

    The result is a new frame containing only the persona ID and three broad axes;
    ``frame`` is never modified or returned. Axes are drawn independently using
    domain-separated SHA-256 hashes. Ages outside 16–64, including 65+, are
    ``unknown`` on every axis because the cited estimates do not support them.

    Args:
        frame: Canonical/local input frame with an ID and integer age column.
        config: Versioned scenario and deterministic seed.
        persona_id_column: Name of the unique identifier column.
        age_column: Name of the age column; age is used only for support eligibility.

    Returns:
        A tuple of an ID-keyed overlay frame and provenance containing no row IDs.

    Raises:
        ValueError: If the version, required columns, IDs, or ID uniqueness is invalid.
    """
    required = {persona_id_column, age_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing overlay input columns: {sorted(missing)}")

    rows = frame.select(persona_id_column, age_column).iter_rows(named=True)
    identifiers: list[str] = []
    orientations: list[str] = []
    gender_identities: list[str] = []
    sex_characteristics: list[str] = []
    eligible = 0

    for row in rows:
        raw_id = row[persona_id_column]
        if raw_id is None or not str(raw_id):
            raise ValueError("Persona IDs must be non-empty")
        identifier = str(raw_id)
        identifiers.append(identifier)
        age = row[age_column]
        supported_age = isinstance(age, int) and 16 <= age <= 64
        if not supported_age:
            orientations.append("unknown")
            gender_identities.append("unknown")
            sex_characteristics.append("unknown")
            continue

        eligible += 1
        orientation_draw = _draw(identifier, "orientation", config)
        gender_draw = _draw(identifier, "gender_identity", config)
        characteristics_draw = _draw(identifier, "sex_characteristics", config)

        orientations.append("minority" if orientation_draw < 0.065 else "not_minority")
        gender_identities.append(
            _range_label(gender_draw, low=0.005, high=0.014, scenario=config.scenario)
        )
        sex_characteristics.append(
            _range_label(
                characteristics_draw,
                low=0.01,
                high=0.017,
                scenario=config.scenario,
            )
        )

    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Persona IDs must be unique")

    overlay = pl.DataFrame(
        {
            persona_id_column: identifiers,
            "sexual_orientation_minority_identity": orientations,
            "trans_or_nonbinary_identity": gender_identities,
            "variation_in_sex_characteristics": sex_characteristics,
        }
    )
    aggregate = {
        "overlay_schema_version": config.version,
        "scenario": config.scenario,
        "seed": config.seed,
        "row_count": len(identifiers),
        "supported_age_range": [16, 64],
        "supported_age_count": eligible,
        "unknown_age_count": len(identifiers) - eligible,
        "marginal_counts": {
            "sexual_orientation_minority_identity": _counts(orientations),
            "trans_or_nonbinary_identity": _counts(gender_identities),
            "variation_in_sex_characteristics": _counts(sex_characteristics),
        },
        "sources": {
            "url": SOURCE_URL,
            "editorial_cutoff": "2025-07",
            "sexual_orientation_minority_identity": "Approximately 6.5%, SHILD 2020",
            "trans_or_nonbinary_identity": "0.5–1.4%, SHILD 2020 and SEXUS 2019",
            "variation_in_sex_characteristics": (
                "Approximately 1% Danish self-report; 1.7% international definition"
            ),
        },
        "limitations": [
            "Synthetic marginal draws are not observed individual data.",
            "Axes are drawn independently; overlaps and joint distributions are "
            "unsupported and are not claimed.",
            "No identity is inferred from sex, partner gender, origin, or status; "
            "relationships do not establish orientation.",
            "No supported estimate is available for ages 65 and older; unsupported "
            "ages are unknown, not negative.",
            "The approximately 8% overall LGBT+ estimate is overlapping and is not "
            "used as a target or additive rate.",
            "Local restricted use only; never include in provider payloads or public "
            "releases without separate approval.",
        ],
    }
    return overlay, aggregate


def _draw(identifier: str, axis: str, config: LgbtOverlayConfig) -> float:
    """Return a stable, axis-independent uniform draw for one persona."""
    material = f"lgbt-overlay:{config.version}:{config.seed}:{axis}:{identifier}"
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") / 2**64


def _range_label(
    draw: float, *, low: float, high: float, scenario: Literal["low", "high"]
) -> str:
    """Encode lower estimates as definite and upper-only prevalence as uncertain."""
    if draw < low:
        return "yes"
    if scenario == "high" and draw < high:
        return "uncertain"
    return "no"


def _counts(values: list[str]) -> dict[str, int]:
    """Count broad labels without retaining row-level information."""
    labels = ("yes", "no", "uncertain", "unknown", "minority", "not_minority")
    return {label: values.count(label) for label in labels if label in values}
