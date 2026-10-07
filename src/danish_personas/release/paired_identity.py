"""Restricted, synthetic paired-gender identity overlay for local analysis only."""

from __future__ import annotations

import hashlib
import json

import polars as pl

ORIENTATIONS = ("homo", "hetero", "bi", "ace", "other", "unknown")
_GENDERS = ("man", "woman", "nonbinary")
_PARTNER_RATES = {
    "male": {"homo": 0.024, "bi": 0.024, "ace": 0.004, "other": 0.010},
    "female": {"homo": 0.017, "bi": 0.032, "ace": 0.007, "other": 0.009},
}


def generate_paired_identity(
    frame: pl.DataFrame,
    orientation_overlay: pl.DataFrame,
    *,
    seed: int = 0,
    persona_id_column: str = "persona_id",
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Create a separate paired identity overlay and aggregate provenance.

    This function is restricted to local synthetic analysis. It never modifies or
    returns the source frame and does not include source or persona text in the
    overlay. Orientation labels can use either the compact overlay values or the
    existing ``lgbt_overlay`` labels ``homosexual``, ``heterosexual``,
    ``bisexual``, and ``asexual``.

    Args:
        frame: Release frame with age, binary source sex, relationship status, and
            existing partner gender.
        orientation_overlay: ID-keyed categorical orientation overlay.
        seed (optional): Non-negative salt for stable, domain-separated draws.
            Defaults to 0.
        persona_id_column (optional): Identifier column shared by both frames.
            Defaults to ``persona_id``.

    Returns:
        The separate ID-keyed overlay and provenance, including changed partner
        gender identifiers and synthetic assumptions.

    Raises:
        ValueError: If required fields, identifiers, categories, or uniqueness are
            invalid.
    """
    if seed < 0:
        raise ValueError("Seed must be non-negative")
    orientations = _orientation_map(
        frame=frame,
        orientation_overlay=orientation_overlay,
        persona_id_column=persona_id_column,
    )

    input_rows = frame.select(
        persona_id_column, "age", "sex", "current_relationship_status", "partner_gender"
    ).iter_rows(named=True)
    rows = list(input_rows)
    prepared_gender: dict[str, tuple[str, str]] = {}
    gender_counts = dict.fromkeys(_GENDERS, 0)
    for row in rows:
        identifier = _identifier(row[persona_id_column])
        if isinstance(row["age"], int) and 18 <= row["age"] <= 64:
            gender_result = _gender(
                identifier=identifier, sex=row["sex"], supported=True, seed=seed
            )
            prepared_gender[identifier] = gender_result
            gender_counts[gender_result[0]] += 1

    records: list[dict[str, object]] = []
    changed_partner_ids: list[str] = []
    for row in rows:
        identifier = _identifier(row[persona_id_column])
        if identifier not in orientations:
            raise ValueError("Every release ID must have an orientation overlay")
        record, changed = _build_record(
            row=row,
            identifier=identifier,
            orientation=orientations[identifier],
            gender_result=prepared_gender.get(identifier, ("unknown", "unknown")),
            gender_weights=gender_counts,
            seed=seed,
            persona_id_column=persona_id_column,
        )
        records.append(record)
        if changed:
            changed_partner_ids.append(identifier)

    identifiers = [str(row[persona_id_column]) for row in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Release IDs must be unique")
    if set(orientations) != set(identifiers):
        raise ValueError("Orientation overlay IDs must exactly match release IDs")

    result = pl.DataFrame(records)
    return result, {
        "schema_version": 1,
        "seed": seed,
        "row_count": len(records),
        "changed_partner_gender_ids": changed_partner_ids,
        "assumptions": [
            "For ages 18-64, non-trans binary gender is proxied from source sex.",
            "SEXUS rates are applied as 0.05% trans men, 0.05% trans women, "
            "and 0.44% nonbinary; nonbinary transgender status is unknown.",
            "Partner gender is assigned from self orientation under the requested "
            "scenario; this is synthetic, not observed relationship evidence.",
            "Partner orientation uses source binary-sex marginal rates by proxying "
            "binary partner gender to source sex, then filtering for scenario "
            "compatibility; neither the proxy nor the joint is observed.",
            "Partner age is not observed; ages 18–64 for the persona are used as "
            "a proxy for the chart's partner age eligibility.",
            "Partner transgender status is a separate source-based synthetic draw; "
            "nonbinary and ages 65+ are unknown.",
            "Ages 65+ retain existing partner gender for partnered records and "
            "have unknown self gender, orientation, and transgender status.",
            "The available figures do not split another-gender identities from "
            "nonbinary, so no additional gender category is sampled.",
        ],
        "sources": {
            "binary_trans_and_nonbinary_rates": (
                "Frisch et al. (2019), Sex i Danmark, Projekt SEXUS 2017–2018, "
                "gender-identity estimates: 0.05% trans men, 0.05% trans women, "
                "0.44% nonbinary; "
                "https://files.projektsexus.dk/2019-10-26_SEXUS-rapport_2017-2018.pdf"
            ),
            "partner_orientation_rates": (
                "SHILD 2020 sex-specific marginals in lgbt_overlay.py"
            ),
        },
        "draw_method": "SHA-256 domain-separated deterministic draws",
    }


def _build_record(
    *,
    row: dict[str, object],
    identifier: str,
    orientation: str,
    gender_result: tuple[str, str],
    gender_weights: dict[str, int],
    seed: int,
    persona_id_column: str,
) -> tuple[dict[str, object], bool]:
    sex = row["sex"]
    age = row["age"]
    relationship = row["current_relationship_status"]
    if sex not in ("male", "female"):
        raise ValueError("Source sex must be male or female")
    if relationship not in ("partnered", "not_partnered"):
        raise ValueError("Relationship status must be partnered or not_partnered")
    supported = isinstance(age, int) and 18 <= age <= 64
    if not supported:
        orientation = "unknown"
    gender, transgender = gender_result
    changed = False
    if relationship == "not_partnered":
        partner_gender = None
        partner_orientation = None
        partner_transgender = None
    elif not supported:
        partner_gender = _existing_partner_gender(row["partner_gender"])
        partner_orientation = "unknown"
        partner_transgender = "unknown"
    else:
        partner_gender = _assign_partner_gender(
            identifier=identifier,
            gender=gender,
            orientation=orientation,
            seed=seed,
            gender_weights=gender_weights,
        )
        partner_orientation = _partner_orientation(
            identifier=identifier,
            partner_gender=partner_gender,
            self_gender=gender,
            seed=seed,
        )
        partner_transgender = _partner_transgender(
            identifier=identifier, partner_gender=partner_gender, seed=seed
        )
        changed = _existing_partner_gender(row["partner_gender"]) != partner_gender
    return {
        persona_id_column: identifier,
        "gender": gender,
        "partner_gender": partner_gender,
        "sexual_orientation": orientation,
        "partner_sexual_orientation": partner_orientation,
        "transgender": transgender,
        "partner_transgender": partner_transgender,
    }, changed


def _assign_partner_gender(
    *,
    identifier: str,
    gender: str,
    orientation: str,
    seed: int,
    gender_weights: dict[str, int],
) -> str:
    if orientation == "homo":
        return gender
    if orientation == "hetero":
        if gender == "man":
            return "woman"
        if gender == "woman":
            return "man"
        return _weighted_gender(
            identifier=identifier,
            seed=seed,
            domain="hetero_partner_gender",
            gender_weights=gender_weights,
            excluded="nonbinary",
        )
    return _weighted_gender(
        identifier=identifier,
        seed=seed,
        domain="other_partner_gender",
        gender_weights=gender_weights,
    )


def _weighted_gender(
    *,
    identifier: str,
    seed: int,
    domain: str,
    gender_weights: dict[str, int],
    excluded: str | None = None,
) -> str:
    options = [gender for gender in _GENDERS if gender != excluded]
    total = sum(gender_weights[gender] for gender in options)
    if total == 0:
        return "unknown"
    threshold = _draw(identifier=identifier, domain=domain, seed=seed) * total
    for gender in options:
        threshold -= gender_weights[gender]
        if threshold < 0:
            return gender
    return options[-1]


def _draw(*, identifier: str, domain: str, seed: int) -> float:
    payload = json.dumps(
        ["paired-identity-v1", seed, domain, identifier],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return integer / 2**64


def _existing_partner_gender(value: object) -> str | None:
    if value is None:
        return None
    mapping = {"male": "man", "female": "woman", "man": "man", "woman": "woman"}
    if value not in mapping:
        raise ValueError("Existing partner gender must be male, female, or null")
    return mapping[value]


def _partner_orientation(
    *, identifier: str, partner_gender: str, self_gender: str, seed: int
) -> str:
    if partner_gender not in ("man", "woman") or self_gender not in ("man", "woman"):
        return "unknown"
    sex = "male" if partner_gender == "man" else "female"
    rates = _PARTNER_RATES[sex]
    options = [("hetero", 1.0 - sum(rates.values()))]
    options.extend(rates.items())
    compatible = [
        (label, rate)
        for label, rate in options
        if not (
            label == "hetero"
            and self_gender == partner_gender
            and self_gender in ("man", "woman")
        )
        and not (
            label == "homo"
            and self_gender != partner_gender
            and self_gender in ("man", "woman")
        )
    ]
    return _weighted_label(
        identifier=identifier,
        domain="partner_orientation",
        seed=seed,
        options=compatible,
    )


def _weighted_label(
    *, identifier: str, domain: str, seed: int, options: list[tuple[str, float]]
) -> str:
    draw = _draw(identifier=identifier, domain=domain, seed=seed)
    total = sum(weight for _, weight in options)
    threshold = draw * total
    for label, weight in options:
        threshold -= weight
        if threshold < 0:
            return label
    return options[-1][0]


def _partner_transgender(*, identifier: str, partner_gender: str, seed: int) -> str:
    if partner_gender == "nonbinary":
        return "unknown"
    draw = _draw(identifier=identifier, domain="partner_transgender", seed=seed)
    return "true" if draw < 0.001 else "false"


def _gender(
    *, identifier: str, sex: str, supported: bool, seed: int
) -> tuple[str, str]:
    if not supported:
        return "unknown", "unknown"
    draw = _draw(identifier=identifier, domain="self_gender", seed=seed)
    if draw < 0.0005:
        return "man", "true"
    if draw < 0.001:
        return "woman", "true"
    if draw < 0.0054:
        return "nonbinary", "unknown"
    return ("man" if sex == "male" else "woman"), "false"


def _identifier(value: object) -> str:
    if value is None or not str(value):
        raise ValueError("Persona IDs must be non-empty")
    return str(value)


def _orientation_map(
    *, frame: pl.DataFrame, orientation_overlay: pl.DataFrame, persona_id_column: str
) -> dict[str, str]:
    required = {
        persona_id_column,
        "age",
        "sex",
        "current_relationship_status",
        "partner_gender",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing paired identity input columns: {sorted(missing)}")
    overlay_required = {persona_id_column, "sexual_orientation_identity"}
    if overlay_required.difference(orientation_overlay.columns):
        raise ValueError(
            "Orientation overlay requires ID and sexual_orientation_identity"
        )
    orientations: dict[str, str] = {}
    rows = orientation_overlay.select(
        persona_id_column, "sexual_orientation_identity"
    ).iter_rows()
    for raw_id, raw_orientation in rows:
        identifier = _identifier(raw_id)
        if identifier in orientations:
            raise ValueError("Orientation overlay IDs must be unique")
        orientations[identifier] = _normalise_orientation(raw_orientation)
    return orientations


def _normalise_orientation(value: object) -> str:
    aliases = {
        "homo": "homo",
        "homosexual": "homo",
        "hetero": "hetero",
        "heterosexual": "hetero",
        "bi": "bi",
        "bisexual": "bi",
        "ace": "ace",
        "asexual": "ace",
        "other": "other",
        "unknown": "unknown",
    }
    if not isinstance(value, str) or value not in aliases:
        raise ValueError(f"Unsupported orientation category: {value!r}")
    return aliases[value]
