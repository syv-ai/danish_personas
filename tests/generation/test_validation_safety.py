"""Objective safety and natural-prose tests for persona validation."""

# Test functions intentionally omit repetitive docstrings.
# ruff: noqa: D103

import json

import pytest
from validation_test_helpers import attributes, demographic, persona

from danish_personas.generation.validation import (
    parse_attributes,
    parse_descriptions,
    parse_generated_persona,
)


@pytest.mark.parametrize(
    "claim",
    [
        "ansigter",
        "ansigtstræk",
        "hårene",
        "hudfarve",
        "højde",
        "vægt",
        "kroppen",
        "kropsbygning",
        "udseende",
        "ser ud",
    ],
)
def test_all_unsupported_appearance_claims_fail(claim: str) -> None:
    context = demographic()
    text = persona(context=context)["persona"] + f" Hun nævner {claim}."
    with pytest.raises(ValueError, match="unsupported"):
        parse_descriptions(json.dumps({"persona": text}), context, attributes())


def test_alternative_official_origin_label_is_rejected_without_echo() -> None:
    context = demographic()
    text = persona(context=context)["persona"] + " Hun kommer fra Denmark."
    with pytest.raises(ValueError, match="origin") as error:
        parse_descriptions(json.dumps({"persona": text}), context, attributes())
    assert "Denmark" not in str(error.value)


def test_natural_biography_and_varied_personality_wording_are_allowed() -> None:
    context = demographic()
    text = persona(
        context=context,
        extra=(
            "Tidligere arbejdede hun i en boghandel, og hun drømmer nu om at "
            "starte en lille læseklub. Hun er social, men sætter også pris på "
            "stille morgener og kan være spontan, når venner foreslår en tur."
        ),
    )["persona"]

    parsed = parse_descriptions(json.dumps({"persona": text}), context, attributes())

    assert parsed.persona == text


def test_relationship_pronouns_and_child_age_are_allowed() -> None:
    context = demographic()
    text = persona(
        context=context,
        extra=(
            "Hendes kæreste er 38 år, og han arbejder på et værksted i "
            "Roskilde. Deres datter er 8 år, og hun går til håndbold."
        ),
    )["persona"]

    parsed = parse_descriptions(json.dumps({"persona": text}), context, attributes())

    assert parsed.persona == text


def test_unrelated_kan_vaere_idiom_remains_allowed() -> None:
    context = demographic()
    text = persona(
        context=context, extra="At læse kan være en rolig afslutning på dagen."
    )["persona"]

    parsed = parse_descriptions(json.dumps({"persona": text}), context, attributes())

    assert parsed.persona == text


@pytest.mark.parametrize(
    "extra",
    [
        " Maja er 35 år.",
        " Hun hedder Maja.",
        " Hun er ved navn Maja.",
        " Personaens navn er Maja.",
        " Navnet er Maja.",
        " Hendes datter hedder Emma.",
        " Partneren hedder Lars.",
        " Partneren kaldes Lars.",
        " Hendes datter ved navn Emma går til håndbold.",
        " Partneren Lars bor i byen.",
        " Hendes datter er Emma.",
        " Maja er hendes kæreste.",
        " Hun bor sammen med sin kæreste Maja.",
        " Deres datter Emma går til håndbold.",
    ],
)
def test_explicit_person_names_are_rejected(extra: str) -> None:
    context = demographic()
    text = persona(context=context, extra=extra)["persona"]

    with pytest.raises(ValueError, match="person name"):
        parse_descriptions(json.dumps({"persona": text}), context, attributes())


@pytest.mark.parametrize(
    "construction",
    [
        "Hendes datter hedder Emma.",
        "Partneren hedder Lars.",
        "Partneren kaldes Lars.",
        "Hendes datter ved navn Emma.",
        "Partner ved navn Lars.",
        "Partneren Lars.",
        "Personaens navn er Maja.",
        "Navnet er Maja.",
        "Maja er hendes kæreste.",
    ],
)
def test_explicit_person_names_are_rejected_in_attributes(construction: str) -> None:
    payload = attributes()
    payload["cultural_context"] = f"En almindelig dansk hverdag. {construction}"

    with pytest.raises(ValueError, match="person name"):
        parse_attributes(json.dumps(payload), demographic())


def test_combined_generation_rejects_partner_name() -> None:
    context = demographic()
    payload = attributes()
    payload["persona"] = persona(
        context=context, extra=" Hun bor sammen med sin kæreste Maja."
    )["persona"]

    with pytest.raises(ValueError, match="person name"):
        parse_generated_persona(json.dumps(payload), context)


def test_required_place_and_origin_labels_are_allowed_in_name_patterns() -> None:
    context = demographic()
    text = persona(
        context=context,
        extra=(
            " Partneren hedder København. Hendes datter ved navn Danmark. "
            "København er hendes kæreste."
        ),
    )["persona"]

    parsed = parse_descriptions(json.dumps({"persona": text}), context, attributes())

    assert parsed.persona == text


def test_grounded_proper_nouns_and_sentence_starts_are_allowed() -> None:
    context = demographic()
    text = persona(
        context=context,
        extra=(
            " Hun møder ofte nye naboer i Aarhus. Danmark er et vigtigt faktum. "
            "Hverdagen er rolig."
        ),
    )["persona"]

    parsed = parse_descriptions(json.dumps({"persona": text}), context, attributes())

    assert parsed.persona == text
