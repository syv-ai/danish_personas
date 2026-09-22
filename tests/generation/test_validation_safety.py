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
        " Hun hedder Maja og bor i Aarhus.",
        " Hun er ved navn Maja.",
        " Jeg er Maja.",
        " Hun er Maja.",
        " Han er Maja.",
        " Personaen er Maja.",
        " Personen er Maja. Personaens navn er Maja.",
        " Navnet er Maja.",
        " Hendes datter hedder Emma.",
        " Hendes datter hedder Emma og går til håndbold.",
        " Partneren hedder Lars.",
        " Partneren kaldes Lars.",
        " Hendes datter ved navn Emma går til håndbold.",
        " Partneren Lars bor i byen.",
        " Partneren Maja bor ofte i byen.",
        " Hendes datter er Emma.",
        " Maja er hendes kæreste.",
        " Hun bor sammen med sin kæreste Maja.",
        " Majas partner bor i byen.",
        " Maja's partner bor i byen.",
        " Maja’s partner bor i byen.",
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


@pytest.mark.parametrize(
    "origin", ["Danmark", "San Marino", "Serbien og Montenegro", "Europa uoplyst"]
)
def test_single_and_multiword_origin_labels_are_allowed_in_name_patterns(
    origin: str,
) -> None:
    context = demographic()
    context["origin_country_da"] = origin
    rendered_origin = origin[:1] + origin[1:].lower()
    text = persona(
        context=context,
        extra=(
            f" Hun er {rendered_origin}. {rendered_origin} er hendes kæreste. "
            f"{rendered_origin}s partner bor i byen."
        ),
    )["persona"]

    parsed = parse_descriptions(json.dumps({"persona": text}), context, attributes())

    assert parsed.persona == text


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


@pytest.mark.parametrize(
    "construction",
    [
        "Aarhus-Lars er 35 år.",
        "San Marino-Lars's partner bor i byen.",
        "San Marino-Lars’s partner bor i byen.",
        "San Marino-Lars´s partner bor i byen.",
    ],
)
def test_grounded_label_name_extensions_are_rejected_in_attributes_and_persona(
    construction: str,
) -> None:
    context = demographic()
    context["municipality"] = "Aarhus"
    context["origin_country_da"] = "San Marino"

    attribute_payload = attributes()
    attribute_payload["cultural_context"] = (
        f"En almindelig dansk hverdag. {construction}"
    )
    with pytest.raises(ValueError, match="person name"):
        parse_attributes(json.dumps(attribute_payload), context)

    text = persona(context=context, extra=construction)["persona"]
    with pytest.raises(ValueError, match="person name"):
        parse_descriptions(json.dumps({"persona": text}), context, attributes())


@pytest.mark.parametrize("possessive", ["Aarhus's", "Aarhus’s", "Aarhus´s"])
def test_exact_grounded_label_possessives_are_allowed(possessive: str) -> None:
    context = demographic()
    context["municipality"] = "Aarhus"
    attribute_payload = attributes()
    attribute_payload["cultural_context"] = (
        f"En almindelig dansk hverdag. {possessive} partner bor i byen."
    )
    parse_attributes(json.dumps(attribute_payload), context)

    text = persona(context=context, extra=f" {possessive} partner bor i byen.")[
        "persona"
    ]
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
