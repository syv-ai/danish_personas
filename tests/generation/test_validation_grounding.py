"""Grounding and contradiction tests for generated persona descriptions."""

# Test functions intentionally omit repetitive docstrings.
# ruff: noqa: D103

import json

import pytest
from validation_test_helpers import attributes, demographic, persona

from danish_personas.generation.grounding import build_persona_grounding_facts
from danish_personas.generation.validation import parse_attributes, parse_descriptions


@pytest.mark.parametrize(
    "education_level",
    ["primary", "secondary_or_vocational", "higher_education", "not_stated"],
)
def test_broad_education_wording_preserves_level(education_level: str) -> None:
    context = demographic(education_level=education_level)
    parsed = parse_attributes(json.dumps(attributes()), context)
    assert parse_descriptions(json.dumps(persona(context=context)), context, parsed)


def test_contradictory_age_is_rejected() -> None:
    context = demographic()
    payload = persona(context=context)
    payload["persona"] = payload["persona"].replace("35 år", "36 år")
    with pytest.raises(ValueError, match="pronoun.*age"):
        parse_descriptions(json.dumps(payload), context, attributes())


@pytest.mark.parametrize(
    ("original", "contradiction"),
    [
        ("arbejder som forretningsspecialist", "arbejder som analytiker"),
        ("arbejder som forretningsspecialist", "er studerende"),
    ],
)
def test_contradictory_current_work_is_rejected(
    original: str, contradiction: str
) -> None:
    context = demographic()
    payload = persona(context=context)
    payload["persona"] = payload["persona"].replace(original, contradiction)
    with pytest.raises(ValueError, match="work status"):
        parse_descriptions(json.dumps(payload), context, attributes())


@pytest.mark.parametrize(
    ("original", "contradiction", "message"),
    [
        ("bor i København", "bor i Roskilde, men nævner København", "municipality"),
        ("kommer fra Danmark", "kommer fra Sverige, men nævner Danmark", "origin"),
    ],
)
def test_contradictory_location_and_origin_are_rejected(
    original: str, contradiction: str, message: str
) -> None:
    context = demographic()
    payload = persona(context=context)
    payload["persona"] = payload["persona"].replace(original, contradiction)
    with pytest.raises(ValueError, match=message):
        parse_descriptions(json.dumps(payload), context, attributes())


@pytest.mark.parametrize(
    ("status", "rendering"),
    [
        ("unemployed", "ledig"),
        ("student", "studerende"),
        ("retired", "pensionist"),
        ("other", "uden for arbejdsmarkedet"),
        ("outside_labour_force", "uden for arbejdsmarkedet"),
    ],
    ids=["unemployed", "student", "retired", "other", "outside-labour-force"],
)
def test_current_status_renderings_are_grounded(status: str, rendering: str) -> None:
    context = demographic(status=status, job_title=None)
    generated = attributes(job_title=None)
    assert parse_descriptions(
        json.dumps(persona(context=context, employment=f"er {rendering}")),
        context,
        generated,
    )


def test_harmless_consistent_everyday_detail_is_allowed() -> None:
    context = demographic()
    text = persona(context=context)["persona"] + " Hun holder af stille morgener."
    assert parse_descriptions(json.dumps({"persona": text}), context, attributes())


@pytest.mark.parametrize(
    "contradiction",
    [
        "Hun er fra Sverige.",
        "Hun kommer fra Sverige.",
        "Hun er bosat i Roskilde.",
        "Hun bor i Roskilde.",
        "Hun har en ungdomsuddannelse.",
        "Hun er studerende.",
        "Hun arbejder som administrativ specialist.",
    ],
)
def test_later_contradictory_grounding_cannot_be_hidden_by_correct_fact(
    contradiction: str,
) -> None:
    context = demographic()
    text = persona(context=context)["persona"] + " " + contradiction
    with pytest.raises(ValueError):
        parse_descriptions(json.dumps({"persona": text}), context, attributes())


@pytest.mark.parametrize(
    ("original", "contradiction", "message"),
    [
        ("Hun er 35 år", "Hun er ikke 35 år", "pronoun or age"),
        ("bor i København", "bor ikke i København", "municipality"),
        ("kommer fra Danmark", "kommer ikke fra Danmark", "origin"),
        (
            "har en videregående uddannelse",
            "har ikke en videregående uddannelse",
            "education",
        ),
        (
            "arbejder som forretningsspecialist",
            "arbejder ikke som forretningsspecialist",
            "current work status",
        ),
    ],
)
def test_negated_grounding_facts_are_rejected(
    original: str, contradiction: str, message: str
) -> None:
    context = demographic()
    payload = persona(context=context)
    payload["persona"] = payload["persona"].replace(original, contradiction)
    with pytest.raises(ValueError, match=message):
        parse_descriptions(json.dumps(payload), context, attributes())


def test_non_employee_status_is_grounded() -> None:
    context = demographic(status="retired", job_title=None)
    generated = attributes(job_title=None)
    text = persona(context=context, employment="er pensionist")
    assert parse_descriptions(json.dumps(text), context, generated)


def test_origin_label_is_not_used_to_infer_sensitive_detail() -> None:
    context = demographic()
    text = persona(context=context)["persona"][:-1] + " Hun omtaler religion."
    with pytest.raises(ValueError, match="sensitive"):
        parse_descriptions(json.dumps({"persona": text}), context, attributes())


def test_persona_can_be_one_sentence_without_interest_or_personality_counts() -> None:
    context = demographic()
    text = (
        "Hun er 35 år, bor i København, kommer fra Danmark, har en videregående "
        "uddannelse og arbejder som forretningsspecialist."
    )
    assert parse_descriptions(json.dumps({"persona": text}), context, attributes())


def test_personality_hedge_does_not_leak_across_assertions() -> None:
    context = demographic()
    text = persona(context=context)["persona"] + " Hun kan være rolig, men er social."
    with pytest.raises(ValueError, match="cautious|generic|contract"):
        parse_descriptions(json.dumps({"persona": text}), context, attributes())


def test_secondary_wording_does_not_force_legacy_banned_phrase() -> None:
    facts = build_persona_grounding_facts(
        demographic(education_level="secondary_or_vocational"), attributes()
    )
    assert facts.education == "har en ungdomsuddannelse eller erhvervsuddannelse"
    assert "ungdoms- eller erhvervsuddannelse" not in facts.education


def test_supplied_facts_can_be_naturally_paraphrased() -> None:
    context = demographic()
    text = (
        "Hun er 35 år og har base i København. Hun har Danmark som leveret "
        "oprindelsesoplysning og har læst videregående uddannelse. Hun arbejder "
        "som forretningsspecialist. Hun holder af en rolig hverdag."
    )
    assert parse_descriptions(json.dumps({"persona": text}), context, attributes())


@pytest.mark.parametrize(
    "trailing",
    [
        "Hun arbejder som forretningsspecialist, men ikke længere.",
        "Hun kommer fra Danmark, men ikke længere.",
        "Hun har en videregående uddannelse, men ikke længere.",
    ],
)
def test_trailing_ikke_laengere_rejects_current_grounding(trailing: str) -> None:
    context = demographic()
    text = persona(context=context)["persona"] + " " + trailing
    with pytest.raises(ValueError):
        parse_descriptions(json.dumps({"persona": text}), context, attributes())
