"""Objective grounding tests for generated persona descriptions."""

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


def test_missing_age_is_rejected() -> None:
    context = demographic()
    payload = persona(context=context)
    payload["persona"] = payload["persona"].replace("35 år", "midt i trediverne")
    with pytest.raises(ValueError, match="pronoun or age"):
        parse_descriptions(json.dumps(payload), context, attributes())


@pytest.mark.parametrize(
    ("original", "replacement", "message"),
    [
        ("bor i København", "bor i Roskilde", "municipality"),
        ("kommer fra Danmark", "kommer fra Sverige", "origin"),
        ("har en videregående uddannelse", "har en ungdomsuddannelse", "education"),
        (
            "arbejder som forretningsspecialist",
            "arbejder som analytiker",
            "work status",
        ),
    ],
)
def test_missing_supplied_fact_is_rejected(
    original: str, replacement: str, message: str
) -> None:
    context = demographic()
    payload = persona(context=context)
    payload["persona"] = payload["persona"].replace(original, replacement)
    with pytest.raises(ValueError, match=message):
        parse_descriptions(json.dumps(payload), context, attributes())


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


def test_origin_label_is_not_used_to_infer_sensitive_detail() -> None:
    context = demographic()
    text = persona(context=context)["persona"][:-1] + " Hun omtaler religion."
    with pytest.raises(ValueError, match="sensitive"):
        parse_descriptions(json.dumps({"persona": text}), context, attributes())


def test_other_people_may_have_different_grounded_facts() -> None:
    context = demographic()
    text = persona(
        context=context,
        extra=(
            "Hendes partner er 41 år, kommer fra Sverige og arbejder som analytiker "
            "i Roskilde. Han har en ungdomsuddannelse, mens deres datter er 9 år."
        ),
    )["persona"]

    parsed = parse_descriptions(json.dumps({"persona": text}), context, attributes())

    assert parsed.persona == text


def test_persona_rejects_a_bare_one_sentence_fact_list() -> None:
    context = demographic()
    text = (
        "Hun er 35 år, bor i København, kommer fra Danmark, har en videregående "
        "uddannelse og arbejder som forretningsspecialist."
    )
    with pytest.raises(ValueError, match="at least 300 characters"):
        parse_descriptions(json.dumps({"persona": text}), context, attributes())


def test_secondary_persona_must_choose_one_branch() -> None:
    context = demographic(education_level="secondary_or_vocational")
    payload = persona(context=context)
    payload["persona"] = payload["persona"].replace(
        "har en erhvervsuddannelse", "har en ungdomsuddannelse og en erhvervsuddannelse"
    )
    with pytest.raises(ValueError, match="one specific education branch"):
        parse_descriptions(json.dumps(payload), context, attributes())


def test_secondary_wording_does_not_force_legacy_banned_phrase() -> None:
    facts = build_persona_grounding_facts(
        demographic(education_level="secondary_or_vocational"), attributes()
    )
    assert facts.education == "har en ungdomsuddannelse eller erhvervsuddannelse"
    assert "ungdoms- eller erhvervsuddannelse" not in facts.education


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
