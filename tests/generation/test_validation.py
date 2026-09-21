"""Behavioural tests for the generation-contract v4 validators."""

# Test functions intentionally omit repetitive docstrings.
# ruff: noqa: D103, E501

import json
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

import danish_personas.generation.validation as validation_module
from danish_personas.generation.grounding import build_persona_grounding_facts
from danish_personas.generation.job_titles import load_job_title_mapping
from danish_personas.generation.validation import (
    EDUCATION_DANISH,
    parse_attributes,
    parse_descriptions,
)

ROOT = Path(__file__).parents[2]
CATEGORIES = yaml.safe_load(
    (ROOT / "config" / "categories.yaml").read_text(encoding="utf-8")
)
EDUCATION_POOLING_VALUES = tuple(
    dict.fromkeys(CATEGORIES["education_pooling"].values())
)
JOB_TITLE = "forretningsspecialist"


@pytest.mark.parametrize(
    "status", ["unemployed", "student", "retired", "other", "outside_labour_force"]
)
def test_all_current_statuses_are_grounded(status: str) -> None:
    context = demographic(status=status, job_title=None)
    generated = attributes(job_title=None)
    expected = {
        "unemployed": "ledig",
        "student": "studerende",
        "retired": "pensionist",
        "other": "uden for arbejdsmarkedet",
        "outside_labour_force": "uden for arbejdsmarkedet",
    }[status]
    assert parse_descriptions(
        json.dumps(persona(context=context, employment=f"er {expected}")),
        context,
        generated,
    )


def attributes(*, job_title: str | None = JOB_TITLE) -> dict[str, object]:
    return {
        "cultural_context": "En almindelig dansk hverdag med plads til fællesskab.",
        "skills_and_expertise": ["planlægning", "samarbejde", "formidling"],
        "hobbies_and_interests": ["at læse", "musik", "brætspil"],
        "career_goals_and_ambitions": None,
        "job_title": job_title,
    }


def demographic(
    *,
    education_level: str = "higher_education",
    sex: str = "female",
    status: str = "employed",
    job_title: str | None = JOB_TITLE,
) -> dict[str, object]:
    eligible = status == "employed" and job_title is not None
    return {
        "persona_id": "persona-1",
        "age": 35,
        "sex": sex,
        "municipality": "København",
        "origin_country_code": "5100",
        "origin_country": "Denmark",
        "origin_country_da": "Danmark",
        "education_level": education_level,
        "labour_market_status": status,
        "job_function": "24 Business and administration professionals"
        if eligible
        else None,
        "job_function_resolution": "lons20_sex_marginal"
        if eligible
        else "not_applicable",
        "openness_score": 50.0,
        "openness_label": "average",
        "conscientiousness_score": 50.0,
        "conscientiousness_label": "average",
        "extraversion_score": 50.0,
        "extraversion_label": "average",
        "agreeableness_score": 50.0,
        "agreeableness_label": "average",
        "neuroticism_score": 50.0,
        "neuroticism_label": "average",
        "job_title": job_title,
    }


def persona(
    *,
    context: Mapping[str, object] | None = None,
    education: str | None = None,
    employment: str | None = None,
    pronoun: str | None = None,
    extra: str = "Den syntetiske hverdag rummer rolige rutiner og små oplevelser.",
) -> dict[str, str]:
    context = context or demographic()
    pronoun = pronoun or ("hun" if context["sex"] == "female" else "han")
    education = (
        education
        or {
            "primary": "har gået i grundskolen",
            "secondary_or_vocational": "har en ungdomsuddannelse eller erhvervsuddannelse",
            "higher_education": "har en videregående uddannelse",
            "not_stated": "har en uddannelse, der ikke er oplyst",
        }[str(context["education_level"])]
    )
    employment = employment or (
        f"arbejder som {JOB_TITLE}" if context["job_title"] else "er pensionist"
    )
    text = (
        f"{pronoun.capitalize()} er {context['age']} år og bor i "
        f"{context['municipality']}. {pronoun.capitalize()} kommer fra "
        f"{context['origin_country_da']} og {education}. {pronoun.capitalize()} "
        f"{employment}, og {extra[0].lower() + extra[1:]}"
    )
    return {"persona": text}


@pytest.mark.parametrize(
    ("sex", "opposing"),
    [
        ("female", "han"),
        ("female", "ham"),
        ("female", "hans"),
        ("male", "hun"),
        ("male", "hende"),
        ("male", "hendes"),
    ],
)
def test_all_danish_pronoun_paradigms_are_consistent(sex: str, opposing: str) -> None:
    context = demographic(sex=sex)
    payload = persona(context=context)
    payload["persona"] = payload["persona"][:-1] + f" {opposing.capitalize()} læser."
    with pytest.raises(ValueError, match="pronoun"):
        parse_descriptions(json.dumps(payload), context, attributes())


@pytest.mark.parametrize(
    "noun",
    [
        "mand",
        "manden",
        "mandens",
        "mands",
        "mænd",
        "mændene",
        "mændenes",
        "kvinde",
        "kvinden",
        "kvindens",
        "kvinder",
        "kvinderne",
        "kvindernes",
    ],
)
def test_all_sex_noun_inflections_are_rejected(noun: str) -> None:
    context = demographic()
    payload = persona(context=context)
    payload["persona"] = payload["persona"][:-1] + f" {noun} står her."
    with pytest.raises(ValueError, match="statistical sex"):
        parse_descriptions(json.dumps(payload), context, attributes())


@pytest.mark.parametrize(
    "term", ["vedkommende", "personen", "kan være", "ungdoms- eller erhvervsuddannelse"]
)
def test_banned_persona_terms_are_case_insensitive_and_token_bounded(term: str) -> None:
    context = demographic()
    text = persona(context=context)["persona"]
    payload = {"persona": text[:-1] + f" {term.upper()}."}
    with pytest.raises(ValueError, match="prohibited|contract"):
        parse_descriptions(json.dumps(payload), context, attributes())


def test_banned_terms_do_not_match_larger_words() -> None:
    context = demographic()
    text = persona(context=context)["persona"][:-1] + " Personenhed er ikke nævnt."
    # ``personen`` is bounded: a larger word is not a generic subject substitute.
    assert parse_descriptions(json.dumps({"persona": text}), context, attributes())


@pytest.mark.parametrize(
    "education_level",
    ["primary", "secondary_or_vocational", "higher_education", "not_stated"],
)
def test_broad_education_wording_preserves_level(education_level: str) -> None:
    context = demographic(education_level=education_level)
    parsed = parse_attributes(json.dumps(attributes()), context)
    assert parse_descriptions(json.dumps(persona(context=context)), context, parsed)


def test_cautious_ocean_paraphrase_is_allowed() -> None:
    context = demographic()
    payload = persona(context=context)
    payload["persona"] = payload["persona"][:-1] + " Hun er ofte social."
    assert parse_descriptions(json.dumps(payload), context, attributes())


@pytest.mark.parametrize(
    "term",
    [
        "vedkommendes",
        "personen",
        "personens",
        "personer",
        "personerne",
        "kan måske være",
        "kan ofte godt være",
        "ungdoms - eller erhvervsuddannelse",
        "ungdoms–eller erhvervsuddannelse",
    ],
)
def test_contract_variants_are_rejected(term: str) -> None:
    context = demographic()
    payload = persona(context=context)
    payload["persona"] = payload["persona"][:-1] + f" {term}."
    with pytest.raises(ValueError, match="prohibited|contract"):
        parse_descriptions(json.dumps(payload), context, attributes())


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


def test_description_schema_rejects_removed_fields() -> None:
    payload = persona(context=demographic())
    payload["removed_field"] = "ikke en aktiv kontrakt"
    with pytest.raises(ValueError, match="Extra inputs"):
        parse_descriptions(json.dumps(payload), demographic(), attributes())


def test_former_work_and_list_syntax_are_rejected() -> None:
    context = demographic()
    payload = persona(context=context)
    payload["persona"] = payload["persona"].replace(
        "arbejder som", "arbejdede tidligere som"
    )
    with pytest.raises(ValueError, match="former|current work"):
        parse_descriptions(json.dumps(payload), context, attributes())

    payload = persona(context=context)
    payload["persona"] = "- " + payload["persona"]
    with pytest.raises(ValueError, match="prose"):
        parse_descriptions(json.dumps(payload), context, attributes())


def test_job_title_eligibility_and_unsafe_titles() -> None:
    context = demographic()
    assert parse_attributes(json.dumps(attributes()), context).job_title == JOB_TITLE
    with pytest.raises(ValueError, match="allowlisted reviewed title"):
        parse_attributes(json.dumps(attributes(job_title="analytiker")), context)

    retired = demographic(status="retired", job_title=None)
    with pytest.raises(ValueError, match="null job_title"):
        parse_attributes(json.dumps(attributes()), retired)

    for title in ("sundhedsprofessionel med angst", "- forretningsspecialist"):
        with pytest.raises(ValueError, match="job_title"):
            parse_attributes(json.dumps(attributes(job_title=title)), context)


def test_job_title_must_be_allowlisted() -> None:
    context = demographic()
    with pytest.raises(ValueError, match="allowlisted reviewed title"):
        parse_attributes(json.dumps(attributes(job_title="opfinder")), context)


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


def test_ocean_terms_are_checked_independently_and_hedged() -> None:
    context = demographic()
    payload = persona(context=context)
    payload["persona"] = payload["persona"][:-1] + " Hun er social."
    with pytest.raises(ValueError, match="cautious"):
        parse_descriptions(json.dumps(payload), context, attributes())

    context["openness_label"] = "high"
    payload = persona(context=context)
    payload["persona"] = payload["persona"][:-1] + " Hun er praktisk."
    with pytest.raises(ValueError, match="incompatible"):
        parse_descriptions(json.dumps(payload), context, attributes())


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


def test_pronoun_must_remain_consistent() -> None:
    context = demographic(sex="female")
    text = persona(context=context)["persona"][:-1] + " Han læser gerne."
    with pytest.raises(ValueError, match="pronoun"):
        parse_descriptions(json.dumps({"persona": text}), context, attributes())


def test_schema_contains_only_persona() -> None:
    assert set(validation_module.PersonaDescriptions.model_fields) == {"persona"}


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
    "phrase",
    [
        "oprindelsesland",
        "oprindelsesetiket",
        "brede uddannelsesbaggrund",
        "uddannelsesniveau",
        "aktuelle arbejdsforhold",
    ],
)
def test_technical_grounding_wording_is_rejected(phrase: str) -> None:
    context = demographic()
    payload = persona(context=context)
    payload["persona"] = payload["persona"][:-1] + f" {phrase}."
    with pytest.raises(ValueError, match="redundant or technical"):
        parse_descriptions(json.dumps(payload), context, attributes())


def test_unsupported_family_and_appearance_claims_are_rejected() -> None:
    context = demographic()
    for claim in ("Hun har børn.", "Hun beskriver sin højde."):
        text = persona(context=context)["persona"][:-1] + " " + claim
        with pytest.raises(ValueError, match="unsupported"):
            parse_descriptions(json.dumps({"persona": text}), context, attributes())


def test_valid_v4_persona_allows_no_interest_or_tendency_copy() -> None:
    context = demographic()
    parsed_attributes = parse_attributes(json.dumps(attributes()), context)
    result = parse_descriptions(
        json.dumps(persona(context=context, extra="Hun holder af hverdage med ro.")),
        context,
        parsed_attributes,
    )
    assert result.persona.startswith("Hun er 35 år")


def test_validator_and_education_exports_are_current() -> None:
    assert validation_module.VALIDATOR_VERSION == "persona-safety-v16"
    assert set(EDUCATION_DANISH) == set(EDUCATION_POOLING_VALUES)
    assert load_job_title_mapping().version == 1
