"""Contextual tests for the generation-contract v2 validators."""

import json
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from danish_personas.generation.models import GeneratedAttributes
from danish_personas.generation.validation import (
    EDUCATION_DANISH,
    parse_attributes,
    parse_descriptions,
)

CATEGORIES_PATH = Path(__file__).parents[2] / "config" / "categories.yaml"
CATEGORIES = yaml.safe_load(CATEGORIES_PATH.read_text(encoding="utf-8"))
EDUCATION_POOLING_VALUES = tuple(
    dict.fromkeys(CATEGORIES["education_pooling"].values())
)
EXPECTED_EDUCATION_RENDERINGS = {
    "primary": "grundskole",
    "secondary_or_vocational": "ungdomsuddannelse eller erhvervsuddannelse",
    "higher_education": "videregående uddannelse",
    "not_stated": "uddannelse ikke oplyst",
}


@pytest.mark.parametrize(
    "field",
    [
        "professional_persona",
        "sports_persona",
        "arts_persona",
        "travel_persona",
        "culinary_persona",
    ],
)
def test_all_six_description_fields_must_be_distinct(field: str) -> None:
    """Exact normalised duplicate text is rejected for every field."""
    context = demographic()
    text = descriptions(context=context)
    text[field] = text["persona"]
    with pytest.raises(ValueError, match="exact duplicates"):
        parse_descriptions(json.dumps(text), context, attributes())


def attributes(*, job_title: str | None = "forretningsspecialist") -> dict[str, object]:
    """Return a valid first-stage payload."""
    return {
        "cultural_context": "Personen har en dansk hverdag og deltager i fællesskaber.",
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
    job_title: str | None = "forretningsspecialist",
) -> dict[str, object]:
    """Return a small valid demographic context for validator tests."""
    eligible = status == "employed"
    return {
        "persona_id": "persona-1",
        "age": 35,
        "sex": sex,
        "municipality": "København",
        "origin_country": "Danmark",
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


def descriptions(
    *, context: Mapping[str, object] | None = None, interests: list[str] | None = None
) -> dict[str, str]:
    """Return six distinct Danish fields with grounded summary facts."""
    context = context or demographic()
    interests = interests or ["at læse", "musik", "brætspil"]
    education = EXPECTED_EDUCATION_RENDERINGS[str(context["education_level"])]
    sex = "kvinde" if context["sex"] == "female" else "mand"
    status = "arbejder som forretningsspecialist"
    if context["labour_market_status"] != "employed":
        status = "er pensionist"
    persona = (
        f"Personen er {context['age']} år og {sex} fra {context['municipality']} "
        f"i {context['origin_country']} med en {education} og {status}. "
        f"Personen kan være rolig og holder af {', '.join(interests)}."
    )
    return {
        "professional_persona": (
            "På hverdage kan personen lide tydelige opgaver og roligt "
            "samarbejde med andre."
        ),
        "sports_persona": (
            "Motion kan være en enkel aktivitet, hvor personen ofte finder "
            "tid til bevægelse."
        ),
        "arts_persona": (
            "Kunst og musik kan give personen en rolig stund med plads til fordybelse."
        ),
        "travel_persona": (
            "På rejser kan personen foretrække en enkel plan og tid til nye oplevelser."
        ),
        "culinary_persona": (
            "I køkkenet kan personen lide enkle retter og hyggelige måltider "
            "i hverdagen."
        ),
        "persona": persona,
    }


def test_appearance_boundary_does_not_reject_hardt() -> None:
    """A longer word containing hår is not an appearance claim."""
    context = demographic(status="retired", job_title=None)
    text = descriptions(context=context, interests=["at læse", "musik"])
    text["professional_persona"] += " Personen arbejder hårdt."
    parse_descriptions(json.dumps(text), context, attributes(job_title=None))


def test_current_title_or_non_employee_status_is_required() -> None:
    """The summary names the exact current title or canonical status."""
    context = demographic()
    text = descriptions(context=context)
    text["persona"] = text["persona"].replace("forretningsspecialist", "analytiker")
    with pytest.raises(ValueError, match="current work status"):
        parse_descriptions(json.dumps(text), context, attributes())


def test_education_renderings_cover_phase_two_pool_domain() -> None:
    """The validator covers exactly the categories emitted by Phase 2 pooling."""
    assert set(EXPECTED_EDUCATION_RENDERINGS) == set(EDUCATION_POOLING_VALUES)
    assert EDUCATION_DANISH == EXPECTED_EDUCATION_RENDERINGS


@pytest.mark.parametrize(
    "claim",
    [
        "familie",
        "børn",
        "børnenes",
        "barnet",
        "ansigter",
        "ansigtstræk",
        "hårene",
        "hudfarve",
        "arbejdede tidligere",
        "udseende",
    ],
)
def test_family_former_work_and_appearance_claims_fail(claim: str) -> None:
    """Unsupported family, former-work, and appearance claims are rejected."""
    context = demographic(status="retired", job_title=None)
    text = descriptions(context=context, interests=["at læse", "musik"])
    text["persona"] = text["persona"].replace(
        "Personen kan være rolig", f"Personen kan være rolig og {claim}"
    )
    with pytest.raises(ValueError):
        parse_descriptions(json.dumps(text), context, attributes(job_title=None))


@pytest.mark.parametrize("eligible", [True, False])
def test_job_title_eligibility_is_contextual(eligible: bool) -> None:
    """Only eligible job-function contexts may contain a title."""
    context = demographic(
        status="employed" if eligible else "retired",
        job_title="forretningsspecialist" if eligible else None,
    )
    content = attributes(job_title="forretningsspecialist" if eligible else None)
    assert (
        parse_attributes(json.dumps(content), context).job_title == content["job_title"]
    )

    invalid = attributes(job_title=None if eligible else "forretningsspecialist")
    with pytest.raises(ValueError, match="title|job_title"):
        parse_attributes(json.dumps(invalid), context)


@pytest.mark.parametrize("punctuation", ["- ", "1. ", "[", ";"])
def test_list_syntax_is_rejected(punctuation: str) -> None:
    """The summary must remain prose rather than list syntax."""
    context = demographic()
    text = descriptions(context=context)
    text["persona"] = punctuation + text["persona"]
    with pytest.raises(ValueError, match="prose"):
        parse_descriptions(json.dumps(text), context, attributes())


def test_ocean_tendency_requires_compatibility_and_hedging() -> None:
    """A compatible tendency is allowed only with cautious wording."""
    context = demographic()
    text = descriptions(context=context)
    text["persona"] = text["persona"].replace("kan være rolig", "er altid rolig")
    with pytest.raises(ValueError, match="cautious|hedged"):
        parse_descriptions(json.dumps(text), context, attributes())

    context["openness_label"] = "high"
    text["persona"] = text["persona"].replace("er altid rolig", "kan være praktisk")
    with pytest.raises(ValueError, match="compatible"):
        parse_descriptions(json.dumps(text), context, attributes())


def test_ocean_tendency_requires_literal_lexicon_terms() -> None:
    """Inflected or otherwise nonliteral terms do not satisfy the contract."""
    context = demographic()
    text = descriptions(context=context)
    text["persona"] = text["persona"].replace("kan være rolig", "kan være rolighed")

    with pytest.raises(ValueError, match="1-2 compatible"):
        parse_descriptions(json.dumps(text), context, attributes())


@pytest.mark.parametrize("count", [1, 4])
def test_one_or_four_literal_interests_fail(count: int) -> None:
    """Too few or too many literal interests fail the contract."""
    context = demographic()
    selected = ["at læse", "musik", "brætspil", "vandring"][:count]
    generated = attributes()
    generated["hobbies_and_interests"] = selected
    with pytest.raises(ValueError, match="interests|at least"):
        parse_descriptions(
            json.dumps(descriptions(context=context, interests=selected)),
            context,
            generated,
        )


@pytest.mark.parametrize("education", EDUCATION_POOLING_VALUES)
def test_real_sample_education_values_parse_contextually(education: str) -> None:
    """Every pooled Phase-2 education value grounds a realistic persona summary."""
    context = demographic(education_level=education, sex="male")
    result = parse_descriptions(
        json.dumps(descriptions(context=context)),
        context,
        GeneratedAttributes.model_validate(attributes()),
    )
    assert EXPECTED_EDUCATION_RENDERINGS[education] in result.persona


@pytest.mark.parametrize("missing", ["age", "sex", "municipality", "origin_country"])
def test_required_demographic_facts_are_literal(missing: str) -> None:
    """Changing any required demographic fact is rejected."""
    context = demographic()
    text = descriptions(context=context)
    replacements = {
        "age": "34 år",
        "sex": "mand",
        "municipality": "Roskilde",
        "origin_country": "Sverige",
    }
    original = "kvinde" if missing == "sex" else str(context[missing])
    text["persona"] = text["persona"].replace(original, replacements[missing])
    with pytest.raises(
        ValueError, match=missing if missing != "origin_country" else "origin"
    ):
        parse_descriptions(json.dumps(text), context, attributes())


def test_status_ten_allows_only_grounded_phrase_in_persona() -> None:
    """Status 10 permits its canonical phrase, but not related claims."""
    context = demographic(status="employed", job_title=None)
    context.update(
        detailed_status_code="10",
        job_function=None,
        job_function_code=None,
        job_function_resolution="not_applicable",
    )
    text = descriptions(context=context, interests=["at læse", "musik"])
    text["persona"] = text["persona"].replace(
        "arbejder som forretningsspecialist", "er medarbejdende ægtefælle"
    )
    assert (
        parse_descriptions(
            json.dumps(text), context, attributes(job_title=None)
        ).persona
        == text["persona"]
    )

    for field in ("professional_persona", "sports_persona"):
        rejected = dict(text)
        rejected[field] = "Personen er medarbejdende ægtefælle i hverdagen."
        with pytest.raises(ValueError):
            parse_descriptions(
                json.dumps(rejected), context, attributes(job_title=None)
            )

    rejected = dict(text)
    rejected["persona"] += " Personen har børnene med sig."
    with pytest.raises(ValueError):
        parse_descriptions(json.dumps(rejected), context, attributes(job_title=None))


@pytest.mark.parametrize("count", [2, 3])
def test_two_or_three_literal_interests_pass(count: int) -> None:
    """The summary embeds exactly two or three complete interest values."""
    context = demographic()
    selected = ["at læse", "musik", "brætspil"][:count]
    assert parse_descriptions(
        json.dumps(descriptions(context=context, interests=selected)),
        context,
        attributes(),
    )


def test_valid_v2_attributes_and_description() -> None:
    """A complete employee record passes both contextual stages."""
    context = demographic()
    parsed = parse_attributes(json.dumps(attributes()), context)
    assert parse_descriptions(
        json.dumps(descriptions(context=context)), context, parsed
    )
