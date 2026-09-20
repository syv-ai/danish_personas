"""Contextual tests for the generation-contract v3 validators."""

import json
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml
from lingua import Language

import danish_personas.generation.validation as validation_module
from danish_personas.generation.grounding import build_persona_grounding_facts
from danish_personas.generation.job_titles import load_job_title_mapping
from danish_personas.generation.models import GeneratedAttributes
from danish_personas.generation.validation import (
    EDUCATION_DANISH,
    parse_attributes,
    parse_descriptions,
)

ROOT = Path(__file__).parents[2]
CATEGORIES_PATH = ROOT / "config" / "categories.yaml"
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
EXPECTED_EDUCATION_CLAUSES = {
    "primary": "har ingen uddannelse efter folkeskolen",
    "secondary_or_vocational": "har en ungdoms- eller erhvervsuddannelse",
    "higher_education": "har en videregående uddannelse",
    "not_stated": "uddannelsen er ikke oplyst",
}
JOB_TITLE_CASES = tuple(
    (code, entry.label, title)
    for code, entry in load_job_title_mapping().job_functions.items()
    for title in entry.titles
)


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


def descriptions(
    *, context: Mapping[str, object] | None = None, interests: list[str] | None = None
) -> dict[str, str]:
    """Return six distinct Danish fields with grounded summary facts."""
    context = context or demographic()
    interests = interests or ["at læse", "musik", "brætspil"]
    education = EXPECTED_EDUCATION_CLAUSES[str(context["education_level"])]
    pronoun = "hun" if context["sex"] == "female" else "han"
    status = "arbejder som forretningsspecialist"
    if context.get("detailed_status_code") == "05":
        status = "er selvstændig"
    elif context["labour_market_status"] != "employed":
        status = "er pensionist"
    persona = (
        f"{pronoun.capitalize()} er {context['age']} år, bor i "
        f"{context['municipality']}, kommer fra {context['origin_country_da']}, "
        f"{education} og {status}. {pronoun.capitalize()} kan være rolig og "
        f"holder af {', '.join(interests)}."
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


@pytest.mark.parametrize(
    ("field", "rejected"),
    [
        (
            "cultural_context",
            "SENTINEL_REJECTED_TEXT is deliberately written in English.",
        ),
        (
            "career_goals_and_ambitions",
            "SENTINEL_REJECTED_TEXT describes an English career ambition.",
        ),
        ("job_title", "SENTINEL_REJECTED_TITLE"),
    ],
)
def test_attribute_diagnostics_identify_field_without_content(
    field: str, rejected: str
) -> None:
    """Attribute failures identify their field without exposing rejected text."""
    context = demographic()
    payload = attributes()
    payload[field] = rejected
    with pytest.raises(ValueError) as error:
        parse_attributes(json.dumps(payload), context)

    message = str(error.value)
    assert message.startswith(f"{field}:")
    assert rejected not in message


@pytest.mark.parametrize("field", ["skills_and_expertise", "hobbies_and_interests"])
def test_attribute_list_diagnostics_identify_category_without_content(
    field: str,
) -> None:
    """List validation failures identify the response list without its values."""
    context = demographic()
    payload = attributes()
    payload[field] = [
        "SENTINEL_REJECTED_TEXT one",
        "SENTINEL_REJECTED_TEXT two",
        "SENTINEL_REJECTED_TEXT three",
    ]
    with pytest.raises(ValueError) as error:
        parse_attributes(json.dumps(payload), context)

    message = str(error.value)
    assert message.startswith(f"{field}:")
    assert "natural Danish" in message
    assert "SENTINEL_REJECTED_TEXT" not in message


def test_benign_interests_are_activities_or_topics() -> None:
    """Ordinary activity and topic interests remain valid."""
    payload = attributes()
    payload["hobbies_and_interests"] = ["at fotografere", "musik", "brætspil"]

    parse_attributes(json.dumps(payload), demographic())


def test_current_title_or_non_employee_status_is_required() -> None:
    """The summary names the exact current title or canonical status."""
    context = demographic()
    text = descriptions(context=context)
    text["persona"] = text["persona"].replace("forretningsspecialist", "analytiker")
    with pytest.raises(ValueError, match="current work status"):
        parse_descriptions(json.dumps(text), context, attributes())


def test_danish_prompts_use_skema_not_schema() -> None:
    """Danish provider instructions consistently use the Danish word skema."""
    for prompt in ("attributes-da.md", "personas-da.md"):
        content = (ROOT / "config" / "prompts" / prompt).read_text(encoding="utf-8")
        assert "skema" in content.casefold()
        assert "schema" not in content.casefold()


@pytest.mark.parametrize(
    "field",
    [
        "professional_persona",
        "sports_persona",
        "arts_persona",
        "travel_persona",
        "culinary_persona",
        "persona",
    ],
)
def test_description_diagnostics_identify_field_without_content(field: str) -> None:
    """Description failures identify every response field without its text."""
    context = demographic()
    payload = descriptions(context=context)
    rejected = (
        "SENTINEL_REJECTED_TEXT is deliberately written in English and fails "
        "the Danish language validation."
    )
    payload[field] = rejected
    with pytest.raises(ValueError) as error:
        parse_descriptions(json.dumps(payload), context, attributes())

    message = str(error.value)
    assert message.startswith(f"{field}:")
    assert "natural Danish" in message
    assert "SENTINEL_REJECTED_TEXT" not in message


def test_education_renderings_cover_phase_two_pool_domain() -> None:
    """The validator covers exactly the categories emitted by Phase 2 pooling."""
    assert set(EXPECTED_EDUCATION_RENDERINGS) == set(EDUCATION_POOLING_VALUES)
    assert EDUCATION_DANISH == EXPECTED_EDUCATION_RENDERINGS


@pytest.mark.parametrize(("code", "label", "title"), JOB_TITLE_CASES)
def test_every_reviewed_job_title_ignores_language_detection(
    code: str, label: str, title: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every reviewed title passes even when Lingua labels it non-Danish."""
    context = demographic()
    context["job_function_code"] = code
    context["job_function"] = label
    monkeypatch.setattr(
        validation_module, "LANGUAGE_DETECTOR", _RejectTitleLanguageDetector(title)
    )

    payload = attributes(job_title=title)
    assert parse_attributes(json.dumps(payload), context).job_title == title


class _RejectTitleLanguageDetector:
    """Return a non-Danish result only for the title under test."""

    def __init__(self, title: str) -> None:
        self.title = title

    def detect_language_of(self, text: str) -> Language:
        return Language.ENGLISH if text == self.title else Language.DANISH


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
        "Hun kan være rolig", f"Hun kan være rolig og {claim}"
    )
    with pytest.raises(ValueError):
        parse_descriptions(json.dumps(text), context, attributes(job_title=None))


@pytest.mark.parametrize(
    ("sentence", "passes"),
    (
        ("Hun kan være rolig. Musik fylder i fritiden sammen med brætspil.", True),
        ("Hun kan være rolig og holder af Musik og brætspil.", False),
    ),
)
def test_interest_capitalisation_is_allowed_only_at_sentence_start(
    sentence: str, passes: bool
) -> None:
    """Copied interests preserve lowercase except at the start of a sentence."""
    context = demographic()
    payload = descriptions(context=context, interests=["musik", "brætspil"])
    first_sentence = payload["persona"].split(". ", maxsplit=1)[0]
    payload["persona"] = f"{first_sentence}. {sentence}"

    if passes:
        assert parse_descriptions(json.dumps(payload), context, attributes())
    else:
        with pytest.raises(ValueError, match="generated interests"):
            parse_descriptions(json.dumps(payload), context, attributes())


@pytest.mark.parametrize(
    "interest",
    [
        "kan være rolig",
        "rolig",
        "åben for nye ideer",
        "kan være rolig i naturen",
        "glad for det velkendte",
    ],
)
def test_interests_reject_ocean_terms_and_phrases(interest: str) -> None:
    """Interests cannot reserve or smuggle OCEAN language into the persona."""
    payload = attributes()
    payload["hobbies_and_interests"] = [interest, "musik", "brætspil"]

    with pytest.raises(ValueError, match="hobbies_and_interests"):
        parse_attributes(json.dumps(payload), demographic())


@pytest.mark.parametrize("interest", ("Musik", "AT læse", "brætspil.", "madlavning!"))
def test_interests_require_lowercase_phrases_without_terminal_punctuation(
    interest: str,
) -> None:
    """Stage-one interests remain lowercase phrases without sentence punctuation."""
    payload = attributes()
    payload["hobbies_and_interests"] = [interest, "musik", "brætspil"]

    with pytest.raises(ValueError, match="hobbies_and_interests"):
        parse_attributes(json.dumps(payload), demographic())


def test_interests_use_boundaries_for_ocean_terms() -> None:
    """A word containing an OCEAN term's letters is not itself a match."""
    payload = attributes()
    payload["hobbies_and_interests"] = ["roligere", "musik", "brætspil"]

    parse_attributes(json.dumps(payload), demographic())


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


@pytest.mark.parametrize(
    "title",
    [
        "invented specialist",
        "software engineer",
        "analytiker",
        "sundhedsprofessionel med angst",
        "- forretningsspecialist",
        "forretningsspecialist\nadministrativ specialist",
    ],
)
def test_job_title_rejects_unreviewed_or_unsafe_titles(title: str) -> None:
    """Invented, foreign, sensitive, and list-like titles remain invalid."""
    with pytest.raises(ValueError, match="job_title"):
        parse_attributes(json.dumps(attributes(job_title=title)), demographic())


@pytest.mark.parametrize("punctuation", ["- ", "1. ", "[", ";"])
def test_list_syntax_is_rejected(punctuation: str) -> None:
    """The summary must remain prose rather than list syntax."""
    context = demographic()
    text = descriptions(context=context)
    text["persona"] = punctuation + text["persona"]
    with pytest.raises(ValueError, match="prose"):
        parse_descriptions(json.dumps(text), context, attributes())


def test_ocean_tendency_requires_compatible_complete_phrase() -> None:
    """A tendency is allowed only when its supplied phrase is copied exactly."""
    context = demographic()
    text = descriptions(context=context)
    text["persona"] = text["persona"].replace("kan være rolig", "er altid rolig")
    with pytest.raises(ValueError):
        parse_descriptions(json.dumps(text), context, attributes())

    context["openness_label"] = "high"
    text["persona"] = text["persona"].replace("er altid rolig", "kan være praktisk")
    with pytest.raises(ValueError, match="compatible"):
        parse_descriptions(json.dumps(text), context, attributes())


def test_ocean_tendency_requires_complete_supplied_phrase() -> None:
    """A bare term, detached hedge, and incompatible phrase are rejected."""
    context = demographic()
    for replacement in ("rolig", "kan muligvis være rolig"):
        candidate = descriptions(context=context)
        candidate["persona"] = candidate["persona"].replace(
            "kan være rolig", replacement
        )
        with pytest.raises(ValueError):
            parse_descriptions(json.dumps(candidate), context, attributes())

    context["openness_label"] = "high"
    candidate = descriptions(context=context)
    candidate["persona"] = candidate["persona"].replace(
        "kan være rolig", "kan være praktisk"
    )
    with pytest.raises(ValueError, match="incompatible"):
        parse_descriptions(json.dumps(candidate), context, attributes())


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


@pytest.mark.parametrize(
    ("municipality", "origin"), (("Mandø", "Normandiet"), ("Kvindestrup", "Kvindeland"))
)
def test_persona_allows_sex_noun_letters_inside_grounded_labels(
    municipality: str, origin: str
) -> None:
    """Grounded municipality and country labels are matched as complete tokens."""
    context = demographic()
    context["municipality"] = municipality
    context["origin_country_da"] = origin
    payload = descriptions(context=context)

    result = parse_descriptions(json.dumps(payload), context, attributes())

    assert municipality in result.persona
    assert origin in result.persona


@pytest.mark.parametrize(
    "compound", ("brandmand", "romandebut", "mandolin", "kvindelig", "kvindekamp")
)
def test_persona_allows_words_that_embed_sex_noun_letters(compound: str) -> None:
    """Unicode token boundaries do not turn embedded letters into sex nouns."""
    context = demographic()
    payload = descriptions(context=context)
    payload["persona"] = payload["persona"].replace(
        "Hun kan være rolig", f"Ordet {compound} står her. Hun kan være rolig"
    )

    result = parse_descriptions(json.dumps(payload), context, attributes())

    assert compound in result.persona


def test_persona_needs_a_separate_tendency_phrase_after_interests() -> None:
    """Copied interests cannot satisfy the separate OCEAN phrase contract."""
    context = demographic()
    text = descriptions(context=context, interests=["at læse", "musik"])
    text["persona"] = text["persona"].replace("Hun kan være rolig og ", "Hun ")

    with pytest.raises(ValueError, match="personality"):
        parse_descriptions(json.dumps(text), context, attributes())


@pytest.mark.parametrize(
    "noun",
    (
        "MAND",
        "mandens",
        "mænd",
        "MÆNDENE",
        "KVINDE",
        "kvindens",
        "kvinder",
        "KVINDERNES",
    ),
)
def test_persona_rejects_case_and_inflected_sex_nouns(noun: str) -> None:
    """Case-folded standalone singular, plural, and genitive nouns are rejected."""
    context = demographic()
    payload = descriptions(context=context)
    payload["persona"] = payload["persona"].replace(
        "Hun kan være rolig", f"{noun} står som ord her. Hun kan være rolig"
    )

    with pytest.raises(ValueError, match="statistical sex only through its pronoun"):
        parse_descriptions(json.dumps(payload), context, attributes())


@pytest.mark.parametrize(
    "phrase",
    (
        "oprindelsesland",
        "oprindelsesetiket",
        "brede uddannelsesbaggrund",
        "uddannelsesniveau",
        "aktuelle arbejdsforhold",
    ),
)
def test_persona_rejects_redundant_and_technical_phrases(phrase: str) -> None:
    """Dan's reviewed redundant and data-model wording fails closed."""
    context = demographic()
    payload = descriptions(context=context)
    payload["persona"] = payload["persona"].replace(
        "Hun kan være rolig", f"Hun kan være rolig, og {phrase}"
    )

    with pytest.raises(ValueError, match="redundant or technical wording"):
        parse_descriptions(json.dumps(payload), context, attributes())


@pytest.mark.parametrize(
    ("sex", "forbidden_sentence"),
    (("female", "Hun er en mand."), ("male", "Han er en kvinde.")),
)
def test_persona_rejects_standalone_sex_nouns_despite_pronoun(
    sex: str, forbidden_sentence: str
) -> None:
    """A valid pronoun clause never permits a standalone statistical-sex noun."""
    context = demographic(sex=sex)
    payload = descriptions(context=context)
    pronoun = "Hun" if sex == "female" else "Han"
    payload["persona"] = payload["persona"].replace(
        f"{pronoun} kan være rolig", f"{forbidden_sentence} {pronoun} kan være rolig"
    )

    with pytest.raises(ValueError, match="statistical sex only through its pronoun"):
        parse_descriptions(json.dumps(payload), context, attributes())


def test_persona_sex_noun_fix_remains_validator_v15() -> None:
    """The boundary fix closes v15's existing sex-noun rule, not a new contract."""
    prompt = (ROOT / "config" / "prompts" / "personas-da.md").read_text(
        encoding="utf-8"
    )

    assert validation_module.VALIDATOR_VERSION == "persona-safety-v15"
    assert "Brug aldrig `mand` eller `kvinde` som selvstændige" in prompt


@pytest.mark.parametrize(
    ("sex", "origin", "expected_pronoun"),
    (("female", "Danmark", "Hun"), ("male", "Libanon", "Han")),
)
def test_pronoun_and_origin_clauses_are_natural(
    sex: str, origin: str, expected_pronoun: str
) -> None:
    """Danish and Lebanese origins use the same exact natural clause contract."""
    context = demographic(sex=sex)
    context["origin_country_da"] = origin
    payload = descriptions(context=context)

    result = parse_descriptions(json.dumps(payload), context, attributes())

    assert result.persona.startswith(f"{expected_pronoun} er 35 år")
    assert f"kommer fra {origin}" in result.persona


@pytest.mark.parametrize(
    ("status", "expected"),
    (
        ("unemployed", "ledig"),
        ("student", "studerende"),
        ("retired", "pensionist"),
        ("other", "uden for arbejdsmarkedet"),
    ),
)
def test_public_grounding_facts_render_canonical_status(
    status: str, expected: str
) -> None:
    """Non-employees use the exact canonical current-status clause."""
    context = demographic(status=status, job_title=None)
    facts = build_persona_grounding_facts(
        demographic=context, attributes=attributes(job_title=None)
    )

    assert facts.employment == f"er {expected}"


@pytest.mark.parametrize(("education", "expected"), EXPECTED_EDUCATION_CLAUSES.items())
def test_public_grounding_facts_render_pool_and_labels(
    education: str, expected: str
) -> None:
    """The public renderer preserves labels and every pooled education clause."""
    context = demographic(education_level=education)
    context.update(municipality="Hjørring", origin_country_da="Côte d’Ivoire")
    facts = build_persona_grounding_facts(demographic=context, attributes=attributes())

    assert facts.model_dump() == {
        "pronoun_age": "hun er 35 år",
        "municipality": "bor i Hjørring",
        "origin": "kommer fra Côte d’Ivoire",
        "education": expected,
        "employment": "arbejder som forretningsspecialist",
    }


@pytest.mark.parametrize(
    ("detailed_code", "expected"),
    (("05", "selvstændig"), ("10", "medarbejdende ægtefælle")),
)
def test_public_grounding_facts_render_special_employee_status(
    detailed_code: str, expected: str
) -> None:
    """Special employee statuses remain canonical when no title is eligible."""
    context = demographic(status="employed", job_title=None)
    context.update(
        detailed_status_code=detailed_code,
        job_function=None,
        job_function_code=None,
        job_function_resolution="not_applicable",
    )

    facts = build_persona_grounding_facts(
        demographic=context, attributes=attributes(job_title=None)
    )

    assert facts.employment == f"er {expected}"


@pytest.mark.parametrize("education", EDUCATION_POOLING_VALUES)
def test_real_sample_education_values_parse_contextually(education: str) -> None:
    """Every pooled Phase-2 education value grounds a realistic persona summary."""
    context = demographic(education_level=education, sex="male")
    result = parse_descriptions(
        json.dumps(descriptions(context=context)),
        context,
        GeneratedAttributes.model_validate(attributes()),
    )
    assert EXPECTED_EDUCATION_CLAUSES[education] in result.persona


@pytest.mark.parametrize(
    ("missing", "original", "replacement"),
    (
        ("pronoun and age", "Hun er 35 år", "Hun er 34 år"),
        ("pronoun and age", "Hun er 35 år", "Han er 35 år"),
        ("municipality", "bor i København", "bor i Roskilde"),
        ("origin", "kommer fra Danmark", "kommer fra Sverige"),
    ),
)
def test_required_demographic_facts_are_literal(
    missing: str, original: str, replacement: str
) -> None:
    """Changing any required demographic clause is rejected."""
    context = demographic()
    text = descriptions(context=context)
    text["persona"] = text["persona"].replace(original, replacement)
    with pytest.raises(ValueError, match=missing):
        parse_descriptions(json.dumps(text), context, attributes())


def test_schema_diagnostics_identify_attribute_field_without_raw_input() -> None:
    """Schema errors retain a safe field location and omit rejected input."""
    payload = attributes()
    payload["cultural_context"] = "SENTINEL_REJECTED_TEXT"
    with pytest.raises(ValueError) as error:
        parse_attributes(json.dumps(payload), demographic())

    message = str(error.value)
    assert message.startswith("cultural_context:")
    assert "SENTINEL_REJECTED_TEXT" not in message


def test_standalone_sex_noun_rule_applies_only_to_persona() -> None:
    """The pronoun-only statistical-sex contract is scoped to the short persona."""
    context = demographic()
    payload = descriptions(context=context)
    payload["professional_persona"] += " Hun er en kvinde."

    result = parse_descriptions(json.dumps(payload), context, attributes())

    assert result.professional_persona.endswith("Hun er en kvinde.")


def test_status_05_does_not_count_as_personality_tendency() -> None:
    """The self-employed status remains a grounding fact, not an OCEAN phrase."""
    context = demographic(status="employed", job_title=None)
    context.update(
        detailed_status_code="05",
        job_function=None,
        job_function_code=None,
        job_function_resolution="not_applicable",
    )
    text = descriptions(context=context)
    generated = attributes(job_title=None)

    parse_descriptions(json.dumps(text), context, generated)


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


def test_valid_v3_attributes_and_description() -> None:
    """A complete employee record passes both contextual stages."""
    context = demographic()
    parsed = parse_attributes(json.dumps(attributes()), context)
    assert parse_descriptions(
        json.dumps(descriptions(context=context)), context, parsed
    )
