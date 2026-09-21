"""Attribute, schema, and validator-contract tests."""

# Test functions intentionally omit repetitive docstrings.
# ruff: noqa: D103

import json
from pathlib import Path

import pytest
import yaml
from validation_test_helpers import JOB_TITLE, attributes, demographic, persona

import danish_personas.generation.validation as validation_module
from danish_personas.generation.job_titles import load_job_title_mapping
from danish_personas.generation.personality import all_personality_tendencies
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
JOB_TITLE_CASES = tuple(
    (code, entry.label, title)
    for code, entry in load_job_title_mapping().job_functions.items()
    for title in entry.titles
)


@pytest.mark.parametrize(
    "field", ["cultural_context", "career_goals_and_ambitions", "job_title"]
)
def test_attribute_diagnostics_redact_rejected_values(field: str) -> None:
    payload = attributes()
    rejected = "SENTINEL_REJECTED_ATTRIBUTE"
    payload[field] = rejected
    with pytest.raises(ValueError) as error:
        parse_attributes(json.dumps(payload), demographic())
    message = str(error.value)
    assert message.startswith(f"{field}:")
    assert rejected not in message


@pytest.mark.parametrize("field", ["skills_and_expertise", "hobbies_and_interests"])
def test_attribute_list_diagnostics_redact_rejected_values(field: str) -> None:
    payload = attributes()
    rejected = "SENTINEL_REJECTED_LIST"
    payload[field] = [rejected, rejected + " two"]
    with pytest.raises(ValueError) as error:
        parse_attributes(json.dumps(payload), demographic())
    message = str(error.value)
    assert message.startswith(f"{field}:")
    assert rejected not in message


def test_description_diagnostics_redact_rejected_values() -> None:
    rejected = "SENTINEL_REJECTED_DESCRIPTION"
    with pytest.raises(ValueError) as error:
        parse_descriptions(
            json.dumps({"persona": rejected}), demographic(), attributes()
        )
    message = str(error.value)
    assert message.startswith("persona:")
    assert rejected not in message


def test_description_schema_rejects_removed_fields() -> None:
    payload = persona(context=demographic())
    payload["removed_field"] = "ikke en aktiv kontrakt"
    with pytest.raises(ValueError, match="Extra inputs"):
        parse_descriptions(json.dumps(payload), demographic(), attributes())


@pytest.mark.parametrize("term", all_personality_tendencies())
def test_every_compatible_personality_term_is_locally_hedged(term: str) -> None:
    context = demographic()
    text = persona(context=context)["persona"] + f" Hun er ofte {term}."
    assert (
        parse_descriptions(json.dumps({"persona": text}), context, attributes()).persona
        == text
    )


@pytest.mark.parametrize("code,label,title", JOB_TITLE_CASES)
def test_every_reviewed_title_is_accepted(code: str, label: str, title: str) -> None:
    context = demographic()
    context["job_function_code"] = code
    context["job_function"] = label
    parsed = parse_attributes(json.dumps(attributes(job_title=title)), context)
    assert parsed.job_title == title


def test_job_title_eligibility_and_unsafe_titles() -> None:
    context = demographic()
    assert parse_attributes(json.dumps(attributes()), context).job_title == JOB_TITLE
    with pytest.raises(ValueError, match="allowlisted reviewed title"):
        parse_attributes(json.dumps(attributes(job_title="analytiker")), context)

    retired = demographic(status="retired", job_title=None)
    with pytest.raises(ValueError, match="null job_title"):
        parse_attributes(json.dumps(attributes()), retired)


def test_job_title_must_be_allowlisted() -> None:
    context = demographic()
    with pytest.raises(ValueError, match="allowlisted reviewed title"):
        parse_attributes(json.dumps(attributes(job_title="opfinder")), context)


def test_schema_contains_only_persona() -> None:
    assert set(validation_module.PersonaDescriptions.model_fields) == {"persona"}


@pytest.mark.parametrize(
    "title",
    ["sundhedsprofessionel med angst", "- forretningsspecialist"],
    ids=["sensitive-phrase", "punctuation"],
)
def test_unsafe_job_titles_are_rejected(title: str) -> None:
    context = demographic()
    with pytest.raises(ValueError, match="job_title"):
        parse_attributes(json.dumps(attributes(job_title=title)), context)


def test_valid_v4_persona_preserves_rich_grounded_text() -> None:
    context = demographic()
    parsed_attributes = parse_attributes(json.dumps(attributes()), context)
    result = parse_descriptions(
        json.dumps(persona(context=context, extra="Hun holder af hverdage med ro.")),
        context,
        parsed_attributes,
    )
    assert result.persona.startswith("Maja er 35 år")


def test_validator_and_education_exports_are_current() -> None:
    assert validation_module.VALIDATOR_VERSION == "persona-safety-v17"
    assert set(EDUCATION_DANISH) == set(EDUCATION_POOLING_VALUES)
    assert load_job_title_mapping().version == 1
