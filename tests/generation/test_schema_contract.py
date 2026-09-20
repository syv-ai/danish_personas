"""Tests for strict JSON-schema response contracts."""

import pytest
from pydantic import BaseModel, ValidationError

from danish_personas.generation.models import GeneratedAttributes, PersonaDescriptions


def test_nullable_attribute_fields_require_explicit_null() -> None:
    """Nullable attributes accept null but do not allow omitted response keys."""
    payload: dict[str, object] = {
        "cultural_context": "Personen har en dansk hverdag og lokale fællesskaber.",
        "skills_and_expertise": ["planlægning", "samarbejde", "formidling"],
        "hobbies_and_interests": ["læsning", "musik", "brætspil"],
        "career_goals_and_ambitions": None,
        "job_title": None,
    }

    attributes = GeneratedAttributes.model_validate(payload)

    assert attributes.career_goals_and_ambitions is None
    assert attributes.job_title is None
    for field in ("career_goals_and_ambitions", "job_title"):
        omitted = payload.copy()
        omitted.pop(field)
        with pytest.raises(ValidationError):
            GeneratedAttributes.model_validate(omitted)


@pytest.mark.parametrize("model", [GeneratedAttributes, PersonaDescriptions])
def test_response_schemas_are_openai_strict(model: type[BaseModel]) -> None:
    """Every object in a response schema satisfies OpenAI strict mode."""
    schema = model.model_json_schema()
    _assert_strict_object_schemas(schema)


def _assert_strict_object_schemas(schema: object) -> None:
    """Assert strict object requirements throughout a JSON schema tree."""
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            assert isinstance(properties, dict)
            assert set(required) == set(properties)
            assert schema.get("additionalProperties") is False
        for value in schema.values():
            _assert_strict_object_schemas(value)
    elif isinstance(schema, list):
        for value in schema:
            _assert_strict_object_schemas(value)
