"""Tests for the deterministic internal partner target."""

import json
from pathlib import Path

import pytest
from validation_test_helpers import attributes, demographic

from danish_personas.generation.config import load_generation_config
from danish_personas.generation.models import GenerationConfig
from danish_personas.generation.partner_target import (
    required_partner_gender,
    same_sex_partner_draw,
    same_sex_partner_target,
)
from danish_personas.generation.pipeline import _generation_payload
from danish_personas.generation.validation import parse_attributes


def test_config_default_and_bounds() -> None:
    """The default is loaded and probabilities are bounded."""
    config = load_generation_config(Path("config/config.yaml"))

    assert config.same_sex_partner_probability == 0.00701
    for probability in (-0.1, 1.1):
        with pytest.raises(ValueError):
            GenerationConfig.model_validate(
                {**config.model_dump(), "same_sex_partner_probability": probability}
            )


def test_partner_gender_mapping() -> None:
    """The target maps to same or different source sex as required."""
    assert required_partner_gender(sex="male", same_sex_target=True) == "male"
    assert required_partner_gender(sex="female", same_sex_target=True) == "female"
    assert required_partner_gender(sex="male", same_sex_target=False) == "female"
    assert required_partner_gender(sex="female", same_sex_target=False) == "male"


def test_provider_payload_contains_internal_target() -> None:
    """The target reaches the provider without becoming a public record field."""
    payload = _generation_payload(
        demographics={"sex": "female"},
        allowed_job_titles=[],
        personality_tendencies=(),
        same_sex_partner_target=True,
    )

    assert payload["same_sex_partner_target"] is True


def test_target_is_stable_at_threshold_edges() -> None:
    """Zero and one provide deterministic Bernoulli edge cases."""
    assert not same_sex_partner_target(persona_id="p-1", probability=0.0)
    assert same_sex_partner_target(persona_id="p-1", probability=1.0)
    assert same_sex_partner_draw("p-1") == same_sex_partner_draw("p-1")


def test_validation_rejects_wrong_target_and_accepts_matching_target() -> None:
    """Contextual validation recomputes the target rather than trusting output."""
    config = load_generation_config(Path("config/config.yaml"))
    config = config.model_copy(update={"same_sex_partner_probability": 1.0})
    context = demographic(sex="female")
    valid = attributes()
    valid["partner_gender"] = "female"
    parse_attributes(json.dumps(valid), context, generation_config=config)
    valid["partner_gender"] = "male"
    with pytest.raises(ValueError, match="stable relationship target"):
        parse_attributes(json.dumps(valid), context, generation_config=config)
