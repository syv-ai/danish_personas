"""Tests for the deterministic internal partner target."""

from pathlib import Path

import pytest

import danish_personas.generation.partner_target as partner_target
from danish_personas.generation.config import load_generation_config
from danish_personas.generation.models import GenerationConfig
from danish_personas.generation.partner_target import (
    required_partner_gender,
    same_sex_partner_draw,
    same_sex_partner_target,
)
from danish_personas.generation.pipeline import _generation_payload


def test_config_default_and_bounds() -> None:
    """The default is loaded and probabilities are bounded."""
    config = load_generation_config(Path("config/config.yaml"))

    assert config.same_sex_partner_probability == 0.00701
    for probability in (-0.1, 1.1):
        with pytest.raises(ValueError):
            GenerationConfig.model_validate(
                {**config.model_dump(), "same_sex_partner_probability": probability}
            )


def test_partner_draw_golden_vectors() -> None:
    """Policy-version changes cannot silently alter established draws."""
    assert same_sex_partner_draw("p-1") == 9_331_940_782_940_116_072
    assert same_sex_partner_draw("persona-1") == 16_907_542_642_995_462_103
    assert same_sex_partner_draw("golden-persona") == 9_522_114_356_481_481_835


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


def test_target_uses_exact_digest_threshold_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Digest values immediately around a threshold retain integer semantics."""

    class Digest:
        def __init__(self, value: int) -> None:
            self._value = value

        def digest(self) -> bytes:
            return self._value.to_bytes(8, "big")

    threshold = 2**63
    monkeypatch.setattr(
        partner_target.hashlib, "sha256", lambda _material: Digest(threshold - 1)
    )
    assert same_sex_partner_target(persona_id="boundary", probability=0.5)

    monkeypatch.setattr(
        partner_target.hashlib, "sha256", lambda _material: Digest(threshold)
    )
    assert not same_sex_partner_target(persona_id="boundary", probability=0.5)

    monkeypatch.setattr(partner_target.hashlib, "sha256", lambda _material: Digest(0))
    assert not same_sex_partner_target(persona_id="boundary", probability=0.0)
    monkeypatch.setattr(
        partner_target.hashlib, "sha256", lambda _material: Digest(2**64 - 1)
    )
    assert same_sex_partner_target(persona_id="boundary", probability=1.0)
