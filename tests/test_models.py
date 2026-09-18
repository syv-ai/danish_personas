"""Tests for strict pipeline contracts."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from danish_personas.io import load_yaml_model
from danish_personas.models import SourceSelection, ValidationConfig


def test_source_selection_accepts_explicit_values() -> None:
    """Explicit StatBank value codes satisfy the source contract."""
    selection = SourceSelection(values=["2024"])
    assert selection.values == ["2024"]


def test_source_selection_requires_one_mechanism() -> None:
    """A source dimension must use exactly one selection mechanism."""
    with pytest.raises(ValidationError):
        SourceSelection()
    with pytest.raises(ValidationError):
        SourceSelection(selector="all", values=["x"])


def test_validation_config_rejects_unsupported_schema_version() -> None:
    """Validation thresholds reject versions outside the supported schema."""
    config = load_yaml_model(
        path=Path("config/validation.yaml"), model=ValidationConfig
    )

    assert config.version == 4
    with pytest.raises(ValidationError, match="Unsupported validation config version"):
        ValidationConfig.model_validate(config.model_dump(mode="json") | {"version": 3})
