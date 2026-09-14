"""Tests for strict pipeline contracts."""

import pytest
from pydantic import ValidationError

from danish_personas.models import SourceSelection


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
