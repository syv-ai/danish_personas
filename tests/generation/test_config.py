"""Tests for the single Hydra generation configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from danish_personas.generation.config import load_generation_config


def test_hydra_resolves_values_and_rejects_obsolete_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hydra interpolation works while removed guard fields remain invalid."""
    monkeypatch.setenv("TEST_GENERATION_MODEL", "resolved-model")
    content = (
        Path("config/config.yaml")
        .read_text(encoding="utf-8")
        .replace("model: gpt-5.6-sol", "model: ${oc.env:TEST_GENERATION_MODEL}")
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content, encoding="utf-8")

    assert load_generation_config(config_path).model == "resolved-model"

    config_path.write_text(f"{content}version: 4\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_generation_config(config_path)


def test_root_generation_config_has_local_defaults() -> None:
    """The canonical config names the default local endpoint and model."""
    config = load_generation_config(Path("config/config.yaml"))

    assert config.base_url == "http://127.0.0.1:18080/v1"
    assert config.model == "gpt-5.6-sol"
