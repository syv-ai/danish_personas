"""Tests for the shared Hydra generation configuration."""

from pathlib import Path

import pytest
import yaml
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from pydantic import ValidationError

from danish_personas.generation.config import (
    load_generation_config,
    persist_effective_generation_config,
)
from danish_personas.generation.models import GenerationConfig


def test_effective_snapshot_is_stable_and_immutable(tmp_path: Path) -> None:
    """Resolved LLM provenance is reused but never silently overwritten."""
    config = load_generation_config(Path("config/config.yaml"))

    first = persist_effective_generation_config(config=config, output_dir=tmp_path)
    second = persist_effective_generation_config(config=config, output_dir=tmp_path)

    assert first == second
    assert load_generation_config(first) == config
    legacy_bytes = yaml.safe_dump(
        config.model_dump(mode="json"), allow_unicode=True, sort_keys=False
    ).encode("utf-8")
    assert first.read_bytes() != legacy_bytes
    assert first.read_text(encoding="utf-8").startswith(
        "# Effective generation configuration format: hydra-v1\n"
    )
    first.write_text("tampered: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance does not match"):
        persist_effective_generation_config(config=config, output_dir=tmp_path)


def test_flat_generation_config_remains_supported(tmp_path: Path) -> None:
    """Service callers can continue to load focused flat YAML fixtures."""
    content = """\
base_url: http://localhost/v1
model: fixture-model
api_key_env: null
timeout_seconds: 30
maximum_http_attempts: 1
maximum_total_requests: 2
retry_backoff_seconds: 0
maximum_rows_per_shard: 5
max_tokens: null
enable_thinking: null
reasoning_effort: null
prompt: prompt.md
job_title_mapping: null
origin_label_contract: config/folk2-ieland-labels-da.yaml
"""
    config_path = tmp_path / "flat.yaml"
    config_path.write_text(content, encoding="utf-8")

    assert load_generation_config(config_path).model == "fixture-model"


def test_generation_config_rejects_schema_less_response_mode() -> None:
    """Provider requests cannot be configured to omit the Pydantic JSON schema."""
    config = load_generation_config(Path("config/config.yaml"))

    with pytest.raises(ValidationError):
        GenerationConfig.model_validate(
            config.model_dump() | {"response_format": "json_object"}
        )


def test_hydra_resolves_nested_values_and_rejects_obsolete_llm_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hydra interpolation works while removed LLM guard fields remain invalid."""
    monkeypatch.setenv("TEST_GENERATION_MODEL", "resolved-model")
    content = (
        Path("config/config.yaml")
        .read_text(encoding="utf-8")
        .replace(
            "  model: deepseek-v4-flash-0731",
            "  model: ${oc.env:TEST_GENERATION_MODEL}",
        )
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(content, encoding="utf-8")

    assert load_generation_config(config_path).model == "resolved-model"

    obsolete = content.replace(
        "  model: ${oc.env:TEST_GENERATION_MODEL}\n",
        "  model: ${oc.env:TEST_GENERATION_MODEL}\n  version: 4\n",
    )
    config_path.write_text(obsolete, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_generation_config(config_path)


def test_load_generation_config_preserves_an_active_hydra_context() -> None:
    """Nested composition restores the caller's Hydra state after every call."""
    config_path = Path("config/config.yaml").resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_path.parent)):
        active_hydra = GlobalHydra.instance().hydra
        first = load_generation_config(config_path)
        assert GlobalHydra.instance().hydra is active_hydra
        second = load_generation_config(config_path)
        assert second == first
        assert GlobalHydra.instance().hydra is active_hydra

    assert not GlobalHydra.instance().is_initialized()


def test_root_generation_config_has_provider_defaults() -> None:
    """The canonical config retains the configured provider endpoint and model."""
    config = load_generation_config(Path("config/config.yaml"))

    assert config.base_url == "https://api.melious.ai/v1"
    assert config.model == "deepseek-v4-flash-0731"


def test_root_prompts_render_origin_as_natural_prose() -> None:
    """Origin guidance avoids exposing metadata terminology in persona prose."""
    prompts = Path("config/persona-da.md").read_text(encoding="utf-8")

    assert "oprindelsesetiket" not in prompts.casefold()
    assert "Han kommer fra Rumænien" in prompts
    assert (
        "Skriv aldrig om inputfelter, metadata, etiketter eller kategorier" in prompts
    )
