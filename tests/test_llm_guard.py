"""Tests for the explicit Phase 3 execution guard."""

from pathlib import Path

from click.testing import CliRunner

from scripts.generate_personas import main


def test_llm_generation_is_disabled() -> None:
    """The current configuration blocks every LLM call."""
    config = Path("config/generation.yaml")
    result = CliRunner().invoke(main, ["--config", str(config)])
    assert result.exit_code != 0
    assert "LLM generation is disabled" in result.output
