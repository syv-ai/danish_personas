"""Tests for the explicit Phase 3 execution guard."""

from pathlib import Path

from click.testing import CliRunner

from scripts.generate_personas import main


def test_llm_generation_is_disabled() -> None:
    """The committed configuration blocks every live LLM call."""
    result = CliRunner().invoke(
        main,
        [
            "--input",
            "missing.parquet",
            "--sample-manifest",
            "missing.json",
            "--config",
            str(Path("config/generation.yaml")),
            "--output-dir",
            "data/test-output",
            "--rows",
            "1",
            "--live",
        ],
    )
    assert result.exit_code != 0
    assert "LLM generation is disabled" in result.output
