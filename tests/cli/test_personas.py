"""Tests for persona commands and the explicit LLM execution guard."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from danish_personas import cli
from scripts.generate_personas import main

RUNNER = CliRunner()


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


def test_persona_live_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shards default to planning while pilots require explicit approval."""
    live_values: list[bool] = []

    def generate(**kwargs: object) -> Path:
        live_values.append(bool(kwargs["live"]))
        return Path("shard")

    monkeypatch.setattr(cli, "generate_personas", generate)
    shard_args = [
        "personas",
        "shard",
        "--input",
        "input",
        "--sample-manifest",
        "manifest",
        "--config",
        "config",
        "--output-dir",
        "output",
        "--rows",
        "1",
    ]
    result = RUNNER.invoke(cli.main, shard_args)
    assert result.exit_code == 0
    assert live_values == [False]

    result = RUNNER.invoke(
        cli.main,
        [
            "personas",
            "pilot",
            "--input",
            "input",
            "--sample-manifest",
            "manifest",
            "--config",
            "config",
            "--output-dir",
            "output",
            "--rows",
            "1",
            "--maximum-total-requests",
            "2",
            "--input-price-per-million",
            "0",
            "--output-price-per-million",
            "0",
        ],
    )
    assert result.exit_code != 0
    assert "--live" in result.output
