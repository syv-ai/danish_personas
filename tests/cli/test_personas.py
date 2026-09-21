"""Tests for persona commands and the configuration execution guard."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from danish_personas import cli

RUNNER = CliRunner()


def test_persona_commands_execute_without_live_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persona commands execute directly without a separate approval flag."""
    calls: list[dict[str, object]] = []

    def generate(**kwargs: object) -> Path:
        calls.append(kwargs)
        return Path("shard")

    def pilot(**kwargs: object) -> Path:
        calls.append(kwargs)
        return Path("pilot")

    monkeypatch.setattr(cli, "generate_personas", generate)
    monkeypatch.setattr(cli, "run_pilot", pilot)
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
    assert result.exit_code == 0
    assert len(calls) == 2
    assert all("live" not in call for call in calls)
