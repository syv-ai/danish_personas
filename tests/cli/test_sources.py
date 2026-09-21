"""Tests for source CLI commands and network guards."""

from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from danish_personas import cli

RUNNER = CliRunner()


def test_source_network_gate_allows_service_with_explicit_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit network flag reaches the resolver exactly once."""
    called = False

    def resolve(**_: object) -> SimpleNamespace:
        nonlocal called
        called = True
        return SimpleNamespace(sources=[], classifications=[])

    monkeypatch.setattr(cli, "resolve_sources", resolve)
    monkeypatch.setattr(cli, "load_yaml_model", lambda **_: SimpleNamespace())
    result = RUNNER.invoke(
        cli.main,
        [
            "sources",
            "resolve",
            "--config",
            "config/sources.yaml",
            "--lock",
            "lock",
            "--network",
        ],
    )
    assert result.exit_code == 0
    assert called


def test_source_network_gate_precedes_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """Source operations require approval before their public service is called."""
    called = False

    def resolve(**_: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(cli, "resolve_sources", resolve)
    result = RUNNER.invoke(
        cli.main,
        ["sources", "resolve", "--config", "config/sources.yaml", "--lock", "lock"],
    )
    assert result.exit_code != 0
    assert "--network" in result.output
    assert not called
