"""Tests for validation CLI commands."""

from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from danish_personas import cli

RUNNER = CliRunner()


@pytest.mark.parametrize(
    ("command", "service"),
    [
        (["validate", "sources", "--bundle", "bundle"], "validate_sources"),
        (
            ["validate", "demographics", "--run", "run", "--bundle", "bundle"],
            "validate_demographics",
        ),
        (["validate", "personas", "--run", "run"], "validate_persona_run"),
        (["validate", "pilot", "--pilot", "pilot"], "validate_persona_pilot"),
    ],
    ids=["sources", "demographics", "personas", "pilot"],
)
def test_validation_commands_fail_on_failed_reports(
    monkeypatch: pytest.MonkeyPatch, command: list[str], service: str
) -> None:
    """Every validation command converts a failed report into a CLI failure."""
    failed = SimpleNamespace(passed=False)
    monkeypatch.setattr(cli, service, lambda **_: failed)

    result = RUNNER.invoke(cli.main, command)
    assert result.exit_code != 0
