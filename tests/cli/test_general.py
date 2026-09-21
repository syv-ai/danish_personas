"""Tests for general unified command-line interface contracts."""

from click.testing import CliRunner

from danish_personas import cli

RUNNER = CliRunner()


def test_cli_hierarchy_and_help() -> None:
    """The public command tree exposes every supported boundary."""
    result = RUNNER.invoke(cli.main, ["--help"])
    assert result.exit_code == 0
    for command in (
        "sources",
        "validate",
        "demographics",
        "sample",
        "personas",
        "workflow",
    ):
        assert command in result.output

    result = RUNNER.invoke(cli.main, ["sources", "--help"])
    assert result.exit_code == 0
    for command in ("restore", "pack", "prepare", "resolve", "fetch"):
        assert command in result.output

    result = RUNNER.invoke(cli.main, ["workflow", "deterministic", "--help"])
    assert result.exit_code == 0
    assert "--force-restore" in result.output
    assert "--raw-parent" in result.output
    assert "--raw-dir" not in result.output
    assert "--sample-rows" in result.output
