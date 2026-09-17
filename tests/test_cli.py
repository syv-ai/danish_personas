"""Tests for the installed unified command-line interface."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from danish_personas import cli
from danish_personas.io import load_yaml_model
from danish_personas.models import SamplingConfig

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
    assert "--sample-rows" in result.output


def test_deterministic_workflow_hands_off_paths_and_stops_at_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke workflows use returned paths and do not cross the statistical gate."""
    order: list[str] = []
    sampling = load_yaml_model(path=Path("config/sampling.yaml"), model=SamplingConfig)
    monkeypatch.setattr(cli, "load_yaml_model", lambda **_: sampling)
    monkeypatch.setattr(
        cli, "restore_raw_sources", lambda **_: order.append("restore") or 1
    )
    monkeypatch.setattr(
        cli,
        "prepare_bundle",
        lambda **_: order.append("prepare") or Path("returned-bundle"),
    )
    monkeypatch.setattr(
        cli,
        "validate_sources",
        lambda **kwargs: (
            order.append(f"sources:{kwargs['bundle_dir']}")
            or SimpleNamespace(passed=True)
        ),
    )
    monkeypatch.setattr(
        cli,
        "generate_records",
        lambda **kwargs: (
            order.append(f"generate:{kwargs['output_dir']}") or Path("returned-smoke")
        ),
    )
    monkeypatch.setattr(
        cli,
        "validate_demographics",
        lambda **kwargs: (
            order.append(f"demographics:{kwargs['run_dir']}")
            or SimpleNamespace(passed=True)
        ),
    )

    result = RUNNER.invoke(cli.main, ["workflow", "deterministic", "--target", "smoke"])
    assert result.exit_code == 0
    assert order == [
        "restore",
        "prepare",
        "sources:returned-bundle",
        "generate:data/runs/smoke",
        "demographics:returned-smoke",
    ]


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


def test_statistical_workflow_runs_only_after_smoke_and_freezes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Statistical workflows perform both validation gates before freezing."""
    order: list[str] = []
    sampling = load_yaml_model(path=Path("config/sampling.yaml"), model=SamplingConfig)
    monkeypatch.setattr(cli, "load_yaml_model", lambda **_: sampling)
    monkeypatch.setattr(
        cli, "restore_raw_sources", lambda **_: order.append("restore") or 1
    )
    monkeypatch.setattr(cli, "prepare_bundle", lambda **_: Path("bundle"))
    monkeypatch.setattr(
        cli,
        "validate_sources",
        lambda **_: order.append("sources") or SimpleNamespace(passed=True),
    )
    run_paths = iter((Path("smoke"), Path("statistical")))
    monkeypatch.setattr(
        cli, "generate_records", lambda **_: order.append("generate") or next(run_paths)
    )
    monkeypatch.setattr(
        cli,
        "validate_demographics",
        lambda **_: order.append("demographics") or SimpleNamespace(passed=True),
    )
    monkeypatch.setattr(
        cli, "freeze_sample", lambda **_: order.append("freeze") or Path("sample")
    )

    result = RUNNER.invoke(
        cli.main, ["workflow", "deterministic", "--target", "statistical"]
    )
    assert result.exit_code == 0
    assert order == [
        "restore",
        "sources",
        "generate",
        "demographics",
        "generate",
        "demographics",
        "freeze",
    ]


def test_validation_commands_fail_on_failed_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every validation command converts a failed report into a CLI failure."""
    failed = SimpleNamespace(passed=False)
    monkeypatch.setattr(cli, "validate_sources", lambda **_: failed)
    monkeypatch.setattr(cli, "validate_demographics", lambda **_: failed)
    monkeypatch.setattr(cli, "validate_persona_run", lambda **_: failed)
    monkeypatch.setattr(cli, "validate_persona_pilot", lambda **_: failed)

    commands = [
        ["validate", "sources", "--bundle", "bundle"],
        ["validate", "demographics", "--run", "run", "--bundle", "bundle"],
        ["validate", "personas", "--run", "run"],
        ["validate", "pilot", "--pilot", "pilot"],
    ]
    for command in commands:
        result = RUNNER.invoke(cli.main, command)
        assert result.exit_code != 0


def test_workflow_stops_when_source_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed source report prevents demographic generation."""
    generated = False
    monkeypatch.setattr(cli, "restore_raw_sources", lambda **_: 1)
    monkeypatch.setattr(cli, "prepare_bundle", lambda **_: Path("bundle"))
    monkeypatch.setattr(
        cli, "validate_sources", lambda **_: SimpleNamespace(passed=False)
    )

    def generate(**_: object) -> Path:
        nonlocal generated
        generated = True
        return Path("run")

    monkeypatch.setattr(cli, "generate_records", generate)
    result = RUNNER.invoke(cli.main, ["workflow", "deterministic", "--target", "smoke"])
    assert result.exit_code != 0
    assert not generated
