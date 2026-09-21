"""Tests for deterministic CLI workflows."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from danish_personas import cli
from danish_personas.io import load_yaml_model
from danish_personas.models import SamplingConfig
from danish_personas.sources.archive import RAW_DIRECTORY

RUNNER = CliRunner()


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
        f"generate:{Path('data/runs/smoke')}",
        "demographics:returned-smoke",
    ]


def test_deterministic_workflow_rejects_restore_conflict_before_services(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The skip/force conflict fails before the workflow can touch the filesystem."""
    called = False

    def restore(**_: object) -> int:
        nonlocal called
        called = True
        return 1

    monkeypatch.setattr(cli, "restore_raw_sources", restore)
    raw_parent = tmp_path / "custom-parent"
    result = RUNNER.invoke(
        cli.main,
        [
            "workflow",
            "deterministic",
            "--target",
            "smoke",
            "--raw-parent",
            str(raw_parent),
            "--skip-restore",
            "--force-restore",
        ],
    )

    assert result.exit_code != 0
    assert "cannot be used with" in result.output
    assert not called
    assert not raw_parent.exists()


def test_deterministic_workflow_uses_only_fixed_custom_raw_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Custom parents reach both services without exposing a raw child path."""
    sampling = load_yaml_model(path=Path("config/sampling.yaml"), model=SamplingConfig)
    raw_parent = tmp_path / "custom-parent"
    sibling = tmp_path / "sibling"
    calls: dict[str, dict[str, object]] = {}
    monkeypatch.setattr(cli, "load_yaml_model", lambda **_: sampling)

    def restore(**kwargs: object) -> int:
        calls["restore"] = kwargs
        return 1

    def prepare(**kwargs: object) -> Path:
        calls["prepare"] = kwargs
        return Path("bundle")

    monkeypatch.setattr(cli, "restore_raw_sources", restore)
    monkeypatch.setattr(cli, "prepare_bundle", prepare)
    monkeypatch.setattr(
        cli, "validate_sources", lambda **_: SimpleNamespace(passed=True)
    )
    monkeypatch.setattr(cli, "generate_records", lambda **_: Path("smoke"))
    monkeypatch.setattr(
        cli, "validate_demographics", lambda **_: SimpleNamespace(passed=True)
    )

    result = RUNNER.invoke(
        cli.main,
        [
            "workflow",
            "deterministic",
            "--target",
            "smoke",
            "--raw-parent",
            str(raw_parent),
            "--force-restore",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls["restore"] == {
        "archive_path": cli.DEFAULT_ARCHIVE,
        "output_dir": raw_parent,
        "force": True,
    }
    raw_dir = raw_parent / RAW_DIRECTORY
    assert calls["prepare"]["raw_dir"] == raw_dir
    assert str(raw_dir) in result.output
    assert calls["prepare"]["raw_dir"] != sibling
    assert calls["prepare"]["raw_dir"] != raw_parent


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
    statistical_dir = Path("statistical") / "content-addressed-run"
    run_paths = iter((Path("smoke"), statistical_dir))
    monkeypatch.setattr(
        cli, "generate_records", lambda **_: order.append("generate") or next(run_paths)
    )
    monkeypatch.setattr(
        cli,
        "validate_demographics",
        lambda **_: order.append("demographics") or SimpleNamespace(passed=True),
    )

    def freeze(**kwargs: object) -> Path:
        order.append("freeze")
        assert kwargs == {
            "run_dir": statistical_dir,
            "rows": 1000,
            "output": statistical_dir / "text-development-seeds.parquet",
        }
        output = kwargs["output"]
        assert isinstance(output, Path)
        return output

    monkeypatch.setattr(cli, "freeze_sample", freeze)

    result = RUNNER.invoke(
        cli.main, ["workflow", "deterministic", "--target", "statistical"]
    )
    assert result.exit_code == 0
    assert result.output.splitlines()[-1] == str(
        statistical_dir / "text-development-seeds.parquet"
    )
    assert order == [
        "restore",
        "sources",
        "generate",
        "demographics",
        "generate",
        "demographics",
        "freeze",
    ]


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
