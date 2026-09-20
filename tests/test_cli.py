"""Tests for the installed unified command-line interface."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from danish_personas import cli
from danish_personas.io import load_yaml_model
from danish_personas.models import SamplingConfig
from danish_personas.origin_labels import (
    DEFAULT_ORIGIN_LABEL_CONTRACT_PATH,
    load_origin_label_contract,
    origin_label_contract_sha256,
)
from danish_personas.release.models import ReleaseManifest, ReleasePackageResult
from danish_personas.release.packager import ReleasePackagingError
from danish_personas.release.verifier import ReleaseVerificationError
from danish_personas.sources.archive import RAW_DIRECTORY

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


def test_release_cli_package_and_verify_delegate_exact_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Release CLI commands expose successful service results and exact arguments."""
    calls: list[tuple[str, dict[str, object]]] = []
    package_result = ReleasePackageResult(
        path=tmp_path / "release-id", manifest_sha256="a" * 64
    )

    def package(**kwargs: object) -> ReleasePackageResult:
        calls.append(("package", kwargs))
        return package_result

    def verify(**kwargs: object) -> ReleaseManifest:
        calls.append(("verify", kwargs))
        contract = load_origin_label_contract()
        return ReleaseManifest(
            version=2,
            release_id="b" * 32,
            created_at="2026-09-17T00:00:00Z",
            pilot_id="c" * 16,
            model="provider/model",
            rows=10_000,
            git_head="d" * 40,
            origin_url="https://example.invalid/repo.git",
            uv_lock_sha256="e" * 64,
            evidence_sha256="f" * 64,
            origin_label_contract_file=DEFAULT_ORIGIN_LABEL_CONTRACT_PATH,
            origin_label_contract_sha256=origin_label_contract_sha256(),
            origin_label_contract_version=contract.version,
            origin_label_contract_content=contract,
            artifacts=[
                {
                    "path": "README.md",
                    "role": "dataset-card",
                    "sha256": "0" * 64,
                    "size": 1,
                }
            ],
        )

    monkeypatch.setattr("danish_personas.release.cli.package_release", package)
    monkeypatch.setattr("danish_personas.release.cli.verify_release", verify)
    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            "release",
            "package",
            "--pilot-dir",
            str(tmp_path / "pilot"),
            "--attestation",
            str(tmp_path / "attestation"),
            "--policy",
            str(tmp_path / "policy"),
            "--dataset-card",
            str(tmp_path / "card"),
            "--licence",
            str(tmp_path / "licence"),
            "--repository-root",
            str(tmp_path / "repo"),
            "--output-parent",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert str(package_result.path) in result.output
    assert package_result.manifest_sha256 in result.output
    assert calls[0] == (
        "package",
        {
            "pilot_dir": tmp_path / "pilot",
            "attestation_path": tmp_path / "attestation",
            "policy_path": tmp_path / "policy",
            "dataset_card_path": tmp_path / "card",
            "licence_path": tmp_path / "licence",
            "repository_root": tmp_path / "repo",
            "output_parent": tmp_path / "out",
        },
    )

    result = runner.invoke(
        cli.main,
        [
            "release",
            "verify",
            "--release-dir",
            str(tmp_path / "relocated"),
            "--expected-manifest-sha256",
            "a" * 64,
        ],
    )
    assert result.exit_code == 0, result.output
    assert "b" * 32 in result.output
    assert calls[1] == (
        "verify",
        {"release_dir": tmp_path / "relocated", "expected_manifest_sha256": "a" * 64},
    )


def test_release_cli_reports_service_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Packaging and verification failures become nonzero Click results."""
    monkeypatch.setattr(
        "danish_personas.release.cli.package_release",
        lambda **_: (_ for _ in ()).throw(ReleasePackagingError("bad package")),
    )
    result = CliRunner().invoke(
        cli.main,
        [
            "release",
            "package",
            "--pilot-dir",
            str(tmp_path / "pilot"),
            "--attestation",
            str(tmp_path / "attestation"),
            "--policy",
            str(tmp_path / "policy"),
            "--dataset-card",
            str(tmp_path / "card"),
            "--licence",
            str(tmp_path / "licence"),
            "--repository-root",
            str(tmp_path / "repo"),
            "--output-parent",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "bad package" in result.output

    monkeypatch.setattr(
        "danish_personas.release.cli.verify_release",
        lambda **_: (_ for _ in ()).throw(ReleaseVerificationError("bad release")),
    )
    result = CliRunner().invoke(
        cli.main,
        [
            "release",
            "verify",
            "--release-dir",
            str(tmp_path / "release"),
            "--expected-manifest-sha256",
            "a" * 64,
        ],
    )
    assert result.exit_code != 0
    assert "bad release" in result.output


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
