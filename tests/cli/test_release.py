"""Tests for release CLI commands."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from danish_personas import cli
from danish_personas.origin_labels import (
    DEFAULT_ORIGIN_LABEL_CONTRACT_PATH,
    load_origin_label_contract,
    origin_label_contract_sha256,
)
from danish_personas.release.models import ReleaseManifest, ReleasePackageResult
from danish_personas.release.packager import ReleasePackagingError
from danish_personas.release.verifier import ReleaseVerificationError

RUNNER = CliRunner()


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
    result = RUNNER.invoke(
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

    result = RUNNER.invoke(
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
    result = RUNNER.invoke(
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
    result = RUNNER.invoke(
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
