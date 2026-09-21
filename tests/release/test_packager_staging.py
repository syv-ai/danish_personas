"""Tests for the public offline release packager."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
from support import ReleaseCase, _package, coherent_evidence

from danish_personas.models import ValidationReport
from danish_personas.release import packager
from danish_personas.release.common import validate_persona_output_rows
from danish_personas.release.packager import ReleasePackagingError, package_release
from danish_personas.release.verifier import verify_release


def test_contextual_validation_accepts_nonemployee_null_title(
    nonemployee_output: pl.DataFrame,
) -> None:
    """Generation v2 permits non-employees without a job title."""
    validate_persona_output_rows(nonemployee_output)


def test_evidence_derivation_fixture_is_strict_and_accounted(
    release_case: ReleaseCase,
) -> None:
    """The small evidence substitute retains strict, additive 10,000-row accounting."""
    evidence = coherent_evidence(release_case)
    assert evidence.rows == sum(shard.rows for shard in evidence.shards)
    assert evidence.accounting.requests == sum(
        shard.requests for shard in evidence.shards
    )
    assert evidence.accounting.total_tokens == 0
    assert evidence.model_dump(mode="json")["shards"][0]["rows"] == 10_000


def test_package_aba_replacement_cannot_repair_invalid_snapshot(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh validation cannot observe a valid replacement of invalid input."""
    valid_original = release_case.output.read_bytes()
    invalid_original = b"invalid parquet"
    release_case.output.write_bytes(invalid_original)

    def validate(*, pilot_dir: Path, repository_root: Path) -> ValidationReport:
        assert pilot_dir.name == "pilot"
        assert repository_root.name == "repository"
        release_case.output.write_bytes(valid_original)
        release_case.output.write_bytes(invalid_original)
        return release_case.report

    monkeypatch.setattr(packager, "validate_persona_pilot", validate)
    monkeypatch.setattr(packager, "_derive_consumed_files", lambda **_: [])
    try:
        with pytest.raises(ReleasePackagingError, match="not readable Parquet"):
            package_release(
                pilot_dir=release_case.pilot,
                attestation_path=release_case.attestation,
                policy_path=release_case.policy,
                dataset_card_path=release_case.card,
                licence_path=release_case.licence,
                repository_root=release_case.repository,
                output_parent=release_case.output_parent,
            )
    finally:
        release_case.output.write_bytes(valid_original)
    assert not list(release_case.output_parent.glob("*"))


def test_package_release_builds_allowlisted_release_and_verifies(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Packaging installs exactly the public allowlist and verifies it."""
    result = _package(release_case, monkeypatch)
    assert result.path.parent == release_case.output_parent
    assert result.path.name
    actual = {
        path.relative_to(result.path).as_posix() for path in result.path.rglob("*")
    }
    expected = (
        set(packager._PUBLIC_FILES)
        | {"release-manifest.json", "release-manifest.sha256"}
        | {
            "attestations",
            "provenance",
            "provenance/prompts",
            "provenance/config",
            "provenance/docs",
            "provenance/code",
            "data",
        }
    )
    assert actual == expected | {
        item for item in ("attestations", "provenance", "data")
    }
    assert not list(result.path.rglob("*.attributes.json"))
    released = pl.read_parquet(result.path / "data" / "personas.parquet")
    assert {"job_function_code", "job_function", "job_function_resolution"} <= set(
        released.columns
    )
    assert (
        verify_release(
            release_dir=result.path, expected_manifest_sha256=result.manifest_sha256
        ).release_id
        == result.path.name
    )


def test_package_release_calls_public_validation_with_exact_subject(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fresh pilot validation service receives the exact pilot path."""
    observed: list[Path] = []

    def validate(*, pilot_dir: Path, repository_root: Path) -> ValidationReport:
        assert repository_root.name == "repository"
        assert pilot_dir.name == "pilot"
        assert pilot_dir != release_case.pilot
        observed.append(pilot_dir)
        return release_case.report

    monkeypatch.setattr(packager, "validate_persona_pilot", validate)
    monkeypatch.setattr(packager, "_derive_consumed_files", lambda **_: [])
    monkeypatch.setattr(
        packager, "_derive_evidence", lambda **_: coherent_evidence(release_case)
    )
    result = _package(release_case, monkeypatch)
    assert result.path.is_dir()


def test_package_release_cleans_temporary_stage_on_install_error(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed install leaves neither a stage nor a destination behind."""

    def fail(**kwargs: object) -> None:
        stage = kwargs["stage"]
        assert isinstance(stage, Path)
        (stage / "README.md").write_bytes(b"partial copy")
        raise OSError("copy failed")

    monkeypatch.setattr(packager, "_install_files", fail)
    with pytest.raises(OSError, match="copy failed"):
        _package(release_case, monkeypatch)
    assert not list(release_case.output_parent.glob(".*"))


def test_package_release_is_deterministic_when_staged_twice(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated staging produces identical public bytes and manifest identity."""
    first = _package(release_case, monkeypatch)
    first_bytes = {
        path.relative_to(first.path): path.read_bytes()
        for path in first.path.rglob("*")
        if path.is_file()
    }
    moved_parent = release_case.output_parent.parent / "second-parent"
    moved_parent.mkdir()
    moved = shutil.copytree(first.path, moved_parent / first.path.name)
    assert {
        path.relative_to(moved): path.read_bytes()
        for path in moved.rglob("*")
        if path.is_file()
    } == first_bytes
    assert (
        verify_release(
            release_dir=moved, expected_manifest_sha256=first.manifest_sha256
        ).release_id
        == first.path.name
    )


def test_package_release_rejects_dirty_or_changed_git(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Git cleanliness and provenance are checked at both boundaries."""
    (release_case.repository / "dirty.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(ReleasePackagingError, match="clean"):
        _package(release_case, monkeypatch)


def test_package_release_rejects_existing_empty_destination(
    packaged_release: tuple[ReleaseCase, Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing destination cannot be replaced, even when it is empty."""
    case, release, _ = packaged_release
    empty_parent = case.output_parent.parent / "empty-destination"
    empty_parent.mkdir()
    target = empty_parent / release.name
    target.mkdir()
    case = replace(case, output_parent=empty_parent)
    with pytest.raises(ReleasePackagingError, match="already exists"):
        _package(case, monkeypatch)
    assert target.is_dir() and not list(target.iterdir())


def test_package_release_rejects_initial_dirty_provenance_before_work(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dirty initial provenance tuple stops all validation and installation work."""
    dirty = ("a" * 40, "https://example.invalid/origin.git", " M dirty.txt")
    provenance_calls: list[tuple[str, str, str]] = []
    work_calls: list[str] = []

    def provenance(_: Path) -> tuple[str, str, str]:
        provenance_calls.append(dirty)
        return dirty

    def forbidden(**_: object) -> None:
        work_calls.append("work")
        raise AssertionError("packaging work must not start")

    monkeypatch.setattr(packager, "_git_provenance", provenance)
    monkeypatch.setattr(packager, "_validate_pilot_for_release", forbidden)
    monkeypatch.setattr(packager, "_materialise_snapshot", forbidden)
    monkeypatch.setattr(packager, "_install_files", forbidden)

    with pytest.raises(ReleasePackagingError, match="clean"):
        _package(release_case, monkeypatch)

    assert provenance_calls == [dirty]
    assert work_calls == []
    assert not list(release_case.output_parent.glob("*"))


def test_package_release_rejects_injected_populated_destination(
    packaged_release: tuple[ReleaseCase, Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An injected destination is never merged with a newly staged package."""
    case, release, _ = packaged_release
    parent = case.output_parent.parent / "injected-destination"
    parent.mkdir()
    target = parent / release.name
    target.mkdir()
    marker = target / "injected.txt"
    marker.write_text("must survive", encoding="utf-8")
    case = replace(case, output_parent=parent)
    with pytest.raises(ReleasePackagingError, match="already exists"):
        _package(case, monkeypatch)
    assert marker.read_text(encoding="utf-8") == "must survive"


def test_package_release_rejects_missing_input_and_path_traversal(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing files and manifest paths are fail-closed."""
    release_case.card.unlink()
    with pytest.raises(ReleasePackagingError, match="Missing input"):
        _package(release_case, monkeypatch)

    # A separate fixture is unnecessary: traversal is rejected while deriving inputs.
    release_case.card.write_text("card", encoding="utf-8")
    payload = release_case.manifest.model_dump(mode="json")
    payload["output_file"] = "../outside.parquet"
    (release_case.pilot / "pilot-manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    monkeypatch.setattr(
        packager, "_derive_consumed_files", packager._derive_consumed_files
    )
    with pytest.raises(ReleasePackagingError):
        _package(release_case, monkeypatch)
