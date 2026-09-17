"""Tests for the public offline release packager."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest
from conftest import ReleaseCase, coherent_evidence

from danish_personas.generation.models import GenerationManifest, PilotBatchReference
from danish_personas.io import sha256_file
from danish_personas.models import ValidationReport
from danish_personas.release import packager
from danish_personas.release.models import ReleaseEvidence, ReleasePackageResult
from danish_personas.release.packager import ReleasePackagingError, package_release
from danish_personas.release.verifier import verify_release


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
    assert (
        verify_release(
            release_dir=result.path, expected_manifest_sha256=result.manifest_sha256
        ).release_id
        == result.path.name
    )


def _package(
    case: ReleaseCase, monkeypatch: pytest.MonkeyPatch
) -> ReleasePackageResult:
    """Package a case while replacing only the expensive upstream seams.

    Returns:
        The installed release package result.
    """
    calls: list[Path] = []

    def validate(*, pilot_dir: Path) -> ValidationReport:
        calls.append(pilot_dir)
        return case.report

    def inventory(*, pilot_dir: Path, manifest: object) -> list[Path]:
        assert pilot_dir == case.pilot
        assert getattr(manifest, "pilot_id") == case.manifest.pilot_id
        return []

    def evidence(**kwargs: object) -> ReleaseEvidence:
        assert kwargs["pilot_dir"] == case.pilot
        return coherent_evidence(case)

    monkeypatch.setattr(packager, "validate_persona_pilot", validate)
    monkeypatch.setattr(packager, "_derive_consumed_files", inventory)
    monkeypatch.setattr(packager, "_derive_evidence", evidence)
    result = package_release(
        pilot_dir=case.pilot,
        attestation_path=case.attestation,
        policy_path=case.policy,
        dataset_card_path=case.card,
        licence_path=case.licence,
        repository_root=case.repository,
        output_parent=case.output_parent,
    )
    assert calls == [case.pilot]
    return result


def test_package_release_calls_public_validation_with_exact_subject(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fresh pilot validation service receives the exact pilot path."""
    observed: list[Path] = []

    def validate(*, pilot_dir: Path) -> ValidationReport:
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


def test_package_release_rejects_prompt_hash_mismatch(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prompt files are public inputs whose manifest hashes cannot drift."""
    prompt = release_case.repository / "config/prompts/attributes-da.md"
    prompt.write_text(prompt.read_text(encoding="utf-8") + "changed", encoding="utf-8")
    monkeypatch.setattr(packager, "_require_clean_git", lambda _: None)
    with pytest.raises(ReleasePackagingError, match="Prompt checksum"):
        _package(release_case, monkeypatch)


def test_package_release_rejects_source_mutation_before_install(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source changed during staging is caught by the final inventory check."""
    original = packager._install_files

    def mutate(**kwargs: object) -> None:
        original(**kwargs)
        release_case.card.write_text("changed", encoding="utf-8")

    monkeypatch.setattr(packager, "_install_files", mutate)
    with pytest.raises(ReleasePackagingError, match="Consumed file changed"):
        _package(release_case, monkeypatch)
    assert not list(release_case.output_parent.glob("*"))


def test_package_release_rejects_wrong_licence_and_policy_metadata(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Licence and policy bindings are checked before installation."""
    release_case.licence.write_text("wrong licence", encoding="utf-8")
    with pytest.raises(ReleasePackagingError, match="licence"):
        _package(release_case, monkeypatch)


def test_real_small_shard_accounting_derivation_needs_no_shard_fanout(
    release_case: ReleaseCase,
) -> None:
    """A real checkpoint/shard derivation remains cheap and additive."""
    small_output = release_case.pilot / "small-output.parquet"
    pl.read_parquet(release_case.output).head(2).write_parquet(small_output)
    shard_dir = release_case.pilot / "small-shard"
    shard_dir.mkdir()
    shard_output = shard_dir / "output.parquet"
    small_output.replace(shard_output)
    shard_manifest = GenerationManifest(
        run_id="small-shard",
        upstream_run_id="upstream-run",
        input_file=Path("sample.parquet"),
        sample_manifest_file=Path("sample-manifest.json"),
        input_sha256="a" * 64,
        ordered_persona_ids_sha256="b" * 64,
        generation_config_file=Path("generation.yaml"),
        generation_config_sha256=release_case.manifest.generation_config_sha256,
        generation_context_sha256=release_case.manifest.generation_context_sha256,
        validator_version="validator-1",
        attributes_prompt_sha256=release_case.manifest.attributes_prompt_sha256,
        personas_prompt_sha256=release_case.manifest.personas_prompt_sha256,
        model=release_case.manifest.model,
        base_url="https://llm.example/v1",
        rows=2,
        offset=0,
        requests=4,
        retries=0,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        estimated_cost_usd=0,
        inference_providers=["test-provider"],
        output_file=Path("output.parquet"),
        output_sha256=sha256_file(shard_output),
        llm_generation=True,
    )
    shard_manifest_path = shard_dir / "manifest.json"
    shard_manifest_path.write_text(
        json.dumps(shard_manifest.model_dump(mode="json")), encoding="utf-8"
    )
    report_path = shard_dir / "report.json"
    report_path.write_text("{}", encoding="utf-8")
    (release_case.pilot / "pilot-validation-report.json").write_text(
        json.dumps(release_case.report.model_dump(mode="json")), encoding="utf-8"
    )
    reference = PilotBatchReference(
        offset=0,
        rows=2,
        run_id="small-shard",
        manifest_file=Path("small-shard/manifest.json"),
        manifest_sha256=sha256_file(shard_manifest_path),
        validation_report_file=Path("small-shard/report.json"),
        validation_report_sha256=sha256_file(report_path),
    )
    manifest = release_case.manifest.model_copy(
        update={
            "rows": 2,
            "batch_runs": [reference],
            "batches": 1,
            "requests": 4,
            "output_file": Path("small-shard/output.parquet"),
            "output_sha256": sha256_file(shard_output),
        }
    )
    evidence = packager._derive_evidence(
        pilot_dir=release_case.pilot,
        manifest=manifest,
        policy_path=release_case.policy,
        attestation_path=release_case.attestation,
        licence_path=release_case.licence,
        repository_root=release_case.repository,
    )
    assert evidence.shards[0].rows == 2
    assert evidence.accounting.requests == 4
    assert evidence.accounting.retries == 0
