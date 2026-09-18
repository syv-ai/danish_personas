"""Tests for the public offline release packager."""

from __future__ import annotations

import json
import shutil
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from conftest import ReleaseCase, coherent_evidence

from danish_personas.generation.models import (
    GenerationConfig,
    GenerationManifest,
    PilotBatchReference,
)
from danish_personas.generation.pipeline import generation_context_sha256
from danish_personas.io import load_yaml_model, sha256_file, write_json
from danish_personas.models import ValidationReport
from danish_personas.release import packager
from danish_personas.release.models import (
    ReleaseEvidence,
    ReleaseManifest,
    ReleasePackageResult,
)
from danish_personas.release.packager import ReleasePackagingError, package_release
from danish_personas.release.verifier import verify_release


def _clean_provenance(_: Path) -> tuple[str, str, str]:
    return "a" * 40, "https://example.invalid/origin.git", ""


def test_aba_replacement_cannot_change_validation_snapshot(tmp_path: Path) -> None:
    """A valid replacement at the original path cannot repair captured input."""
    repository = tmp_path / "repository"
    pilot = tmp_path / "pilot"
    repository.mkdir()
    pilot.mkdir()
    checkpoint = pilot / "checkpoint.json"
    checkpoint.write_bytes(b"invalid")
    inventory = packager._snapshot_inventory(paths=[checkpoint])
    snapshot = packager._materialise_snapshot(
        inventory=inventory, pilot_dir=pilot, repository_root=repository
    )
    try:
        checkpoint.write_bytes(b"valid")
        checkpoint.write_bytes(b"invalid")
        assert (snapshot / "pilot/checkpoint.json").read_bytes() == b"invalid"
    finally:
        shutil.rmtree(snapshot)


def test_capture_accepts_windows_path_and_descriptor_identity_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows identity fields may differ between path and descriptor stats."""
    source = tmp_path / "source.bin"
    content = b"captured"
    source.write_bytes(content)
    actual = source.stat()
    path_calls = 0
    fd_calls = 0

    def stat_view(offset: int, *, path_observer: bool) -> SimpleNamespace:
        return SimpleNamespace(
            st_mode=actual.st_mode ^ (0o1 if path_observer else 0),
            st_nlink=actual.st_nlink,
            st_size=actual.st_size,
            st_dev=actual.st_dev + offset,
            st_ino=actual.st_ino + offset,
            st_mtime_ns=actual.st_mtime_ns + (1000 if path_observer else 0),
            st_ctime_ns=actual.st_ctime_ns + offset,
        )

    def fake_lstat(_: object) -> SimpleNamespace:
        nonlocal path_calls
        path_calls += 1
        return stat_view(path_calls, path_observer=True)

    def fake_fstat(_: int) -> SimpleNamespace:
        nonlocal fd_calls
        fd_calls += 1
        return stat_view(100 + fd_calls, path_observer=False)

    monkeypatch.setattr(packager, "_WINDOWS", True)
    monkeypatch.setattr(packager, "_WINDOWS_NATIVE", False)
    monkeypatch.setattr(packager.os, "lstat", fake_lstat)
    monkeypatch.setattr(packager.os, "fstat", fake_fstat)

    item = packager._capture_file(source)

    assert item.content == content
    assert item.sha256 == packager.sha256_bytes(content)


def test_capture_inventory_keeps_captured_bytes_after_replacement(
    tmp_path: Path,
) -> None:
    """Packaging bytes remain bound to the descriptor capture."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"captured")
    inventory = packager._snapshot_inventory(paths=[source])
    source.write_bytes(b"replacement")
    assert packager._captured_bytes(inventory, source) == b"captured"
    with pytest.raises(ReleasePackagingError, match="Consumed file changed"):
        packager._recheck_inventory(inventory)


def test_capture_rejects_windows_size_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows capture rejects a path observation with a changed size."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"captured")
    actual = source.stat()
    path_calls = 0

    def fake_lstat(_: object) -> SimpleNamespace:
        nonlocal path_calls
        path_calls += 1
        return SimpleNamespace(
            st_mode=actual.st_mode,
            st_nlink=actual.st_nlink,
            st_size=actual.st_size + (path_calls > 1),
            st_dev=actual.st_dev,
            st_ino=actual.st_ino,
            st_mtime_ns=actual.st_mtime_ns + path_calls,
            st_ctime_ns=actual.st_ctime_ns,
        )

    monkeypatch.setattr(packager, "_WINDOWS", True)
    monkeypatch.setattr(packager, "_WINDOWS_NATIVE", False)
    monkeypatch.setattr(packager.os, "lstat", fake_lstat)

    with pytest.raises(ReleasePackagingError, match="Input metadata changed"):
        packager._capture_file(source)


def test_capture_uses_binary_flag_and_preserves_binary_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Descriptor capture preserves binary bytes and requests binary mode."""
    source = tmp_path / "source.bin"
    content = b"prefix\r\n\x1a\r\nparquet-bytes\x1a\r\n"
    source.write_bytes(content)
    if packager._WINDOWS_NATIVE:
        # Native capture uses CreateFileW and an explicit reparse-point denial
        # contract instead of the POSIX os.open flags below.
        assert (
            packager._WINDOWS_FINAL_FLAGS
            & packager._WINDOWS_FILE_FLAG_OPEN_REPARSE_POINT
        )
        assert packager._WINDOWS_FINAL_SHARE_MODE == packager._WINDOWS_FILE_SHARE_READ
        item = packager._capture_file(source)
    else:
        binary_flag = getattr(packager.os, "O_BINARY", 0)
        captured_flags: list[int] = []
        original_open = packager.os.open

        def capture_open(path: Path, flags: int, *args: int) -> int:
            captured_flags.append(flags)
            return original_open(path, flags, *args)

        monkeypatch.setattr(packager.os, "open", capture_open)
        item = packager._capture_file(source)

        assert captured_flags
        assert binary_flag == 0 or captured_flags[0] & binary_flag == binary_flag

    assert item.content == content
    assert item.size == len(content)
    assert item.sha256 == packager.sha256_bytes(content)


def test_capture_uses_separate_windows_observer_stability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows descriptor mtime drift does not mask path stability checks."""
    source = tmp_path / "source.bin"
    content = b"captured"
    source.write_bytes(content)
    actual = source.stat()
    fd_calls = 0

    def fake_lstat(_: object) -> SimpleNamespace:
        return SimpleNamespace(
            st_mode=actual.st_mode,
            st_nlink=actual.st_nlink,
            st_size=actual.st_size,
            st_dev=actual.st_dev + 1,
            st_ino=actual.st_ino + 1,
            st_mtime_ns=actual.st_mtime_ns,
            st_ctime_ns=actual.st_ctime_ns + 1,
        )

    def fake_fstat(_: int) -> SimpleNamespace:
        nonlocal fd_calls
        fd_calls += 1
        return SimpleNamespace(
            st_mode=actual.st_mode,
            st_nlink=actual.st_nlink,
            st_size=actual.st_size,
            st_dev=actual.st_dev + 100 + fd_calls,
            st_ino=actual.st_ino + 100 + fd_calls,
            st_mtime_ns=actual.st_mtime_ns + fd_calls,
            st_ctime_ns=actual.st_ctime_ns + 100 + fd_calls,
        )

    monkeypatch.setattr(packager, "_WINDOWS", True)
    monkeypatch.setattr(packager, "_WINDOWS_NATIVE", False)
    monkeypatch.setattr(packager.os, "lstat", fake_lstat)
    monkeypatch.setattr(packager.os, "fstat", fake_fstat)

    item = packager._capture_file(source)

    assert item.content == content


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

    def validate(*, pilot_dir: Path, repository_root: Path) -> ValidationReport:
        assert repository_root.name == "repository"
        assert pilot_dir.name == "pilot"
        assert pilot_dir != case.pilot
        calls.append(pilot_dir)
        return case.report

    def inventory(
        *, pilot_dir: Path, repository_root: Path, manifest: object
    ) -> list[Path]:
        assert pilot_dir == case.pilot
        assert repository_root == case.repository
        assert getattr(manifest, "pilot_id") == case.manifest.pilot_id
        return []

    def evidence(**kwargs: object) -> ReleaseEvidence:
        snapshot_pilot = kwargs["pilot_dir"]
        assert isinstance(snapshot_pilot, Path)
        assert snapshot_pilot.name == "pilot"
        evidence = coherent_evidence(case)
        return evidence.model_copy(
            update={
                "pilot_validation_report_sha256": sha256_file(
                    snapshot_pilot / "pilot-validation-report.json"
                )
            }
        )

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
    assert len(calls) == 1
    assert calls[0].name == "pilot"
    return result


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


@pytest.mark.parametrize(
    "mutation",
    ["restore_checkpoint", "add_checkpoint", "remove_ledger", "add_unrelated"],
)
def test_package_release_rejects_pilot_tree_membership_changes(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """Pilot checkpoint and ledger membership cannot change during staging."""
    shard = release_case.pilot / "shard"
    checkpoints = shard / "checkpoints"
    ledger = shard / "request-ledger.json"
    if mutation == "add_checkpoint":
        checkpoints.mkdir(parents=True)
        (checkpoints / "existing.json").write_text("{}", encoding="utf-8")
    elif mutation == "remove_ledger":
        shard.mkdir()
        ledger.write_text("{}", encoding="utf-8")

    original = packager._install_files

    def mutate(**kwargs: object) -> None:
        original(**kwargs)
        if mutation in {"restore_checkpoint", "add_checkpoint"}:
            checkpoints.mkdir(parents=True, exist_ok=True)
            (checkpoints / "restored.json").write_text("{}", encoding="utf-8")
        elif mutation == "remove_ledger":
            ledger.unlink()
        else:
            (release_case.pilot / "unexpected.txt").write_text(
                "unexpected", encoding="utf-8"
            )

    monkeypatch.setattr(packager, "_install_files", mutate)
    with pytest.raises(ReleasePackagingError, match="Pilot tree membership changed"):
        _package(release_case, monkeypatch)
    assert not list(release_case.output_parent.glob("*"))


def test_package_release_rejects_prompt_hash_mismatch(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prompt files are public inputs whose manifest hashes cannot drift."""
    prompt = release_case.repository / "config/prompts/attributes-da.md"
    prompt.write_text(prompt.read_text(encoding="utf-8") + "changed", encoding="utf-8")
    monkeypatch.setattr(packager, "_git_provenance", _clean_provenance)
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


def test_package_uses_captured_bytes_during_copy_time_replacement(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replacement cannot alter bytes selected for the package."""
    captured_card = release_case.card.read_bytes()
    original = packager._install_files
    observed: list[bytes] = []

    def replace_before_copy(**kwargs: object) -> None:
        release_case.card.write_bytes(b"replacement")
        original(**kwargs)
        stage = kwargs["stage"]
        assert isinstance(stage, Path)
        observed.append((stage / "README.md").read_bytes())

    monkeypatch.setattr(packager, "_install_files", replace_before_copy)
    with pytest.raises(ReleasePackagingError, match="Consumed file changed"):
        _package(release_case, monkeypatch)
    assert observed == [captured_card]


def test_package_uses_manifest_effective_generation_config(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ignored local config is snapshotted as the effective generation config."""
    local_path = release_case.repository / "config/generation.local.yaml"
    local_path.write_bytes(
        (release_case.repository / "config/generation.yaml")
        .read_bytes()
        .replace(b"TEST_TOKEN", b"LOCAL_TOKEN")
    )
    config = load_yaml_model(path=local_path, model=GenerationConfig)
    context = generation_context_sha256(
        config=config,
        attributes_prompt=(
            release_case.repository / "config/prompts/attributes-da.md"
        ).read_text(encoding="utf-8"),
        personas_prompt=(
            release_case.repository / "config/prompts/personas-da.md"
        ).read_text(encoding="utf-8"),
    )
    manifest = release_case.manifest.model_copy(
        update={
            "generation_config_file": Path("config/generation.local.yaml"),
            "generation_config_sha256": sha256_file(local_path),
            "generation_context_sha256": context,
        }
    )
    write_json(path=release_case.pilot / "pilot-manifest.json", payload=manifest)
    case = replace(release_case, manifest=manifest)
    config_hashes = {
        **coherent_evidence(case).config_hashes,
        "generation.yaml": sha256_file(local_path),
    }

    def evidence(**kwargs: object) -> ReleaseEvidence:
        snapshot_pilot = kwargs["pilot_dir"]
        assert isinstance(snapshot_pilot, Path)
        return coherent_evidence(case).model_copy(
            update={
                "config_hashes": config_hashes,
                "pilot_validation_report_sha256": sha256_file(
                    snapshot_pilot / "pilot-validation-report.json"
                ),
            }
        )

    monkeypatch.setattr(packager, "_git_provenance", _clean_provenance)
    monkeypatch.setattr(packager, "validate_persona_pilot", lambda **_: case.report)
    monkeypatch.setattr(packager, "_derive_consumed_files", lambda **_: [])
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
    assert (
        result.path / "provenance/config/generation.yaml"
    ).read_bytes() == local_path.read_bytes()


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


@pytest.mark.parametrize("version", [True, 1.0, "1", b"1"])
def test_release_versions_require_exact_integer_one(
    release_case: ReleaseCase, version: object
) -> None:
    """Release contracts reject values that Pydantic could coerce to one."""
    manifest = release_case.manifest
    manifest_payload = {
        "version": version,
        "release_id": "a" * 32,
        "created_at": "2026-09-17T00:00:00+00:00",
        "pilot_id": manifest.pilot_id,
        "model": manifest.model,
        "rows": 1,
        "git_head": "a" * 40,
        "origin_url": "https://example.invalid/repo.git",
        "uv_lock_sha256": "a" * 64,
        "evidence_sha256": "a" * 64,
        "artifacts": [
            {"path": "README.md", "role": "dataset-card", "sha256": "a" * 64, "size": 0}
        ],
    }
    evidence_payload = coherent_evidence(release_case).model_dump(mode="json")
    evidence_payload["version"] = version
    with pytest.raises(ValueError):
        ReleaseManifest.model_validate(manifest_payload)
    with pytest.raises(ValueError):
        ReleaseEvidence.model_validate(evidence_payload)


def test_scanner_accepts_only_nullable_career_goals(release_case: ReleaseCase) -> None:
    """Only the optional all-null career field may use Polars Null dtype."""
    output = pl.read_parquet(release_case.output)
    safe = output.with_columns(
        pl.lit(None, dtype=pl.Null).alias("career_goals_and_ambitions")
    )
    packager._scan_dataframe(safe)
    unsafe = output.with_columns(pl.lit(None, dtype=pl.Null).alias("cultural_context"))
    with pytest.raises(ReleasePackagingError, match="logical dtype"):
        packager._scan_dataframe(unsafe)


def test_snapshot_materialisation_preserves_repository_and_pilot_paths(
    tmp_path: Path,
) -> None:
    """Captured files are copied without links and with relative semantics."""
    repository = tmp_path / "repository"
    pilot = tmp_path / "pilot"
    repository.mkdir()
    pilot.mkdir()
    repository_file = repository / "config" / "generation.yaml"
    pilot_file = pilot / "checkpoints" / "persona.json"
    repository_file.parent.mkdir()
    pilot_file.parent.mkdir()
    repository_file.write_bytes(b"repository")
    pilot_file.write_bytes(b"pilot")
    inventory = packager._snapshot_inventory(paths=[repository_file, pilot_file])
    snapshot = packager._materialise_snapshot(
        inventory=inventory, pilot_dir=pilot, repository_root=repository
    )
    try:
        assert (
            snapshot / "repository/config/generation.yaml"
        ).read_bytes() == b"repository"
        assert (snapshot / "pilot/checkpoints/persona.json").read_bytes() == b"pilot"
        assert not (snapshot / "repository/config/generation.yaml").is_symlink()
        assert not (snapshot / "pilot/checkpoints/persona.json").is_symlink()
    finally:
        shutil.rmtree(snapshot)


def test_supplied_path_with_symlink_parent_is_rejected(tmp_path: Path) -> None:
    """Path checks inspect the first relative component and every parent."""
    real = tmp_path / "real"
    real.mkdir()
    (real / "input.txt").write_text("input", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ReleasePackagingError, match="symlink"):
        packager._require_regular_file(link / "input.txt")


def test_windows_capture_contract_uses_no_follow_and_stable_identity() -> None:
    """Native file and parent handles deny mutation sharing."""
    first = packager._WindowsFileInfo(
        attributes=0,
        volume_serial=7,
        file_index=2**40,
        size=17,
        number_of_links=1,
        write_time=19,
    )
    same = replace(first)
    changed = replace(first, file_index=2**40 + 1)

    assert (
        packager._WINDOWS_FINAL_FLAGS & packager._WINDOWS_FILE_FLAG_OPEN_REPARSE_POINT
    )
    assert packager._WINDOWS_FINAL_SHARE_MODE == packager._WINDOWS_FILE_SHARE_READ
    assert not packager._WINDOWS_FINAL_SHARE_MODE & packager._WINDOWS_FILE_SHARE_WRITE
    assert not packager._WINDOWS_FINAL_SHARE_MODE & packager._WINDOWS_FILE_SHARE_DELETE
    assert packager._WINDOWS_DIRECTORY_SHARE_MODE == packager._WINDOWS_FILE_SHARE_READ
    assert not (
        packager._WINDOWS_DIRECTORY_SHARE_MODE & packager._WINDOWS_FILE_SHARE_WRITE
    )
    assert not packager._WINDOWS_DIRECTORY_SHARE_MODE & (
        packager._WINDOWS_FILE_SHARE_DELETE
    )
    assert packager._windows_observations_match(first, same)
    assert not packager._windows_observations_match(first, changed)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "nonregular"])
@pytest.mark.parametrize("observer", ["path", "fd"])
def test_windows_capture_rejects_unsafe_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, observer: str
) -> None:
    """Windows checks reject unsafe path and descriptor observations."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"captured")
    actual = source.stat()
    unsafe = _unsafe_stat(kind=kind, actual=actual)

    def fake_lstat(_: object) -> object:
        return unsafe if observer == "path" else actual

    def fake_fstat(_: int) -> object:
        return unsafe if observer == "fd" else actual

    monkeypatch.setattr(packager, "_WINDOWS", True)
    monkeypatch.setattr(packager, "_WINDOWS_NATIVE", False)
    monkeypatch.setattr(packager.os, "lstat", fake_lstat)
    monkeypatch.setattr(packager.os, "fstat", fake_fstat)

    with pytest.raises(ReleasePackagingError):
        packager._capture_file(source)


def _unsafe_stat(kind: str, actual: object) -> SimpleNamespace:
    if kind == "symlink":
        mode = stat.S_IFLNK | 0o777
        nlink = 1
    elif kind == "hardlink":
        mode = getattr(actual, "st_mode")
        nlink = 2
    else:
        mode = stat.S_IFDIR | 0o755
        nlink = 1
    return SimpleNamespace(
        st_mode=mode,
        st_nlink=nlink,
        st_size=getattr(actual, "st_size"),
        st_dev=getattr(actual, "st_dev"),
        st_ino=getattr(actual, "st_ino"),
        st_mtime_ns=getattr(actual, "st_mtime_ns"),
        st_ctime_ns=getattr(actual, "st_ctime_ns"),
    )


@pytest.mark.skipif(not packager._WINDOWS_NATIVE, reason="Windows-only mutation test")
def test_windows_parent_handles_block_directory_rename_during_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retained parent handles block renames until capture finishes."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "source.bin"
    source.write_bytes(b"captured")
    renamed = tmp_path / "renamed"
    original_read = packager._read_capture_descriptor

    def read_with_mutation(descriptor: int, *, path: Path) -> bytes:
        with pytest.raises(OSError):
            source_dir.rename(renamed)
        return original_read(descriptor, path=path)

    monkeypatch.setattr(packager, "_read_capture_descriptor", read_with_mutation)
    item = packager._capture_file(source)

    assert item.content == b"captured"
    source_dir.rename(renamed)
    assert (renamed / source.name).read_bytes() == b"captured"


def test_windows_recheck_ignores_unreliable_identity_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows rechecks accept stable files with changed identity fields."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"captured")
    monkeypatch.setattr(packager, "_WINDOWS", True)
    monkeypatch.setattr(packager, "_WINDOWS_NATIVE", False)
    inventory = packager._snapshot_inventory(paths=[source])
    item = inventory[0]
    changed_identity = replace(
        item,
        mode=item.mode ^ 0o1,
        device=item.device + 1,
        inode=item.inode + 1,
        mtime_ns=item.mtime_ns + 1,
        ctime_ns=item.ctime_ns + 1,
    )
    monkeypatch.setattr(packager, "_capture_file", lambda _: changed_identity)

    packager._recheck_inventory(inventory)


@pytest.mark.parametrize("field", ["size", "sha256"])
def test_windows_recheck_rejects_size_or_hash_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """Windows rechecks reject changed size or content hashes."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"captured")
    monkeypatch.setattr(packager, "_WINDOWS", True)
    monkeypatch.setattr(packager, "_WINDOWS_NATIVE", False)
    inventory = packager._snapshot_inventory(paths=[source])
    item = inventory[0]
    value = "0" * 64 if field == "sha256" else getattr(item, field) + 1
    changed = replace(item, **{field: value})
    monkeypatch.setattr(packager, "_capture_file", lambda _: changed)

    with pytest.raises(ReleasePackagingError, match="Consumed file changed"):
        packager._recheck_inventory(inventory)
