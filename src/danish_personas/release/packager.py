"""Offline release packaging with fail-closed provenance checks."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import inspect
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import typing as t
from dataclasses import dataclass, replace
from pathlib import Path

import polars as pl
import yaml

from ..generation.models import (
    GeneratedAttributes,
    GenerationConfig,
    GenerationManifest,
    PersonaCheckpoint,
    PersonaDescriptions,
    PilotManifest,
)
from ..generation.report import validate_persona_pilot
from ..io import canonical_json, sha256_file, write_json
from ..models import DemographicRecord, RunManifest, StrictModel, ValidationReport
from .common import persona_output_dtypes_are_valid, release_id, role
from .models import (
    Accounting,
    Artifact,
    ReleaseEvidence,
    ReleaseManifest,
    ReleasePackageResult,
    ReleasePolicy,
    ReviewAttestation,
    ShardEvidence,
)
from .policy import validate_release_approval
from .verifier import _verify_release


class ReleasePackagingError(ValueError):
    """Raised when a release cannot be safely packaged."""


@dataclass(frozen=True)
class _InventoryItem:
    path: Path
    size: int
    sha256: str
    device: int
    inode: int
    mode: int
    nlink: int
    mtime_ns: int
    ctime_ns: int
    content: bytes
    snapshot_path: Path | None = None


_PUBLIC_FILES = (
    "README.md",
    "LICENSE.txt",
    "data/personas.parquet",
    "attestations/human-review.json",
    "provenance/release-policy.yaml",
    "provenance/evidence.json",
    "provenance/pilot-validation-report.json",
    "provenance/prompts/attributes-da.md",
    "provenance/prompts/personas-da.md",
    "provenance/config/generation.yaml",
    "provenance/config/sources.lock.yaml",
    "provenance/config/categories.yaml",
    "provenance/config/sampling.yaml",
    "provenance/config/validation.yaml",
    "provenance/docs/source-register.md",
    "provenance/docs/privacy-risk-register.md",
    "provenance/docs/acceptance-criteria.md",
    "provenance/code/uv.lock",
    "provenance/code/LICENSE",
)


def package_release(
    *,
    pilot_dir: Path,
    attestation_path: Path,
    policy_path: Path,
    dataset_card_path: Path,
    licence_path: Path,
    repository_root: Path,
    output_parent: Path,
) -> ReleasePackageResult:
    """Create and atomically install a deterministic offline release package.

    All input paths are treated as untrusted, and only the fixed public layout is
    copied.

    Returns:
        The installed package path and externally retainable manifest digest.

    Raises:
        ReleasePackagingError:
            If any release gate or source-integrity check fails.
    """
    _require_directory(pilot_dir)
    _require_directory(repository_root)
    pilot_dir = _lexical_absolute(pilot_dir)
    repository_root = _lexical_absolute(repository_root)
    output_parent = _lexical_absolute(output_parent)
    _require_no_symlink_components(output_parent)
    _require_clean_git(repository_root)
    git_head, origin_url = _git_provenance(repository_root)
    inventory = _snapshot_inventory(paths=[pilot_dir / "pilot-manifest.json"])
    pilot_manifest = _load_captured_json(
        inventory=inventory, path=pilot_dir / "pilot-manifest.json", model=PilotManifest
    )
    _capture_path(path=policy_path, inventory=inventory)
    policy = _load_captured_yaml(
        inventory=inventory, path=policy_path, model=ReleasePolicy
    )
    _capture_path(path=attestation_path, inventory=inventory)
    attestation = _load_captured_json(
        inventory=inventory, path=attestation_path, model=ReviewAttestation
    )
    config_path = _repository_path(
        repository_root, pilot_manifest.generation_config_file
    )
    _capture_path(path=config_path, inventory=inventory)
    config = _load_captured_yaml(
        inventory=inventory, path=config_path, model=GenerationConfig
    )
    attributes_path = _repository_path(repository_root, config.attributes_prompt)
    personas_path = _repository_path(repository_root, config.personas_prompt)
    _capture_path(path=attributes_path, inventory=inventory)
    _capture_path(path=personas_path, inventory=inventory)
    _require_public_prompt(
        attributes_path,
        pilot_manifest.attributes_prompt_sha256,
        content=_captured_bytes(inventory, attributes_path),
    )
    _require_public_prompt(
        personas_path,
        pilot_manifest.personas_prompt_sha256,
        content=_captured_bytes(inventory, personas_path),
    )
    _capture_path(path=licence_path, inventory=inventory)
    licence_bytes = _captured_bytes(inventory, licence_path)
    if not licence_bytes:
        raise ReleasePackagingError("The supplied licence is empty")
    if sha256_bytes(licence_bytes) != policy.licence_file_sha256:
        raise ReleasePackagingError("Supplied licence does not match policy")
    _capture_path(path=dataset_card_path, inventory=inventory)
    card_bytes = _captured_bytes(inventory, dataset_card_path)
    if not card_bytes.strip():
        raise ReleasePackagingError("Dataset card must be non-empty")

    output_path = _output_path(pilot_dir, pilot_manifest.output_file)
    _derive_consumed_for_package(
        pilot_dir=pilot_dir,
        repository_root=repository_root,
        manifest=pilot_manifest,
        inventory=inventory,
    )
    _capture_path(path=output_path, inventory=inventory)
    for name in (
        "uv.lock",
        "LICENSE",
        "config/sources.lock.yaml",
        "config/categories.yaml",
        "config/sampling.yaml",
        "config/validation.yaml",
        "docs/source-register.md",
        "docs/privacy-risk-register.md",
        "docs/acceptance-criteria.md",
    ):
        _capture_path(path=repository_root / name, inventory=inventory)

    snapshot_root = _materialise_snapshot(
        inventory=inventory,
        pilot_dir=pilot_dir,
        repository_root=repository_root,
        external_paths=(attestation_path, policy_path, dataset_card_path, licence_path),
    )
    snapshot_repository = snapshot_root / "repository"
    snapshot_pilot = snapshot_root / "pilot"
    try:
        report, evidence, output = _validate_pilot_for_release(
            pilot_dir=snapshot_pilot,
            pilot_manifest=pilot_manifest,
            policy=policy,
            attestation=attestation,
            config=config,
            config_path=snapshot_repository / config_path.relative_to(repository_root),
            attributes_path=snapshot_repository
            / attributes_path.relative_to(repository_root),
            personas_path=(
                snapshot_repository / personas_path.relative_to(repository_root)
            ),
            policy_path=policy_path,
            attestation_path=attestation_path,
            licence_path=licence_path,
            repository_root=snapshot_repository,
            inventory=inventory,
            card_bytes=card_bytes,
            licence_bytes=licence_bytes,
        )
    finally:
        shutil.rmtree(snapshot_root, ignore_errors=True)

    _recheck_inventory(inventory)
    _require_clean_git(repository_root)
    git_head_after, origin_after = _git_provenance(repository_root)
    if (git_head_after, origin_after) != (git_head, origin_url):
        raise ReleasePackagingError("Git provenance changed during packaging")

    output_parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_parent / ".release-package.lock"
    lock_owned = False
    stage: Path | None = None
    try:
        _acquire_lock(lock_path)
        lock_owned = True
        release_identifier = release_id(
            pilot_id=pilot_manifest.pilot_id,
            output_sha256=pilot_manifest.output_sha256,
            reviewed_at=attestation.reviewed_at,
            git_head=git_head,
            origin_url=origin_url,
        )
        destination = output_parent / release_identifier
        if destination.exists() or destination.is_symlink():
            raise ReleasePackagingError("Release destination already exists")
        stage = Path(
            tempfile.mkdtemp(prefix=f".{release_identifier}.", dir=output_parent)
        )
        _install_files(
            stage=stage,
            card=card_bytes,
            licence=licence_bytes,
            output_path=_snapshot_path(inventory, output_path),
            attestation_path=attestation_path,
            policy_path=policy_path,
            evidence=evidence,
            report=report,
            config_path=_snapshot_path(inventory, config_path),
            attributes_path=_snapshot_path(inventory, attributes_path),
            personas_path=_snapshot_path(inventory, personas_path),
            repository_root=snapshot_repository,
            inventory=inventory,
            card_path=dataset_card_path,
            licence_path=licence_path,
        )
        artifacts = tuple(
            Artifact(
                path=path,
                role=role(path),
                sha256=sha256_file(stage / path),
                size=(stage / path).stat().st_size,
            )
            for path in _PUBLIC_FILES
        )
        manifest = ReleaseManifest(
            version=1,
            release_id=release_identifier,
            created_at=attestation.reviewed_at,
            pilot_id=pilot_manifest.pilot_id,
            model=pilot_manifest.model,
            rows=pilot_manifest.rows,
            git_head=git_head,
            origin_url=origin_url,
            uv_lock_sha256=_source_sha256(repository_root / "uv.lock", inventory),
            evidence_sha256=sha256_file(stage / "provenance/evidence.json"),
            artifacts=artifacts,
        )
        write_json(path=stage / "release-manifest.json", payload=manifest)
        manifest_sha256 = sha256_file(stage / "release-manifest.json")
        (stage / "release-manifest.sha256").write_text(
            f"{manifest_sha256}  release-manifest.json\n", encoding="ascii"
        )
        _verify_release(
            release_dir=stage,
            expected_manifest_sha256=manifest_sha256,
            allow_staging=True,
        )
        _recheck_inventory(inventory)
        _require_clean_git(repository_root)
        _rename_noreplace(stage, destination)
        stage = None
        return ReleasePackageResult(path=destination, manifest_sha256=manifest_sha256)
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        if lock_owned:
            lock_path.unlink(missing_ok=True)


def _capture_path(*, path: Path, inventory: list[_InventoryItem]) -> None:
    key = _lexical_absolute(path)
    if any(item.path.absolute() == key for item in inventory):
        return
    inventory.extend(_snapshot_inventory(paths=[path]))


def _snapshot_inventory(paths: list[Path]) -> list[_InventoryItem]:
    """Capture each source once with descriptor-anchored reads.

    Returns:
        Immutable source records containing bytes and file metadata.
    """
    result: list[_InventoryItem] = []
    seen: set[Path] = set()
    for path in paths:
        key = _lexical_absolute(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(_capture_file(path))
    return result


def _capture_file(path: Path) -> _InventoryItem:
    """Read a regular file through one descriptor and bind its pathname.

    Returns:
        The immutable bytes and metadata captured from ``path``.

    Raises:
        ReleasePackagingError:
            If the path is not a stable, regular, single-link file.
    """
    candidate = _lexical_absolute(path)
    _require_no_symlink_components(candidate)
    try:
        before_path = os.lstat(candidate)
    except OSError as error:
        raise ReleasePackagingError(f"Missing input file: {path}") from error
    _require_stat_file(path=path, stat_result=before_path)
    flags = os.O_RDONLY
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    flags |= nofollow
    try:
        descriptor = os.open(candidate, flags)
    except OSError as error:
        raise ReleasePackagingError(f"Cannot open input file: {path}") from error
    try:
        before_fd = os.fstat(descriptor)
        _require_stat_file(path=path, stat_result=before_fd)
        if _file_metadata(before_path) != _file_metadata(before_fd):
            raise ReleasePackagingError(f"Input metadata changed: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        content = b"".join(chunks)
        after_fd = os.fstat(descriptor)
        after_path = os.lstat(candidate)
        if (
            _file_metadata(before_fd) != _file_metadata(after_fd)
            or _file_metadata(before_path) != _file_metadata(after_path)
            or len(content) != before_fd.st_size
        ):
            raise ReleasePackagingError(f"Input metadata changed: {path}")
        return _InventoryItem(
            path=candidate,
            size=before_fd.st_size,
            sha256=sha256_bytes(content),
            device=before_fd.st_dev,
            inode=before_fd.st_ino,
            mode=before_fd.st_mode,
            nlink=before_fd.st_nlink,
            mtime_ns=before_fd.st_mtime_ns,
            ctime_ns=before_fd.st_ctime_ns,
            content=content,
        )
    finally:
        os.close(descriptor)


def _file_metadata(stat: os.stat_result) -> tuple[int, ...]:
    return (
        stat.st_mode,
        stat.st_nlink,
        stat.st_size,
        stat.st_dev,
        stat.st_ino,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )


def _require_stat_file(*, path: Path, stat_result: os.stat_result) -> None:
    if not stat.S_ISREG(stat_result.st_mode):
        raise ReleasePackagingError(f"Input must be a regular file: {path}")
    if stat_result.st_nlink != 1:
        raise ReleasePackagingError(f"Input must be a regular non-linked file: {path}")


def _captured_bytes(inventory: list[_InventoryItem], path: Path) -> bytes:
    key = _lexical_absolute(path)
    for item in inventory:
        if item.path.absolute() == key or (
            item.snapshot_path is not None and item.snapshot_path.absolute() == key
        ):
            return item.content
    raise ReleasePackagingError(f"Source was not captured: {path}")


def _derive_consumed_for_package(
    *,
    pilot_dir: Path,
    repository_root: Path,
    manifest: PilotManifest,
    inventory: list[_InventoryItem],
) -> None:
    """Derive consumed files while retaining compatibility with test seams."""
    parameters = inspect.signature(_derive_consumed_files).parameters
    if "inventory" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        _derive_consumed_files(
            pilot_dir=pilot_dir,
            repository_root=repository_root,
            manifest=manifest,
            inventory=inventory,
        )
        return
    derived = _derive_consumed_files(
        pilot_dir=pilot_dir, repository_root=repository_root, manifest=manifest
    )
    inventory.extend(_snapshot_inventory(paths=derived))


def _derive_consumed_files(
    *,
    pilot_dir: Path,
    repository_root: Path,
    manifest: PilotManifest,
    inventory: list[_InventoryItem] | None = None,
) -> list[Path]:
    """Derive the strict input inventory from pilot and shard manifests.

    Returns:
        Paths consumed by validation and evidence derivation.

    Raises:
        ReleasePackagingError:
            If a manifest references a missing, unsafe, or malformed file.
    """
    captured = inventory if inventory is not None else []
    paths = [pilot_dir / "pilot-manifest.json"]
    _capture_path(path=paths[0], inventory=captured)
    paths.append(_output_path(pilot_dir, manifest.output_file))
    _capture_path(path=paths[-1], inventory=captured)
    paths.append(_repository_path(repository_root, manifest.input_file))
    _capture_path(path=paths[-1], inventory=captured)
    paths.append(_repository_path(repository_root, manifest.sample_manifest_file))
    _capture_path(path=paths[-1], inventory=captured)
    paths.append(_repository_path(repository_root, manifest.generation_config_file))
    _capture_path(path=paths[-1], inventory=captured)
    sample_path = _repository_path(repository_root, manifest.input_file)
    upstream_manifest_path = sample_path.parent / "run-manifest.json"
    upstream_report_path = sample_path.parent / "validation-report.json"
    paths.extend((upstream_manifest_path, upstream_report_path))
    _capture_path(path=upstream_manifest_path, inventory=captured)
    _capture_path(path=upstream_report_path, inventory=captured)
    upstream = _load_captured_json(
        inventory=captured, path=upstream_manifest_path, model=RunManifest
    )
    paths.append(_output_path(sample_path.parent, upstream.data_file))
    _capture_path(path=paths[-1], inventory=captured)
    for reference in manifest.batch_runs:
        manifest_path = _pilot_path(pilot_dir, reference.manifest_file)
        report_path = _pilot_path(pilot_dir, reference.validation_report_file)
        _capture_path(path=manifest_path, inventory=captured)
        _capture_path(path=report_path, inventory=captured)
        shard = _load_captured_json(
            inventory=captured, path=manifest_path, model=GenerationManifest
        )
        paths.extend((manifest_path, report_path))
        paths.append(_output_path(manifest_path.parent, shard.output_file))
        _capture_path(path=paths[-1], inventory=captured)
        paths.extend(
            (
                _repository_path(repository_root, shard.input_file),
                _repository_path(repository_root, shard.sample_manifest_file),
            )
        )
        _capture_path(path=paths[-2], inventory=captured)
        _capture_path(path=paths[-1], inventory=captured)
        if shard.generation_config_file is not None:
            paths.append(
                _repository_path(repository_root, shard.generation_config_file)
            )
            _capture_path(path=paths[-1], inventory=captured)
        checkpoint_dir = manifest_path.parent / "checkpoints"
        if checkpoint_dir.is_symlink():
            raise ReleasePackagingError("Checkpoint directory must not be a symlink")
        if checkpoint_dir.is_dir():
            checkpoint_paths = sorted(checkpoint_dir.glob("*.json"))
            paths.extend(checkpoint_paths)
            for checkpoint_path in checkpoint_paths:
                _capture_path(path=checkpoint_path, inventory=captured)
        ledger = manifest_path.parent / "request-ledger.json"
        if ledger.exists():
            paths.append(ledger)
            _capture_path(path=ledger, inventory=captured)
    return paths


def _load_captured_json(
    *, inventory: list[_InventoryItem], path: Path, model: type[ModelType]
) -> ModelType:
    try:
        return model.model_validate_json(_captured_bytes(inventory, path))
    except Exception as error:
        raise ReleasePackagingError(f"Invalid contract: {path}") from error


def _install_files(**kwargs: object) -> None:
    stage = t.cast(Path, kwargs["stage"])
    output_path = t.cast(Path, kwargs["output_path"])
    attestation_path = t.cast(Path, kwargs["attestation_path"])
    policy_path = t.cast(Path, kwargs["policy_path"])
    evidence = t.cast(ReleaseEvidence, kwargs["evidence"])
    report = t.cast(ValidationReport, kwargs["report"])
    attributes_path = t.cast(Path, kwargs["attributes_path"])
    personas_path = t.cast(Path, kwargs["personas_path"])
    repository_root = t.cast(Path, kwargs["repository_root"])
    config_path = t.cast(Path, kwargs["config_path"])
    inventory = t.cast(list[_InventoryItem], kwargs["inventory"])
    payloads: dict[str, bytes] = {
        "README.md": _captured_bytes(inventory, t.cast(Path, kwargs["card_path"])),
        "LICENSE.txt": _captured_bytes(inventory, t.cast(Path, kwargs["licence_path"])),
        "data/personas.parquet": _captured_bytes(inventory, output_path),
        "attestations/human-review.json": _captured_bytes(inventory, attestation_path),
        "provenance/release-policy.yaml": _captured_bytes(inventory, policy_path),
        "provenance/evidence.json": (
            json.dumps(
                evidence.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
        "provenance/pilot-validation-report.json": (
            json.dumps(
                report.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
        "provenance/prompts/attributes-da.md": _captured_bytes(
            inventory, attributes_path
        ),
        "provenance/prompts/personas-da.md": _captured_bytes(inventory, personas_path),
    }
    for name in (
        "generation.yaml",
        "sources.lock.yaml",
        "categories.yaml",
        "sampling.yaml",
        "validation.yaml",
    ):
        source = (
            config_path
            if name == "generation.yaml"
            else repository_root / "config" / name
        )
        payloads[f"provenance/config/{name}"] = _captured_bytes(inventory, source)
    for name in (
        "source-register.md",
        "privacy-risk-register.md",
        "acceptance-criteria.md",
    ):
        payloads[f"provenance/docs/{name}"] = _captured_bytes(
            inventory, repository_root / "docs" / name
        )
    payloads["provenance/code/uv.lock"] = _captured_bytes(
        inventory, repository_root / "uv.lock"
    )
    payloads["provenance/code/LICENSE"] = _captured_bytes(
        inventory, repository_root / "LICENSE"
    )
    for relative, content in payloads.items():
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def _load_captured_yaml(
    *, inventory: list[_InventoryItem], path: Path, model: type[ModelType]
) -> ModelType:
    try:
        payload: object = yaml.safe_load(
            _captured_bytes(inventory, path).decode("utf-8")
        )
        return model.model_validate(payload)
    except Exception as error:
        raise ReleasePackagingError(f"Invalid contract: {path}") from error


def _materialise_snapshot(
    *,
    inventory: list[_InventoryItem],
    pilot_dir: Path,
    repository_root: Path,
    external_paths: tuple[Path, ...] = (),
) -> Path:
    """Build a private, link-free tree from immutable source records.

    Returns:
        The private snapshot root containing ``repository`` and ``pilot`` trees.

    Raises:
        ReleasePackagingError:
            If a consumed file is outside the explicit snapshot roots.
    """
    snapshot_root = Path(
        os.path.realpath(tempfile.mkdtemp(prefix=".release-snapshot-"))
    )
    repository_root = _lexical_absolute(repository_root)
    pilot_dir = _lexical_absolute(pilot_dir)
    allowed_external = {_lexical_absolute(path) for path in external_paths}
    try:
        for index, item in enumerate(inventory):
            if item.path.is_relative_to(pilot_dir):
                root = snapshot_root / "pilot"
                relative = item.path.relative_to(pilot_dir)
            elif item.path.is_relative_to(repository_root):
                root = snapshot_root / "repository"
                relative = item.path.relative_to(repository_root)
            else:
                if item.path not in allowed_external:
                    raise ReleasePackagingError(
                        f"Consumed file escapes snapshot roots: {item.path}"
                    )
                # Policy, attestation, licence, and card are explicit byte inputs
                # and do not participate in manifest-relative validation paths.
                continue
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                output.write(item.content)
            inventory[index] = replace(item, snapshot_path=target)
    except BaseException:
        shutil.rmtree(snapshot_root, ignore_errors=True)
        raise
    return snapshot_root


def _recheck_inventory(items: list[_InventoryItem]) -> None:
    for item in items:
        try:
            current = _capture_file(item.path)
        except ReleasePackagingError as error:
            raise ReleasePackagingError(
                f"Consumed file changed: {item.path}"
            ) from error
        if (
            current.size,
            current.sha256,
            current.device,
            current.inode,
            current.mode,
            current.nlink,
            current.mtime_ns,
            current.ctime_ns,
        ) != (
            item.size,
            item.sha256,
            item.device,
            item.inode,
            item.mode,
            item.nlink,
            item.mtime_ns,
            item.ctime_ns,
        ):
            raise ReleasePackagingError(f"Consumed file changed: {item.path}")


def _snapshot_path(inventory: list[_InventoryItem], path: Path) -> Path:
    for item in inventory:
        if item.path.absolute() == _lexical_absolute(path):
            if item.snapshot_path is None:
                raise ReleasePackagingError(f"Source has no private snapshot: {path}")
            return item.snapshot_path
    raise ReleasePackagingError(f"Source was not captured: {path}")


def _source_sha256(path: Path, inventory: list[_InventoryItem] | None) -> str:
    if inventory is not None:
        try:
            return sha256_bytes(_captured_bytes(inventory, path))
        except ReleasePackagingError:
            pass
    return sha256_file(path)


def _validate_pilot_for_release(
    *,
    pilot_dir: Path,
    pilot_manifest: PilotManifest,
    policy: ReleasePolicy,
    attestation: ReviewAttestation,
    config: GenerationConfig,
    config_path: Path,
    attributes_path: Path,
    personas_path: Path,
    policy_path: Path,
    attestation_path: Path,
    licence_path: Path,
    repository_root: Path,
    inventory: list[_InventoryItem],
    card_bytes: bytes,
    licence_bytes: bytes,
) -> tuple[ValidationReport, ReleaseEvidence, pl.DataFrame]:
    """Run fresh validation and derive the release evidence boundary.

    Returns:
        The deterministic report, portable evidence, and public output frame.

    Raises:
        ReleasePackagingError:
            If pilot validation or release eligibility fails.
    """
    report = validate_persona_pilot(
        pilot_dir=pilot_dir, repository_root=repository_root
    )
    if not report.passed or report.kind != "persona_pilot":
        raise ReleasePackagingError("Fresh pilot validation did not pass")
    if report.subject_id != pilot_manifest.pilot_id:
        raise ReleasePackagingError("Pilot validation subject does not match")
    output_path = _output_path(pilot_dir, pilot_manifest.output_file)
    output = _read_output(output_path)
    if (
        output.height != pilot_manifest.rows
        or sha256_file(output_path) != pilot_manifest.output_sha256
    ):
        raise ReleasePackagingError("Pilot output does not match its manifest")
    expected_columns = (
        *DemographicRecord.model_fields,
        *GeneratedAttributes.model_fields,
        *PersonaDescriptions.model_fields,
    )
    if set(output.columns) != set(expected_columns) or len(output.columns) != len(
        expected_columns
    ):
        raise ReleasePackagingError("Persona output schema must match exactly")
    if not persona_output_dtypes_are_valid(output):
        raise ReleasePackagingError("Persona output contains an invalid logical dtype")
    validate_release_approval(
        policy=policy,
        attestation=attestation,
        model=pilot_manifest.model,
        pilot_id=pilot_manifest.pilot_id,
        output_sha256=pilot_manifest.output_sha256,
        population_rows=pilot_manifest.rows,
        output_persona_ids=_output_ids(output),
    )
    _assert_manifest_bindings(
        manifest=pilot_manifest,
        config=config,
        config_path=config_path,
        attributes_path=attributes_path,
        personas_path=personas_path,
    )
    report = report.model_copy(
        update={"created_at": attestation.reviewed_at.isoformat()}
    )
    write_json(path=pilot_dir / "pilot-validation-report.json", payload=report)
    evidence = _derive_evidence(
        pilot_dir=pilot_dir,
        manifest=pilot_manifest,
        policy_path=policy_path,
        attestation_path=attestation_path,
        licence_path=licence_path,
        repository_root=repository_root,
        config_path=config_path,
        inventory=inventory,
    )
    _scan_public_values(
        values=[
            card_bytes,
            licence_bytes,
            _captured_bytes(inventory, policy_path),
            _captured_bytes(inventory, attestation_path),
            canonical_json(evidence.model_dump(mode="json")).encode(),
            canonical_json(report.model_dump(mode="json")).encode(),
            _captured_bytes(inventory, attributes_path),
            _captured_bytes(inventory, personas_path),
        ]
    )
    _scan_dataframe(output)
    return report, evidence, output


def _assert_manifest_bindings(
    *,
    manifest: PilotManifest,
    config: GenerationConfig,
    config_path: Path,
    attributes_path: Path,
    personas_path: Path,
) -> None:
    if manifest.llm_generation is not True or not config.llm_generation_enabled:
        raise ReleasePackagingError("Release requires enabled LLM generation")
    if attributes_path != config_path.parent / "prompts/attributes-da.md":
        raise ReleasePackagingError("Generation attributes prompt path binding failed")
    if personas_path != config_path.parent / "prompts/personas-da.md":
        raise ReleasePackagingError("Generation personas prompt path binding failed")
    if sha256_file(config_path) != manifest.generation_config_sha256:
        raise ReleasePackagingError("Generation config binding failed")
    if sha256_file(attributes_path) != manifest.attributes_prompt_sha256:
        raise ReleasePackagingError("Attributes prompt binding failed")
    if sha256_file(personas_path) != manifest.personas_prompt_sha256:
        raise ReleasePackagingError("Personas prompt binding failed")
    if config.model != manifest.model:
        raise ReleasePackagingError("Generation model binding failed")


def _derive_evidence(
    *,
    pilot_dir: Path,
    manifest: PilotManifest,
    policy_path: Path,
    attestation_path: Path,
    licence_path: Path,
    repository_root: Path,
    config_path: Path | None = None,
    inventory: list[_InventoryItem] | None = None,
) -> ReleaseEvidence:
    """Derive portable evidence and reconcile every shard with the pilot.

    Returns:
        Strict, portable release evidence.

    Raises:
        ReleasePackagingError:
            If shard accounting is inconsistent.
    """
    shards: list[ShardEvidence] = []
    for reference in manifest.batch_runs:
        manifest_path = _pilot_path(pilot_dir, reference.manifest_file)
        shard = _load_json_model(manifest_path, GenerationManifest)
        output_path = _output_path(manifest_path.parent, shard.output_file)
        rejected = 0
        checkpoint_dir = manifest_path.parent / "checkpoints"
        for checkpoint_path in sorted(checkpoint_dir.glob("*.json")):
            if checkpoint_path.name.endswith(".attributes.json"):
                continue
            checkpoint = _load_json_model(checkpoint_path, PersonaCheckpoint)
            rejected += sum(response.content == "" for response in checkpoint.responses)
        shards.append(
            ShardEvidence(
                shard_id=shard.run_id,
                offset=shard.offset,
                rows=shard.rows,
                manifest_sha256=_source_sha256(manifest_path, inventory),
                report_sha256=_source_sha256(
                    _pilot_path(pilot_dir, reference.validation_report_file), inventory
                ),
                output_sha256=_source_sha256(output_path, inventory),
                requests=shard.requests,
                retries=shard.retries,
                rejected_validation_responses=rejected,
                prompt_tokens=shard.prompt_tokens,
                completion_tokens=shard.completion_tokens,
                total_tokens=shard.total_tokens,
                provider_cost_usd=shard.estimated_cost_usd,
                providers=tuple(sorted(shard.inference_providers)),
            )
        )
    sums = {
        name: sum(getattr(item, name) for item in shards)
        for name in (
            "rows",
            "requests",
            "retries",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        )
    }
    if any(getattr(manifest, name) != value for name, value in sums.items()):
        raise ReleasePackagingError("Shard accounting does not match pilot manifest")
    output = _read_output(_output_path(pilot_dir, manifest.output_file))
    dropped_rows = manifest.rows - output.height
    provider_costs = [item.provider_cost_usd for item in shards]
    provider_cost = (
        sum(item for item in provider_costs if item is not None)
        if all(item is not None for item in provider_costs)
        else None
    )
    accounting = Accounting(
        requests=manifest.requests,
        retries=manifest.retries,
        rejected_validation_responses=sum(
            item.rejected_validation_responses for item in shards
        ),
        dropped_rows=dropped_rows,
        prompt_tokens=manifest.prompt_tokens,
        completion_tokens=manifest.completion_tokens,
        total_tokens=manifest.total_tokens,
        input_price_per_million_usd=manifest.input_price_per_million_usd,
        output_price_per_million_usd=manifest.output_price_per_million_usd,
        list_price_estimated_cost_usd=manifest.list_price_estimated_cost_usd,
        provider_estimated_cost_usd=provider_cost,
        providers=tuple(sorted(manifest.inference_providers)),
    )
    sample_source_run_id = None
    source_bundle_id = None
    sample_path = _repository_path(repository_root, manifest.sample_manifest_file)
    try:
        sample_source_run_id = json.loads(sample_path.read_text(encoding="utf-8"))[
            "source_run_id"
        ]
    except KeyError, OSError, TypeError, UnicodeError, ValueError:
        pass
    try:
        upstream = _load_json_model(
            sample_path.parent / "run-manifest.json", RunManifest
        )
        source_bundle_id = upstream.bundle_id
    except ReleasePackagingError:
        pass
    effective_config_path = config_path or _repository_path(
        repository_root, manifest.generation_config_file
    )
    config_hashes = {
        "generation.yaml": _source_sha256(effective_config_path, inventory),
        **{
            name: _source_sha256(repository_root / "config" / name, inventory)
            for name in (
                "sources.lock.yaml",
                "categories.yaml",
                "sampling.yaml",
                "validation.yaml",
            )
        },
    }
    return ReleaseEvidence(
        version=1,
        pilot_id=manifest.pilot_id,
        model=manifest.model,
        rows=manifest.rows,
        output_sha256=manifest.output_sha256,
        input_sha256=manifest.input_sha256,
        sample_manifest_sha256=manifest.sample_manifest_sha256,
        generation_config_sha256=manifest.generation_config_sha256,
        generation_context_sha256=manifest.generation_context_sha256,
        validator_version=manifest.validator_version,
        attributes_prompt_sha256=manifest.attributes_prompt_sha256,
        personas_prompt_sha256=manifest.personas_prompt_sha256,
        upstream_run_id=manifest.upstream_run_id,
        sample_source_run_id=sample_source_run_id,
        source_bundle_id=source_bundle_id,
        policy_sha256=_source_sha256(policy_path, inventory),
        attestation_sha256=_source_sha256(attestation_path, inventory),
        licence_sha256=_source_sha256(licence_path, inventory),
        code_license_sha256=_source_sha256(repository_root / "LICENSE", inventory),
        uv_lock_sha256=_source_sha256(repository_root / "uv.lock", inventory),
        pilot_validation_report_sha256=_source_sha256(
            pilot_dir / "pilot-validation-report.json", inventory
        ),
        config_hashes=config_hashes,
        shards=tuple(shards),
        accounting=accounting,
    )


def _read_output(path: Path) -> pl.DataFrame:
    try:
        return pl.read_parquet(path)
    except Exception as error:
        raise ReleasePackagingError("Persona output is not readable Parquet") from error


def _output_ids(output: pl.DataFrame) -> list[str]:
    if "persona_id" not in output.columns:
        raise ReleasePackagingError("Persona output lacks persona_id")
    values = output.get_column("persona_id").to_list()
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ReleasePackagingError("Persona IDs must be non-empty strings")
    ids = t.cast(list[str], values)
    if len(ids) != len(set(ids)) or not ids:
        raise ReleasePackagingError("Persona IDs must be unique and non-empty")
    return ids


ModelType = t.TypeVar("ModelType", bound=StrictModel)


def _acquire_lock(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
    except FileExistsError as error:
        raise ReleasePackagingError(
            "Another release package operation is active"
        ) from error


def _git_provenance(root: Path) -> tuple[str, str]:
    head = _git(root, "rev-parse", "HEAD")
    origin = _git(root, "remote", "get-url", "origin")
    if len(head) != 40 or not origin:
        raise ReleasePackagingError("Git HEAD and origin URL are required")
    return head, origin


def _git(repository_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository_root), *args],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReleasePackagingError("Git provenance is unavailable") from error


def _load_json_model(path: Path, model: type[ModelType]) -> ModelType:
    try:
        _require_regular_file(path)
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ReleasePackagingError(f"Invalid contract: {path}") from error


def _require_regular_file(path: Path) -> None:
    candidate = _lexical_absolute(path)
    _require_no_symlink_components(candidate)
    try:
        stat = candidate.lstat()
    except OSError as error:
        raise ReleasePackagingError(f"Missing input file: {path}") from error
    if candidate.is_symlink() or not os.path.isfile(candidate) or stat.st_nlink != 1:
        raise ReleasePackagingError(f"Input must be a regular non-linked file: {path}")


def _lexical_absolute(path: Path) -> Path:
    """Make an absolute normalised path without following symlinks.

    Returns:
        An absolute, lexically normalised path.
    """
    return Path(os.path.abspath(os.path.normpath(path)))


def _require_no_symlink_components(path: Path) -> None:
    candidate = _lexical_absolute(path)
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current /= component
        try:
            if current.is_symlink():
                raise ReleasePackagingError(
                    f"Input must be a regular non-linked file; path contains a "
                    f"symlink component: {path}"
                )
        except OSError as error:
            raise ReleasePackagingError(f"Cannot inspect input path: {path}") from error


def _output_path(owner_dir: Path, value: Path) -> Path:
    """Resolve an output path relative to its owning run or pilot directory.

    Returns:
        The resolved output path.

    Raises:
        ReleasePackagingError:
            If the source is not a regular file or escapes its owner.
    """
    candidate = _lexical_absolute(value if value.is_absolute() else owner_dir / value)
    _require_regular_file(candidate)
    try:
        candidate.relative_to(_lexical_absolute(owner_dir))
    except ValueError as error:
        raise ReleasePackagingError("Output path escapes its owning run") from error
    return candidate


def _pilot_path(pilot_dir: Path, value: Path) -> Path:
    """Resolve a batch reference relative to the pilot directory.

    Returns:
        The resolved pilot path.

    Raises:
        ReleasePackagingError:
            If the source is not a regular file or escapes the pilot.
    """
    candidate = _lexical_absolute(value if value.is_absolute() else pilot_dir / value)
    _require_regular_file(candidate)
    try:
        candidate.relative_to(_lexical_absolute(pilot_dir))
    except ValueError as error:
        raise ReleasePackagingError(
            "Pilot batch path escapes pilot directory"
        ) from error
    return candidate


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Install without replacing a destination, failing closed if unsupported.

    Raises:
        ReleasePackagingError:
            If no atomic no-replace primitive is available or installation fails.
    """
    if destination.exists() or destination.is_symlink():
        raise ReleasePackagingError("Release destination already exists")
    if os.name != "posix":
        _rename_windows(source, destination)
        return
    _rename_posix(source, destination)


def _rename_posix(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        operation = getattr(libc, "renameat2", None)
        flags = 1
        directory_fd = -100
    elif sys.platform == "darwin":
        operation = getattr(libc, "renameatx_np", None)
        flags = 4
        directory_fd = -2
    else:
        raise ReleasePackagingError("Atomic no-replace install is unsupported")
    if operation is None:
        raise ReleasePackagingError("Atomic no-replace install is unavailable")
    result = operation(
        directory_fd,
        str(source).encode(),
        directory_fd,
        str(destination).encode(),
        flags,
    )
    if result == 0:
        return
    if ctypes.get_errno() == errno.EEXIST:
        raise ReleasePackagingError("Release destination already exists")
    raise ReleasePackagingError("Atomic no-replace install failed")


def _rename_windows(source: Path, destination: Path) -> None:
    try:
        os.rename(source, destination)
    except FileExistsError as error:
        raise ReleasePackagingError("Release destination already exists") from error
    except OSError as error:
        raise ReleasePackagingError("Atomic no-replace install failed") from error


def _repository_path(repository_root: Path, value: Path) -> Path:
    """Resolve a manifest path inside the repository.

    Returns:
        The resolved repository path.

    Raises:
        ReleasePackagingError:
            If the source is not a regular file or escapes the repository.
    """
    candidate = _lexical_absolute(
        value if value.is_absolute() else repository_root / value
    )
    _require_regular_file(candidate)
    try:
        candidate.relative_to(_lexical_absolute(repository_root))
    except ValueError as error:
        raise ReleasePackagingError(
            "Repository manifest path escapes repository root"
        ) from error
    return candidate


def _require_clean_git(root: Path) -> None:
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ReleasePackagingError("Git checkout must be clean")


def _require_directory(path: Path) -> None:
    candidate = _lexical_absolute(path)
    _require_no_symlink_components(candidate)
    try:
        candidate.lstat()
    except OSError as error:
        raise ReleasePackagingError(f"Missing directory: {path}") from error
    if candidate.is_symlink() or not os.path.isdir(candidate):
        raise ReleasePackagingError(f"Input must be a regular directory: {path}")


def _require_public_prompt(
    path: Path, expected: str, content: bytes | None = None
) -> None:
    if content is None:
        content = _regular_bytes(path)
    if sha256_bytes(content) != expected:
        raise ReleasePackagingError(f"Prompt checksum mismatch: {path}")


def _regular_bytes(path: Path) -> bytes:
    _require_regular_file(path)
    return path.read_bytes()


def sha256_bytes(content: bytes) -> str:
    """Return the SHA-256 digest of bytes."""
    return hashlib.sha256(content).hexdigest()


def _scan_dataframe(output: pl.DataFrame) -> None:
    """Recursively scan every scalar in every public Parquet column.

    Raises:
        ReleasePackagingError:
            If the output schema or a public value is unsafe.
    """
    if not persona_output_dtypes_are_valid(output):
        raise ReleasePackagingError("Persona output contains an invalid logical dtype")
    for column in output.columns:
        for value in output[column].to_list():
            _scan_public_value(value)


def _scan_public_value(value: object) -> None:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ReleasePackagingError("Public binary value is not UTF-8") from error
    if isinstance(value, str):
        _scan_public_values([value.encode("utf-8")])
    elif isinstance(value, dict):
        for key, nested in value.items():
            _scan_public_value(key)
            _scan_public_value(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _scan_public_value(nested)
    elif value is not None:
        _scan_public_values([str(value).encode("utf-8")])


def _scan_public_values(values: list[bytes]) -> None:
    """Reject obvious local paths and credentials in public text.

    Raises:
        ReleasePackagingError:
            If a forbidden path or credential pattern is found.
    """
    patterns = (
        rb"(?:^|[^A-Za-z0-9_])/(?:Users|private|tmp|home|var|etc|opt|Volumes|mnt|root)/",
        rb"\b[A-Za-z]:[\\/]",
        rb"\\\\[^\\/\s]+[\\/]",
        rb"(?i:file://)",
        rb"https?://[^\s/@]+:[^\s/@]+@",
        rb"(?i)(?:https?://[^\s]+[?&](?:token|api[_-]?key|secret|password|access[_-]?token)=)",
        rb"(?i)(?:https?://[^\s]+#[^\s]*(?:token|secret|key))",
        rb"(?i)authorization\s+(?:basic|bearer)\s+",
        rb"(?i)\b(?:api[_-]?key|token|secret|password)(?![_-]?env)\s*[:=]\s*\S+",
        rb"(?:^|[^A-Za-z0-9_])(?:hf_|ghp_|github_pat_|sk-)",
    )
    for content in values:
        text = content.decode("utf-8", errors="strict")
        # ``hf_xet`` is a normal package name in uv.lock, not a token.
        scan_bytes = text.replace("hf_xet", "hf-xet").encode()
        if any(re.search(pattern, scan_bytes) for pattern in patterns):
            raise ReleasePackagingError("Public evidence contains a path or secret")
