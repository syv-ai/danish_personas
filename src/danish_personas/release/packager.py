"""Offline release packaging with fail-closed provenance checks."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import typing as t
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ..generation.models import (
    GeneratedAttributes,
    GenerationConfig,
    GenerationManifest,
    PersonaCheckpoint,
    PersonaDescriptions,
    PilotManifest,
)
from ..generation.report import validate_persona_pilot
from ..io import canonical_json, load_yaml_model, sha256_file, write_json
from ..models import RunManifest, StrictModel, ValidationReport
from .common import release_id, role
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
    pilot_dir = pilot_dir.resolve()
    repository_root = repository_root.resolve()
    output_parent = output_parent.resolve()
    _require_clean_git(repository_root)
    git_head, origin_url = _git_provenance(repository_root)
    pilot_manifest = _load_json_model(pilot_dir / "pilot-manifest.json", PilotManifest)
    policy = load_yaml_model(path=policy_path, model=ReleasePolicy)
    attestation = _load_json_model(attestation_path, ReviewAttestation)
    config_path = _manifest_path(pilot_dir, pilot_manifest.generation_config_file)
    config = load_yaml_model(path=config_path, model=GenerationConfig)
    attributes_path = _repository_path(repository_root, config.attributes_prompt)
    personas_path = _repository_path(repository_root, config.personas_prompt)
    _require_public_prompt(attributes_path, pilot_manifest.attributes_prompt_sha256)
    _require_public_prompt(personas_path, pilot_manifest.personas_prompt_sha256)
    _require_regular_file(licence_path)
    licence_bytes = licence_path.read_bytes()
    if not licence_bytes:
        raise ReleasePackagingError("The supplied licence is empty")
    if sha256_bytes(licence_bytes) != policy.licence_file_sha256:
        raise ReleasePackagingError("Supplied licence does not match policy")
    card_bytes = _regular_bytes(dataset_card_path)
    if not card_bytes.strip():
        raise ReleasePackagingError("Dataset card must be non-empty")

    consumed = _derive_consumed_files(pilot_dir=pilot_dir, manifest=pilot_manifest)
    consumed.extend(
        _inventory_inputs(
            paths=[
                attestation_path,
                policy_path,
                dataset_card_path,
                licence_path,
                attributes_path,
                personas_path,
                repository_root / "uv.lock",
                repository_root / "LICENSE",
                *[
                    repository_root / name
                    for name in (
                        "config/sources.lock.yaml",
                        "config/categories.yaml",
                        "config/sampling.yaml",
                        "config/validation.yaml",
                        "docs/source-register.md",
                        "docs/privacy-risk-register.md",
                        "docs/acceptance-criteria.md",
                    )
                ],
            ]
        )
    )
    inventory = _snapshot_inventory(consumed)

    output_path = _manifest_path(pilot_dir, pilot_manifest.output_file)
    report, evidence, output = _validate_pilot_for_release(
        pilot_dir=pilot_dir,
        pilot_manifest=pilot_manifest,
        policy=policy,
        attestation=attestation,
        config=config,
        config_path=config_path,
        attributes_path=attributes_path,
        personas_path=personas_path,
        policy_path=policy_path,
        attestation_path=attestation_path,
        licence_path=licence_path,
        repository_root=repository_root,
        inventory=inventory,
        card_bytes=card_bytes,
        licence_bytes=licence_bytes,
    )

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
            output_path=output_path,
            attestation_path=attestation_path,
            policy_path=policy_path,
            evidence=evidence,
            report=report,
            attributes_path=attributes_path,
            personas_path=personas_path,
            repository_root=repository_root,
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
            uv_lock_sha256=sha256_file(repository_root / "uv.lock"),
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


def _derive_consumed_files(*, pilot_dir: Path, manifest: PilotManifest) -> list[Path]:
    """Derive the strict input inventory from pilot and shard manifests.

    Returns:
        Paths consumed by validation and evidence derivation.
    """
    paths = [pilot_dir / "pilot-manifest.json"]
    paths.append(_manifest_path(pilot_dir, manifest.output_file))
    paths.append(_manifest_path(pilot_dir, manifest.input_file))
    paths.append(_manifest_path(pilot_dir, manifest.sample_manifest_file))
    paths.append(_manifest_path(pilot_dir, manifest.generation_config_file))
    sample_path = _manifest_path(pilot_dir, manifest.input_file)
    upstream_manifest_path = sample_path.parent / "run-manifest.json"
    upstream_report_path = sample_path.parent / "validation-report.json"
    paths.extend((upstream_manifest_path, upstream_report_path))
    upstream = _load_json_model(upstream_manifest_path, RunManifest)
    paths.append(_manifest_path(sample_path.parent, upstream.data_file))
    for reference in manifest.batch_runs:
        manifest_path = _pilot_path(pilot_dir, reference.manifest_file)
        report_path = _pilot_path(pilot_dir, reference.validation_report_file)
        shard = _load_json_model(manifest_path, GenerationManifest)
        paths.extend((manifest_path, report_path))
        paths.append(_pilot_path(manifest_path.parent, shard.output_file))
        checkpoint_dir = manifest_path.parent / "checkpoints"
        if checkpoint_dir.is_dir():
            paths.extend(sorted(checkpoint_dir.glob("*.json")))
        ledger = manifest_path.parent / "request-ledger.json"
        if ledger.exists():
            paths.append(ledger)
    return paths


def _install_files(**kwargs: object) -> None:
    stage = t.cast(Path, kwargs["stage"])
    card = t.cast(bytes, kwargs["card"])
    licence = t.cast(bytes, kwargs["licence"])
    output_path = t.cast(Path, kwargs["output_path"])
    attestation_path = t.cast(Path, kwargs["attestation_path"])
    policy_path = t.cast(Path, kwargs["policy_path"])
    evidence = t.cast(ReleaseEvidence, kwargs["evidence"])
    report = t.cast(ValidationReport, kwargs["report"])
    attributes_path = t.cast(Path, kwargs["attributes_path"])
    personas_path = t.cast(Path, kwargs["personas_path"])
    repository_root = t.cast(Path, kwargs["repository_root"])
    payloads: dict[str, bytes] = {
        "README.md": card,
        "LICENSE.txt": licence,
        "data/personas.parquet": output_path.read_bytes(),
        "attestations/human-review.json": attestation_path.read_bytes(),
        "provenance/release-policy.yaml": policy_path.read_bytes(),
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
        "provenance/prompts/attributes-da.md": attributes_path.read_bytes(),
        "provenance/prompts/personas-da.md": personas_path.read_bytes(),
    }
    for name in (
        "sources.lock.yaml",
        "categories.yaml",
        "sampling.yaml",
        "validation.yaml",
    ):
        payloads[f"provenance/config/{name}"] = (
            repository_root / "config" / name
        ).read_bytes()
    for name in (
        "source-register.md",
        "privacy-risk-register.md",
        "acceptance-criteria.md",
    ):
        payloads[f"provenance/docs/{name}"] = (
            repository_root / "docs" / name
        ).read_bytes()
    payloads["provenance/code/uv.lock"] = (repository_root / "uv.lock").read_bytes()
    payloads["provenance/code/LICENSE"] = (repository_root / "LICENSE").read_bytes()
    for relative, content in payloads.items():
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def _inventory_inputs(*, paths: list[Path]) -> list[Path]:
    return paths


def _recheck_inventory(items: list[_InventoryItem]) -> None:
    for item in items:
        _require_regular_file(item.path)
        stat = item.path.stat()
        if (stat.st_size, stat.st_dev, stat.st_ino, sha256_file(item.path)) != (
            item.size,
            item.device,
            item.inode,
            item.sha256,
        ):
            raise ReleasePackagingError(f"Consumed file changed: {item.path}")


def _snapshot_inventory(paths: list[Path]) -> list[_InventoryItem]:
    result: list[_InventoryItem] = []
    seen: set[Path] = set()
    for path in paths:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        _require_regular_file(path)
        stat = path.stat()
        result.append(
            _InventoryItem(
                path, stat.st_size, sha256_file(path), stat.st_dev, stat.st_ino
            )
        )
    return result


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
    report = validate_persona_pilot(pilot_dir=pilot_dir)
    if not report.passed or report.kind != "persona_pilot":
        raise ReleasePackagingError("Fresh pilot validation did not pass")
    if report.subject_id != pilot_manifest.pilot_id:
        raise ReleasePackagingError("Pilot validation subject does not match")
    output_path = _manifest_path(pilot_dir, pilot_manifest.output_file)
    output = _read_output(output_path)
    if (
        output.height != pilot_manifest.rows
        or sha256_file(output_path) != pilot_manifest.output_sha256
    ):
        raise ReleasePackagingError("Pilot output does not match its manifest")
    required_columns = {
        "persona_id",
        *GeneratedAttributes.model_fields,
        *PersonaDescriptions.model_fields,
    }
    if not required_columns.issubset(output.columns):
        raise ReleasePackagingError("Persona output schema is incomplete")
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
    inventory.extend(
        _snapshot_inventory(paths=[pilot_dir / "pilot-validation-report.json"])
    )
    evidence = _derive_evidence(
        pilot_dir=pilot_dir,
        manifest=pilot_manifest,
        policy_path=policy_path,
        attestation_path=attestation_path,
        licence_path=licence_path,
        repository_root=repository_root,
    )
    _scan_public_values(
        values=[
            card_bytes,
            licence_bytes,
            policy_path.read_bytes(),
            attestation_path.read_bytes(),
            canonical_json(evidence.model_dump(mode="json")).encode(),
            canonical_json(report.model_dump(mode="json")).encode(),
            attributes_path.read_bytes(),
            personas_path.read_bytes(),
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
        output_path = _pilot_path(manifest_path.parent, shard.output_file)
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
                manifest_sha256=sha256_file(manifest_path),
                report_sha256=sha256_file(
                    _pilot_path(pilot_dir, reference.validation_report_file)
                ),
                output_sha256=sha256_file(output_path),
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
    output = _read_output(_manifest_path(pilot_dir, manifest.output_file))
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
    sample_path = _manifest_path(pilot_dir, manifest.sample_manifest_file)
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
    config_hashes = {
        name: sha256_file(repository_root / "config" / name)
        for name in (
            "sources.lock.yaml",
            "categories.yaml",
            "sampling.yaml",
            "validation.yaml",
        )
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
        policy_sha256=sha256_file(policy_path),
        attestation_sha256=sha256_file(attestation_path),
        licence_sha256=sha256_file(licence_path),
        code_license_sha256=sha256_file(repository_root / "LICENSE"),
        uv_lock_sha256=sha256_file(repository_root / "uv.lock"),
        pilot_validation_report_sha256=sha256_file(
            pilot_dir / "pilot-validation-report.json"
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
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ReleasePackagingError(f"Invalid contract: {path}") from error


def _pilot_path(base: Path, value: Path) -> Path:
    """Resolve a shard path while refusing traversal outside its pilot root.

    Returns:
        The validated path.

    Raises:
        ReleasePackagingError:
            If the path escapes its containing directory.
    """
    path = _manifest_path(base, value)
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError as error:
        raise ReleasePackagingError(
            "Pilot manifest path escapes its directory"
        ) from error
    return path


def _manifest_path(base: Path, value: Path) -> Path:
    if value.is_absolute():
        path = value
    else:
        path = base / value
    return path


def _regular_bytes(path: Path) -> bytes:
    _require_regular_file(path)
    return path.read_bytes()


def _require_regular_file(path: Path) -> None:
    try:
        stat = path.lstat()
    except OSError as error:
        raise ReleasePackagingError(f"Missing input file: {path}") from error
    if not os.path.isfile(path) or stat.st_nlink != 1 or path.is_symlink():
        raise ReleasePackagingError(f"Input must be a regular non-linked file: {path}")


def _rename_noreplace(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise ReleasePackagingError("Release destination already exists")
    if os.name == "posix":
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is not None:
            result = renameat2(
                -100, str(source).encode(), -100, str(destination).encode(), 1
            )
            if result == 0:
                return
            if ctypes.get_errno() == 17:
                raise ReleasePackagingError("Release destination already exists")
        renameatx_np = getattr(libc, "renameatx_np", None)
        if renameatx_np is not None:
            result = renameatx_np(
                -2, str(source).encode(), -2, str(destination).encode(), 4
            )
            if result == 0:
                return
            if ctypes.get_errno() == 17:
                raise ReleasePackagingError("Release destination already exists")
    try:
        os.rename(source, destination)
    except FileExistsError as error:
        raise ReleasePackagingError("Release destination already exists") from error


def _repository_path(root: Path, value: Path) -> Path:
    path = value if value.is_absolute() else root / value
    try:
        path.resolve().relative_to(root)
    except ValueError as error:
        raise ReleasePackagingError("Prompt is outside repository root") from error
    return path


def _require_clean_git(root: Path) -> None:
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ReleasePackagingError("Git checkout must be clean")


def _require_public_prompt(path: Path, expected: str) -> None:
    _require_regular_file(path)
    if sha256_file(path) != expected:
        raise ReleasePackagingError(f"Prompt checksum mismatch: {path}")


def _scan_dataframe(output: pl.DataFrame) -> None:
    """Scan portable string values before they enter the public package."""
    for column in output.columns:
        if output[column].dtype == pl.String or output[column].dtype == pl.List(
            pl.String
        ):
            for value in output[column].to_list():
                _scan_public_values(
                    [json.dumps(value, ensure_ascii=False).encode("utf-8")]
                )


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
        rb"(?:hf_|ghp_|github_pat_|sk-)",
    )
    for content in values:
        text = content.decode("utf-8", errors="strict")
        # ``hf_xet`` is a normal package name in uv.lock, not a token.
        scan_bytes = text.replace("hf_xet", "hf-xet").encode()
        if any(re.search(pattern, scan_bytes) for pattern in patterns):
            raise ReleasePackagingError("Public evidence contains a path or secret")


def sha256_bytes(content: bytes) -> str:
    """Return the SHA-256 digest of bytes."""
    return hashlib.sha256(content).hexdigest()
