"""Relocation-safe, offline verification for release packages."""

from __future__ import annotations

import hmac
import json
import os
import re
import typing as t
from pathlib import Path

import polars as pl
from pydantic import ValidationError

from ..generation.job_titles import load_job_title_mapping
from ..generation.models import GenerationConfig
from ..generation.pipeline import generation_context_sha256
from ..io import load_yaml_model, sha256_file
from ..models import StrictModel, ValidationReport
from ..origin_labels import (
    DEFAULT_ORIGIN_LABEL_CONTRACT_PATH,
    ORIGIN_LABEL_CONTRACT_SHA256,
    OriginLabelContract,
    load_origin_label_contract,
)
from .common import (
    PERSONA_OUTPUT_COLUMNS,
    persona_output_dtypes_are_valid,
    release_id,
    role,
    validate_persona_output_rows,
)
from .models import ReleaseEvidence, ReleaseManifest, ReleasePolicy, ReviewAttestation
from .policy import validate_release_approval

ModelType = t.TypeVar("ModelType", bound=StrictModel)


class ReleaseVerificationError(ValueError):
    """Raised when a release package is not independently verifiable."""


_PUBLIC_FILES = {
    "README.md",
    "LICENSE.txt",
    "data/personas.parquet",
    "attestations/human-review.json",
    "provenance/release-policy.yaml",
    "provenance/evidence.json",
    "provenance/pilot-validation-report.json",
    "provenance/persona-da.md",
    "provenance/config/config.yaml",
    "provenance/config/job-function-titles.yaml",
    "provenance/config/folk2-ieland-labels-da.yaml",
    "provenance/config/sources.lock.yaml",
    "provenance/config/categories.yaml",
    "provenance/config/sampling.yaml",
    "provenance/config/validation.yaml",
    "provenance/docs/source-register.md",
    "provenance/docs/privacy-risk-register.md",
    "provenance/docs/acceptance-criteria.md",
    "provenance/code/uv.lock",
    "provenance/code/LICENSE",
}
_PUBLIC_DIRS = {
    "attestations",
    "provenance",
    "provenance/config",
    "provenance/docs",
    "provenance/code",
    "data",
}


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting ambiguous duplicate keys.

    Returns:
        The object with every key present exactly once.

    Raises:
        ValueError: If a key occurs more than once.
    """
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("Public JSON contract contains a duplicate key")
        payload[key] = value
    return payload


def verify_release(
    *, release_dir: Path, expected_manifest_sha256: str
) -> ReleaseManifest:
    """Verify a release using a mandatory externally retained manifest digest.

    Returns:
        The strictly parsed release manifest.
    """
    return _verify_release(
        release_dir=release_dir,
        expected_manifest_sha256=expected_manifest_sha256,
        allow_staging=False,
    )


def _verify_release(
    *, release_dir: Path, expected_manifest_sha256: str, allow_staging: bool
) -> ReleaseManifest:
    """Verify a package, optionally while it is still private staging.

    The external digest comparison is deliberately the first trust decision.  No
    sidecar, evidence, or release content is trusted before it succeeds.

    Returns:
        The strictly parsed release manifest.

    Raises:
        ReleaseVerificationError:
            If the digest, layout, contracts, or bindings fail.
    """
    if not isinstance(expected_manifest_sha256, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_manifest_sha256
    ):
        raise ReleaseVerificationError("Expected manifest SHA-256 is mandatory")
    release_dir = _lexical_absolute(Path(release_dir))
    _require_no_symlink_components(release_dir)
    manifest_path = release_dir / "release-manifest.json"
    try:
        actual_digest = sha256_file(manifest_path)
    except OSError as error:
        raise ReleaseVerificationError("Release manifest is missing") from error
    if not hmac.compare_digest(actual_digest, expected_manifest_sha256.lower()):
        raise ReleaseVerificationError("External release manifest digest mismatch")
    manifest = _load_json(manifest_path, ReleaseManifest)
    _check_layout(release_dir, manifest)
    sidecar = release_dir / "release-manifest.sha256"
    try:
        sidecar_content = sidecar.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise ReleaseVerificationError(
            "Manifest sidecar is missing or invalid"
        ) from error
    if sidecar_content != f"{actual_digest}  release-manifest.json\n":
        raise ReleaseVerificationError("Manifest sidecar mismatch")
    if not allow_staging and manifest.release_id != release_dir.name:
        raise ReleaseVerificationError("Release directory does not match release ID")
    return _verify_contents(release_dir=release_dir, manifest=manifest)


def _check_layout(release_dir: Path, manifest: ReleaseManifest) -> None:
    if not release_dir.is_dir() or release_dir.is_symlink():
        raise ReleaseVerificationError("Release directory is not a directory")
    expected = (
        _PUBLIC_FILES
        | _PUBLIC_DIRS
        | {"release-manifest.json", "release-manifest.sha256"}
    )
    names: set[str] = set()
    for root, dirs, files in os.walk(release_dir, followlinks=False):
        relative_root = Path(root).relative_to(release_dir)
        for name in [*dirs, *files]:
            relative = (
                (relative_root / name).as_posix()
                if relative_root != Path(".")
                else name
            )
            if relative.lower() in names:
                raise ReleaseVerificationError("Case-colliding release paths")
            names.add(relative.lower())
            path = release_dir / relative
            stat = path.lstat()
            if path.is_symlink() or not path.is_file() and not path.is_dir():
                raise ReleaseVerificationError("Release contains a non-regular path")
            if path.is_file() and stat.st_nlink != 1:
                raise ReleaseVerificationError("Release contains a hard link")
    actual = {item for item in names}
    if actual != {item.lower() for item in expected}:
        raise ReleaseVerificationError("Release has missing, extra, or empty paths")
    listed = {item.path for item in manifest.artifacts}
    if listed != _PUBLIC_FILES or len(manifest.artifacts) != len(_PUBLIC_FILES):
        raise ReleaseVerificationError(
            "Manifest does not cover the fixed public layout"
        )


def _lexical_absolute(path: Path) -> Path:
    """Make an absolute normalised path without following symlinks.

    Returns:
        An absolute, lexically normalised path.
    """
    return Path(os.path.abspath(os.path.normpath(path)))


def _load_json(path: Path, model: type[ModelType]) -> ModelType:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        return model.model_validate(payload)
    except ValidationError as error:
        diagnostics = "; ".join(
            f"{_safe_validation_location(item=item, model=model)}: "
            f"{_safe_validation_error_type(item)}"
            for item in error.errors(
                include_url=False, include_context=False, include_input=False
            )
        )
        raise ReleaseVerificationError(
            f"Invalid public contract: {path} ({diagnostics})"
        ) from error
    except Exception as error:
        raise ReleaseVerificationError(f"Invalid public contract: {path}") from error


def _safe_validation_error_type(item: t.Mapping[str, object]) -> str:
    """Return a fixed diagnostic code without echoing validation input."""
    location = item.get("loc")
    message = str(item.get("msg", ""))
    if location == ("origin_label_contract_content",):
        origin_failures = (
            ("Unsupported origin-label contract version", "origin_version"),
            ("contract table_id must be FOLK2", "origin_table"),
            ("contract dimension must be IELAND", "origin_dimension"),
            ("contract language must be da", "origin_language"),
            ("English metadata checksum does not match", "origin_metadata_en"),
            ("Danish metadata checksum does not match", "origin_metadata_da"),
            ("must contain 241 labels", "origin_label_count"),
            ("Malformed FOLK2 IELAND code", "origin_code"),
            ("label is blank or padded", "origin_label_padding"),
            ("label is not NFC-normalised", "origin_label_nfc"),
            ("labels must be unique", "origin_label_uniqueness"),
            ("English keys differ from the reviewed contract", "origin_en_keys"),
            ("English whitespace differs", "origin_en_whitespace"),
            ("English values differ from the reviewed contract", "origin_en_values"),
            ("Danish keys differ from the reviewed contract", "origin_da_keys"),
            ("Danish whitespace differs", "origin_da_whitespace"),
            ("Danish values differ from the reviewed contract", "origin_da_values"),
            ("English and Danish code order differs", "origin_code_order"),
            ("English labels must be unique", "origin_english_uniqueness"),
        )
        for fragment, diagnostic in origin_failures:
            if fragment in message:
                return diagnostic
    return str(item.get("type", "validation_error"))


def _safe_validation_location(
    *, item: t.Mapping[str, object], model: type[ModelType]
) -> str:
    """Return only an allowlisted top-level model field."""
    location = item.get("loc")
    if not isinstance(location, tuple) or not location:
        return "<model>"
    field = location[0]
    if isinstance(field, str) and field in model.model_fields:
        return field
    return "<model>"


def _require_no_symlink_components(path: Path) -> None:
    """Reject symlink components in a supplied verifier path.

    Raises:
        ReleaseVerificationError:
            If a path component is a symlink or cannot be inspected.
    """
    candidate = _lexical_absolute(path)
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current /= component
        try:
            if current.is_symlink():
                raise ReleaseVerificationError(
                    "Release directory path contains a symlink component"
                )
        except OSError as error:
            raise ReleaseVerificationError(
                "Release path cannot be inspected"
            ) from error


def _verify_contents(
    *, release_dir: Path, manifest: ReleaseManifest
) -> ReleaseManifest:
    """Verify contracts and content after the manifest trust boundary.

    Returns:
        The strictly parsed release manifest.

    Raises:
        ReleaseVerificationError:
            If a contract or binding fails.
    """
    if (
        release_id(
            pilot_id=manifest.pilot_id,
            output_sha256=_artifact_hash(manifest, "data/personas.parquet"),
            reviewed_at=manifest.created_at,
            git_head=manifest.git_head,
            origin_url=manifest.origin_url,
        )
        != manifest.release_id
    ):
        raise ReleaseVerificationError("Release ID binding failed")
    _check_artifacts(release_dir, manifest)
    policy = _load_yaml(release_dir / "provenance/release-policy.yaml", ReleasePolicy)
    attestation = _load_json(
        release_dir / "attestations/human-review.json", ReviewAttestation
    )
    evidence = _load_json(release_dir / "provenance/evidence.json", ReleaseEvidence)
    if attestation.reviewed_at != manifest.created_at:
        raise ReleaseVerificationError("Manifest timestamp is not attested")
    report = _load_json(
        release_dir / "provenance/pilot-validation-report.json", ValidationReport
    )
    output = _read_output(release_dir / "data/personas.parquet")
    ids = _output_ids(output)
    _check_output(
        release_dir=release_dir, manifest=manifest, output=output, report=report
    )
    if evidence.pilot_id != manifest.pilot_id or evidence.rows != manifest.rows:
        raise ReleaseVerificationError("Evidence identity binding failed")
    if (
        manifest.origin_label_contract_file != evidence.origin_label_contract_file
        or manifest.origin_label_contract_sha256
        != evidence.origin_label_contract_sha256
        or manifest.origin_label_contract_version
        != evidence.origin_label_contract_version
        or manifest.origin_label_contract_content
        != evidence.origin_label_contract_content
    ):
        raise ReleaseVerificationError("Origin-label contract identity binding failed")
    if evidence.output_sha256 != _artifact_hash(manifest, "data/personas.parquet"):
        raise ReleaseVerificationError("Evidence output binding failed")
    if evidence.pilot_validation_report_sha256 != sha256_file(
        release_dir / "provenance/pilot-validation-report.json"
    ):
        raise ReleaseVerificationError("Pilot report checksum binding failed")
    _check_provenance_bindings(
        release_dir=release_dir, manifest=manifest, evidence=evidence
    )
    _check_approval(
        policy=policy,
        attestation=attestation,
        manifest=manifest,
        evidence=evidence,
        ids=ids,
    )
    _check_evidence(evidence=evidence, rows=manifest.rows)
    _scan_release_text(release_dir=release_dir)
    return manifest


def _artifact_hash(manifest: ReleaseManifest, path: str) -> str:
    for artifact in manifest.artifacts:
        if artifact.path == path:
            return artifact.sha256
    raise ReleaseVerificationError("Manifest artifact is missing")


def _check_approval(
    *,
    policy: ReleasePolicy,
    attestation: ReviewAttestation,
    manifest: ReleaseManifest,
    evidence: ReleaseEvidence,
    ids: list[str],
) -> None:
    """Apply the strict human-review eligibility contract.

    Raises:
        ReleaseVerificationError:
            If approval is not valid.
    """
    try:
        validate_release_approval(
            policy=policy,
            attestation=attestation,
            model=manifest.model,
            pilot_id=manifest.pilot_id,
            output_sha256=evidence.output_sha256,
            population_rows=manifest.rows,
            output_persona_ids=ids,
        )
    except ValueError as error:
        raise ReleaseVerificationError(str(error)) from error


def _check_artifacts(release_dir: Path, manifest: ReleaseManifest) -> None:
    for artifact in manifest.artifacts:
        if artifact.path not in _PUBLIC_FILES or artifact.role != role(artifact.path):
            raise ReleaseVerificationError("Invalid manifest artifact role")
        path = release_dir / artifact.path
        if path.stat().st_size != artifact.size or sha256_file(path) != artifact.sha256:
            raise ReleaseVerificationError(
                f"Artifact checksum mismatch: {artifact.path}"
            )


def _check_evidence(*, evidence: ReleaseEvidence, rows: int) -> None:
    if sum(item.rows for item in evidence.shards) != rows:
        raise ReleaseVerificationError("Shard row sums do not match output")
    for shard in evidence.shards:
        if (
            shard.generation_config_sha256 != evidence.generation_config_sha256
            or shard.generation_context_sha256 != evidence.generation_context_sha256
            or shard.origin_label_contract_file != evidence.origin_label_contract_file
            or shard.origin_label_contract_sha256
            != evidence.origin_label_contract_sha256
            or shard.origin_label_contract_version
            != evidence.origin_label_contract_version
        ):
            raise ReleaseVerificationError(
                "Shard generation contract bindings disagree"
            )
    _check_shard_accounting(evidence=evidence)
    _check_accounting_values(evidence=evidence)


def _check_accounting_values(*, evidence: ReleaseEvidence) -> None:
    """Check token, price, provider, and dropped-row counters.

    Raises:
        ReleaseVerificationError:
            If accounting is inconsistent.
    """
    accounting = evidence.accounting
    if accounting.dropped_rows != 0:
        raise ReleaseVerificationError("Dropped rows are not releasable")
    if (
        accounting.total_tokens
        != accounting.prompt_tokens + accounting.completion_tokens
    ):
        raise ReleaseVerificationError("Token accounting mismatch")
    expected_cost = (
        accounting.prompt_tokens * accounting.input_price_per_million_usd
        + accounting.completion_tokens * accounting.output_price_per_million_usd
    ) / 1_000_000
    if accounting.list_price_estimated_cost_usd != expected_cost:
        raise ReleaseVerificationError("List-price accounting mismatch")
    if accounting.providers != tuple(
        sorted(set(provider for item in evidence.shards for provider in item.providers))
    ):
        raise ReleaseVerificationError("Provider accounting mismatch")
    provider_costs = [item.provider_cost_usd for item in evidence.shards]
    expected_provider_cost = (
        sum(item for item in provider_costs if item is not None)
        if all(item is not None for item in provider_costs)
        else None
    )
    if accounting.provider_estimated_cost_usd != expected_provider_cost:
        raise ReleaseVerificationError("Provider cost accounting mismatch")


def _check_shard_accounting(*, evidence: ReleaseEvidence) -> None:
    """Check shard ranges and additive counters.

    Raises:
        ReleaseVerificationError:
            If ranges or counters are inconsistent.
    """
    expected_offset = 0
    for shard in evidence.shards:
        if shard.offset != expected_offset or shard.requests < shard.rows:
            raise ReleaseVerificationError("Shard ranges or request sums are invalid")
        if shard.retries != shard.requests - shard.rows:
            raise ReleaseVerificationError("Shard retry accounting mismatch")
        expected_offset += shard.rows
    accounting = evidence.accounting
    for field in (
        "requests",
        "retries",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ):
        if sum(getattr(item, field) for item in evidence.shards) != getattr(
            accounting, field
        ):
            raise ReleaseVerificationError("Shard accounting sum mismatch")
    if (
        sum(item.rejected_validation_responses for item in evidence.shards)
        != accounting.rejected_validation_responses
    ):
        raise ReleaseVerificationError("Rejected-response accounting mismatch")


def _check_output(
    *,
    release_dir: Path,
    manifest: ReleaseManifest,
    output: pl.DataFrame,
    report: ValidationReport,
) -> None:
    """Check the public output schema and validation report.

    Raises:
        ReleaseVerificationError:
            If the schema or report is invalid.
    """
    if output.height != manifest.rows:
        raise ReleaseVerificationError("Output row count does not match manifest")
    if set(output.columns) != set(PERSONA_OUTPUT_COLUMNS) or len(output.columns) != len(
        PERSONA_OUTPUT_COLUMNS
    ):
        raise ReleaseVerificationError(
            "Persona output schema must match generation contract v5"
        )
    if not persona_output_dtypes_are_valid(output):
        raise ReleaseVerificationError(
            "Persona output contains an invalid logical dtype"
        )
    try:
        validate_persona_output_rows(
            output,
            job_title_mapping=load_job_title_mapping(
                release_dir / "provenance/config/job-function-titles.yaml"
            ),
            origin_label_contract=load_origin_label_contract(
                release_dir / "provenance/config/folk2-ieland-labels-da.yaml"
            ),
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise ReleaseVerificationError(
            "Persona output fails contextual generation-v5 validation"
        ) from error
    if not report.passed or report.kind != "persona_pilot":
        raise ReleaseVerificationError("Pilot validation report is not passing")
    if report.subject_id != manifest.pilot_id:
        raise ReleaseVerificationError("Pilot report subject does not match")


def _check_provenance_bindings(
    *, release_dir: Path, manifest: ReleaseManifest, evidence: ReleaseEvidence
) -> None:
    hashes = {
        "provenance/release-policy.yaml": evidence.policy_sha256,
        "attestations/human-review.json": evidence.attestation_sha256,
        "LICENSE.txt": evidence.licence_sha256,
        "provenance/code/LICENSE": evidence.code_license_sha256,
        "provenance/code/uv.lock": evidence.uv_lock_sha256,
    }
    for path, expected in hashes.items():
        if sha256_file(release_dir / path) != expected:
            raise ReleaseVerificationError(f"Provenance checksum mismatch: {path}")
    if manifest.uv_lock_sha256 != evidence.uv_lock_sha256:
        raise ReleaseVerificationError("Lock hash binding failed")
    if (
        sha256_file(release_dir / "provenance/evidence.json")
        != manifest.evidence_sha256
    ):
        raise ReleaseVerificationError("Evidence checksum binding failed")
    if sha256_file(release_dir / "provenance/persona-da.md") != evidence.prompt_sha256:
        raise ReleaseVerificationError("Generation prompt checksum mismatch")
    _check_origin_contract(
        release_dir=release_dir, manifest=manifest, evidence=evidence
    )
    _check_config_hashes(release_dir=release_dir, evidence=evidence)
    policy = load_yaml_model(
        path=release_dir / "provenance/release-policy.yaml", model=ReleasePolicy
    )
    if evidence.licence_sha256 != policy.licence_file_sha256:
        raise ReleaseVerificationError("Policy licence binding failed")
    if evidence.rows != manifest.rows or evidence.model != manifest.model:
        raise ReleaseVerificationError("Evidence model binding failed")


def _check_config_hashes(*, release_dir: Path, evidence: ReleaseEvidence) -> None:
    """Check hashes for configuration files included in the fixed layout.

    Raises:
        ReleaseVerificationError:
            If a configuration hash does not match.
    """
    expected_names = {
        "config.yaml",
        "job-function-titles.yaml",
        "sources.lock.yaml",
        "categories.yaml",
        "sampling.yaml",
        "validation.yaml",
        "folk2-ieland-labels-da.yaml",
    }
    if set(evidence.config_hashes) != expected_names:
        raise ReleaseVerificationError("Configuration hash set is incomplete")
    for name, expected in evidence.config_hashes.items():
        path = f"provenance/config/{name}"
        if path not in _PUBLIC_FILES or sha256_file(release_dir / path) != expected:
            raise ReleaseVerificationError("Configuration checksum mismatch")
    _check_generation_context(release_dir=release_dir, evidence=evidence)


def _check_generation_context(*, release_dir: Path, evidence: ReleaseEvidence) -> None:
    """Check the packaged effective config, prompts, and context digest.

    Raises:
        ReleaseVerificationError:
            If a packaged generation input or digest is inconsistent.
    """
    config_path = release_dir / "provenance/config/config.yaml"
    config, prompt_path = _load_generation_inputs(
        config_path=config_path, release_dir=release_dir, evidence=evidence
    )
    try:
        mapping_path = release_dir / "provenance/config/job-function-titles.yaml"
        mapping = load_job_title_mapping(mapping_path)
        origin_contract = _load_bound_origin_contract(
            release_dir=release_dir, evidence=evidence
        )
        origin_path = release_dir / "provenance/config/folk2-ieland-labels-da.yaml"
        context = generation_context_sha256(
            config=config,
            prompt=prompt_path.read_text(encoding="utf-8"),
            job_title_mapping=mapping,
            job_title_mapping_sha256=sha256_file(mapping_path),
            origin_label_contract=origin_contract,
            origin_label_contract_sha256=sha256_file(origin_path),
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise ReleaseVerificationError(
            "Generation context cannot be computed"
        ) from error
    if context != evidence.generation_context_sha256:
        raise ReleaseVerificationError("Generation context checksum binding failed")


def _load_bound_origin_contract(
    *, release_dir: Path, evidence: ReleaseEvidence
) -> OriginLabelContract:
    """Load the packaged contract and compare its portable evidence binding.

    Returns:
        The strictly validated packaged origin-label contract.

    Raises:
        ReleaseVerificationError:
            If the contract content, version, or checksum differs from evidence.
    """
    path = release_dir / "provenance/config/folk2-ieland-labels-da.yaml"
    contract = load_origin_label_contract(path)
    if (
        contract != evidence.origin_label_contract_content
        or contract.version != evidence.origin_label_contract_version
        or sha256_file(path) != evidence.origin_label_contract_sha256
    ):
        raise ReleaseVerificationError("Origin-label contract content mismatch")
    return contract


def _load_generation_inputs(
    *, release_dir: Path, config_path: Path, evidence: ReleaseEvidence
) -> tuple[GenerationConfig, Path]:
    """Load and check the packaged generation inputs.

    Returns:
        The effective config and the packaged prompt path.

    Raises:
        ReleaseVerificationError:
            If a generation input is missing, changed, or misbound.
    """
    config = _load_yaml(config_path, GenerationConfig)
    if evidence.generation_config_sha256 != sha256_file(config_path):
        raise ReleaseVerificationError("Generation config checksum binding failed")
    prompt_path = release_dir / "provenance/persona-da.md"
    if config.prompt != Path("config/persona-da.md"):
        raise ReleaseVerificationError("Generation prompt path binding failed")
    if config.job_title_mapping != Path("config/job-function-titles.yaml"):
        raise ReleaseVerificationError("Job-title mapping path binding failed")
    if config.origin_label_contract != DEFAULT_ORIGIN_LABEL_CONTRACT_PATH:
        raise ReleaseVerificationError("Origin-label contract path binding failed")
    if sha256_file(prompt_path) != evidence.prompt_sha256:
        raise ReleaseVerificationError("Generation prompt checksum mismatch")
    return config, prompt_path


def _load_yaml(path: Path, model: type[ModelType]) -> ModelType:
    try:
        return load_yaml_model(path=path, model=model)
    except Exception as error:
        raise ReleaseVerificationError(
            f"Invalid public YAML contract: {path}"
        ) from error


def _check_origin_contract(
    *, release_dir: Path, manifest: ReleaseManifest, evidence: ReleaseEvidence
) -> None:
    """Verify the immutable Danish origin-label contract bytes and identity.

    Raises:
        ReleaseVerificationError:
            If the packaged contract is missing, altered, or misbound.
    """
    path = release_dir / "provenance/config/folk2-ieland-labels-da.yaml"
    if evidence.origin_label_contract_file != DEFAULT_ORIGIN_LABEL_CONTRACT_PATH:
        raise ReleaseVerificationError("Origin-label contract path binding failed")
    if manifest.origin_label_contract_file != DEFAULT_ORIGIN_LABEL_CONTRACT_PATH:
        raise ReleaseVerificationError("Origin-label manifest path binding failed")
    try:
        contract = load_origin_label_contract(path)
    except (OSError, UnicodeError, ValueError) as error:
        raise ReleaseVerificationError("Invalid origin-label contract") from error
    digest = sha256_file(path)
    if digest != ORIGIN_LABEL_CONTRACT_SHA256:
        raise ReleaseVerificationError("Origin-label contract checksum mismatch")
    if (
        digest != evidence.origin_label_contract_sha256
        or digest != manifest.origin_label_contract_sha256
        or contract != evidence.origin_label_contract_content
        or contract != manifest.origin_label_contract_content
        or contract.version != evidence.origin_label_contract_version
        or contract.version != manifest.origin_label_contract_version
    ):
        raise ReleaseVerificationError("Origin-label contract binding failed")


def _output_ids(output: pl.DataFrame) -> list[str]:
    if "persona_id" not in output.columns:
        raise ReleaseVerificationError("Persona output lacks persona_id")
    ids = output.get_column("persona_id").to_list()
    if (
        not ids
        or any(not isinstance(item, str) or not item.strip() for item in ids)
        or len(ids) != len(set(ids))
    ):
        raise ReleaseVerificationError("Persona IDs are not unique non-empty strings")
    return t.cast(list[str], ids)


def _read_output(path: Path) -> pl.DataFrame:
    try:
        return pl.read_parquet(path)
    except Exception as error:
        raise ReleaseVerificationError(
            "Persona output is not readable Parquet"
        ) from error


def _scan_release_text(*, release_dir: Path) -> None:
    text_files = [
        path
        for path in release_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".md", ".json", ".yaml", ".yml", ".txt", ".lock"}
    ]
    for path in text_files:
        try:
            content = path.read_bytes()
            content.decode("utf-8")
            _scan_text(content)
        except UnicodeDecodeError as error:
            raise ReleaseVerificationError("Public text is not UTF-8") from error
    output = _read_output(release_dir / "data/personas.parquet")
    for column in output.columns:
        for value in output[column].to_list():
            _scan_value(value)


def _scan_text(content: bytes) -> None:
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
    scan_bytes = content.replace(b"hf_xet", b"hf-xet")
    if any(re.search(pattern, scan_bytes) for pattern in patterns):
        raise ReleaseVerificationError("Public release contains a path or secret")


def _scan_value(value: object) -> None:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ReleaseVerificationError(
                "Public binary value is not UTF-8"
            ) from error
    if isinstance(value, str):
        _scan_text(value.encode("utf-8"))
    elif isinstance(value, dict):
        for key, nested in value.items():
            _scan_value(key)
            _scan_value(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _scan_value(nested)
    elif value is not None:
        _scan_text(str(value).encode("utf-8"))
