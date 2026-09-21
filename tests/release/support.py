"""Neutral helpers shared by release tests."""

from __future__ import annotations

import collections.abc as c
import json
import typing as t
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from danish_personas.generation.models import PilotManifest
from danish_personas.io import sha256_file
from danish_personas.models import ValidationReport
from danish_personas.release import packager
from danish_personas.release.models import (
    Accounting,
    ReleaseApproval,
    ReleaseEvidence,
    ReleasePackageResult,
    ReleasePolicy,
    ReviewAttestation,
    ShardEvidence,
)
from danish_personas.release.packager import package_release
from danish_personas.release.policy import validate_release_approval

_HASH = "a" * 64
_PILOT_ID = "b" * 16
_MODEL = "provider/model@revision"
_PILOT_ROWS = 10_000


@dataclass(frozen=True)
class ReleaseCase:
    """Paths and contracts for a minimal but complete release input."""

    repository: Path
    pilot: Path
    attestation: Path
    policy: Path
    card: Path
    licence: Path
    output_parent: Path
    output: Path
    manifest: PilotManifest
    attestation_model: ReviewAttestation
    policy_model: ReleasePolicy
    report: ValidationReport


def _approve_pilot(
    *,
    policy: ReleasePolicy | None = None,
    attestation: ReviewAttestation | None = None,
    output_persona_ids: object | None = None,
) -> ReleaseApproval:
    ids = _output_ids(_PILOT_ROWS) if output_persona_ids is None else output_persona_ids
    return validate_release_approval(
        policy or make_policy(),
        attestation or make_attestation(),
        model=_MODEL,
        pilot_id=_PILOT_ID,
        output_sha256=_HASH,
        population_rows=_PILOT_ROWS,
        output_persona_ids=t.cast(c.Iterable[str], ids),
    )


def _output_ids(rows: int) -> c.Iterator[str]:
    return (f"persona-{index}" for index in range(rows))


def make_attestation(**overrides: object) -> ReviewAttestation:
    """Build a valid explicit blinded-review attestation for tests.

    Returns:
        A valid review attestation.
    """
    values: dict[str, object] = {
        "version": 1,
        "release_approved": True,
        "blinded": True,
        "reviewer_id": "reviewer-1",
        "protocol": "blinded-human-review",
        "protocol_version": "1",
        "reviewed_at": datetime(2026, 9, 17, tzinfo=timezone.utc),
        "pilot_id": _PILOT_ID,
        "output_sha256": _HASH,
        "population_rows": _PILOT_ROWS,
        "reviewed_persona_ids": [f"persona-{index}" for index in range(300)],
    }
    values.update(overrides)
    return ReviewAttestation.model_validate(values)


def make_policy(**overrides: object) -> ReleasePolicy:
    """Build a valid enabled release policy for tests.

    Returns:
        A valid enabled policy.
    """
    values: dict[str, object] = {
        "version": 1,
        "enabled": True,
        "approved_models": [_MODEL],
        "dataset_licence": "CC-BY-4.0",
        "licence_file_sha256": _HASH,
    }
    values.update(overrides)
    return ReleasePolicy.model_validate(values)


def _clean_provenance(_: Path) -> tuple[str, str, str]:
    return "a" * 40, "https://example.invalid/origin.git", ""


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


def coherent_evidence(case: ReleaseCase) -> ReleaseEvidence:
    """Build evidence whose accounting and file bindings are internally coherent.

    Returns:
        Strict evidence for the fixture release.
    """
    manifest = case.manifest
    shard = ShardEvidence(
        shard_id="shard-1",
        offset=0,
        rows=10_000,
        manifest_sha256="a" * 64,
        report_sha256="b" * 64,
        output_sha256=manifest.output_sha256,
        generation_config_sha256=manifest.generation_config_sha256,
        generation_context_sha256=manifest.generation_context_sha256,
        origin_label_contract_file=manifest.origin_label_contract_file,
        origin_label_contract_sha256=manifest.origin_label_contract_sha256,
        origin_label_contract_version=manifest.origin_label_contract_version,
        requests=10_000,
        retries=0,
        rejected_validation_responses=0,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        provider_cost_usd=0,
        providers=("test-provider",),
    )
    config_hashes = {
        "config.yaml": sha256_file(case.repository / "config.yaml"),
        **{
            name: sha256_file(case.repository / "config" / name)
            for name in (
                "job-function-titles.yaml",
                "sources.lock.yaml",
                "categories.yaml",
                "sampling.yaml",
                "validation.yaml",
                "folk2-ieland-labels-da.yaml",
            )
        },
    }
    return ReleaseEvidence(
        version=2,
        pilot_id=manifest.pilot_id,
        model=manifest.model,
        rows=manifest.rows,
        output_sha256=manifest.output_sha256,
        input_sha256=manifest.input_sha256,
        sample_manifest_sha256=manifest.sample_manifest_sha256,
        generation_config_sha256=manifest.generation_config_sha256,
        generation_context_sha256=manifest.generation_context_sha256,
        origin_label_contract_file=manifest.origin_label_contract_file,
        origin_label_contract_sha256=manifest.origin_label_contract_sha256,
        origin_label_contract_version=manifest.origin_label_contract_version,
        origin_label_contract_content=manifest.origin_label_contract_content,
        validator_version=manifest.validator_version,
        attributes_prompt_sha256=manifest.attributes_prompt_sha256,
        personas_prompt_sha256=manifest.personas_prompt_sha256,
        upstream_run_id=manifest.upstream_run_id,
        sample_source_run_id="source-run",
        source_bundle_id="bundle",
        policy_sha256=sha256_file(case.policy),
        attestation_sha256=sha256_file(case.attestation),
        licence_sha256=sha256_file(case.licence),
        code_license_sha256=sha256_file(case.repository / "LICENSE"),
        uv_lock_sha256=sha256_file(case.repository / "uv.lock"),
        pilot_validation_report_sha256=(
            sha256_file(case.pilot / "pilot-validation-report.json")
            if (case.pilot / "pilot-validation-report.json").exists()
            else "c" * 64
        ),
        config_hashes=config_hashes,
        shards=(shard,),
        accounting=Accounting(
            requests=10_000,
            retries=0,
            rejected_validation_responses=0,
            dropped_rows=0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            input_price_per_million_usd=0,
            output_price_per_million_usd=0,
            list_price_estimated_cost_usd=0,
            provider_estimated_cost_usd=0,
            providers=("test-provider",),
        ),
    )


def _refresh_artifact(release: Path, relative: str) -> str:
    """Refresh one manifest artifact and the manifest sidecar.

    Args:
        release:
            Release directory.
        relative:
            Manifest-relative artifact path.

    Returns:
        The refreshed manifest digest.

    Raises:
        AssertionError:
            If the artifact is not listed in the manifest.
    """
    manifest = json.loads(
        (release / "release-manifest.json").read_text(encoding="utf-8")
    )
    for artifact in manifest["artifacts"]:
        if artifact["path"] == relative:
            artifact["sha256"] = sha256_file(release / relative)
            artifact["size"] = (release / relative).stat().st_size
            break
    else:
        raise AssertionError(relative)
    return _refresh_manifest(release, artifacts=manifest["artifacts"])


def _refresh_manifest(release: Path, **changes: object) -> str:
    """Apply manifest changes and return its new externally retained digest.

    Returns:
        The refreshed manifest digest.
    """
    path = release / "release-manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(changes)
    path.write_bytes(
        (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    )
    digest = sha256_file(path)
    (release / "release-manifest.sha256").write_bytes(
        f"{digest}  release-manifest.json\n".encode("ascii")
    )
    return digest
