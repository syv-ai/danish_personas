"""Shared fixtures for release packaging and verification tests."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest
import yaml

from danish_personas.generation.models import (
    GeneratedAttributes,
    GenerationConfig,
    PersonaDescriptions,
    PilotManifest,
)
from danish_personas.io import sha256_file
from danish_personas.models import DemographicRecord, ValidationReport
from danish_personas.release import packager
from danish_personas.release.models import (
    Accounting,
    ReleaseEvidence,
    ReleasePolicy,
    ReviewAttestation,
    ShardEvidence,
)

ROOT = Path(__file__).parents[2]
PILOT_ID = "b" * 16
MODEL = "provider/model@revision"
REVIEWED_AT = datetime(2026, 9, 17, tzinfo=timezone.utc)


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


@pytest.fixture
def packaged_release(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch
) -> tuple[ReleaseCase, Path, str]:
    """Package a fixture release through the public packager.

    Returns:
        The fixture, release path, and external manifest digest.
    """
    calls: list[Path] = []

    def validate(*, pilot_dir: Path, repository_root: Path) -> ValidationReport:
        assert repository_root == release_case.repository
        calls.append(pilot_dir)
        return release_case.report

    def no_inventory(
        *, pilot_dir: Path, repository_root: Path, manifest: PilotManifest
    ) -> list[Path]:
        assert pilot_dir == release_case.pilot
        assert repository_root == release_case.repository
        assert manifest.pilot_id == PILOT_ID
        return []

    def evidence(**kwargs: object) -> ReleaseEvidence:
        assert kwargs["pilot_dir"] == release_case.pilot
        assert kwargs["manifest"] == release_case.manifest
        return coherent_evidence(release_case)

    monkeypatch.setattr(packager, "validate_persona_pilot", validate)
    monkeypatch.setattr(packager, "_derive_consumed_files", no_inventory)
    monkeypatch.setattr(packager, "_derive_evidence", evidence)
    result = packager.package_release(
        pilot_dir=release_case.pilot,
        attestation_path=release_case.attestation,
        policy_path=release_case.policy,
        dataset_card_path=release_case.card,
        licence_path=release_case.licence,
        repository_root=release_case.repository,
        output_parent=release_case.output_parent,
    )
    assert calls == [release_case.pilot]
    return release_case, result.path, result.manifest_sha256


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
        requests=20_000,
        retries=0,
        rejected_validation_responses=0,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        provider_cost_usd=0,
        providers=("test-provider",),
    )
    config_hashes = {
        name: sha256_file(case.repository / "config" / name)
        for name in (
            "generation.yaml",
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
            requests=20_000,
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


@pytest.fixture
def release_case(tmp_path: Path) -> ReleaseCase:
    """Create a clean temporary repository and a 10,000-row pilot.

    Returns:
        A complete release input fixture.
    """
    repository = tmp_path / "repository"
    repository.mkdir()
    for relative in (
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
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    for relative in ("attributes-da.md", "personas-da.md"):
        destination = repository / "config/prompts" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "config/prompts" / relative, destination)

    generation_config = GenerationConfig(
        version=1,
        llm_generation_enabled=True,
        base_url="https://llm.example/v1",
        model=MODEL,
        api_key_env="TEST_TOKEN",
        timeout_seconds=30,
        maximum_http_attempts=3,
        maximum_validation_attempts=2,
        maximum_total_requests=15,
        retry_backoff_seconds=0,
        maximum_smoke_rows=5,
        response_format="json_schema",
        attributes_prompt=Path("config/prompts/attributes-da.md"),
        personas_prompt=Path("config/prompts/personas-da.md"),
    )
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    config_path = repository / "config/generation.yaml"
    config_path.write_text(
        yaml.safe_dump(generation_config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    sample = repository / "sample.parquet"
    sample.write_bytes(b"sample")
    sample_manifest = repository / "sample-manifest.json"
    sample_manifest.write_text('{"source_run_id":"source-run"}\n', encoding="utf-8")
    output = pilot / "generated-personas.parquet"
    ids = [f"persona-{index:05d}" for index in range(10_000)]
    sentence = "Personen finder ro og fællesskab i hverdagen. "
    output_frame = pl.DataFrame(
        {
            "persona_id": ids,
            "cultural_context": ["Dansk hverdagsliv med lokale fællesskaber."] * 10_000,
            "skills_and_expertise": [["planlægning", "samarbejde", "formidling"]]
            * 10_000,
            "hobbies_and_interests": [["vandring", "musik", "madlavning"]] * 10_000,
            "career_goals_and_ambitions": ["At udvikle nye færdigheder."] * 10_000,
            "professional_persona": [sentence + "Arbejdet giver plads til læring."]
            * 10_000,
            "sports_persona": [sentence + "Motion passer naturligt ind."] * 10_000,
            "arts_persona": [sentence + "Kunst og kultur inspirerer."] * 10_000,
            "travel_persona": [sentence + "Nye steder opleves i roligt tempo."]
            * 10_000,
            "culinary_persona": [sentence + "Måltider deles gerne med andre."] * 10_000,
            "persona": [sentence + "Hverdagen rummer nysgerrighed og nærvær."] * 10_000,
            "visual_persona": [
                sentence + "Et neutralt motiv uden personlige kendetegn."
            ]
            * 10_000,
        }
    )
    for name in DemographicRecord.model_fields:
        if name not in output_frame.columns:
            value: object = (
                35 if name == "age" else 50.0 if name.endswith("_score") else "Danmark"
            )
            output_frame = output_frame.with_columns(pl.lit(value).alias(name))
    output_frame = output_frame.select(
        [
            *DemographicRecord.model_fields,
            *GeneratedAttributes.model_fields,
            *PersonaDescriptions.model_fields,
        ]
    )
    output_frame.write_parquet(output)

    input_hash = hashlib.sha256(b"input").hexdigest()
    zero_hash = "a" * 64
    shard_reference = {
        "offset": 0,
        "rows": 5,
        "run_id": "shard-1",
        "manifest_file": "shard/manifest.json",
        "manifest_sha256": zero_hash,
        "validation_report_file": "shard/report.json",
        "validation_report_sha256": zero_hash,
    }
    manifest = PilotManifest(
        pilot_id=PILOT_ID,
        created_at=REVIEWED_AT.isoformat(),
        model=MODEL,
        base_url="https://llm.example/v1",
        upstream_run_id="upstream-run",
        input_file=Path("sample.parquet"),
        input_sha256=input_hash,
        sample_manifest_file=Path("sample-manifest.json"),
        sample_manifest_sha256=sha256_file(sample_manifest),
        generation_config_file=Path("config/generation.yaml"),
        generation_config_sha256=sha256_file(config_path),
        generation_context_sha256=zero_hash,
        validator_version="validator-1",
        attributes_prompt_sha256=sha256_file(
            repository / "config/prompts/attributes-da.md"
        ),
        personas_prompt_sha256=sha256_file(
            repository / "config/prompts/personas-da.md"
        ),
        rows=10_000,
        batch_size=5,
        batches=1,
        batch_runs=[shard_reference],
        maximum_total_requests=20_000,
        maximum_shard_requests=20,
        requests=20_000,
        retries=0,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        input_price_per_million_usd=0,
        output_price_per_million_usd=0,
        list_price_estimated_cost_usd=0,
        provider_estimated_cost_usd=0,
        inference_providers=["test-provider"],
        output_file=Path("generated-personas.parquet"),
        output_sha256=sha256_file(output),
        llm_generation=True,
    )
    (pilot / "pilot-manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )

    licence = tmp_path / "licence.txt"
    licence.write_bytes(b"Creative Commons Attribution 4.0 International\n")
    policy_model = ReleasePolicy(
        version=1,
        enabled=True,
        approved_models=(MODEL,),
        dataset_licence="CC-BY-4.0",
        licence_file_sha256=sha256_file(licence),
    )
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        yaml.safe_dump(policy_model.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    attestation_model = ReviewAttestation(
        version=1,
        release_approved=True,
        blinded=True,
        reviewer_id="reviewer-1",
        protocol="blinded-human-review",
        protocol_version="1",
        reviewed_at=REVIEWED_AT,
        pilot_id=PILOT_ID,
        output_sha256=manifest.output_sha256,
        population_rows=10_000,
        reviewed_persona_ids=ids[:300],
    )
    attestation = tmp_path / "attestation.json"
    attestation.write_text(
        json.dumps(attestation_model.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    card = tmp_path / "card.md"
    card.write_text("# Public synthetic persona dataset\n", encoding="utf-8")
    report = ValidationReport(
        kind="persona_pilot",
        passed=True,
        created_at="before-packaging",
        subject_id=PILOT_ID,
        metrics=[],
    )
    _initialise_git(repository)
    return ReleaseCase(
        repository=repository,
        pilot=pilot,
        attestation=attestation,
        policy=policy,
        card=card,
        licence=licence,
        output_parent=tmp_path / "releases",
        output=output,
        manifest=manifest,
        attestation_model=attestation_model,
        policy_model=policy_model,
        report=report,
    )


def _initialise_git(repository: Path) -> None:
    subprocess.run(["git", "-C", str(repository), "init", "-q"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "tests@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Release Tests"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-qm", "fixture"], check=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "remote",
            "add",
            "origin",
            "https://example.invalid/repo.git",
        ],
        check=True,
    )


@pytest.fixture
def verifier_package(
    packaged_release: tuple[ReleaseCase, Path, str],
) -> tuple[Path, str]:
    """Return a successfully packaged release for verifier tests.

    Returns:
        The release path and external manifest digest.
    """
    _, path, digest = packaged_release
    return path, digest
