"""Shared fixtures for release packaging and verification tests."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest
import yaml
from support import ReleaseCase, coherent_evidence

from danish_personas.generation.models import (
    GeneratedAttributes,
    GenerationConfig,
    PersonaDescriptions,
    PilotManifest,
)
from danish_personas.generation.pipeline import generation_context_sha256
from danish_personas.generation.validation import VALIDATOR_VERSION
from danish_personas.io import sha256_file
from danish_personas.models import DemographicRecord, ValidationReport
from danish_personas.release import packager
from danish_personas.release.models import (
    ReleaseEvidence,
    ReleasePolicy,
    ReviewAttestation,
)

ROOT = Path(__file__).parents[2]
PILOT_ID = "b" * 16
MODEL = "provider/model@revision"
REVIEWED_AT = datetime(2026, 9, 17, tzinfo=timezone.utc)


@pytest.fixture
def nonemployee_output(release_case: ReleaseCase) -> pl.DataFrame:
    """Return a v5 row with a non-employee status and null job title."""
    return (
        pl.read_parquet(release_case.output)
        .head(1)
        .with_columns(
            pl.lit("unemployed").alias("labour_market_status"),
            pl.lit("50").alias("detailed_status_code"),
            pl.lit("Ledig").alias("detailed_status"),
            pl.lit("status").alias("detailed_status_resolution"),
            pl.lit(None, dtype=pl.String).alias("job_function_code"),
            pl.lit(None, dtype=pl.String).alias("job_function"),
            pl.lit("not_applicable").alias("job_function_resolution"),
            pl.lit(None, dtype=pl.String).alias("job_title"),
            pl.lit(
                "Han er 35 år og bor i Aarhus. Han kommer fra Danmark, har en "
                "ungdomsuddannelse og er ledig. Han har aldrig været gift og er "
                "single. I hverdagen bruger han planlægning, og i fritiden dyrker "
                "han vandring og musik med lokale fællesskaber. Han har ofte "
                "tendens til at være nysgerrig, når han møder nye muligheder. "
                "At udvikle nye færdigheder."
            ).alias("persona"),
        )
    )


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
        assert repository_root.name == "repository"
        assert pilot_dir.name == "pilot"
        assert pilot_dir != release_case.pilot
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
        snapshot_pilot = kwargs["pilot_dir"]
        assert isinstance(snapshot_pilot, Path)
        assert snapshot_pilot.name == "pilot"
        assert kwargs["manifest"] == release_case.manifest
        evidence = coherent_evidence(release_case)
        report_path = snapshot_pilot / "pilot-validation-report.json"
        return evidence.model_copy(
            update={"pilot_validation_report_sha256": sha256_file(report_path)}
        )

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
    assert len(calls) == 1
    assert calls[0].name == "pilot"
    return release_case, result.path, result.manifest_sha256


@pytest.fixture
def release_case(tmp_path: Path) -> ReleaseCase:
    """Create a clean temporary repository and a 10,000-row pilot.

    Returns:
        A complete release input fixture.
    """
    repository = tmp_path / "repository"
    repository.mkdir()
    for relative in (
        ".gitattributes",
        "uv.lock",
        "LICENSE",
        "config/sources.lock.yaml",
        "config/categories.yaml",
        "config/sampling.yaml",
        "config/validation.yaml",
        "config/folk2-ieland-labels-da.yaml",
        "docs/source-register.md",
        "docs/privacy-risk-register.md",
        "docs/acceptance-criteria.md",
    ):
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    destination = repository / "config/persona-da.md"
    shutil.copyfile(ROOT / "config/persona-da.md", destination)
    mapping_path = repository / "config/job-function-titles.yaml"
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "config/job-function-titles.yaml", mapping_path)

    generation_config = GenerationConfig(
        base_url="https://llm.example/v1",
        model=MODEL,
        api_key_env="TEST_TOKEN",
        timeout_seconds=30,
        maximum_http_attempts=3,
        maximum_validation_attempts=2,
        maximum_total_requests=15,
        retry_backoff_seconds=0,
        maximum_rows_per_shard=5,
        response_format="json_schema",
        prompt=Path("config/persona-da.md"),
        job_title_mapping=Path("config/job-function-titles.yaml"),
        origin_label_contract=Path("config/folk2-ieland-labels-da.yaml"),
    )
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    config_path = repository / "config/config.yaml"
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
    output_frame = pl.DataFrame(
        {
            "persona_id": ids,
            "cultural_context": ["Dansk hverdagsliv med lokale fællesskaber."] * 10_000,
            "skills_and_expertise": [["planlægning", "samarbejde", "formidling"]]
            * 10_000,
            "hobbies_and_interests": [["vandring", "musik", "madlavning"]] * 10_000,
            "career_goals_and_ambitions": ["At udvikle nye færdigheder."] * 10_000,
            "job_title": ["forretningsspecialist"] * 10_000,
            "current_relationship_status": ["not_partnered"] * 10_000,
            "partner_gender": [None] * 10_000,
            "legal_status_detail": [None] * 10_000,
            "persona": [
                "Han er 35 år og bor i Aarhus. Han kommer fra Danmark, har en "
                "ungdomsuddannelse og arbejder som forretningsspecialist på en mindre "
                "arbejdsplads i byen. I hverdagen bruger han planlægning, og i "
                "fritiden "
                "dyrker han vandring og musik sammen med lokale fællesskaber. Han har "
                "ofte tendens til at være nysgerrig, når han møder nye opgaver. Han er "
                "single og har aldrig været gift. At "
                "udvikle nye færdigheder."
            ]
            * 10_000,
        }
    )
    for name in DemographicRecord.model_fields:
        if name not in output_frame.columns:
            fixture_values: dict[str, object] = {
                "country": "Danmark",
                "origin_country_code": "5100",
                "origin_country": "Denmark",
                "origin_country_da": "Danmark",
                "age": 35,
                "age_resolution": "municipality_age_band",
                "age_band": "30-39",
                "sex": "male",
                "marital_status": "never_married",
                "marital_resolution": "municipality",
                "municipality_code": "0751",
                "municipality": "Aarhus",
                "region_code": "1084",
                "region": "Midtjylland",
                "education_level": "secondary_or_vocational",
                "education_source_code": "h30",
                "education_resolution": "ras209_age_band",
                "labour_market_status": "employed",
                "detailed_status_code": "30",
                "detailed_status": "Beskæftiget",
                "detailed_status_resolution": "status",
                "job_function_code": "24",
                "job_function": "24 Business and administration professionals",
                "job_function_resolution": "lons20_sex_marginal",
                "openness_score": 70.0,
                "openness_label": "high",
                "conscientiousness_score": 50.0,
                "conscientiousness_label": "average",
                "extraversion_score": 50.0,
                "extraversion_label": "average",
                "agreeableness_score": 50.0,
                "agreeableness_label": "average",
                "neuroticism_score": 50.0,
                "neuroticism_label": "average",
            }
            value: object = fixture_values.get(name, "Danmark")
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
        "origin_label_contract_file": "config/folk2-ieland-labels-da.yaml",
        "origin_label_contract_sha256": sha256_file(
            repository / "config/folk2-ieland-labels-da.yaml"
        ),
        "origin_label_contract_version": 1,
        "origin_label_contract_content": yaml.safe_load(
            (repository / "config/folk2-ieland-labels-da.yaml").read_text(
                encoding="utf-8"
            )
        ),
    }
    generation_context = generation_context_sha256(
        config=generation_config,
        prompt=(repository / "config/persona-da.md").read_text(encoding="utf-8"),
        job_title_mapping=packager.load_job_title_mapping(mapping_path),
        job_title_mapping_sha256=sha256_file(mapping_path),
        origin_label_contract=packager.load_origin_label_contract(
            repository / "config/folk2-ieland-labels-da.yaml"
        ),
        origin_label_contract_sha256=sha256_file(
            repository / "config/folk2-ieland-labels-da.yaml"
        ),
    )
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
        generation_config_file=Path("config/config.yaml"),
        generation_config_sha256=sha256_file(config_path),
        generation_context_sha256=generation_context,
        validator_version=VALIDATOR_VERSION,
        job_title_mapping_file=Path("config/job-function-titles.yaml"),
        job_title_mapping_sha256=sha256_file(mapping_path),
        job_title_mapping_version=1,
        origin_label_contract_file=Path("config/folk2-ieland-labels-da.yaml"),
        origin_label_contract_sha256=sha256_file(
            repository / "config/folk2-ieland-labels-da.yaml"
        ),
        origin_label_contract_version=1,
        origin_label_contract_content=packager.load_origin_label_contract(
            repository / "config/folk2-ieland-labels-da.yaml"
        ),
        prompt_sha256=sha256_file(repository / "config/persona-da.md"),
        rows=10_000,
        batch_size=5,
        batches=1,
        batch_runs=[shard_reference],
        maximum_total_requests=10_000,
        maximum_shard_requests=20,
        requests=10_000,
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
