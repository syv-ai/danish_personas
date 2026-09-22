"""Input, prompt-boundary, and upstream-contract pipeline tests."""

import json
from pathlib import Path

import polars as pl
import pytest
import yaml
from generation_test_helpers import MockGenerationClient, write_generation_inputs

from danish_personas.generation.models import FrozenSampleManifest, GenerationConfig
from danish_personas.generation.personality import allowed_personality_tendencies
from danish_personas.generation.pipeline import (
    generate_personas,
    models_match,
    validate_upstream_sample,
)
from danish_personas.generation.report import validate_persona_run
from danish_personas.generation.validation import VALIDATOR_VERSION
from danish_personas.io import sha256_file, write_json
from danish_personas.models import RunManifest


def test_generation_rejects_origin_contract_and_row_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Obsolete config fields and mismatched origin inputs fail closed."""
    paths = write_generation_inputs(root=tmp_path)
    config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))

    config["version"] = 4
    paths["config"].write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        generate_personas(
            input_path=paths["sample"],
            sample_manifest_path=paths["sample_manifest"],
            config_path=paths["config"],
            output_dir=tmp_path / "v2",
            rows=1,
        )

    config.pop("version")
    config.pop("origin_label_contract")
    paths["config"].write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        GenerationConfig.model_validate(config)

    config["origin_label_contract"] = str(
        Path("config/folk2-ieland-labels-da.yaml").resolve()
    )
    with pytest.raises(ValueError, match="canonical path"):
        GenerationConfig.model_validate(config)

    config["origin_label_contract"] = "missing-origin-labels.yaml"
    paths["config"].write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical path"):
        generate_personas(
            input_path=paths["sample"],
            sample_manifest_path=paths["sample_manifest"],
            config_path=paths["config"],
            output_dir=tmp_path / "missing",
            rows=1,
        )

    tampered_contract = tmp_path / "tampered-origin-labels.yaml"
    tampered = yaml.safe_load(
        Path("config/folk2-ieland-labels-da.yaml").read_text(encoding="utf-8")
    )
    tampered["labels_da"]["5100"] = "Libanon"
    tampered_contract.write_text(
        yaml.safe_dump(tampered, allow_unicode=True), encoding="utf-8"
    )
    config["origin_label_contract"] = tampered_contract.name
    config["job_title_mapping"] = str(Path("config/job-function-titles.yaml").resolve())
    paths["config"].write_text(yaml.safe_dump(config), encoding="utf-8")
    with monkeypatch.context() as patch:
        patch.chdir(tmp_path)
        with pytest.raises(ValueError):
            generate_personas(
                input_path=paths["sample"],
                sample_manifest_path=paths["sample_manifest"],
                config_path=paths["config"],
                output_dir=tmp_path / "tampered",
                rows=1,
            )

    config["origin_label_contract"] = "config/folk2-ieland-labels-da.yaml"
    config.pop("job_title_mapping")
    paths["config"].write_text(yaml.safe_dump(config), encoding="utf-8")
    sample = pl.read_parquet(paths["sample"]).with_columns(
        pl.when(pl.col("origin_country_code") == "5100")
        .then(pl.lit("Libanon"))
        .otherwise(pl.col("origin_country_da"))
        .alias("origin_country_da")
    )
    sample.write_parquet(paths["sample"])
    sample_manifest = FrozenSampleManifest.model_validate_json(
        paths["sample_manifest"].read_text(encoding="utf-8")
    ).model_copy(update={"sha256": sha256_file(paths["sample"])})
    write_json(path=paths["sample_manifest"], payload=sample_manifest)
    source_path = paths["sample"].parent / "structured-records.parquet"
    sample.write_parquet(source_path)
    run_manifest_path = source_path.parent / "run-manifest.json"
    run_manifest = RunManifest.model_validate_json(
        run_manifest_path.read_text(encoding="utf-8")
    ).model_copy(update={"data_sha256": sha256_file(source_path)})
    write_json(path=run_manifest_path, payload=run_manifest)
    with pytest.raises(ValueError, match="triple"):
        generate_personas(
            input_path=paths["sample"],
            sample_manifest_path=paths["sample_manifest"],
            config_path=paths["config"],
            output_dir=tmp_path / "mismatch",
            rows=1,
        )


def test_generation_withholds_resolution_provenance_from_both_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The LLM request receives values without sampler resolution metadata."""
    paths = write_generation_inputs(root=tmp_path)
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", MockGenerationClient
    )
    MockGenerationClient.requests = 0
    MockGenerationClient.payloads = []

    run_dir = generate_personas(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "outputs",
        rows=1,
    )

    resolution_columns = (
        "age_resolution",
        "marital_resolution",
        "education_resolution",
        "detailed_status_resolution",
        "origin_country_code",
        "origin_country",
        "job_function_code",
        "job_function_resolution",
        "municipality_code",
        "municipality",
    )
    sample = pl.read_parquet(paths["sample"])
    assert set(resolution_columns) <= set(sample.columns)
    assert len(MockGenerationClient.payloads) == 1
    generation_request = MockGenerationClient.payloads[0]
    generation_payload = generation_request["demographics_and_personality"]
    assert isinstance(generation_payload, dict)

    allowed_fields = {
        "origin_country_da",
        "municipality",
        "job_function",
        "age",
        "sex",
        "education_level",
        "labour_market_status",
        "openness_score",
        "openness_label",
        "conscientiousness_score",
        "conscientiousness_label",
        "extraversion_score",
        "extraversion_label",
        "agreeableness_score",
        "agreeableness_label",
        "neuroticism_score",
        "neuroticism_label",
        "current_status",
        "marital_status",
    }
    forbidden_fields = {
        "age_resolution",
        "marital_resolution",
        "education_resolution",
        "detailed_status_resolution",
        "origin_country_code",
        "origin_country",
        "job_function_code",
        "job_function_resolution",
        "municipality_code",
        "region_code",
        "country",
        "detailed_status_code",
        "education_source_code",
        "persona_id",
    }
    assert set(generation_request) == {
        "demographics_and_personality",
        "allowed_job_titles",
        "allowed_personality_tendencies",
        "same_sex_partner_target",
    }
    assert set(generation_payload) == allowed_fields
    assert forbidden_fields.isdisjoint(generation_payload)
    assert allowed_fields.isdisjoint(forbidden_fields)
    assert {"municipality", "origin_country_da", "job_function"} <= set(
        generation_payload
    )
    allowed_phrases = generation_request["allowed_personality_tendencies"]
    assert isinstance(allowed_phrases, list)
    assert allowed_phrases == list(
        allowed_personality_tendencies(context=generation_payload)
    )
    assert allowed_phrases
    assert all(
        isinstance(phrase, str) and phrase.startswith("har ofte tendens til at være ")
        for phrase in allowed_phrases
    )
    assert (
        generation_payload["job_function"]
        == "Business and administration professionals"
    )
    assert generation_payload["municipality"] == "København"
    assert generation_payload["origin_country_da"] == "Danmark"
    assert "origin_country" not in generation_payload
    assert generation_payload["education_level"] == "videregående uddannelse"

    generation_manifest = json.loads(
        (run_dir / "generation-manifest.json").read_text(encoding="utf-8")
    )
    assert generation_manifest["input_sha256"] == sha256_file(paths["sample"])
    assert generation_manifest["validator_version"] == VALIDATOR_VERSION
    assert generation_manifest["origin_label_contract_file"] == (
        "config/folk2-ieland-labels-da.yaml"
    )
    assert generation_manifest["origin_label_contract_version"] == 1
    assert generation_manifest["origin_label_contract_content"]["labels_da"][
        "5100"
    ] == ("Danmark")
    assert validate_persona_run(run_dir=run_dir).passed
    report = json.loads((run_dir / "validation-report.json").read_text())
    assert (
        report["origin_label_contract_sha256"]
        == generation_manifest["origin_label_contract_sha256"]
    )
    output = pl.read_parquet(run_dir / "generated-personas.parquet")
    assert set(resolution_columns) <= set(output.columns)
    assert output.select(list(resolution_columns)).equals(
        sample.head(1).select(list(resolution_columns))
    )


def test_pipeline_selects_an_offset_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bounded invocation can select a later frozen-sample shard."""
    paths = write_generation_inputs(root=tmp_path)
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", MockGenerationClient
    )
    MockGenerationClient.requests = 0
    MockGenerationClient.payloads = []
    first_run = generate_personas(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "outputs",
        rows=1,
        offset=0,
    )
    second_run = generate_personas(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "outputs",
        rows=1,
        offset=1,
    )
    assert first_run != second_run
    first_output = pl.read_parquet(first_run / "generated-personas.parquet")
    second_output = pl.read_parquet(second_run / "generated-personas.parquet")
    assert first_output.get_column("persona_id").to_list() == ["persona-1"]
    assert second_output.get_column("persona_id").to_list() == ["persona-2"]
    demographics = MockGenerationClient.payloads[1]["demographics_and_personality"]
    assert isinstance(demographics, dict)
    assert demographics["origin_country_da"] == "Libanon"
    assert "origin_country" not in demographics
    assert validate_persona_run(run_dir=second_run).passed


def test_provider_qualified_model_alias_matches_case_insensitively() -> None:
    """HF provider suffixes and casing do not create false provenance failures."""
    assert models_match(
        configured="meta-llama/Llama-4-Maverick:novita",
        returned="meta-llama/llama-4-maverick",
    )
    assert not models_match(
        configured="meta-llama/Llama-4-Maverick:novita",
        returned="meta-llama/llama-4-scout",
    )


def test_upstream_sample_rejects_legacy_sampler_schema(tmp_path: Path) -> None:
    """A validated legacy run cannot cross the current Phase-3 boundary."""
    paths = write_generation_inputs(root=tmp_path)
    manifest_path = paths["sample"].parent / "run-manifest.json"
    legacy_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy_manifest["sampler_schema_version"] = 2
    write_json(path=manifest_path, payload=legacy_manifest)

    with pytest.raises(ValueError, match="Unsupported sampler schema version"):
        validate_upstream_sample(
            input_path=paths["sample"], sample_manifest_path=paths["sample_manifest"]
        )


def test_upstream_sample_rejects_origin_less_legacy_columns(tmp_path: Path) -> None:
    """Origin-less schema-v2-shaped rows require migration before Phase 3."""
    paths = write_generation_inputs(root=tmp_path)
    sample_path = paths["sample"]
    pl.read_parquet(sample_path).drop(
        "origin_country_code", "origin_country", "origin_country_da"
    ).write_parquet(sample_path)
    sample_manifest = FrozenSampleManifest.model_validate_json(
        paths["sample_manifest"].read_text(encoding="utf-8")
    )
    write_json(
        path=paths["sample_manifest"],
        payload=sample_manifest.model_copy(update={"sha256": sha256_file(sample_path)}),
    )

    with pytest.raises(ValueError, match="columns do not match"):
        validate_upstream_sample(
            input_path=sample_path, sample_manifest_path=paths["sample_manifest"]
        )
