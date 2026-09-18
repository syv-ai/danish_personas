"""Integration tests for guarded persona pipeline provenance and resume."""

import collections.abc as c
import json
import typing as t
from pathlib import Path

import polars as pl
import pytest
import yaml
from click.testing import CliRunner

from danish_personas.generation.client import RequestBudgetExceeded
from danish_personas.generation.models import (
    FrozenSampleManifest,
    GenerationConfig,
    LLMResponse,
)
from danish_personas.generation.pilot import run_pilot
from danish_personas.generation.pipeline import (
    generate_personas,
    models_match,
    validate_upstream_sample,
)
from danish_personas.generation.report import (
    validate_persona_pilot,
    validate_persona_run,
)
from danish_personas.io import sha256_file, write_json
from danish_personas.models import SAMPLER_SCHEMA_VERSION, RunManifest, ValidationReport
from scripts.generate_persona_pilot import main as pilot_main


class _MockClient:
    requests = 0
    payloads: list[dict[str, object]] = []

    def __init__(
        self,
        config: GenerationConfig,
        initial_requests_made: int = 0,
        record_request: c.Callable[[int], None] | None = None,
        **_: object,
    ) -> None:
        self._model = config.model or ""
        self._requests_made = initial_requests_made
        self._record_request = record_request

    def close(self) -> None:
        return None

    def complete(self, schema_name: str, **kwargs: object) -> LLMResponse:
        payload = kwargs["user_payload"]
        assert isinstance(payload, dict)
        type(self).payloads.append(t.cast(dict[str, object], payload))
        type(self).requests += 1
        self._requests_made += 1
        if self._record_request is not None:
            self._record_request(self._requests_made)
        content = (
            _attributes_json()
            if schema_name == "generated_attributes"
            else _descriptions_json()
        )
        return LLMResponse(
            response_id=f"response-{self.requests}",
            model=self._model,
            content=content,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            request_attempts=1,
            latency_seconds=0.1,
            estimated_cost_usd=0.001,
            inference_provider="mock-provider",
            raw_response_sha256="0" * 64,
        )

    @property
    def requests_made(self) -> int:
        return self._requests_made


class _InterruptingClient(_MockClient):
    interrupt_once = True

    def complete(self, schema_name: str, **kwargs: object) -> LLMResponse:
        if schema_name == "persona_descriptions" and self.interrupt_once:
            type(self).interrupt_once = False
            type(self).requests += 1
            self._requests_made += 1
            if self._record_request is not None:
                self._record_request(self._requests_made)
            raise RequestBudgetExceeded("test interruption")
        return super().complete(schema_name=schema_name, **kwargs)


def _attributes_json() -> str:
    return json.dumps(
        {
            "cultural_context": (
                "Personen har en dansk hverdag og deltager i lokale fællesskaber."
            ),
            "skills_and_expertise": ["planlægning", "samarbejde", "formidling"],
            "hobbies_and_interests": [
                "at læse danske romaner",
                "at lytte til musik i fritiden",
                "at spille brætspil med venner",
            ],
            "career_goals_and_ambitions": None,
        },
        ensure_ascii=False,
    )


def _descriptions_json() -> str:
    return json.dumps(
        {
            "professional_persona": (
                "På arbejdet kan personen lide tydelige opgaver og et godt "
                "samarbejde med andre."
            ),
            "sports_persona": (
                "Motion kan være en rolig aktivitet, og personen vælger gerne "
                "fleksible rammer."
            ),
            "arts_persona": (
                "Personen læser gerne og lytter til musik, når der er tid til "
                "fordybelse."
            ),
            "travel_persona": (
                "På rejser kan personen foretrække en enkel plan og tid til nye "
                "oplevelser."
            ),
            "culinary_persona": (
                "I køkkenet er der plads til enkle retter og hyggelige måltider "
                "med andre."
            ),
            "persona": (
                "Personen har en rolig dansk hverdag med plads til læsning, musik "
                "og venner. Nye opgaver mødes med nysgerrighed og samarbejde."
            ),
            "visual_persona": (
                "Personen vælger en blå skjorte og et grønt armbånd. "
                "Baggrunden er en neutral flade."
            ),
        },
        ensure_ascii=False,
    )


class _RejectingClient(_MockClient):
    reject_once = True

    def complete(self, schema_name: str, **kwargs: object) -> LLMResponse:
        response = super().complete(schema_name=schema_name, **kwargs)
        if schema_name == "generated_attributes" and self.reject_once:
            type(self).reject_once = False
            invalid = json.loads(_attributes_json())
            invalid["skills_and_expertise"] = [
                "project management",
                "written communication",
                "data analysis",
            ]
            return response.model_copy(update={"content": json.dumps(invalid)})
        return response


def test_generation_withholds_resolution_provenance_from_both_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both LLM stages receive values without sampler resolution metadata."""
    paths = _write_inputs(root=tmp_path)
    monkeypatch.setattr("danish_personas.generation.pipeline.OpenAIClient", _MockClient)
    _MockClient.requests = 0
    _MockClient.payloads = []

    run_dir = generate_personas(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "outputs",
        rows=1,
        live=True,
    )

    resolution_columns = (
        "age_resolution",
        "marital_resolution",
        "education_resolution",
        "detailed_status_resolution",
        "origin_country_code",
        "origin_country",
        "job_function_code",
        "job_function",
        "job_function_resolution",
        "municipality_code",
        "municipality",
    )
    sample = pl.read_parquet(paths["sample"])
    assert set(resolution_columns) <= set(sample.columns)
    assert len(_MockClient.payloads) == 2
    attributes_payload = _MockClient.payloads[0]["demographics_and_personality"]
    descriptions_payload = _MockClient.payloads[1]["demographics_and_personality"]
    assert isinstance(attributes_payload, dict)
    assert isinstance(descriptions_payload, dict)
    assert set(resolution_columns).isdisjoint(attributes_payload)
    assert set(resolution_columns).isdisjoint(descriptions_payload)
    assert attributes_payload["education_level"] == "masters"
    assert descriptions_payload["education_level"] == "masters"
    assert "generated_attributes" in _MockClient.payloads[1]

    generation_manifest = json.loads(
        (run_dir / "generation-manifest.json").read_text(encoding="utf-8")
    )
    assert generation_manifest["input_sha256"] == sha256_file(paths["sample"])
    output = pl.read_parquet(run_dir / "generated-personas.parquet")
    assert "visual_persona" in output.columns
    assert set(resolution_columns) <= set(output.columns)
    assert output.select(list(resolution_columns)).equals(
        sample.head(1).select(list(resolution_columns))
    )


def _write_inputs(root: Path) -> dict[str, Path]:
    run_dir = root / "upstream"
    run_dir.mkdir()
    source = pl.DataFrame(
        {
            "persona_id": ["persona-1", "persona-2"],
            "country": ["Danmark", "Danmark"],
            "age": [35, 72],
            "age_resolution": ["municipality_age_band_sex", "municipality_age_band"],
            "age_band": ["30-49", "67+"],
            "sex": ["female", "male"],
            "marital_status": ["married_or_separated", "never_married"],
            "marital_resolution": ["municipality_age_band_sex", "municipality"],
            "municipality_code": ["101", "265"],
            "municipality": ["København", "Roskilde"],
            "region_code": ["084", "085"],
            "region": ["Region Hovedstaden", "Region Sjælland"],
            "education_level": ["masters", "vocational"],
            "education_source_code": ["H70", "H40"],
            "education_resolution": ["ras209_age_band", "ras209_67_plus_proxy"],
            "labour_market_status": ["employed", "outside_labour_force"],
            "detailed_status_code": ["30", "90"],
            "detailed_status": ["Employees", "Retired"],
            "detailed_status_resolution": ["status", "sex_status"],
            "job_function_code": ["24", None],
            "job_function": ["24 Business and administration professionals", None],
            "job_function_resolution": ["lons20_sex_marginal", "not_applicable"],
            "openness_score": [50.0, 51.0],
            "openness_label": ["average", "average"],
            "conscientiousness_score": [52.0, 53.0],
            "conscientiousness_label": ["average", "average"],
            "extraversion_score": [54.0, 55.0],
            "extraversion_label": ["average", "high"],
            "agreeableness_score": [56.0, 57.0],
            "agreeableness_label": ["high", "high"],
            "neuroticism_score": [48.0, 47.0],
            "neuroticism_label": ["average", "average"],
            "origin_country_code": ["5100", "5103"],
            "origin_country": ["Denmark", "Stateless"],
        }
    )
    source_path = run_dir / "structured-records.parquet"
    sample_path = run_dir / "text-development-seeds.parquet"
    source.write_parquet(source_path)
    source.write_parquet(sample_path)
    run_manifest = RunManifest(
        run_id="upstream-run",
        sampler_schema_version=SAMPLER_SCHEMA_VERSION,
        created_at="2026-01-01T00:00:00+00:00",
        bundle_id="bundle",
        bundle_manifest_sha256="0" * 64,
        sampling_config_sha256="1" * 64,
        rows=2,
        seed=1,
        data_file=Path(source_path.name),
        data_sha256=sha256_file(source_path),
        logical_content_sha256="2" * 64,
        llm_calls=0,
    )
    write_json(path=run_dir / "run-manifest.json", payload=run_manifest)
    write_json(
        path=run_dir / "validation-report.json",
        payload=ValidationReport(
            kind="demographics",
            passed=True,
            created_at="2026-01-01T00:00:00+00:00",
            subject_id=run_manifest.run_id,
            metrics=[],
        ),
    )
    sample_manifest = FrozenSampleManifest(
        sample_schema_version=2,
        source_run_id=run_manifest.run_id,
        rows=2,
        strata=[],
        method="test",
        data_file=Path(sample_path.name),
        sha256=sha256_file(sample_path),
        llm_calls=0,
    )
    sample_manifest_path = sample_path.with_suffix(".manifest.json")
    write_json(path=sample_manifest_path, payload=sample_manifest)
    attributes_prompt = root / "attributes.md"
    personas_prompt = root / "personas.md"
    attributes_prompt.write_text("Danske attributter")
    personas_prompt.write_text("Danske personaer")
    config_path = root / "generation.yaml"
    config = {
        "version": 1,
        "llm_generation_enabled": True,
        "base_url": "http://test/v1",
        "model": "test-model",
        "api_key_env": None,
        "timeout_seconds": 10.0,
        "maximum_http_attempts": 2,
        "maximum_validation_attempts": 2,
        "maximum_total_requests": 5,
        "retry_backoff_seconds": 0.0,
        "maximum_smoke_rows": 2,
        "max_tokens": None,
        "enable_thinking": None,
        "reasoning_effort": None,
        "response_format": "json_schema",
        "attributes_prompt": str(attributes_prompt),
        "personas_prompt": str(personas_prompt),
    }
    config_path.write_text(yaml.safe_dump(config))
    return {
        "sample": sample_path,
        "sample_manifest": sample_manifest_path,
        "config": config_path,
        "personas_prompt": personas_prompt,
    }


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_checkpoint",
        "malformed_checkpoint",
        "accounting",
        "accepted_content",
        "trailing_response",
        "attempts",
        "ledger_context",
        "llm_flag",
        "run_id",
        "offset",
    ],
)
def test_persona_validation_rejects_independent_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    """Each checkpoint, accounting, and identity tamper fails from a valid run."""
    paths = _write_inputs(root=tmp_path)
    monkeypatch.setattr("danish_personas.generation.pipeline.OpenAIClient", _MockClient)
    _MockClient.requests = 0
    run_dir = generate_personas(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "outputs",
        rows=1,
        live=True,
    )
    assert validate_persona_run(run_dir=run_dir).passed
    manifest_path = run_dir / "generation-manifest.json"
    checkpoint_path = next((run_dir / "checkpoints").glob("*.json"))
    ledger_path = run_dir / "request-ledger.json"

    if tamper == "missing_checkpoint":
        checkpoint_path.unlink()
    elif tamper == "malformed_checkpoint":
        checkpoint_path.write_text("{")
    elif tamper == "accounting":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["prompt_tokens"] += 1
        write_json(path=manifest_path, payload=manifest)
    elif tamper in {"accepted_content", "trailing_response", "attempts"}:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if tamper == "accepted_content":
            accepted = json.loads(checkpoint["responses"][0]["content"])
            accepted["cultural_context"] = (
                "Personen har en anden dansk hverdag og deltager i lokale fællesskaber."
            )
            checkpoint["responses"][0]["content"] = json.dumps(
                accepted, ensure_ascii=False
            )
        elif tamper == "trailing_response":
            checkpoint["responses"].append(checkpoint["responses"][-1])
            checkpoint["attempts"] += 1
        else:
            checkpoint["attempts"] += 1
        write_json(path=checkpoint_path, payload=checkpoint)
    elif tamper == "ledger_context":
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger["generation_context_sha256"] = "f" * 64
        write_json(path=ledger_path, payload=ledger)
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if tamper == "llm_flag":
            manifest["llm_generation"] = False
        elif tamper == "run_id":
            manifest["run_id"] = "forged-run-id"
        else:
            manifest["offset"] = 1
        write_json(path=manifest_path, payload=manifest)

    assert not validate_persona_run(run_dir=run_dir).passed


def test_persona_validation_reports_missing_manifest(tmp_path: Path) -> None:
    """A missing required manifest produces a failed report rather than an error."""
    run_dir = tmp_path / "missing-manifest"
    run_dir.mkdir()

    report = validate_persona_run(run_dir=run_dir)

    assert not report.passed
    assert report.subject_id == run_dir.name


def test_pilot_identity_changes_when_prompt_context_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing an effective prompt creates a new pilot artefact directory."""
    paths = _write_inputs(root=tmp_path)
    monkeypatch.setattr("danish_personas.generation.pipeline.OpenAIClient", _MockClient)
    _MockClient.requests = 0
    arguments = [
        "--input",
        str(paths["sample"]),
        "--sample-manifest",
        str(paths["sample_manifest"]),
        "--config",
        str(paths["config"]),
        "--output-dir",
        str(tmp_path / "pilot"),
        "--rows",
        "1",
        "--batch-size",
        "1",
        "--concurrency",
        "1",
        "--maximum-total-requests",
        "5",
        "--input-price-per-million",
        "0.3",
        "--output-price-per-million",
        "1.2",
        "--live",
    ]
    first = CliRunner().invoke(pilot_main, arguments)
    assert first.exit_code == 0, first.output
    paths["personas_prompt"].write_text("En ændret dansk prompt", encoding="utf-8")
    second = CliRunner().invoke(pilot_main, arguments)
    assert second.exit_code == 0, second.output

    pilot_dirs = list((tmp_path / "pilot").iterdir())
    assert len(pilot_dirs) == 2
    manifests = [
        json.loads((pilot_dir / "pilot-manifest.json").read_text(encoding="utf-8"))
        for pilot_dir in pilot_dirs
    ]
    assert manifests[0]["pilot_id"] != manifests[1]["pilot_id"]
    assert len({manifest["generation_context_sha256"] for manifest in manifests}) == 2


def test_pilot_identity_supports_a_batch_larger_than_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pilot identity retains the requested batch size without a new manifest field."""
    paths = _write_inputs(root=tmp_path)
    monkeypatch.setattr("danish_personas.generation.pipeline.OpenAIClient", _MockClient)
    _MockClient.requests = 0

    pilot_dir = run_pilot(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "pilot",
        rows=1,
        batch_size=2,
        concurrency=1,
        delay_between_batches=0.0,
        maximum_total_requests=5,
        input_price_per_million=0.3,
        output_price_per_million=1.2,
    )

    assert validate_persona_pilot(pilot_dir=pilot_dir).passed


def test_pilot_merges_validated_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pilot runner merges each bounded invocation exactly once."""
    paths = _write_inputs(root=tmp_path)
    monkeypatch.setattr("danish_personas.generation.pipeline.OpenAIClient", _MockClient)
    _MockClient.requests = 0
    result = CliRunner().invoke(
        pilot_main,
        [
            "--input",
            str(paths["sample"]),
            "--sample-manifest",
            str(paths["sample_manifest"]),
            "--config",
            str(paths["config"]),
            "--output-dir",
            str(tmp_path / "pilot"),
            "--rows",
            "2",
            "--batch-size",
            "1",
            "--concurrency",
            "1",
            "--maximum-total-requests",
            "10",
            "--input-price-per-million",
            "0.3",
            "--output-price-per-million",
            "1.2",
            "--live",
        ],
    )
    assert result.exit_code == 0, result.output
    output_path = next((tmp_path / "pilot").glob("*/generated-personas.parquet"))
    output = pl.read_parquet(output_path)
    assert output.get_column("persona_id").to_list() == ["persona-1", "persona-2"]
    assert "visual_persona" in output.columns
    assert _MockClient.requests == 4


def test_pilot_revalidates_shards_without_rewriting_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh shard validation rejects stale reports and preserves their bytes."""
    paths = _write_inputs(root=tmp_path)
    monkeypatch.setattr("danish_personas.generation.pipeline.OpenAIClient", _MockClient)
    _MockClient.requests = 0
    result = CliRunner().invoke(
        pilot_main,
        [
            "--input",
            str(paths["sample"]),
            "--sample-manifest",
            str(paths["sample_manifest"]),
            "--config",
            str(paths["config"]),
            "--output-dir",
            str(tmp_path / "pilot"),
            "--rows",
            "1",
            "--batch-size",
            "1",
            "--concurrency",
            "1",
            "--maximum-total-requests",
            "10",
            "--input-price-per-million",
            "0.3",
            "--output-price-per-million",
            "1.2",
            "--live",
        ],
    )
    assert result.exit_code == 0, result.output
    pilot_dir = next((tmp_path / "pilot").iterdir())
    pilot_manifest_path = pilot_dir / "pilot-manifest.json"
    pilot_manifest = json.loads(pilot_manifest_path.read_text(encoding="utf-8"))
    reference = pilot_manifest["batch_runs"][0]
    report_path = pilot_dir / reference["validation_report_file"]
    report_bytes = report_path.read_bytes()
    report_sha256 = sha256_file(report_path)
    checkpoint_path = next((pilot_dir / "batches").glob("*/checkpoints/*.json"))
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint = json.loads(checkpoint_bytes)
    checkpoint["generation_context_sha256"] = "f" * 64
    checkpoint_path.write_text(json.dumps(checkpoint))

    assert not validate_persona_pilot(pilot_dir=pilot_dir).passed
    assert report_path.read_bytes() == report_bytes
    assert sha256_file(report_path) == report_sha256


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_manifest",
        "malformed_manifest",
        "accounting",
        "llm_flag",
        "pilot_id",
        "rows",
        "batch_reference",
        "config_path",
        "batch_upstream",
        "stored_report",
    ],
)
def test_pilot_validation_rejects_independent_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    """Each pilot identity, aggregate, and artefact tamper starts from validity."""
    paths = _write_inputs(root=tmp_path)
    monkeypatch.setattr("danish_personas.generation.pipeline.OpenAIClient", _MockClient)
    _MockClient.requests = 0
    pilot_dir = run_pilot(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "pilot",
        rows=1,
        batch_size=1,
        concurrency=1,
        delay_between_batches=0.0,
        maximum_total_requests=5,
        input_price_per_million=0.3,
        output_price_per_million=1.2,
    )
    assert validate_persona_pilot(pilot_dir=pilot_dir).passed
    manifest_path = pilot_dir / "pilot-manifest.json"

    if tamper == "missing_manifest":
        manifest_path.unlink()
    elif tamper == "malformed_manifest":
        manifest_path.write_text("{")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if tamper == "accounting":
            manifest["requests"] += 1
        elif tamper == "llm_flag":
            manifest["llm_generation"] = False
        elif tamper == "pilot_id":
            manifest["pilot_id"] = "forged-pilot-id"
        elif tamper == "rows":
            manifest["rows"] += 1
        elif tamper == "batch_reference":
            manifest["batch_runs"][0]["rows"] += 1
        elif tamper == "config_path":
            alternate_config = tmp_path / "same-generation.yaml"
            alternate_config.write_bytes(paths["config"].read_bytes())
            manifest["generation_config_file"] = str(alternate_config)
        elif tamper == "batch_upstream":
            reference = manifest["batch_runs"][0]
            batch_manifest_path = pilot_dir / reference["manifest_file"]
            batch_manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
            batch_manifest["upstream_run_id"] = "different-upstream"
            write_json(path=batch_manifest_path, payload=batch_manifest)
            reference["manifest_sha256"] = sha256_file(batch_manifest_path)
        else:
            reference = manifest["batch_runs"][0]
            report_path = pilot_dir / reference["validation_report_file"]
            stored_report = json.loads(report_path.read_text(encoding="utf-8"))
            stored_report["kind"] = "demographics"
            write_json(path=report_path, payload=stored_report)
            reference["validation_report_sha256"] = sha256_file(report_path)
        write_json(path=manifest_path, payload=manifest)

    assert not validate_persona_pilot(pilot_dir=pilot_dir).passed


def test_pilot_validation_uses_repository_root_from_another_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh pilot validation resolves relative evidence outside process cwd."""
    paths = _write_inputs(root=tmp_path)
    monkeypatch.setattr("danish_personas.generation.pipeline.OpenAIClient", _MockClient)
    _MockClient.requests = 0
    pilot_dir = run_pilot(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "pilot",
        rows=1,
        batch_size=1,
        concurrency=1,
        delay_between_batches=0.0,
        maximum_total_requests=5,
        input_price_per_million=0.3,
        output_price_per_million=1.2,
    )
    manifest_path = pilot_dir / "pilot-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative_input = paths["sample"].relative_to(tmp_path)
    relative_sample_manifest = paths["sample_manifest"].relative_to(tmp_path)
    relative_config = paths["config"].relative_to(tmp_path)
    manifest["input_file"] = str(relative_input)
    manifest["sample_manifest_file"] = str(relative_sample_manifest)
    manifest["generation_config_file"] = str(relative_config)
    for reference in manifest["batch_runs"]:
        shard_path = pilot_dir / reference["manifest_file"]
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        shard["input_file"] = str(relative_input)
        shard["sample_manifest_file"] = str(relative_sample_manifest)
        shard["generation_config_file"] = str(relative_config)
        write_json(path=shard_path, payload=shard)
        reference["manifest_sha256"] = sha256_file(shard_path)
    write_json(path=manifest_path, payload=manifest)
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    assert validate_persona_pilot(pilot_dir=pilot_dir, repository_root=tmp_path).passed


def test_pipeline_rejects_tampering_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validated rows generate once; forged rows and stale checkpoints fail."""
    paths = _write_inputs(root=tmp_path)
    monkeypatch.setattr("danish_personas.generation.pipeline.OpenAIClient", _MockClient)
    _MockClient.requests = 0
    run_dir = generate_personas(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "outputs",
        rows=1,
        live=True,
    )
    assert _MockClient.requests == 2
    assert (run_dir / "generated-personas.parquet").exists()
    assert validate_persona_run(run_dir=run_dir).passed

    generate_personas(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "outputs",
        rows=1,
        live=True,
    )
    assert _MockClient.requests == 2

    checkpoint_path = next((run_dir / "checkpoints").glob("*.json"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["generation_context_sha256"] = "f" * 64
    checkpoint_path.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match="Stale generation context"):
        generate_personas(
            input_path=paths["sample"],
            sample_manifest_path=paths["sample_manifest"],
            config_path=paths["config"],
            output_dir=tmp_path / "outputs",
            rows=1,
            live=True,
        )

    sample = pl.read_parquet(paths["sample"]).with_columns(pl.lit(99).alias("age"))
    sample.write_parquet(paths["sample"])
    manifest = FrozenSampleManifest.model_validate_json(
        paths["sample_manifest"].read_text(encoding="utf-8")
    ).model_copy(update={"sha256": sha256_file(paths["sample"])})
    write_json(path=paths["sample_manifest"], payload=manifest)
    with pytest.raises(ValueError, match="absent from validated"):
        generate_personas(
            input_path=paths["sample"],
            sample_manifest_path=paths["sample_manifest"],
            config_path=paths["config"],
            output_dir=tmp_path / "outputs",
            rows=1,
            live=True,
        )


def test_pipeline_selects_an_offset_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bounded invocation can select a later frozen-sample shard."""
    paths = _write_inputs(root=tmp_path)
    monkeypatch.setattr("danish_personas.generation.pipeline.OpenAIClient", _MockClient)
    _MockClient.requests = 0
    run_dir = generate_personas(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "outputs",
        rows=1,
        live=True,
        offset=1,
    )
    output = pl.read_parquet(run_dir / "generated-personas.parquet")
    assert output.get_column("persona_id").to_list() == ["persona-2"]
    assert validate_persona_run(run_dir=run_dir).passed


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


def test_rejected_completion_text_is_not_checkpointed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rejected text is removed while its usage and response hash remain auditable."""
    paths = _write_inputs(root=tmp_path)
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", _RejectingClient
    )
    _RejectingClient.requests = 0
    _RejectingClient.reject_once = True
    run_dir = generate_personas(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "outputs",
        rows=1,
        live=True,
    )
    checkpoint_path = next((run_dir / "checkpoints").glob("*.json"))
    responses = json.loads(checkpoint_path.read_text(encoding="utf-8"))["responses"]
    assert responses[0]["content"] == ""
    assert responses[0]["raw_response_sha256"] == "0" * 64
    assert len(responses) == 3


def test_stage_checkpoint_avoids_repeating_attributes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A descriptions-stage interruption preserves the billable attributes stage."""
    paths = _write_inputs(root=tmp_path)
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", _InterruptingClient
    )
    _InterruptingClient.requests = 0
    _InterruptingClient.interrupt_once = True
    with pytest.raises(RequestBudgetExceeded):
        generate_personas(
            input_path=paths["sample"],
            sample_manifest_path=paths["sample_manifest"],
            config_path=paths["config"],
            output_dir=tmp_path / "outputs",
            rows=1,
            live=True,
        )
    assert _InterruptingClient.requests == 2
    assert (
        len(list((tmp_path / "outputs").glob("*/checkpoints/*.attributes.json"))) == 1
    )

    run_dir = generate_personas(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "outputs",
        rows=1,
        live=True,
    )
    assert _InterruptingClient.requests == 3
    manifest = json.loads(
        (run_dir / "generation-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["requests"] == 3
    assert manifest["retries"] == 1
    assert manifest["estimated_cost_usd"] == 0.002
    assert manifest["inference_providers"] == ["mock-provider"]


def test_upstream_sample_rejects_legacy_sampler_schema(tmp_path: Path) -> None:
    """A validated legacy run cannot cross the current Phase-3 boundary."""
    paths = _write_inputs(root=tmp_path)
    manifest_path = paths["sample"].parent / "run-manifest.json"
    manifest = RunManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    write_json(
        path=manifest_path,
        payload=manifest.model_copy(update={"sampler_schema_version": 2}),
    )

    with pytest.raises(ValueError, match="unsupported sampler schema version"):
        validate_upstream_sample(
            input_path=paths["sample"], sample_manifest_path=paths["sample_manifest"]
        )


def test_upstream_sample_rejects_origin_less_legacy_columns(tmp_path: Path) -> None:
    """Origin-less schema-v2-shaped rows require migration before Phase 3."""
    paths = _write_inputs(root=tmp_path)
    sample_path = paths["sample"]
    pl.read_parquet(sample_path).drop(
        "origin_country_code", "origin_country"
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
