"""Reusable fixtures for generation integration tests."""

# These public-looking objects are test doubles, not production API.
# ruff: noqa: D101, D102, D103, D107

import collections.abc as c
import json
import typing as t
from pathlib import Path

import polars as pl
import yaml
from manifest_helpers import origin_contract_fields

from danish_personas.generation.client import RequestBudgetExceeded
from danish_personas.generation.models import (
    FrozenSampleManifest,
    GenerationConfig,
    LLMResponse,
)
from danish_personas.generation.personality import allowed_personality_tendencies
from danish_personas.generation.validation import EDUCATION_DANISH
from danish_personas.io import sha256_file, write_json
from danish_personas.models import (
    FROZEN_SAMPLE_SCHEMA_VERSION,
    SAMPLER_SCHEMA_VERSION,
    RunManifest,
    ValidationReport,
)


class MockGenerationClient:
    requests = 0
    payloads: list[dict[str, object]] = []
    job_title = "forretningsspecialist"

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
        demographics = t.cast(
            dict[str, object], payload["demographics_and_personality"]
        )
        employed = demographics["job_function"] is not None
        assert schema_name == "generated_persona"
        content = generated_persona_json(
            employed=employed,
            demographic=demographics,
            job_title=type(self).job_title if employed else None,
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


class InterruptingGenerationClient(MockGenerationClient):
    interrupt_once = True

    def complete(self, schema_name: str, **kwargs: object) -> LLMResponse:
        if schema_name == "generated_persona" and self.interrupt_once:
            type(self).interrupt_once = False
            type(self).requests += 1
            self._requests_made += 1
            if self._record_request is not None:
                self._record_request(self._requests_made)
            raise RequestBudgetExceeded("test interruption")
        return super().complete(schema_name=schema_name, **kwargs)


def generated_persona_json(
    *,
    employed: bool = True,
    demographic: dict[str, object] | None = None,
    job_title: str | None = None,
) -> str:
    """Return one valid combined generation response."""
    payload = json.loads(attributes_json(employed=employed, job_title=job_title))
    partnered = (
        demographic is None
        or demographic.get("marital_status") == "married_or_separated"
    )
    payload.update(
        {
            "current_relationship_status": "partnered"
            if partnered
            else "not_partnered",
            "partner_gender": "male" if partnered else None,
            "legal_status_detail": "married" if partnered else None,
        }
    )
    payload.update(
        json.loads(
            descriptions_json(
                employed=employed, demographic=demographic, job_title=job_title
            )
        )
    )
    return json.dumps(payload, ensure_ascii=False)


def attributes_json(*, employed: bool = True, job_title: str | None = None) -> str:
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
            "job_title": job_title if employed else None,
            "current_relationship_status": "partnered",
            "partner_gender": "male",
            "legal_status_detail": "married",
        },
        ensure_ascii=False,
    )


def descriptions_json(
    *,
    employed: bool = True,
    demographic: dict[str, object] | None = None,
    job_title: str | None = None,
) -> str:
    context = demographic or {
        "age": 35,
        "sex": "female",
        "municipality": "København",
        "origin_country_da": "Danmark",
        "education_level": "higher_education",
    }
    pronoun = "hun" if context["sex"] == "female" else "han"
    education_level = str(context["education_level"])
    education_labels = {
        "grundskole": "har gået i grundskolen",
        "ungdomsuddannelse eller erhvervsuddannelse": "har en erhvervsuddannelse",
        "videregående uddannelse": "har en videregående uddannelse",
        "uddannelse ikke oplyst": "har en uddannelse, der ikke er oplyst",
    }
    rendered_education = EDUCATION_DANISH.get(education_level, education_level)
    education = education_labels[rendered_education]
    tendency = allowed_personality_tendencies(context=context)[0]
    if employed:
        work = (
            f"{pronoun.capitalize()} arbejder som {job_title} i en mindre "
            "rådgivningsvirksomhed i centrum"
        )
    else:
        status = (
            "pensionist"
            if context.get("labour_market_status") == "retired"
            else "uden for arbejdsmarkedet"
        )
        work = f"{pronoun.capitalize()} er {status}"
    married = context.get("marital_status") == "married_or_separated"
    legal = (
        f"{pronoun.capitalize()} er gift. "
        if married
        else f"{pronoun.capitalize()} har aldrig været gift. "
    )
    relationship = (
        f"{pronoun.capitalize()} bor sammen med sin kæreste og deres barn i en "
        "rolig del af kommunen."
        if married
        else f"{pronoun.capitalize()} er single og holder af rolige aftaler med "
        "venner i en rolig del af kommunen."
    )
    persona = (
        f"{pronoun.capitalize()} er {context['age']} år, og {pronoun} bor i "
        f"{context['municipality']} og kommer fra {context['origin_country_da']}. "
        f"{pronoun.capitalize()} {education} med en praktisk retning i Aarhus. "
        f"{work}. I fritiden holder {pronoun} af at læse danske romaner og at "
        f"lytte til musik i fritiden, og planlægning hjælper med at få tid til "
        f"begge dele. {legal}{relationship} {pronoun.capitalize()} "
        f"{tendency} og drømmer om at skabe mere plads til lokale fællesskaber."
    )
    return json.dumps({"persona": persona}, ensure_ascii=False)


class RejectingGenerationClient(MockGenerationClient):
    reject_once = True

    def complete(self, schema_name: str, **kwargs: object) -> LLMResponse:
        response = super().complete(schema_name=schema_name, **kwargs)
        if schema_name == "generated_persona" and self.reject_once:
            type(self).reject_once = False
            invalid = json.loads(attributes_json())
            invalid["skills_and_expertise"] = [
                "project management",
                "written communication",
                "data analysis",
            ]
            return response.model_copy(update={"content": json.dumps(invalid)})
        return response


def write_generation_inputs(root: Path) -> dict[str, Path]:
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
            "education_level": ["higher_education", "secondary_or_vocational"],
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
            "origin_country_code": ["5100", "5456"],
            "origin_country": ["Denmark", "Lebanon"],
            "origin_country_da": ["Danmark", "Libanon"],
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
        **origin_contract_fields(),
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
            **origin_contract_fields(),
        ),
    )
    sample_manifest = FrozenSampleManifest(
        sample_schema_version=FROZEN_SAMPLE_SCHEMA_VERSION,
        source_run_id=run_manifest.run_id,
        rows=2,
        strata=[
            "origin_country_code",
            "municipality_code",
            "education_level",
            "labour_market_status",
        ],
        method=(
            "deterministic largest-remainder quotas for the "
            "origin_country_code marginal, then population-proportional "
            "allocation within each origin"
        ),
        data_file=Path(sample_path.name),
        sha256=sha256_file(sample_path),
        llm_calls=0,
        **origin_contract_fields(),
    )
    sample_manifest_path = sample_path.with_suffix(".manifest.json")
    write_json(path=sample_manifest_path, payload=sample_manifest)
    prompt = root / "prompt.md"
    prompt.write_text("Danske personaer")
    contract_path = root / "config" / "folk2-ieland-labels-da.yaml"
    contract_path.parent.mkdir()
    contract_path.write_bytes(
        (Path("config") / "folk2-ieland-labels-da.yaml").read_bytes()
    )
    mapping_path = root / "config" / "job-function-titles.yaml"
    mapping_path.write_bytes((Path("config") / "job-function-titles.yaml").read_bytes())
    config_path = root / "config/config.yaml"
    config = {
        "base_url": "http://test/v1",
        "model": "test-model",
        "api_key_env": None,
        "timeout_seconds": 10.0,
        "maximum_http_attempts": 2,
        "maximum_validation_attempts": 2,
        "maximum_total_requests": 5,
        "retry_backoff_seconds": 0.0,
        "maximum_rows_per_shard": 5,
        "max_tokens": None,
        "enable_thinking": None,
        "reasoning_effort": None,
        "response_format": "json_schema",
        "prompt": str(prompt),
        "origin_label_contract": "config/folk2-ieland-labels-da.yaml",
    }
    config_path.write_text(yaml.safe_dump(config))
    return {
        "sample": sample_path,
        "sample_manifest": sample_manifest_path,
        "config": config_path,
        "prompt": prompt,
    }
