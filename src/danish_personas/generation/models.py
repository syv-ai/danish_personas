"""Contracts for LLM-generated persona attributes and text."""

import typing as t
from pathlib import Path

from pydantic import (
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from ..models import (
    GENERATION_SCHEMA_VERSION,
    FrozenSampleManifest,
    StrictModel,
    ValidationReport,
)
from ..origin_labels import (
    OriginLabelContract,
    OriginLabelContractPath,
    canonical_origin_label_contract_path,
    validate_origin_contract_reference,
)
from .job_titles import JobFunctionTitleMapping
from .policy import ChecksumValidationPolicy


def _checksum_policy_from_context(info: ValidationInfo) -> ChecksumValidationPolicy:
    """Read the optional scoped checksum policy used by model validation.

    Returns:
        The configured checksum policy, or strict validation by default.
    """
    if isinstance(info.context, dict):
        policy = info.context.get("checksum_policy")
        if isinstance(policy, ChecksumValidationPolicy):
            return policy
    return ChecksumValidationPolicy.STRICT


__all__ = ["FrozenSampleManifest"]


class GeneratedAttributes(StrictModel):
    """Generated structured persona attributes."""

    cultural_context: str = Field(min_length=20, max_length=600)
    skills_and_expertise: list[str] = Field(min_length=3, max_length=6)
    hobbies_and_interests: list[str] = Field(min_length=3, max_length=6)
    career_goals_and_ambitions: str | None = Field(max_length=500)
    job_title: str | None = Field(max_length=80)
    current_relationship_status: t.Literal["partnered", "not_partnered"]
    partner_gender: t.Literal["male", "female"] | None
    legal_status_detail: t.Literal["married", "separated"] | None

    @field_validator("job_title")
    @classmethod
    def require_stripped_single_line_title(_cls, value: str | None) -> str | None:
        """Normalise an optional title while rejecting multiline output.

        Args:
            value:
                Candidate generated title.

        Returns:
            The validated title or null.

        Raises:
            ValueError:
                If the title is not a short, stripped, single-line string.
        """
        if value is None:
            return None
        stripped = value.strip()
        if stripped != value or "\n" in value or "\r" in value:
            raise ValueError("job_title must be a stripped single-line string")
        if not 2 <= len(stripped) <= 80:
            raise ValueError("job_title must contain 2-80 characters")
        return stripped

    @field_validator("skills_and_expertise", "hobbies_and_interests")
    @classmethod
    def require_unique_items(_cls, values: list[str]) -> list[str]:
        """Require non-empty, case-insensitively unique list entries.

        Args:
            values:
                Generated list entries.

        Returns:
            Stripped list entries.

        Raises:
            ValueError:
                If an entry is empty or duplicated.
        """
        stripped = [value.strip() for value in values]
        if any(not value for value in stripped):
            message = "Generated list entries cannot be empty"
            raise ValueError(message)
        if len({value.casefold() for value in stripped}) != len(stripped):
            message = "Generated list entries must be unique"
            raise ValueError(message)
        return stripped

    @model_validator(mode="after")
    def validate_relationship_fields(self) -> "GeneratedAttributes":
        """Require partner fields to agree with the current relationship status.

        Returns:
            The validated attributes.

        Raises:
            ValueError:
                If partner gender disagrees with the relationship status.
        """
        has_gender = self.partner_gender is not None
        if self.current_relationship_status == "partnered" and not has_gender:
            raise ValueError("partnered responses require partner_gender")
        if self.current_relationship_status == "not_partnered" and has_gender:
            raise ValueError("not_partnered responses must not include partner_gender")
        return self


class GeneratedPersona(GeneratedAttributes):
    """Single-response attributes and Danish persona text."""

    persona: str = Field(min_length=300, max_length=900)


class GenerationConfig(StrictModel):
    """Guarded OpenAI-compatible generation configuration."""

    base_url: str | None
    model: str | None
    api_key_env: str | None
    timeout_seconds: float = Field(gt=0.0)
    maximum_http_attempts: int = Field(ge=1, le=5)
    maximum_validation_attempts: int = Field(ge=1, le=3)
    maximum_total_requests: int | None = Field(ge=1, le=15)
    retry_backoff_seconds: float = Field(ge=0.0)
    maximum_rows_per_shard: int = Field(ge=1, le=5)
    same_sex_partner_probability: float = Field(default=0.00701, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=32, le=4_096)
    enable_thinking: bool | None = None
    reasoning_effort: t.Literal["none", "low", "medium", "high"] | None = None
    response_format: t.Literal["json_schema", "json_object"]
    prompt: Path
    job_title_mapping: Path | None = None
    origin_label_contract: OriginLabelContractPath

    @field_validator("origin_label_contract")
    @classmethod
    def require_repository_relative_origin_contract(_cls, value: Path) -> Path:
        """Require the origin-label contract to be repository-relative.

        Args:
            value:
                Configured origin-label contract path.

        Returns:
            The validated relative path.

        """
        canonical_origin_label_contract_path(value)
        return value

    @field_serializer("prompt", "job_title_mapping", when_used="json")
    def serialise_repository_path(self, value: Path | None) -> str | None:
        """Serialise repository paths with portable separators.

        Args:
            value:
                Runtime filesystem path to serialise.

        Returns:
            A repository path using forward slashes, or ``None``.
        """
        if value is None:
            return None
        return value.as_posix().replace("\\", "/")


class GenerationManifest(StrictModel):
    """Manifest for a completed persona generation run."""

    run_id: str
    upstream_run_id: str
    input_file: Path
    sample_manifest_file: Path
    input_sha256: str
    ordered_persona_ids_sha256: str
    generation_config_file: Path | None = None
    generation_config_sha256: str
    generation_context_sha256: str
    validator_version: str
    job_title_mapping_file: Path | None = None
    job_title_mapping_sha256: str | None = None
    job_title_mapping_version: int | None = None
    job_title_mapping_content: JobFunctionTitleMapping | None = None
    origin_label_contract_file: OriginLabelContractPath
    origin_label_contract_sha256: str
    origin_label_contract_version: int
    origin_label_contract_content: OriginLabelContract
    prompt_sha256: str
    model: str
    base_url: str
    rows: int = Field(ge=1, le=5)
    offset: int = Field(default=0, ge=0)
    requests: int = Field(ge=0)
    retries: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)
    inference_providers: list[str]
    output_file: Path
    output_sha256: str
    llm_generation: bool
    generation_schema_version: int = GENERATION_SCHEMA_VERSION

    @model_validator(mode="after")
    def validate_origin_contract_binding(
        self, info: ValidationInfo
    ) -> "GenerationManifest":
        """Require the exact compiled origin contract binding.

        Returns:
            The validated model.

        Raises:
            ValueError:
                If the generation schema version or origin binding is stale.
        """
        if self.generation_schema_version != GENERATION_SCHEMA_VERSION:
            raise ValueError("Unsupported generation schema version")
        validate_origin_contract_reference(
            path=self.origin_label_contract_file,
            version=self.origin_label_contract_version,
            sha256=self.origin_label_contract_sha256,
            content=self.origin_label_contract_content,
            checksum_policy=_checksum_policy_from_context(info),
        )
        return self


class GenerationValidationReport(ValidationReport):
    """Generation report bound to the effective Danish origin-label contract."""

    origin_label_contract_file: OriginLabelContractPath
    origin_label_contract_sha256: str
    origin_label_contract_version: int
    origin_label_contract_content: OriginLabelContract

    @model_validator(mode="after")
    def validate_origin_contract_binding(
        self, info: ValidationInfo
    ) -> "GenerationValidationReport":
        """Require the exact compiled origin contract binding.

        Returns:
            The validated model.
        """
        validate_origin_contract_reference(
            path=self.origin_label_contract_file,
            version=self.origin_label_contract_version,
            sha256=self.origin_label_contract_sha256,
            content=self.origin_label_contract_content,
            checksum_policy=_checksum_policy_from_context(info),
        )
        return self


class LLMResponse(StrictModel):
    """Parsed completion plus auditable response metadata."""

    response_id: str = Field(min_length=1, pattern=r"\S")
    model: str = Field(min_length=1, pattern=r"\S")
    content: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    request_attempts: int = Field(default=1, ge=1)
    latency_seconds: float = Field(ge=0.0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)
    inference_provider: str | None = Field(default=None, min_length=1, pattern=r"\S")
    raw_response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class PersonaDescriptions(StrictModel):
    """Generated Danish persona text."""

    persona: str = Field(min_length=300, max_length=900)


class PersonaCheckpoint(StrictModel):
    """Completed single-response generation checkpoint for one persona."""

    persona_id: str
    input_sha256: str
    generation_context_sha256: str
    validator_version: str
    job_title_mapping_sha256: str | None = None
    job_title_mapping_version: int | None = None
    job_title_mapping_file: Path | None = None
    job_title_mapping_content: JobFunctionTitleMapping | None = None
    origin_label_contract_file: OriginLabelContractPath
    origin_label_contract_sha256: str
    origin_label_contract_version: int
    origin_label_contract_content: OriginLabelContract
    attributes: GeneratedAttributes
    descriptions: PersonaDescriptions
    responses: list[LLMResponse]
    attempts: int = Field(ge=1)
    http_requests: int = Field(default=0, ge=0)
    generation_schema_version: int = GENERATION_SCHEMA_VERSION

    @model_validator(mode="after")
    def validate_origin_contract_binding(
        self, info: ValidationInfo
    ) -> "PersonaCheckpoint":
        """Require the exact compiled origin contract binding.

        Returns:
            The validated model.

        Raises:
            ValueError:
                If the generation schema version or origin binding is stale.
        """
        if self.generation_schema_version != GENERATION_SCHEMA_VERSION:
            raise ValueError("Unsupported generation schema version")
        validate_origin_contract_reference(
            path=self.origin_label_contract_file,
            version=self.origin_label_contract_version,
            sha256=self.origin_label_contract_sha256,
            content=self.origin_label_contract_content,
            checksum_policy=_checksum_policy_from_context(info),
        )
        return self


class PilotBatchReference(StrictModel):
    """Checksummed reference to one validated pilot shard."""

    offset: int = Field(ge=0)
    rows: int = Field(ge=1, le=5)
    run_id: str
    manifest_file: Path
    manifest_sha256: str
    validation_report_file: Path
    validation_report_sha256: str
    job_title_mapping_file: Path | None = None
    job_title_mapping_sha256: str | None = None
    job_title_mapping_version: int | None = None
    job_title_mapping_content: JobFunctionTitleMapping | None = None
    origin_label_contract_file: OriginLabelContractPath
    origin_label_contract_sha256: str
    origin_label_contract_version: int
    origin_label_contract_content: OriginLabelContract


class PilotManifest(StrictModel):
    """Provenance and accounting for a merged persona pilot."""

    pilot_id: str
    created_at: str
    model: str
    base_url: str
    upstream_run_id: str
    input_file: Path
    input_sha256: str
    sample_manifest_file: Path
    sample_manifest_sha256: str
    generation_config_file: Path
    generation_config_sha256: str
    generation_context_sha256: str
    validator_version: str
    job_title_mapping_file: Path | None = None
    job_title_mapping_sha256: str | None = None
    job_title_mapping_version: int | None = None
    job_title_mapping_content: JobFunctionTitleMapping | None = None
    origin_label_contract_file: OriginLabelContractPath
    origin_label_contract_sha256: str
    origin_label_contract_version: int
    origin_label_contract_content: OriginLabelContract
    prompt_sha256: str
    rows: int = Field(ge=1)
    batch_size: int = Field(ge=1, le=5)
    batches: int = Field(ge=1)
    batch_runs: list[PilotBatchReference]
    maximum_total_requests: int | None = Field(ge=1)
    maximum_shard_requests: int | None = Field(ge=1)
    requests: int = Field(ge=0)
    retries: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    input_price_per_million_usd: float = Field(ge=0.0)
    output_price_per_million_usd: float = Field(ge=0.0)
    list_price_estimated_cost_usd: float = Field(ge=0.0)
    provider_estimated_cost_usd: float | None = Field(default=None, ge=0.0)
    inference_providers: list[str]
    output_file: Path
    output_sha256: str
    llm_generation: bool


class RequestLedger(StrictModel):
    """Durable HTTP-attempt budget for one generation run."""

    generation_context_sha256: str
    attempts: int = Field(ge=0)
    maximum_attempts: int | None = Field(ge=1)
