"""Strict typed contracts for release policy and human review evidence."""

import math
import re
import typing as t
from datetime import datetime
from pathlib import Path

from pydantic import (
    AliasChoices,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from ..models import StrictModel


class ReleasePolicyError(ValueError):
    """Raised when a release policy cannot authorise an eligibility check."""


_HEX_16 = re.compile(r"\A[0-9a-fA-F]{16}\Z")
_HEX_64 = re.compile(r"\A[0-9a-fA-F]{64}\Z")
_LICENCE_IDENTIFIER = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9.-]*(?:\+[A-Za-z0-9.-]+)?\Z")
_SAFE_IDENTIFIER = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._:@+\-]*\Z")
_CANONICAL_TIMESTAMP = re.compile(
    r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})\Z"
)
PILOT_ROWS = 10_000
RELEASE_MINIMUM_ROWS = 100_000


class ReleaseApproval(StrictModel):
    """Immutable in-memory result of a successful release eligibility check.

    The result records organisational review evidence; it does not authenticate the
    reviewer cryptographically and does not write or represent a release file.
    """

    pilot_id: StrictStr
    output_sha256: StrictStr
    population_rows: StrictInt = Field(gt=0)
    model: StrictStr
    reviewer_id: StrictStr
    reviewed_persona_ids: tuple[StrictStr, ...]
    required_review_count: StrictInt = Field(gt=0)

    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("model")
    @classmethod
    def _validate_model(_cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("Model identifier must be nonblank and unpadded")
        return value

    @field_validator("output_sha256")
    @classmethod
    def _validate_output_sha256(_cls, value: str) -> str:
        if not _HEX_64.fullmatch(value):
            raise ValueError("Output SHA-256 must contain exactly 64 hex characters")
        return value

    @field_validator("pilot_id")
    @classmethod
    def _validate_pilot_id(_cls, value: str) -> str:
        if not _HEX_16.fullmatch(value):
            raise ValueError("Pilot ID must contain exactly 16 hex characters")
        return value

    @field_validator("reviewed_persona_ids")
    @classmethod
    def _validate_reviewed_persona_ids(
        _cls, values: tuple[str, ...]
    ) -> tuple[str, ...]:
        _require_exact_unique_ids(values=values)
        return values

    @field_validator("reviewer_id")
    @classmethod
    def _validate_reviewer_id(_cls, value: str) -> str:
        if not _SAFE_IDENTIFIER.fullmatch(value):
            raise ValueError("Reviewer ID must be a nonblank safe identifier")
        return value

    @property
    def reviewed_count(self) -> int:
        """Number of unique IDs covered by the attestation."""
        return len(self.reviewed_persona_ids)

    @property
    def reviewed_ids_count(self) -> int:
        """Number of unique IDs covered by the attestation."""
        return self.reviewed_count


class ReleasePolicy(StrictModel):
    """Version-one policy describing whether release eligibility may be checked.

    A disabled policy is useful for the committed default configuration and does not
    need to name a model or a licence. The row boundaries are deliberately constants
    in :mod:`danish_personas.release.policy`; the two minima may only be tightened.
    """

    version: t.Literal[1]
    enabled: StrictBool = Field(
        validation_alias=AliasChoices("enabled", "release_enabled")
    )
    approved_models: tuple[StrictStr, ...] = Field(
        default_factory=tuple,
        validation_alias=AliasChoices("approved_models", "approved_model_identifiers"),
    )
    dataset_licence: StrictStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "dataset_licence",
            "dataset_license",
            "dataset_licence_spdx",
            "dataset_license_spdx",
        ),
    )
    licence_file_sha256: StrictStr | None = Field(
        default=None,
        validation_alias=AliasChoices("licence_file_sha256", "license_file_sha256"),
    )
    pilot_minimum_reviewed_ids: StrictInt = Field(
        default=300,
        ge=300,
        validation_alias=AliasChoices(
            "pilot_minimum_reviewed_ids", "minimum_reviewed_ids_pilot"
        ),
    )
    release_minimum_reviewed_ids: StrictInt = Field(
        default=500,
        ge=500,
        validation_alias=AliasChoices(
            "release_minimum_reviewed_ids", "minimum_reviewed_ids_release"
        ),
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid", frozen=True)

    @field_validator("version", mode="before")
    @classmethod
    def _require_strict_version(_cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Policy version must be an integer")
        return value

    @field_validator("approved_models")
    @classmethod
    def _validate_approved_models(_cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value or value != value.strip() for value in values):
            raise ValueError("Approved model identifiers must be nonblank and unpadded")
        if len(set(values)) != len(values):
            raise ValueError("Approved model identifiers must be unique")
        return values

    @field_validator("dataset_licence")
    @classmethod
    def _validate_dataset_licence(_cls, value: str | None) -> str | None:
        if value is not None and not _LICENCE_IDENTIFIER.fullmatch(value):
            raise ValueError("Dataset licence must be a strict SPDX-like identifier")
        return value

    @field_validator("licence_file_sha256")
    @classmethod
    def _validate_licence_file_sha256(_cls, value: str | None) -> str | None:
        if value is not None and not _HEX_64.fullmatch(value):
            raise ValueError(
                "Licence-file SHA-256 must contain exactly 64 hex characters"
            )
        return value

    @model_validator(mode="after")
    def require_enabled_release_metadata(self) -> "ReleasePolicy":
        """Require complete publication metadata for an enabled policy.

        Returns:
            The validated policy.

        Raises:
            ValueError:
                If an enabled policy lacks model or licence metadata.
        """
        if self.enabled:
            if not self.approved_models:
                raise ValueError(
                    "An enabled release policy must approve at least one model"
                )
            if self.dataset_licence is None:
                raise ValueError(
                    "An enabled release policy must specify a dataset licence"
                )
            if self.licence_file_sha256 is None:
                raise ValueError(
                    "An enabled release policy must specify a licence-file SHA-256"
                )
        return self

    def required_review_count(self, population_rows: int) -> int:
        """Return this policy's required review count for a population size.

        Args:
            population_rows:
                Number of rows in the proposed output.

        Returns:
            The minimum number of unique reviewed IDs.

        Raises:
            ReleasePolicyError:
                If this policy is disabled or the population size is unsupported.
        """
        if type(population_rows) is not int:
            raise ReleasePolicyError("Population row count must be an integer")
        if not self.enabled:
            raise ReleasePolicyError("Release policy is disabled")
        if population_rows == PILOT_ROWS:
            return self.pilot_minimum_reviewed_ids
        if population_rows >= RELEASE_MINIMUM_ROWS:
            return self.release_minimum_reviewed_ids
        raise ReleasePolicyError(
            "Only exactly 10,000 rows or at least 100,000 rows are eligible for release"
        )


ReleaseApprovalResult = ReleaseApproval


class Accounting(StrictModel):
    """Aggregate generation accounting with no provider-identifying metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requests: StrictInt = Field(ge=0)
    retries: StrictInt = Field(ge=0)
    rejected_validation_responses: StrictInt = Field(ge=0)
    dropped_rows: StrictInt = Field(ge=0)
    prompt_tokens: StrictInt = Field(ge=0)
    completion_tokens: StrictInt = Field(ge=0)
    total_tokens: StrictInt = Field(ge=0)
    input_price_per_million_usd: StrictFloat = Field(ge=0.0)
    output_price_per_million_usd: StrictFloat = Field(ge=0.0)
    list_price_estimated_cost_usd: StrictFloat = Field(ge=0.0)
    provider_estimated_cost_usd: StrictFloat | None = Field(default=None, ge=0.0)

    @field_validator(
        "input_price_per_million_usd",
        "output_price_per_million_usd",
        "list_price_estimated_cost_usd",
        "provider_estimated_cost_usd",
    )
    @classmethod
    def _finite_cost(_cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("Costs must be finite")
        return value

    providers: tuple[StrictStr, ...]

    @field_validator("providers")
    @classmethod
    def _unique_providers(_cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(set(value))) != value:
            raise ValueError("Provider names must be sorted and unique")
        return value


class Artifact(StrictModel):
    """One public release file covered by the release manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: StrictStr = Field(min_length=1)
    role: StrictStr = Field(min_length=1)
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    size: StrictInt = Field(ge=0)


class ReleaseManifest(StrictModel):
    """Signed-by-hash description of every file in a public release."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: t.Literal[1]
    release_id: StrictStr = Field(pattern=r"^[0-9a-f]{32}$")
    created_at: datetime
    pilot_id: StrictStr = Field(min_length=1)
    model: StrictStr = Field(min_length=1)
    rows: StrictInt = Field(gt=0)
    git_head: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    origin_url: StrictStr = Field(min_length=1)
    uv_lock_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: tuple[Artifact, ...] = Field(
        min_length=1, validation_alias=AliasChoices("artifacts", "files")
    )

    @model_validator(mode="after")
    def _require_aware_created_at(self) -> "ReleaseManifest":
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("Manifest timestamp must be timezone-aware")
        return self

    @field_validator("created_at", mode="before")
    @classmethod
    def _strict_created_at(_cls, value: object) -> object:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and _CANONICAL_TIMESTAMP.fullmatch(value):
            return value
        raise ValueError("Manifest timestamp must be timezone-aware ISO-8601")

    @property
    def files(self) -> tuple[Artifact, ...]:
        """Manifest-covered files using the common public terminology."""
        return self.artifacts


Manifest = ReleaseManifest


class ReleasePackageResult(StrictModel):
    """Result returned after an atomic release installation."""

    path: Path
    manifest_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path", mode="before")
    @classmethod
    def _strict_path(_cls, value: object) -> object:
        if not isinstance(value, Path):
            raise ValueError("Package path must be a pathlib.Path")
        return value

    model_config = ConfigDict(extra="forbid", frozen=True)


class ReviewAttestation(StrictModel):
    """Blinded human-review attestation bound to one generated output.

    This is organisational evidence of a review decision, not cryptographic proof of
    reviewer identity or authenticity.
    """

    version: t.Literal[1]
    release_approved: t.Literal[True]
    blinded: t.Literal[True]
    reviewer_id: StrictStr = Field(
        validation_alias=AliasChoices("reviewer_id", "reviewer_identifier")
    )
    protocol: StrictStr = Field(
        validation_alias=AliasChoices("protocol", "review_protocol")
    )
    protocol_version: StrictStr = Field(
        validation_alias=AliasChoices("protocol_version", "review_protocol_version")
    )
    reviewed_at: datetime
    pilot_id: StrictStr
    output_sha256: StrictStr
    population_rows: StrictInt = Field(gt=0)
    reviewed_persona_ids: tuple[StrictStr, ...] = Field(
        min_length=1,
        validation_alias=AliasChoices("reviewed_persona_ids", "reviewed_ids"),
    )

    model_config = ConfigDict(populate_by_name=True, extra="forbid", frozen=True)

    @field_validator("release_approved", "blinded", mode="before")
    @classmethod
    def _require_literal_true(_cls, value: object) -> object:
        if value is not True:
            raise ValueError("Approval and blindedness must be literal true values")
        return value

    @field_validator("reviewed_at", mode="before")
    @classmethod
    def _require_strict_timestamp_input(_cls, value: object) -> object:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and _CANONICAL_TIMESTAMP.fullmatch(value):
            return value
        raise ValueError(
            "Review timestamp must be a datetime or canonical timezone-aware ISO-8601"
        )

    @field_validator("version", mode="before")
    @classmethod
    def _require_strict_version(_cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("Attestation version must be an integer")
        return value

    @model_validator(mode="after")
    def _require_timezone_aware_timestamp(self) -> "ReviewAttestation":
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() is None:
            raise ValueError("Review timestamp must be timezone-aware")
        return self

    @field_validator("output_sha256")
    @classmethod
    def _validate_output_sha256(_cls, value: str) -> str:
        if not _HEX_64.fullmatch(value):
            raise ValueError("Output SHA-256 must contain exactly 64 hex characters")
        return value

    @field_validator("pilot_id")
    @classmethod
    def _validate_pilot_id(_cls, value: str) -> str:
        if not _HEX_16.fullmatch(value):
            raise ValueError("Pilot ID must contain exactly 16 hex characters")
        return value

    @field_validator("reviewed_persona_ids")
    @classmethod
    def _validate_reviewed_persona_ids(
        _cls, values: tuple[str, ...]
    ) -> tuple[str, ...]:
        _require_exact_unique_ids(values=values)
        return values

    @field_validator("reviewer_id", "protocol", "protocol_version")
    @classmethod
    def _validate_safe_text_identifier(_cls, value: str) -> str:
        if not _SAFE_IDENTIFIER.fullmatch(value):
            raise ValueError("Review metadata must be nonblank safe identifiers")
        return value


def _require_exact_unique_ids(*, values: tuple[str, ...]) -> None:
    if any(not value or value != value.strip() for value in values):
        raise ValueError("Reviewed persona IDs must be nonblank and unpadded")
    if len(set(values)) != len(values):
        raise ValueError("Reviewed persona IDs must be unique")


class ShardEvidence(StrictModel):
    """Portable accounting and checksums for one generation shard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    shard_id: StrictStr = Field(min_length=1)
    offset: StrictInt = Field(ge=0)
    rows: StrictInt = Field(gt=0)
    manifest_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    report_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    requests: StrictInt = Field(ge=0)
    retries: StrictInt = Field(ge=0)
    rejected_validation_responses: StrictInt = Field(ge=0)
    prompt_tokens: StrictInt = Field(ge=0)
    completion_tokens: StrictInt = Field(ge=0)
    total_tokens: StrictInt = Field(ge=0)
    provider_cost_usd: StrictFloat | None = Field(default=None, ge=0.0)
    providers: tuple[StrictStr, ...]

    @field_validator("provider_cost_usd")
    @classmethod
    def _finite_cost(_cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("Costs must be finite")
        return value

    @field_validator("providers")
    @classmethod
    def _unique_providers(_cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item or item != item.strip() for item in value):
            raise ValueError("Provider names must be nonblank")
        if tuple(sorted(set(value))) != value:
            raise ValueError("Provider names must be sorted and unique")
        return value


class ReleaseEvidence(StrictModel):
    """Portable provenance and accounting for a release."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: t.Literal[1]
    pilot_id: StrictStr = Field(min_length=1)
    model: StrictStr = Field(min_length=1)
    rows: StrictInt = Field(gt=0)
    output_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    input_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    sample_manifest_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    generation_config_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    generation_context_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    validator_version: StrictStr = Field(min_length=1)
    attributes_prompt_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    personas_prompt_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    upstream_run_id: StrictStr = Field(min_length=1)
    sample_source_run_id: StrictStr | None = None
    source_bundle_id: StrictStr | None = None
    policy_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    attestation_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    licence_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    code_license_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    uv_lock_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    pilot_validation_report_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    config_hashes: dict[StrictStr, StrictStr] = Field(min_length=5)
    shards: tuple[ShardEvidence, ...] = Field(min_length=1)
    accounting: Accounting
