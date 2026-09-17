"""Strict typed contracts for release policy and human review evidence."""

import re
import typing as t
from dataclasses import dataclass
from datetime import datetime

from pydantic import AliasChoices, Field, field_validator, model_validator

from ..models import StrictModel


class ReleasePolicyError(ValueError):
    """Raised when a release policy cannot authorise an eligibility check."""


_HEX_16 = re.compile(r"\A[0-9a-fA-F]{16}\Z")
_HEX_64 = re.compile(r"\A[0-9a-fA-F]{64}\Z")
_LICENCE_IDENTIFIER = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9.-]*(?:\+[A-Za-z0-9.-]+)?\Z")
_SAFE_IDENTIFIER = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._:@+\-]*\Z")
PILOT_ROWS = 10_000
RELEASE_MINIMUM_ROWS = 100_000


@dataclass(frozen=True, slots=True)
class ReleaseApproval:
    """Immutable in-memory result of a successful release eligibility check.

    The result records organisational review evidence; it does not authenticate the
    reviewer cryptographically and does not write or represent a release file.
    """

    pilot_id: str
    output_sha256: str
    population_rows: int
    model: str
    reviewer_id: str
    reviewed_persona_ids: tuple[str, ...]
    required_review_count: int

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
    need to name a model or a licence.  The row boundaries are deliberately constants
    in :mod:`danish_personas.release.policy`; the two minima may only be tightened.
    """

    version: t.Literal[1]
    enabled: bool = Field(validation_alias=AliasChoices("enabled", "release_enabled"))
    approved_models: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("approved_models", "approved_model_identifiers"),
    )
    dataset_licence: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "dataset_licence",
            "dataset_license",
            "dataset_licence_spdx",
            "dataset_license_spdx",
        ),
    )
    licence_file_sha256: str | None = Field(
        default=None,
        validation_alias=AliasChoices("licence_file_sha256", "license_file_sha256"),
    )
    pilot_minimum_reviewed_ids: int = Field(
        default=300,
        ge=300,
        validation_alias=AliasChoices(
            "pilot_minimum_reviewed_ids", "minimum_reviewed_ids_pilot"
        ),
    )
    release_minimum_reviewed_ids: int = Field(
        default=500,
        ge=500,
        validation_alias=AliasChoices(
            "release_minimum_reviewed_ids", "minimum_reviewed_ids_release"
        ),
    )

    model_config = {"populate_by_name": True, "extra": "forbid"}

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
        if not isinstance(population_rows, int) or isinstance(population_rows, bool):
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

    @field_validator("approved_models")
    @classmethod
    def validate_approved_models(_cls, values: list[str]) -> list[str]:
        """Require exact, nonblank and unique model identifiers.

        Returns:
            The unchanged model identifiers.

        Raises:
            ValueError:
                If an identifier is blank, padded, or duplicated.
        """
        if any(not value or value != value.strip() for value in values):
            raise ValueError(
                "Approved model identifiers must be nonblank and untrimmed"
            )
        if len(set(values)) != len(values):
            raise ValueError("Approved model identifiers must be unique")
        return values

    @field_validator("dataset_licence")
    @classmethod
    def validate_dataset_licence(_cls, value: str | None) -> str | None:
        """Require an SPDX-like identifier when a licence is supplied.

        Returns:
            The unchanged licence identifier, or ``None``.

        Raises:
            ValueError:
                If the identifier is not SPDX-like.
        """
        if value is not None and not _LICENCE_IDENTIFIER.fullmatch(value):
            raise ValueError("Dataset licence must be a strict SPDX-like identifier")
        return value

    @field_validator("licence_file_sha256")
    @classmethod
    def validate_licence_file_sha256(_cls, value: str | None) -> str | None:
        """Require an exact hexadecimal SHA-256 when a hash is supplied.

        Returns:
            The unchanged checksum, or ``None``.

        Raises:
            ValueError:
                If the checksum is not exactly 64 hexadecimal characters.
        """
        if value is not None and not _HEX_64.fullmatch(value):
            raise ValueError(
                "Licence-file SHA-256 must contain exactly 64 hex characters"
            )
        return value


ReleaseApprovalResult = ReleaseApproval


class ReviewAttestation(StrictModel):
    """Blinded human-review attestation bound to one generated output.

    This is organisational evidence of a review decision, not cryptographic proof of
    reviewer identity or authenticity.
    """

    version: t.Literal[1]
    release_approved: t.Literal[True]
    blinded: t.Literal[True]
    reviewer_id: str = Field(
        validation_alias=AliasChoices("reviewer_id", "reviewer_identifier")
    )
    protocol: str = Field(validation_alias=AliasChoices("protocol", "review_protocol"))
    protocol_version: str = Field(
        validation_alias=AliasChoices("protocol_version", "review_protocol_version")
    )
    reviewed_at: datetime
    pilot_id: str
    output_sha256: str
    population_rows: int = Field(gt=0)
    reviewed_persona_ids: list[str] = Field(
        min_length=1,
        validation_alias=AliasChoices("reviewed_persona_ids", "reviewed_ids"),
    )

    model_config = {"populate_by_name": True, "extra": "forbid"}

    @model_validator(mode="after")
    def require_timezone_aware_timestamp(self) -> "ReviewAttestation":
        """Reject timestamps that do not identify a timezone.

        Returns:
            The validated attestation.

        Raises:
            ValueError:
                If ``reviewed_at`` is naive.
        """
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() is None:
            raise ValueError("Review timestamp must be timezone-aware")
        return self

    @field_validator("output_sha256")
    @classmethod
    def validate_output_sha256(_cls, value: str) -> str:
        """Require the fixed-size hexadecimal output checksum.

        Returns:
            The unchanged output checksum.

        Raises:
            ValueError:
                If the value is not exactly 64 hexadecimal characters.
        """
        if not _HEX_64.fullmatch(value):
            raise ValueError("Output SHA-256 must contain exactly 64 hex characters")
        return value

    @field_validator("pilot_id")
    @classmethod
    def validate_pilot_id(_cls, value: str) -> str:
        """Require the fixed-size hexadecimal pilot identifier.

        Returns:
            The unchanged pilot identifier.

        Raises:
            ValueError:
                If the value is not exactly 16 hexadecimal characters.
        """
        if not _HEX_16.fullmatch(value):
            raise ValueError("Pilot ID must contain exactly 16 hex characters")
        return value

    @field_validator("reviewed_persona_ids")
    @classmethod
    def validate_reviewed_persona_ids(_cls, values: list[str]) -> list[str]:
        """Require nonblank, exactly unique reviewed persona identifiers.

        Returns:
            The unchanged reviewed identifiers.

        Raises:
            ValueError:
                If an identifier is blank or duplicated.
        """
        if any(not value or not value.strip() for value in values):
            raise ValueError("Reviewed persona IDs must be nonblank")
        if len(set(values)) != len(values):
            raise ValueError("Reviewed persona IDs must be unique")
        return values

    @field_validator("reviewer_id", "protocol", "protocol_version")
    @classmethod
    def validate_safe_text_identifier(_cls, value: str) -> str:
        """Require a nonblank safe token for review metadata.

        Returns:
            The unchanged safe identifier.

        Raises:
            ValueError:
                If the value is blank or contains unsafe characters.
        """
        if not _SAFE_IDENTIFIER.fullmatch(value):
            raise ValueError("Review metadata must be nonblank safe identifiers")
        return value
