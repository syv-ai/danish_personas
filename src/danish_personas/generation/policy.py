"""Policies controlling generation integrity and content validation."""

from enum import StrEnum

from ..checksum import ChecksumValidationPolicy


class ContentValidationPolicy(StrEnum):
    """Content-validation modes for generated persona responses."""

    GUARDED = "guarded"
    SCHEMA_ONLY = "schema_only"


__all__ = ["ChecksumValidationPolicy", "ContentValidationPolicy"]
