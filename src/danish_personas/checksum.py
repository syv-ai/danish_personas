"""Policies controlling checksum comparison at validation boundaries."""

from enum import StrEnum


class ChecksumValidationPolicy(StrEnum):
    """Whether a caller must match persisted checksums to current artefacts."""

    STRICT = "strict"
    IGNORE = "ignore"

    @property
    def validates_checksums(self) -> bool:
        """Whether checksum comparisons should reject mismatches."""
        return self is self.STRICT
