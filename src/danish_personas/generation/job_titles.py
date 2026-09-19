"""Reviewed Danish job-title allowlists for generation contract v2."""

import typing as t
from pathlib import Path

from pydantic import Field, field_validator, model_validator

from ..io import load_yaml_model, sha256_file
from ..models import DISCO_TWO_DIGIT_CODES, StrictModel

DEFAULT_JOB_TITLE_MAPPING_PATH = (
    Path(__file__).resolve().parents[3] / "config/job-function-titles.yaml"
)


class JobFunctionTitles(StrictModel):
    """One reviewed allowlist for an official job-function code."""

    label: str = Field(min_length=1)
    titles: list[str] = Field(min_length=1)

    @field_validator("titles")
    @classmethod
    def _require_nonempty_titles(_cls, values: list[str]) -> list[str]:
        if any(not value or value != value.strip() for value in values):
            raise ValueError("Job titles must be non-empty and trimmed")
        if len({value.casefold() for value in values}) != len(values):
            raise ValueError("Job titles must be unique")
        return values


class JobFunctionTitleMapping(StrictModel):
    """Strict, versioned mapping of all official ARBF codes to titles."""

    version: t.Literal[1]
    job_functions: dict[str, JobFunctionTitles]

    @model_validator(mode="after")
    def _require_frozen_partition(self) -> "JobFunctionTitleMapping":
        expected = set(DISCO_TWO_DIGIT_CODES)
        actual = set(self.job_functions)
        if actual != expected:
            raise ValueError("Job-title mapping must contain exactly all 42 ARBF codes")
        for code, entry in self.job_functions.items():
            if not entry.label.startswith(f"{code} "):
                raise ValueError(f"Job-function label must retain its code: {code}")
        return self


def job_title_mapping_sha256(path: Path = DEFAULT_JOB_TITLE_MAPPING_PATH) -> str:
    """Return the checksum of the exact reviewed mapping bytes."""
    load_job_title_mapping(path=path)
    return sha256_file(path)


def load_job_title_mapping(
    path: Path = DEFAULT_JOB_TITLE_MAPPING_PATH,
) -> JobFunctionTitleMapping:
    """Load and strictly validate the reviewed job-title mapping.

    Args:
        path: YAML mapping path.

    Returns:
        The validated mapping.
    """
    return load_yaml_model(path=path, model=JobFunctionTitleMapping)
