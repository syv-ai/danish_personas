"""Typed contracts for the Danish personas data pipeline."""

import typing as t
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StatBankValue(BaseModel):
    """One code and label from StatBank metadata."""

    id: str
    text: str


class StatBankVariable(BaseModel):
    """One variable from StatBank metadata."""

    id: str
    text: str
    values: list[StatBankValue]


class StatBankMetadata(BaseModel):
    """Subset of StatBank table metadata used by the pipeline."""

    id: str
    text: str
    description: str
    unit: str
    updated: str
    variables: list[StatBankVariable]


class StrictModel(BaseModel):
    """Base model that rejects undeclared fields."""

    model_config = ConfigDict(extra="forbid")


class CategoryConfig(StrictModel):
    """Versioned canonical category mappings."""

    version: int
    sex: dict[str, str]
    marital_status: dict[str, str]
    education: dict[str, str]
    education_pooling: dict[str, str]
    labour_market_status: dict[str, list[str]]


class LockedSource(StrictModel):
    """Resolved source query with explicit values."""

    table_id: str
    role: str
    period: str
    metadata_url: str
    data_url: str
    retrieved_metadata_at: str
    table_updated_at: str
    unit: str
    dimensions: dict[str, list[str]]
    estimated_cells: int = Field(gt=0)


class MetricResult(StrictModel):
    """One validation metric result."""

    name: str
    passed: bool
    value: float | int | str
    threshold: float | int | str
    details: str


class OceanConfig(StrictModel):
    """OCEAN sampling parameters."""

    mean: float
    standard_deviation: float = Field(gt=0.0)
    minimum: float
    maximum: float
    label_boundaries: list[float]
    labels: list[str]


class OceanTraits(StrictModel):
    """OCEAN personality scores and categorical labels."""

    openness_score: float = Field(ge=20.0, le=80.0)
    openness_label: str
    conscientiousness_score: float = Field(ge=20.0, le=80.0)
    conscientiousness_label: str
    extraversion_score: float = Field(ge=20.0, le=80.0)
    extraversion_label: str
    agreeableness_score: float = Field(ge=20.0, le=80.0)
    agreeableness_label: str
    neuroticism_score: float = Field(ge=20.0, le=80.0)
    neuroticism_label: str


class DemographicRecord(OceanTraits):
    """One non-LLM synthetic demographic and personality record."""

    persona_id: str
    country: t.Literal["Danmark"]
    age: int = Field(ge=18, le=125)
    age_band: str
    sex: t.Literal["male", "female"]
    marital_status: str
    region_code: str
    region: str
    education_level: str
    education_source_code: str
    education_resolution: t.Literal["ras209_age_band", "ras209_67_plus_proxy"]
    labour_market_status: str
    detailed_status_code: str
    detailed_status: str


class RunManifest(StrictModel):
    """Manifest for one deterministic generation run."""

    run_id: str
    created_at: str
    bundle_id: str
    bundle_manifest_sha256: str
    sampling_config_sha256: str
    rows: int = Field(gt=0)
    seed: int
    data_file: Path
    data_sha256: str
    logical_content_sha256: str
    llm_calls: int = Field(ge=0)


class SamplingConfig(StrictModel):
    """Deterministic demographic generation configuration."""

    version: int
    seed: int
    smoke_rows: int = Field(gt=0)
    statistical_rows: int = Field(gt=0)
    country: t.Literal["Danmark"]
    minimum_age: int = Field(ge=18)
    maximum_age: int = Field(le=125)
    publication_geography: t.Literal["region"]
    smoothing: float = Field(ge=0.0)
    ocean: OceanConfig


class SnapshotManifest(StrictModel):
    """Checksums and provenance for one raw source snapshot."""

    table_id: str
    role: str
    period: str
    metadata_sha256: str
    metadata_da_sha256: str
    query_sha256: str
    data_sha256: str
    response_headers_sha256: str
    retrieved_at: str
    data_bytes: int = Field(gt=0)


class BundleManifest(StrictModel):
    """Manifest for a prepared source bundle."""

    bundle_id: str
    created_at: str
    source_lock_sha256: str
    categories_sha256: str
    source_snapshots: list[SnapshotManifest]
    files: dict[str, str]
    reference_periods: dict[str, str]
    assumptions: list[str]


class SourceLock(StrictModel):
    """Resolved collection of immutable source queries."""

    version: int
    language: str
    release_rows: int = Field(gt=0)
    minimum_source_count: int = Field(ge=0)
    minimum_expected_release_count: int = Field(ge=0)
    resolved_at: str
    sources: list[LockedSource]


class SourceSelection(StrictModel):
    """Selection for one StatBank dimension."""

    selector: str | None = None
    values: list[str] | None = None

    @model_validator(mode="after")
    def validate_choice(self) -> "SourceSelection":
        """Require exactly one selection mechanism.

        Returns:
            The validated selection.

        Raises:
            ValueError:
                If both or neither selection mechanisms are set.
        """
        if (self.selector is None) == (self.values is None):
            message = "Set exactly one of selector and values"
            raise ValueError(message)
        return self


class SourceDefinition(StrictModel):
    """Configured StatBank source table."""

    table_id: str
    role: str
    period: str
    dimensions: dict[str, SourceSelection]


class SourcesConfig(StrictModel):
    """Source acquisition configuration."""

    version: int
    language: str
    release_rows: int = Field(gt=0)
    minimum_source_count: int = Field(ge=0)
    minimum_expected_release_count: int = Field(ge=0)
    sources: list[SourceDefinition]


class ValidationConfig(StrictModel):
    """Statistical and structural validation thresholds."""

    version: int
    absolute_proportion_tolerance: float = Field(gt=0.0)
    standard_error_multiplier: float = Field(gt=0.0)
    minimum_expected_count: float = Field(ge=0.0)
    maximum_ocean_pairwise_correlation: float = Field(ge=0.0)
    maximum_total_variation: dict[str, float]
    smoke_maximum_total_variation: float = Field(gt=0.0)
    smoke_holdout_maximum_total_variation: float = Field(gt=0.0)
    mandatory_marginals: list[str]


class ValidationReport(StrictModel):
    """Machine-readable validation report."""

    kind: t.Literal["sources", "demographics"]
    passed: bool
    created_at: str
    subject_id: str
    metrics: list[MetricResult]
