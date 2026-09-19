"""Typed contracts for the Danish personas data pipeline."""

import typing as t
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Increment when deterministic sampling semantics or generated record columns change.
# The run identity includes this value so incompatible historical outputs cannot be
# silently reused.
SAMPLER_SCHEMA_VERSION: int = 5
# Increment when prepared source artefacts or their interpretation changes.
# The bundle identity includes this value so incompatible historical bundles cannot
# be silently reused.
PREPARED_BUNDLE_SCHEMA_VERSION: int = 5
FROZEN_SAMPLE_SCHEMA_VERSION: int = 2
SUPPORTED_SAMPLING_CONFIG_VERSIONS: frozenset[int] = frozenset({3})
SUPPORTED_VALIDATION_CONFIG_VERSIONS: frozenset[int] = frozenset({5})

ELIGIBLE_JOB_FUNCTION_STATUS_CODES: frozenset[str] = frozenset(
    {"15", "20", "25", "30", "35", "40"}
)
DISCO_TWO_DIGIT_CODES: tuple[str, ...] = (
    "01",
    "02",
    "03",
    "11",
    "12",
    "13",
    "14",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "31",
    "32",
    "33",
    "34",
    "35",
    "41",
    "42",
    "43",
    "44",
    "51",
    "52",
    "53",
    "54",
    "61",
    "62",
    "71",
    "72",
    "73",
    "74",
    "75",
    "81",
    "82",
    "83",
    "91",
    "92",
    "93",
    "94",
    "95",
    "96",
)


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


class ClassificationDefinition(StrictModel):
    """Configured Statistics Denmark classification attachment."""

    classification_id: str
    role: str
    title: str
    valid_from: str
    page_url: str
    attachment_url: str


class ClassificationManifest(StrictModel):
    """Checksums and provenance for one raw classification snapshot."""

    classification_id: str
    role: str
    valid_from: str
    attachment_url: str
    resolved_url: str
    data_sha256: str
    response_headers_sha256: str
    retrieved_at: str
    data_bytes: int = Field(gt=0)


class Lons20Contract(StrictModel):
    """Separately reviewed canonical semantics for LONS20."""

    version: int = Field(gt=0)
    table_id: str
    table_text: str
    description: str
    unit: str
    dimensions: dict[str, str]
    selectors: dict[str, dict[str, str]]
    arbf: dict[str, str]


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
    origin_country_code: str
    origin_country: str
    age: int = Field(ge=18, le=125)
    age_resolution: t.Literal["municipality_age_band_sex", "municipality_age_band"]
    age_band: str
    sex: t.Literal["male", "female"]
    marital_status: str
    marital_resolution: t.Literal[
        "municipality_age_band_sex", "municipality_age_band", "municipality"
    ]
    municipality_code: str
    municipality: str
    region_code: str
    region: str
    education_level: str
    education_source_code: str
    education_resolution: t.Literal["ras209_age_band", "ras209_67_plus_proxy"]
    labour_market_status: str
    detailed_status_code: str
    detailed_status: str
    detailed_status_resolution: t.Literal["age_band_sex_status", "sex_status", "status"]
    job_function_code: str | None = None
    job_function: str | None = None
    job_function_resolution: t.Literal["lons20_sex_marginal", "not_applicable"]

    @model_validator(mode="after")
    def validate_job_function(self) -> "DemographicRecord":
        """Require paired job-function fields exactly for eligible employees.

        Returns:
            The validated record.

        Raises:
            ValueError:
                If fields or resolution disagree with detailed-status eligibility.
        """
        eligible = self.detailed_status_code in ELIGIBLE_JOB_FUNCTION_STATUS_CODES
        expected = "lons20_sex_marginal" if eligible else "not_applicable"
        if self.job_function_resolution != expected:
            message = (
                "Job-function resolution does not match detailed-status eligibility"
            )
            raise ValueError(message)
        if self.job_function_resolution == "not_applicable":
            if self.job_function_code is not None or self.job_function is not None:
                raise ValueError(
                    "Not-applicable job-function fields must both be exactly null"
                )
            return self
        if (
            self.job_function_code is None
            or self.job_function is None
            or not self.job_function_code.strip()
            or not self.job_function.strip()
        ):
            raise ValueError(
                "Eligible job-function fields must both be nonblank strings"
            )
        return self


class RunManifest(StrictModel):
    """Manifest for one deterministic generation run."""

    run_id: str
    sampler_schema_version: int = Field(ge=1)
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
    publication_geography: t.Literal["municipality"]
    smoothing: float = Field(ge=0.0)
    ocean: OceanConfig

    @model_validator(mode="after")
    def validate_version(self) -> "SamplingConfig":
        """Reject sampling files with an unsupported schema version.

        Returns:
            The validated configuration.

        Raises:
            ValueError:
                If the configuration version is not supported.
        """
        if self.version not in SUPPORTED_SAMPLING_CONFIG_VERSIONS:
            message = f"Unsupported sampling config version: {self.version}"
            raise ValueError(message)
        return self


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
    prepared_bundle_schema_version: int
    created_at: str
    source_lock_sha256: str
    categories_sha256: str
    source_snapshots: list[SnapshotManifest]
    classification_snapshots: list[ClassificationManifest]
    files: dict[str, str]
    reference_periods: dict[str, str]
    assumptions: list[str]
    lons20_contract_version: int = Field(ge=1)
    lons20_contract_sha256: str


class SourceMetadataExpectations(StrictModel):
    """Versioned metadata semantics expected for a source table."""

    table_text: str
    description: str
    unit: str
    dimensions: dict[str, str]
    values: dict[str, dict[str, str]]


class LockedSource(StrictModel):
    """Resolved source query with explicit values."""

    table_id: str
    role: str
    period: str
    format: t.Literal["CSV", "BULK"] = "CSV"
    metadata_url: str
    data_url: str
    retrieved_metadata_at: str
    table_updated_at: str
    unit: str
    dimensions: dict[str, list[str]]
    metadata_expectations: SourceMetadataExpectations | None = None
    expected_zero_codes: list[str] = Field(default_factory=list)
    estimated_cells: int = Field(gt=0)


class SourceLock(StrictModel):
    """Resolved collection of immutable source queries."""

    version: int
    language: str
    release_rows: int = Field(gt=0)
    minimum_source_count: int = Field(ge=0)
    minimum_expected_release_count: int = Field(ge=0)
    resolved_at: str
    sources: list[LockedSource]
    classifications: list[ClassificationDefinition]


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
    format: t.Literal["CSV", "BULK"] = "CSV"
    dimensions: dict[str, SourceSelection]
    metadata_expectations: SourceMetadataExpectations | None = None
    expected_zero_codes: list[str] = Field(default_factory=list)


class SourcesConfig(StrictModel):
    """Source acquisition configuration."""

    version: int
    language: str
    release_rows: int = Field(gt=0)
    minimum_source_count: int = Field(ge=0)
    minimum_expected_release_count: int = Field(ge=0)
    sources: list[SourceDefinition]
    classifications: list[ClassificationDefinition]


class ValidationConfig(StrictModel):
    """Statistical and structural validation thresholds."""

    version: int
    absolute_proportion_tolerance: float = Field(gt=0.0)
    standard_error_multiplier: float = Field(gt=0.0)
    minimum_expected_count: float = Field(ge=0.0)
    maximum_ocean_pairwise_correlation: float = Field(ge=0.0)
    maximum_backoff_rate: float = Field(ge=0.0, le=1.0)
    maximum_total_variation: dict[str, float]
    maximum_municipality_joint_total_variation: float = Field(gt=0.0)
    smoke_maximum_total_variation: float = Field(gt=0.0)
    smoke_holdout_maximum_total_variation: float = Field(gt=0.0)
    mandatory_marginals: list[str]

    @model_validator(mode="after")
    def validate_version(self) -> "ValidationConfig":
        """Reject validation files with an unsupported schema version.

        Returns:
            The validated configuration.

        Raises:
            ValueError:
                If the configuration version is not supported.
        """
        if self.version not in SUPPORTED_VALIDATION_CONFIG_VERSIONS:
            message = f"Unsupported validation config version: {self.version}"
            raise ValueError(message)
        required = {"origin_country", "municipality_code", "job_function"}
        missing = sorted(required - set(self.mandatory_marginals))
        if missing:
            message = f"Validation config is missing mandatory marginals: {missing}"
            raise ValueError(message)
        return self


class ValidationReport(StrictModel):
    """Machine-readable validation report."""

    kind: t.Literal["sources", "demographics", "personas", "persona_pilot"]
    passed: bool
    created_at: str
    subject_id: str
    metrics: list[MetricResult]
