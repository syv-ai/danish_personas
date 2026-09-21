"""Demographic job-function schema contract tests."""

import pytest

from danish_personas.models import DemographicRecord


@pytest.mark.parametrize(
    ("status_code", "resolution", "code", "label"),
    [
        ("15", "lons20_sex_marginal", "", "Function"),
        ("15", "lons20_sex_marginal", "01", " "),
        ("15", "not_applicable", None, None),
        ("05", "lons20_sex_marginal", "01", "Function"),
        ("05", "not_applicable", "01", None),
    ],
    ids=[
        "eligible-empty-code",
        "eligible-blank-label",
        "ineligible-valid-null-fields",
        "ineligible-code-and-label",
        "ineligible-resolution",
    ],
)
def test_demographic_record_rejects_job_function_cross_field_tampering(
    status_code: str, resolution: str, code: str | None, label: str | None
) -> None:
    """Eligibility, resolution, and paired values form one strict contract."""
    payload = {
        "persona_id": "p-1",
        "country": "Danmark",
        "origin_country_code": "5100",
        "origin_country": "Denmark",
        "age": 40,
        "age_resolution": "municipality_age_band",
        "age_band": "30-49",
        "sex": "female",
        "marital_status": "Ugift",
        "marital_resolution": "municipality",
        "municipality_code": "101",
        "municipality": "København",
        "region_code": "1081",
        "region": "Region Hovedstaden",
        "education_level": "Grundskole",
        "education_source_code": "10",
        "education_resolution": "ras209_age_band",
        "labour_market_status": "Beskæftigede",
        "detailed_status_code": status_code,
        "detailed_status": "Status",
        "detailed_status_resolution": "status",
        "job_function_code": code,
        "job_function": label,
        "job_function_resolution": resolution,
        "openness_score": 50,
        "openness_label": "average",
        "conscientiousness_score": 50,
        "conscientiousness_label": "average",
        "extraversion_score": 50,
        "extraversion_label": "average",
        "agreeableness_score": 50,
        "agreeableness_label": "average",
        "neuroticism_score": 50,
        "neuroticism_label": "average",
    }

    with pytest.raises(ValueError):
        DemographicRecord.model_validate(payload)


def test_demographic_schema_requires_job_function_resolution() -> None:
    """The release record schema cannot omit job-function provenance."""
    assert DemographicRecord.model_fields["job_function_resolution"].is_required()
    assert not DemographicRecord.model_fields["job_function_code"].is_required()
    assert not DemographicRecord.model_fields["job_function"].is_required()
