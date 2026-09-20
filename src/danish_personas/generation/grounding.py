"""Canonical literal facts required in generated persona summaries."""

import collections.abc as c

from pydantic import Field

from ..models import DemographicRecord, StrictModel
from .models import GeneratedAttributes

EDUCATION_DANISH = {
    "primary": "grundskole",
    "secondary_or_vocational": "ungdomsuddannelse eller erhvervsuddannelse",
    "higher_education": "videregående uddannelse",
    "not_stated": "uddannelse ikke oplyst",
}
SEX_DANISH = {"male": "mand", "female": "kvinde", "m": "mand", "k": "kvinde"}
STATUS_DANISH = {
    "unemployed": "ledig",
    "student": "studerende",
    "retired": "pensionist",
    "other": "uden for arbejdsmarkedet",
    "outside_labour_force": "uden for arbejdsmarkedet",
    "not_applicable": "uden for arbejdsmarkedet",
}
DETAILED_STATUS_DANISH = {
    "05": "selvstændig",
    "10": "medarbejdende ægtefælle",
    "50": "ledig",
    "85": "ledig",
    "90": "ledig",
    "95": "ledig",
    "130": "studerende",
    "154": "studerende",
    "156": "studerende",
    "158": "studerende",
    "160": "studerende",
    "135": "pensionist",
    "138": "pensionist",
    "139": "pensionist",
    "140": "pensionist",
    "145": "pensionist",
    "150": "pensionist",
    "155": "pensionist",
}


class PersonaGroundingFacts(StrictModel):
    """Exact strings that the short persona must reproduce literally."""

    age: str = Field(min_length=3)
    sex: str = Field(min_length=1)
    municipality: str = Field(min_length=1)
    education_level: str = Field(min_length=1)
    origin_country: str = Field(min_length=1)
    current_employment: str = Field(min_length=1)


def build_persona_grounding_facts(
    demographic: DemographicRecord | c.Mapping[str, object],
    attributes: GeneratedAttributes | c.Mapping[str, object],
) -> PersonaGroundingFacts:
    """Render the one canonical set of literal persona facts.

    Args:
        demographic:
            The sampled demographic row, including official human-readable labels.
        attributes:
            The validated first-stage attributes. Its title has already passed the
            reviewed job-title allowlist when a title is eligible.

    Returns:
        The exact strings shared by the stage-two payload and validator.

    Raises:
        ValueError:
            If a demographic value cannot be rendered or title eligibility is
            inconsistent with the stage-one attributes.
    """
    context = _context_values(demographic)
    generated = (
        attributes
        if isinstance(attributes, GeneratedAttributes)
        else GeneratedAttributes.model_validate(attributes)
    )
    sex_key = str(context.get("sex", "")).casefold()
    sex = SEX_DANISH.get(sex_key)
    education_key = str(context.get("education_level", "")).casefold()
    education = EDUCATION_DANISH.get(education_key)
    resolution = context.get("job_function_resolution")
    if resolution == "lons20_sex_marginal" and generated.job_title is None:
        raise ValueError("Eligible job-function context requires a title")
    if resolution == "not_applicable" and generated.job_title is not None:
        raise ValueError("Not-applicable job-function context requires no title")
    current_employment = generated.job_title or canonical_current_status(
        demographic=context
    )
    values = {
        "age": f"{context.get('age')} år",
        "sex": sex,
        "municipality": context.get("municipality"),
        "education_level": education,
        "origin_country": context.get("origin_country"),
        "current_employment": current_employment,
    }
    if any(
        not isinstance(value, str) or not value.strip() for value in values.values()
    ):
        raise ValueError("Demographic context contains an unrenderable grounding fact")
    return PersonaGroundingFacts.model_validate(values)


def _context_values(
    demographic: DemographicRecord | c.Mapping[str, object],
) -> dict[str, object]:
    if isinstance(demographic, DemographicRecord):
        return demographic.model_dump(mode="python")
    return dict(demographic)


def canonical_current_status(
    *, demographic: DemographicRecord | c.Mapping[str, object]
) -> str:
    """Render the canonical current status when no job title is present.

    Args:
        demographic:
            The sampled demographic row.

    Returns:
        The canonical Danish current-status phrase.

    Raises:
        ValueError:
            If the labour-market status cannot be rendered.
    """
    context = _context_values(demographic)
    detailed_code = str(context.get("detailed_status_code", ""))
    if detailed_code in {"05", "10"}:
        return DETAILED_STATUS_DANISH[detailed_code]
    status = str(context.get("labour_market_status", "")).casefold()
    if status == "employed":
        return "lønmodtager"
    rendered = STATUS_DANISH.get(status)
    if rendered is None:
        raise ValueError("Unknown current labour status")
    return rendered
