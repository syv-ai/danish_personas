"""Small shared release identity helpers."""

import hashlib
from datetime import datetime

import polars as pl

from ..generation.job_titles import JobFunctionTitleMapping
from ..generation.models import GeneratedAttributes, PersonaDescriptions
from ..generation.validation import parse_attributes, parse_descriptions
from ..io import canonical_json
from ..models import DemographicRecord
from ..origin_labels import OriginLabelContract, load_origin_label_contract

# The release output contract is deliberately explicit: accepting model-derived
# columns here could silently publish a future field or restore the removed field.
PERSONA_OUTPUT_COLUMNS = (
    *DemographicRecord.model_fields,
    "cultural_context",
    "skills_and_expertise",
    "hobbies_and_interests",
    "career_goals_and_ambitions",
    "job_title",
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "persona",
)


def persona_output_dtypes_are_valid(output: pl.DataFrame) -> bool:
    """Return whether a persona output uses the permitted logical dtypes.

    The nullable career-goals and job-title fields may use Polars' Null dtype only
    when they have no values at all.  A Null dtype on any other field would hide a
    malformed output.
    """
    expected_dtypes: dict[str, object] = {
        name: (
            pl.Int64
            if name == "age"
            else pl.Float64
            if name.endswith("_score")
            else pl.String
        )
        for name in PERSONA_OUTPUT_COLUMNS
        if name not in {"skills_and_expertise", "hobbies_and_interests"}
    }
    expected_dtypes["skills_and_expertise"] = pl.List(pl.String)
    expected_dtypes["hobbies_and_interests"] = pl.List(pl.String)
    if set(output.columns) != set(expected_dtypes) or len(output.columns) != len(
        expected_dtypes
    ):
        return False
    for name, expected_dtype in expected_dtypes.items():
        if name not in output.schema:
            return False
        actual_dtype = output.schema[name]
        if name == "age" and actual_dtype in {pl.Int32, pl.Int64}:
            continue
        if (
            name in {"career_goals_and_ambitions", "job_title"}
            and actual_dtype == pl.Null
        ):
            continue
        if actual_dtype != expected_dtype:
            return False
    return True


def release_id(
    *,
    pilot_id: str,
    output_sha256: str,
    reviewed_at: datetime,
    git_head: str,
    origin_url: str,
) -> str:
    """Derive the deterministic public release identifier.

    Returns:
        The first 32 hexadecimal characters of the identity digest.
    """
    identity = canonical_json(
        {
            "git_head": git_head,
            "origin_url": origin_url,
            "output_sha256": output_sha256,
            "pilot_id": pilot_id,
            "reviewed_at": reviewed_at.isoformat(),
        }
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:32]


def role(path: str) -> str:
    """Return the stable public role for a layout path."""
    if path == "data/personas.parquet":
        return "dataset"
    if path == "README.md":
        return "dataset-card"
    if path == "LICENSE.txt":
        return "licence"
    if path.startswith("provenance/"):
        return "provenance"
    return "attestation"


def validate_persona_output_rows(
    output: pl.DataFrame,
    *,
    job_title_mapping: JobFunctionTitleMapping | None = None,
    origin_label_contract: OriginLabelContract | None = None,
) -> None:
    """Replay generation-v3 contextual validation for every public output row.

    Args:
        output:
            The exact v3 persona output frame.
        job_title_mapping (optional):
            Reviewed title mapping bound to the release inputs. Defaults to the
            production mapping when omitted.
        origin_label_contract (optional):
            Official Danish FOLK2 labels bound to the release inputs. Defaults to
            the checked-in contract when omitted.

    Raises:
        ValueError:
            If a row is not a valid demographic, attribute, or contextual persona
            record.
    """
    if set(output.columns) != set(PERSONA_OUTPUT_COLUMNS) or len(output.columns) != len(
        PERSONA_OUTPUT_COLUMNS
    ):
        raise ValueError("Persona output schema must match generation contract v3")
    contract = origin_label_contract or load_origin_label_contract()
    labels = contract.labels
    validated_rows: set[str] = set()
    for index, row in enumerate(output.iter_rows(named=True)):
        cache_key = canonical_json(
            {name: row[name] for name in PERSONA_OUTPUT_COLUMNS if name != "persona_id"}
        )
        if cache_key in validated_rows:
            continue
        try:
            code = row["origin_country_code"]
            english = row["origin_country"]
            danish = row["origin_country_da"]
            if (
                not isinstance(code, str)
                or code not in labels
                or not isinstance(english, str)
                or not english.strip()
                or not isinstance(danish, str)
                or danish != labels[code]
            ):
                raise ValueError("Origin country labels do not match the contract")
            demographic = DemographicRecord.model_validate(
                {name: row[name] for name in DemographicRecord.model_fields}
            )
            attributes = GeneratedAttributes.model_validate(
                {name: row[name] for name in GeneratedAttributes.model_fields}
            )
            descriptions = PersonaDescriptions.model_validate(
                {name: row[name] for name in PersonaDescriptions.model_fields}
            )
            parse_attributes(
                attributes.model_dump_json(),
                demographic,
                job_title_mapping=job_title_mapping,
            )
            parse_descriptions(descriptions.model_dump_json(), demographic, attributes)
            validated_rows.add(cache_key)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Persona row {index} fails generation-v3 contextual validation"
            ) from error
