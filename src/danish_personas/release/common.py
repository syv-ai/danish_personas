"""Small shared release identity helpers."""

import hashlib
from datetime import datetime

import polars as pl

from ..generation.models import GeneratedAttributes, PersonaDescriptions
from ..io import canonical_json
from ..models import DemographicRecord


def persona_output_dtypes_are_valid(output: pl.DataFrame) -> bool:
    """Return whether a persona output uses the permitted logical dtypes.

    The optional career-goals field may use Polars' Null dtype only when it has no
    values at all.  A Null dtype on any other field would hide a malformed output.
    """
    expected_columns = (
        *DemographicRecord.model_fields,
        *GeneratedAttributes.model_fields,
        *PersonaDescriptions.model_fields,
    )
    expected_dtypes: dict[str, object] = {
        name: (
            pl.Int64
            if name == "age"
            else pl.Float64
            if name.endswith("_score")
            else pl.String
        )
        for name in expected_columns
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
        if name == "career_goals_and_ambitions" and actual_dtype == pl.Null:
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
