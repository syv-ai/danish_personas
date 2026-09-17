"""Pure release-size and blinded-review eligibility checks."""

from collections.abc import Iterable

from .models import (
    PILOT_ROWS,
    RELEASE_MINIMUM_ROWS,
    ReleaseApproval,
    ReleasePolicy,
    ReleasePolicyError,
    ReviewAttestation,
)

PILOT_MINIMUM_REVIEWED_IDS = 300
RELEASE_MINIMUM_REVIEWED_IDS = 500


def validate_release_approval(
    policy: ReleasePolicy,
    attestation: ReviewAttestation,
    *,
    model: str,
    pilot_id: str,
    output_sha256: str,
    population_rows: int,
    output_persona_ids: Iterable[str],
) -> ReleaseApproval:
    """Validate a policy, attestation, and in-memory output ID collection.

    Args:
        policy:
            Release policy to apply.
        attestation:
            Explicit blinded human-review attestation.
        model:
            Exact model identifier used for the output.
        pilot_id:
            Exact pilot identifier bound to the output.
        output_sha256:
            Exact SHA-256 bound to the output.
        population_rows:
            Number of rows in the output.
        output_persona_ids:
            Persona IDs present in the output.  This function does not read a file.

    Returns:
        An immutable approval value containing the checked bindings and review count.

    Raises:
        ReleasePolicyError:
            If the policy, output bindings, review threshold, or output ID set fails.
    """
    if not policy.enabled:
        raise ReleasePolicyError("Release policy is disabled")
    if model not in policy.approved_models:
        raise ReleasePolicyError(
            "Output model is not on the exact approved-model allowlist"
        )

    required = required_review_count(population_rows=population_rows, policy=policy)

    if attestation.pilot_id != pilot_id:
        raise ReleasePolicyError("Attestation pilot ID does not match the output")
    if attestation.output_sha256 != output_sha256:
        raise ReleasePolicyError("Attestation output SHA-256 does not match the output")
    if attestation.population_rows != population_rows:
        raise ReleasePolicyError("Attestation population does not match the output")

    output_id_values = tuple(output_persona_ids)
    if len(set(output_id_values)) != len(output_id_values):
        raise ReleasePolicyError("Output persona IDs must be unique")
    output_id_set = set(output_id_values)
    missing_ids = set(attestation.reviewed_persona_ids) - output_id_set
    if missing_ids:
        raise ReleasePolicyError("Every reviewed persona ID must occur in the output")
    if len(attestation.reviewed_persona_ids) < required:
        raise ReleasePolicyError(
            "The attestation does not meet the required review count"
        )

    return ReleaseApproval(
        pilot_id=pilot_id,
        output_sha256=output_sha256,
        population_rows=population_rows,
        model=model,
        reviewer_id=attestation.reviewer_id,
        reviewed_persona_ids=tuple(attestation.reviewed_persona_ids),
        required_review_count=required,
    )


def required_review_count(
    population_rows: int, policy: ReleasePolicy | None = None
) -> int:
    """Return the required number of unique human-reviewed persona IDs.

    Args:
        population_rows:
            Number of rows in the proposed output.
        policy (optional):
            Policy supplying stricter minima.  If omitted, the fixed minima are used.

    Returns:
        The minimum number of unique IDs that must be reviewed.

    Raises:
        ReleasePolicyError:
            If the population is not one of the two publishable policy sizes, or if a
            supplied policy is disabled.
    """
    if not isinstance(population_rows, int) or isinstance(population_rows, bool):
        raise ReleasePolicyError("Population row count must be an integer")
    if policy is not None:
        return policy.required_review_count(population_rows=population_rows)
    if population_rows == PILOT_ROWS:
        return PILOT_MINIMUM_REVIEWED_IDS
    if population_rows >= RELEASE_MINIMUM_ROWS:
        return RELEASE_MINIMUM_REVIEWED_IDS
    raise ReleasePolicyError(
        "Only exactly 10,000 rows or at least 100,000 rows are eligible for release"
    )
