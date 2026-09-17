"""Pure release-size and blinded-review eligibility checks."""

from collections.abc import Iterable

from pydantic import ValidationError

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
            Persona IDs present in the output. This function does not read a file.

    Returns:
        An immutable approval value containing the checked bindings and review count.

    Raises:
        ReleasePolicyError:
            If the policy, output bindings, review threshold, or output ID set fails.
    """
    checked_policy = _revalidate_policy(policy=policy)
    checked_attestation = _revalidate_attestation(attestation=attestation)
    _validate_output_bindings(
        model=model,
        pilot_id=pilot_id,
        output_sha256=output_sha256,
        population_rows=population_rows,
    )

    if not checked_policy.enabled:
        raise ReleasePolicyError("Release policy is disabled")
    if model not in checked_policy.approved_models:
        raise ReleasePolicyError(
            "Output model is not on the exact approved-model allowlist"
        )

    required = required_review_count(
        population_rows=population_rows, policy=checked_policy
    )

    if checked_attestation.pilot_id != pilot_id:
        raise ReleasePolicyError("Attestation pilot ID does not match the output")
    if checked_attestation.output_sha256 != output_sha256:
        raise ReleasePolicyError("Attestation output SHA-256 does not match the output")
    if checked_attestation.population_rows != population_rows:
        raise ReleasePolicyError("Attestation population does not match the output")

    _validate_output_ids(
        output_persona_ids=output_persona_ids,
        population_rows=checked_attestation.population_rows,
        reviewed_persona_ids=checked_attestation.reviewed_persona_ids,
    )
    if len(checked_attestation.reviewed_persona_ids) < required:
        raise ReleasePolicyError(
            "The attestation does not meet the required review count"
        )

    return ReleaseApproval(
        pilot_id=pilot_id,
        output_sha256=output_sha256,
        population_rows=population_rows,
        model=model,
        reviewer_id=checked_attestation.reviewer_id,
        reviewed_persona_ids=checked_attestation.reviewed_persona_ids,
        required_review_count=required,
    )


def _revalidate_attestation(*, attestation: ReviewAttestation) -> ReviewAttestation:
    if not isinstance(attestation, ReviewAttestation):
        raise ReleasePolicyError(
            "Review attestation must use the review attestation contract"
        )
    try:
        return ReviewAttestation.model_validate(attestation.model_dump(warnings=False))
    except (AttributeError, TypeError, ValidationError, ValueError) as error:
        raise ReleasePolicyError("Review attestation contract is invalid") from error


def _revalidate_policy(*, policy: ReleasePolicy) -> ReleasePolicy:
    if not isinstance(policy, ReleasePolicy):
        raise ReleasePolicyError("Release policy must use the release policy contract")
    try:
        return ReleasePolicy.model_validate(policy.model_dump(warnings=False))
    except (AttributeError, TypeError, ValidationError, ValueError) as error:
        raise ReleasePolicyError("Release policy contract is invalid") from error


def _validate_output_bindings(
    *, model: str, pilot_id: str, output_sha256: str, population_rows: int
) -> None:
    if not isinstance(model, str) or not model or model != model.strip():
        raise ReleasePolicyError("Output model must be an exact nonblank string")
    if not isinstance(pilot_id, str):
        raise ReleasePolicyError("Output pilot ID must be a string")
    if not isinstance(output_sha256, str):
        raise ReleasePolicyError("Output SHA-256 must be a string")
    if type(population_rows) is not int:
        raise ReleasePolicyError("Population row count must be an integer")


def _validate_output_ids(
    *,
    output_persona_ids: Iterable[str],
    population_rows: int,
    reviewed_persona_ids: tuple[str, ...],
) -> None:
    output_id_values = tuple(output_persona_ids)
    if len(output_id_values) != population_rows:
        raise ReleasePolicyError(
            "Output persona ID count must equal the attested population row count"
        )
    if any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in output_id_values
    ):
        raise ReleasePolicyError(
            "Output persona IDs must be exact, nonblank string identifiers"
        )
    output_id_set = set(output_id_values)
    if len(output_id_set) != len(output_id_values):
        raise ReleasePolicyError("Output persona IDs must be unique")
    if set(reviewed_persona_ids) - output_id_set:
        raise ReleasePolicyError("Every reviewed persona ID must occur in the output")


def required_review_count(
    population_rows: int, policy: ReleasePolicy | None = None
) -> int:
    """Return the required number of unique human-reviewed persona IDs.

    Args:
        population_rows:
            Number of rows in the proposed output.
        policy (optional):
            Policy supplying stricter minima. If omitted, the fixed minima are used.

    Returns:
        The minimum number of unique IDs that must be reviewed.

    Raises:
        ReleasePolicyError:
            If the population is not one of the two publishable policy sizes, or if a
            supplied policy is disabled.
    """
    if type(population_rows) is not int:
        raise ReleasePolicyError("Population row count must be an integer")
    if policy is not None:
        checked_policy = _revalidate_policy(policy=policy)
        return checked_policy.required_review_count(population_rows=population_rows)
    if population_rows == PILOT_ROWS:
        return PILOT_MINIMUM_REVIEWED_IDS
    if population_rows >= RELEASE_MINIMUM_ROWS:
        return RELEASE_MINIMUM_REVIEWED_IDS
    raise ReleasePolicyError(
        "Only exactly 10,000 rows or at least 100,000 rows are eligible for release"
    )
