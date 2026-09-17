"""Tests for the pure release eligibility contracts."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from danish_personas.release import (
    ReleasePolicy,
    ReleasePolicyError,
    ReviewAttestation,
    required_review_count,
    validate_release_approval,
)

_HASH = "a" * 64
_PILOT_ID = "b" * 16


def test_approval_rejects_a_review_count_below_the_boundary() -> None:
    """A pilot with fewer than 300 reviewed IDs fails closed."""
    ids = [f"persona-{index}" for index in range(300)]
    with pytest.raises(ReleasePolicyError):
        validate_release_approval(
            make_policy(),
            make_attestation(reviewed_persona_ids=ids[:-1]),
            model="provider/model@revision",
            pilot_id=_PILOT_ID,
            output_sha256=_HASH,
            population_rows=10_000,
            output_persona_ids=ids,
        )


def make_attestation(**overrides: object) -> ReviewAttestation:
    """Build a valid explicit blinded-review attestation for tests.

    Returns:
        A valid review attestation.
    """
    values: dict[str, object] = {
        "version": 1,
        "release_approved": True,
        "blinded": True,
        "reviewer_id": "reviewer-1",
        "protocol": "blinded-human-review",
        "protocol_version": "1",
        "reviewed_at": datetime(2026, 9, 17, tzinfo=timezone.utc),
        "pilot_id": _PILOT_ID,
        "output_sha256": _HASH,
        "population_rows": 10_000,
        "reviewed_persona_ids": [f"persona-{index}" for index in range(300)],
    }
    values.update(overrides)
    return ReviewAttestation.model_validate(values)


def make_policy(**overrides: object) -> ReleasePolicy:
    """Build a valid enabled release policy for tests.

    Returns:
        A valid enabled policy.
    """
    values: dict[str, object] = {
        "version": 1,
        "enabled": True,
        "approved_models": ["provider/model@revision"],
        "dataset_licence": "CC-BY-4.0",
        "licence_file_sha256": _HASH,
    }
    values.update(overrides)
    return ReleasePolicy.model_validate(values)


def test_approval_rejects_duplicate_or_unreviewed_output_ids() -> None:
    """The output ID collection must be unique and contain all reviewed IDs."""
    ids = [f"persona-{index}" for index in range(300)]
    for output_ids in (ids[:-1], [*ids, ids[0]]):
        with pytest.raises(ReleasePolicyError):
            validate_release_approval(
                make_policy(),
                make_attestation(),
                model="provider/model@revision",
                pilot_id=_PILOT_ID,
                output_sha256=_HASH,
                population_rows=10_000,
                output_persona_ids=output_ids,
            )


@pytest.mark.parametrize(
    "changes",
    [
        {"population_rows": 9_999},
        {"pilot_id": "c" * 16},
        {"output_sha256": "c" * 64},
        {"model": "other/model"},
    ],
)
def test_approval_rejects_wrong_bindings_or_size(changes: dict[str, object]) -> None:
    """Eligibility rejects unsupported sizes and every mismatched binding."""
    attestation = make_attestation()
    kwargs: dict[str, object] = {
        "model": "provider/model@revision",
        "pilot_id": _PILOT_ID,
        "output_sha256": _HASH,
        "population_rows": 10_000,
        "output_persona_ids": [f"persona-{index}" for index in range(300)],
    }
    kwargs.update({key: value for key, value in changes.items() if key != "model"})
    model = str(changes.get("model", "provider/model@revision"))
    with pytest.raises(ReleasePolicyError):
        validate_release_approval(
            make_policy(),
            attestation,
            model=model,
            pilot_id=str(kwargs["pilot_id"]),
            output_sha256=str(kwargs["output_sha256"]),
            population_rows=int(kwargs["population_rows"]),
            output_persona_ids=kwargs["output_persona_ids"],  # type: ignore[arg-type]
        )


def test_attestation_rejects_bad_bindings_and_duplicate_review_ids() -> None:
    """Attestation identifiers, hashes, and reviewed IDs have fixed shapes."""
    with pytest.raises(ValidationError):
        make_attestation(pilot_id="short")
    with pytest.raises(ValidationError):
        make_attestation(output_sha256="short")
    with pytest.raises(ValidationError):
        make_attestation(reviewed_persona_ids=["same", "same"])


def test_attestation_requires_explicit_true_fields_and_timezone() -> None:
    """Approval and blindedness must be explicit, and timestamps timezone-aware."""
    for field in ("release_approved", "blinded"):
        with pytest.raises(ValidationError):
            make_attestation(**{field: False})
        values = {
            "version": 1,
            "reviewer_id": "reviewer-1",
            "protocol": "blinded-human-review",
            "protocol_version": "1",
            "reviewed_at": datetime(2026, 9, 17, tzinfo=timezone.utc),
            "pilot_id": _PILOT_ID,
            "output_sha256": _HASH,
            "population_rows": 10_000,
            "reviewed_persona_ids": ["persona-1"],
        }
        with pytest.raises(ValidationError):
            ReviewAttestation.model_validate(
                {key: value for key, value in values.items()}
            )
    with pytest.raises(ValidationError):
        make_attestation(reviewed_at=datetime(2026, 9, 17))
    with pytest.raises(ValidationError):
        make_attestation(reviewer_id="reviewer/1")
    with pytest.raises(ValidationError):
        make_attestation(protocol="")
    with pytest.raises(ValidationError):
        make_attestation(protocol_version=" ")


def test_disabled_policy_may_be_incomplete_but_enabled_policy_may_not() -> None:
    """The committed policy can be disabled without publication metadata."""
    assert not ReleasePolicy(version=1, enabled=False).enabled
    with pytest.raises(ValidationError):
        ReleasePolicy(version=1, enabled=True)


def test_policy_can_only_tighten_review_minima() -> None:
    """Configured review minima cannot weaken the fixed boundaries."""
    policy = make_policy(
        pilot_minimum_reviewed_ids=301, release_minimum_reviewed_ids=501
    )
    assert policy.required_review_count(population_rows=10_000) == 301
    assert required_review_count(100_000, policy) == 501
    with pytest.raises(ValidationError):
        make_policy(pilot_minimum_reviewed_ids=299)
    with pytest.raises(ValidationError):
        make_policy(release_minimum_reviewed_ids=499)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("approved_models", ["model", "model"]),
        ("approved_models", [" model"]),
        ("dataset_licence", "CC BY 4.0"),
        ("dataset_licence", ""),
        ("licence_file_sha256", "a" * 63),
        ("licence_file_sha256", "g" * 64),
    ],
)
def test_policy_rejects_incomplete_or_non_exact_metadata(
    field: str, value: object
) -> None:
    """Model allowlists and licence metadata are exact contracts."""
    with pytest.raises(ValidationError):
        make_policy(**{field: value})


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (9_999, None),
        (10_000, 300),
        (10_001, None),
        (99_999, None),
        (100_000, 500),
        (250_000, 500),
    ],
)
def test_review_boundaries(rows: int, expected: int | None) -> None:
    """Only the fixed pilot and release population boundaries are eligible."""
    if expected is None:
        with pytest.raises(ReleasePolicyError):
            required_review_count(rows)
    else:
        assert required_review_count(rows) == expected


def test_valid_pilot_and_release_approvals() -> None:
    """Both supported release sizes return immutable approval results."""
    pilot_ids = [f"persona-{index}" for index in range(300)]
    pilot = validate_release_approval(
        make_policy(),
        make_attestation(),
        model="provider/model@revision",
        pilot_id=_PILOT_ID,
        output_sha256=_HASH,
        population_rows=10_000,
        output_persona_ids=pilot_ids,
    )
    assert pilot.reviewed_count == 300
    with pytest.raises((AttributeError, TypeError)):
        pilot.__setattr__("population_rows", 1)

    release_attestation = make_attestation(
        population_rows=100_000,
        reviewed_persona_ids=[f"persona-{index}" for index in range(500)],
    )
    release = validate_release_approval(
        make_policy(),
        release_attestation,
        model="provider/model@revision",
        pilot_id=_PILOT_ID,
        output_sha256=_HASH,
        population_rows=100_000,
        output_persona_ids=[f"persona-{index}" for index in range(500)],
    )
    assert release.required_review_count == 500
