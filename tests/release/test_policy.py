"""Tests for the pure release eligibility contracts."""

import collections.abc as c
import json
import typing as t
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from danish_personas.release import (
    ReleaseApproval,
    ReleasePolicy,
    ReleasePolicyError,
    ReviewAttestation,
    required_review_count,
    validate_release_approval,
)

_HASH = "a" * 64
_PILOT_ID = "b" * 16
_MODEL = "provider/model@revision"
_PILOT_ROWS = 10_000


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", 1),
        ("model", b"provider/model@revision"),
        ("model", f" {_MODEL}"),
        ("pilot_id", 1),
        ("pilot_id", b"b" * 16),
        ("output_sha256", 1),
        ("output_sha256", b"a" * 64),
        ("population_rows", True),
        ("population_rows", "10000"),
        ("population_rows", b"10000"),
    ],
)
def test_approval_bindings_do_not_coerce(field: str, value: object) -> None:
    """Approval bindings reject coercible security values."""
    kwargs: dict[str, object] = {
        "model": _MODEL,
        "pilot_id": _PILOT_ID,
        "output_sha256": _HASH,
        "population_rows": _PILOT_ROWS,
    }
    kwargs[field] = value
    with pytest.raises(ReleasePolicyError):
        validate_release_approval(
            make_policy(),
            make_attestation(),
            model=t.cast(str, kwargs["model"]),
            pilot_id=t.cast(str, kwargs["pilot_id"]),
            output_sha256=t.cast(str, kwargs["output_sha256"]),
            population_rows=t.cast(int, kwargs["population_rows"]),
            output_persona_ids=_output_ids(_PILOT_ROWS),
        )


def _output_ids(rows: int) -> c.Iterator[str]:
    return (f"persona-{index}" for index in range(rows))


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
        "population_rows": _PILOT_ROWS,
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
        "approved_models": [_MODEL],
        "dataset_licence": "CC-BY-4.0",
        "licence_file_sha256": _HASH,
    }
    values.update(overrides)
    return ReleasePolicy.model_validate(values)


def test_approval_defensively_revalidates_constructed_contracts() -> None:
    """Bypassed Pydantic construction cannot evade approval invariants."""
    invalid_policy = ReleasePolicy.model_construct(
        version=1,
        enabled=True,
        approved_models=(_MODEL,),
        dataset_licence=None,
        licence_file_sha256=_HASH,
        pilot_minimum_reviewed_ids=300,
        release_minimum_reviewed_ids=500,
    )
    invalid_attestation = ReviewAttestation.model_construct(
        _fields_set=None, **{**make_attestation().model_dump(), "release_approved": 1}
    )
    for policy, attestation in (
        (invalid_policy, make_attestation()),
        (make_policy(), invalid_attestation),
    ):
        with pytest.raises(ReleasePolicyError):
            _approve_pilot(policy=policy, attestation=attestation)


def _approve_pilot(
    *,
    policy: ReleasePolicy | None = None,
    attestation: ReviewAttestation | None = None,
    output_persona_ids: object | None = None,
) -> ReleaseApproval:
    ids = _output_ids(_PILOT_ROWS) if output_persona_ids is None else output_persona_ids
    return validate_release_approval(
        policy or make_policy(),
        attestation or make_attestation(),
        model=_MODEL,
        pilot_id=_PILOT_ID,
        output_sha256=_HASH,
        population_rows=_PILOT_ROWS,
        output_persona_ids=t.cast(c.Iterable[str], ids),
    )


def test_approval_rejects_a_review_count_below_the_boundary() -> None:
    """A pilot with fewer than 300 reviewed IDs fails closed."""
    reviewed_ids = [f"persona-{index}" for index in range(299)]
    with pytest.raises(ReleasePolicyError):
        _approve_pilot(attestation=make_attestation(reviewed_persona_ids=reviewed_ids))


@pytest.mark.parametrize("bad_id", [1, b"persona-1", " persona-1", "persona-1 ", ""])
def test_approval_rejects_non_exact_output_ids(bad_id: object) -> None:
    """Output populations reject non-string, blank, and padded identifiers."""
    ids: list[object] = [f"persona-{index}" for index in range(_PILOT_ROWS)]
    ids[-1] = bad_id
    with pytest.raises(ReleasePolicyError):
        _approve_pilot(output_persona_ids=ids)


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
    kwargs: dict[str, object] = {
        "model": _MODEL,
        "pilot_id": _PILOT_ID,
        "output_sha256": _HASH,
        "population_rows": _PILOT_ROWS,
    }
    kwargs.update(changes)
    with pytest.raises(ReleasePolicyError):
        validate_release_approval(
            make_policy(),
            make_attestation(),
            model=t.cast(str, kwargs["model"]),
            pilot_id=t.cast(str, kwargs["pilot_id"]),
            output_sha256=t.cast(str, kwargs["output_sha256"]),
            population_rows=t.cast(int, kwargs["population_rows"]),
            output_persona_ids=_output_ids(_PILOT_ROWS),
        )


def test_approval_requires_exact_complete_unique_output_population() -> None:
    """Output IDs must exactly account for all attested rows without duplicates."""
    too_few = _output_ids(_PILOT_ROWS - 1)
    too_many = _output_ids(_PILOT_ROWS + 1)
    duplicated = [f"persona-{index}" for index in range(_PILOT_ROWS)]
    duplicated[-1] = duplicated[0]
    missing_reviewed = make_attestation(
        reviewed_persona_ids=(
            *(f"persona-{index}" for index in range(299)),
            "persona-outside-output",
        )
    )

    for attestation, ids in (
        (make_attestation(), too_few),
        (make_attestation(), too_many),
        (make_attestation(), duplicated),
        (missing_reviewed, _output_ids(_PILOT_ROWS)),
    ):
        with pytest.raises(ReleasePolicyError):
            _approve_pilot(attestation=attestation, output_persona_ids=ids)


def test_attestation_accepts_canonical_timezone_aware_json_timestamp() -> None:
    """Canonical timezone-aware ISO-8601 JSON timestamps remain supported."""
    payload = {
        "version": 1,
        "release_approved": True,
        "blinded": True,
        "reviewer_id": "reviewer-1",
        "protocol": "blinded-human-review",
        "protocol_version": "1",
        "reviewed_at": "2026-09-17T00:00:00Z",
        "pilot_id": _PILOT_ID,
        "output_sha256": _HASH,
        "population_rows": _PILOT_ROWS,
        "reviewed_persona_ids": ["persona-1"],
    }
    attestation = ReviewAttestation.model_validate_json(json.dumps(payload))
    assert attestation.reviewed_at.utcoffset() == timezone.utc.utcoffset(None)


def test_attestation_rejects_bad_bindings_and_duplicate_review_ids() -> None:
    """Attestation identifiers, hashes, and reviewed IDs have fixed shapes."""
    with pytest.raises(ValidationError):
        make_attestation(pilot_id="short")
    with pytest.raises(ValidationError):
        make_attestation(output_sha256="short")
    with pytest.raises(ValidationError):
        make_attestation(reviewed_persona_ids=["same", "same"])
    with pytest.raises(ValidationError):
        make_attestation(reviewed_persona_ids=[" padded "])


@pytest.mark.parametrize(
    "value",
    [
        0,
        1_789_603_200,
        "1789603200",
        b"2026-09-17T00:00:00Z",
        "2026-09-17 00:00:00Z",
        "2026-09-17T00:00:00",
        datetime(2026, 9, 17),
    ],
)
def test_attestation_rejects_noncanonical_or_naive_timestamps(value: object) -> None:
    """Timestamps reject numeric, byte, noncanonical and timezone-naive inputs."""
    with pytest.raises(ValidationError):
        make_attestation(reviewed_at=value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reviewer_id", " reviewer-1"),
        ("reviewer_id", "reviewer-1 "),
        ("protocol", " blinded-human-review"),
        ("protocol_version", "1 "),
    ],
)
def test_attestation_rejects_padded_bound_identifiers(
    field: str, value: object
) -> None:
    """Bound attestation identifiers are rejected rather than silently stripped."""
    with pytest.raises(ValidationError):
        make_attestation(**{field: value})


@pytest.mark.parametrize("field", ["release_approved", "blinded"])
@pytest.mark.parametrize("value", [False, 0, 1, "true", "True"])
def test_attestation_requires_literal_true(field: str, value: object) -> None:
    """Approval and blindedness reject false and coercible true-like values."""
    with pytest.raises(ValidationError):
        make_attestation(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", "1"),
        ("version", True),
        ("population_rows", "10000"),
        ("population_rows", b"10000"),
        ("population_rows", True),
        ("reviewer_id", 1),
        ("reviewer_id", b"reviewer-1"),
        ("protocol", 1),
        ("protocol", b"blinded-human-review"),
        ("protocol_version", 1),
        ("protocol_version", b"1"),
        ("pilot_id", 1),
        ("pilot_id", b"b" * 16),
        ("output_sha256", 1),
        ("output_sha256", b"a" * 64),
        ("reviewed_persona_ids", [1]),
        ("reviewed_persona_ids", [b"persona-1"]),
    ],
)
def test_attestation_security_fields_do_not_coerce(field: str, value: object) -> None:
    """Security-relevant attestation fields reject coercible representations."""
    with pytest.raises(ValidationError):
        make_attestation(**{field: value})


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
    assert policy.required_review_count(population_rows=_PILOT_ROWS) == 301
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
    ("field", "value"),
    [
        ("version", "1"),
        ("version", True),
        ("enabled", "true"),
        ("enabled", 1),
        ("approved_models", [1]),
        ("approved_models", [b"model"]),
        ("dataset_licence", 1),
        ("dataset_licence", b"CC-BY-4.0"),
        ("licence_file_sha256", 1),
        ("licence_file_sha256", b"a" * 64),
        ("pilot_minimum_reviewed_ids", "300"),
        ("pilot_minimum_reviewed_ids", b"300"),
        ("pilot_minimum_reviewed_ids", True),
        ("release_minimum_reviewed_ids", "500"),
        ("release_minimum_reviewed_ids", b"500"),
        ("release_minimum_reviewed_ids", True),
    ],
)
def test_policy_security_fields_do_not_coerce(field: str, value: object) -> None:
    """Security-relevant policy fields reject coercible representations."""
    with pytest.raises(ValidationError):
        make_policy(**{field: value})


def test_release_contracts_are_deeply_immutable() -> None:
    """Policy, attestation, and approval expose tuples and reject all mutation."""
    models = [_MODEL]
    reviewed_ids = [f"persona-{index}" for index in range(300)]
    policy = make_policy(approved_models=models)
    attestation = make_attestation(reviewed_persona_ids=reviewed_ids)
    approval = _approve_pilot(policy=policy, attestation=attestation)

    models[0] = "changed"
    reviewed_ids[0] = "changed"
    assert policy.approved_models == (_MODEL,)
    assert attestation.reviewed_persona_ids[0] == "persona-0"
    assert approval.reviewed_persona_ids[0] == "persona-0"

    for contract, field, value in (
        (policy, "enabled", False),
        (attestation, "reviewer_id", "changed"),
        (approval, "population_rows", 1),
    ):
        with pytest.raises(ValidationError):
            setattr(contract, field, value)

    with pytest.raises(TypeError):
        t.cast(c.MutableSequence[str], policy.approved_models)[0] = "changed"
    with pytest.raises(TypeError):
        t.cast(c.MutableSequence[str], attestation.reviewed_persona_ids)[0] = "changed"
    with pytest.raises(TypeError):
        t.cast(c.MutableSequence[str], approval.reviewed_persona_ids)[0] = "changed"


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


@pytest.mark.parametrize("rows", [True, "10000", b"10000", 10_000.0])
def test_review_boundaries_reject_coercible_row_counts(rows: object) -> None:
    """Boundary checks require actual integers and reject Boolean values."""
    with pytest.raises(ReleasePolicyError):
        required_review_count(t.cast(int, rows))


def test_valid_pilot_and_large_release_approvals() -> None:
    """Complete 10,000 and 100,000-plus populations pass eligibility checks."""
    pilot = _approve_pilot()
    assert pilot.reviewed_count == 300

    release_rows = 100_001
    release_attestation = make_attestation(
        population_rows=release_rows,
        reviewed_persona_ids=[f"persona-{index}" for index in range(500)],
    )
    release = validate_release_approval(
        make_policy(),
        release_attestation,
        model=_MODEL,
        pilot_id=_PILOT_ID,
        output_sha256=_HASH,
        population_rows=release_rows,
        output_persona_ids=_output_ids(release_rows),
    )
    assert release.required_review_count == 500
