"""Tests for the pure release eligibility contracts."""

import typing as t

import pytest
from support import _approve_pilot, _output_ids, make_attestation, make_policy

from danish_personas.release import (
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
    ids=[
        "model-int",
        "model-bytes",
        "model-padded",
        "pilot-int",
        "pilot-bytes",
        "hash-int",
        "hash-bytes",
        "rows-bool",
        "rows-string",
        "rows-bytes",
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


def test_approval_rejects_a_review_count_below_the_boundary() -> None:
    """A pilot with fewer than 300 reviewed IDs fails closed."""
    reviewed_ids = [f"persona-{index}" for index in range(299)]
    with pytest.raises(ReleasePolicyError):
        _approve_pilot(attestation=make_attestation(reviewed_persona_ids=reviewed_ids))


@pytest.mark.parametrize(
    "bad_id",
    [1, b"persona-1", " persona-1", "persona-1 ", ""],
    ids=["integer", "bytes", "leading-space", "trailing-space", "empty"],
)
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
    ids=["row-count", "pilot-id", "output-hash", "model"],
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
    ids=[
        "below-pilot",
        "pilot-boundary",
        "between-boundaries",
        "below-release",
        "release-boundary",
        "above-release",
    ],
)
def test_review_boundaries(rows: int, expected: int | None) -> None:
    """Only the fixed pilot and release population boundaries are eligible."""
    if expected is None:
        with pytest.raises(ReleasePolicyError):
            required_review_count(rows)
    else:
        assert required_review_count(rows) == expected


@pytest.mark.parametrize(
    "rows",
    [True, "10000", b"10000", 10_000.0],
    ids=["bool", "string", "bytes", "float"],
)
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
