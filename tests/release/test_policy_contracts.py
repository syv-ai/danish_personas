"""Tests for the pure release eligibility contracts."""

import collections.abc as c
import json
import typing as t
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError
from support import _approve_pilot, make_attestation, make_policy

from danish_personas.release import (
    ReleasePolicy,
    ReviewAttestation,
    required_review_count,
)

_HASH = "a" * 64
_PILOT_ID = "b" * 16
_MODEL = "provider/model@revision"
_PILOT_ROWS = 10_000


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
    ids=[
        "zero",
        "unix-int",
        "unix-string",
        "bytes",
        "space-separated",
        "naive-iso",
        "naive-datetime",
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
    ids=[
        "reviewer-leading-space",
        "reviewer-trailing-space",
        "protocol-leading-space",
        "version-trailing-space",
    ],
)
def test_attestation_rejects_padded_bound_identifiers(
    field: str, value: object
) -> None:
    """Bound attestation identifiers are rejected rather than silently stripped."""
    with pytest.raises(ValidationError):
        make_attestation(**{field: value})


@pytest.mark.parametrize(
    "field", ["release_approved", "blinded"], ids=["approval", "blindedness"]
)
@pytest.mark.parametrize(
    "value",
    [False, 0, 1, "true", "True"],
    ids=["false", "zero", "one", "lowercase-true", "titlecase-true"],
)
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
    ids=[
        "version-string",
        "version-bool",
        "rows-string",
        "rows-bytes",
        "rows-bool",
        "reviewer-int",
        "reviewer-bytes",
        "protocol-int",
        "protocol-bytes",
        "protocol-version-int",
        "protocol-version-bytes",
        "pilot-int",
        "pilot-bytes",
        "hash-int",
        "hash-bytes",
        "reviewed-id-int",
        "reviewed-id-bytes",
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
    ids=[
        "duplicate-models",
        "padded-model",
        "licence-spelling",
        "empty-licence",
        "short-hash",
        "invalid-hash",
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
    ids=[
        "version-string",
        "version-bool",
        "enabled-string",
        "enabled-int",
        "model-int",
        "model-bytes",
        "licence-int",
        "licence-bytes",
        "licence-hash-int",
        "licence-hash-bytes",
        "pilot-min-string",
        "pilot-min-bytes",
        "pilot-min-bool",
        "release-min-string",
        "release-min-bytes",
        "release-min-bool",
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
