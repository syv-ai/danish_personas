"""Tests for local prose-review validation."""

from __future__ import annotations

import json
import typing as t
from dataclasses import FrozenInstanceError

import pytest

from danish_personas.generation.prose_review import (
    UNCHANGED_CONSISTENT_NOTE,
    ProseReviewError,
    ProseReviewResponse,
    validate_prose_review,
)


def test_extra_fields_and_unsafe_changed_facts_fail_closed() -> None:
    """Use local Pydantic extra-forbid and bounded verified fact deltas."""
    with pytest.raises(ProseReviewError):
        validate_prose_review(
            original_text=_persona_text("Før ændring"),
            changed_facts={"marital_status": {"old": "single", "new": "married"}},
            response={
                "disposition": "needs_manual_review",
                "patches": [],
                "unchanged_evidence": [],
                "manual_review_reason": "ambiguous",
                "completion_text": "do not retain me",
            },
        )
    with pytest.raises(ProseReviewError):
        validate_prose_review(
            original_text=_persona_text("Før ændring"),
            changed_facts={"age": {"old": 41, "new": 41}},
            response={
                "disposition": "needs_manual_review",
                "patches": [],
                "unchanged_evidence": [],
                "manual_review_reason": "ambiguous",
            },
        )


def _persona_text(unique_phrase: str) -> str:
    return f"{unique_phrase}. " + "Dette er en syntetisk dansk persona. " * 12


def test_invalid_completion_text_is_not_retained_in_error_message() -> None:
    """Fail closed without surfacing invalid freeform completion content."""
    raw = "freeform invalid completion with private-secret-token"

    with pytest.raises(ProseReviewError) as exc_info:
        validate_prose_review(
            original_text=_persona_text("Før ændring"),
            changed_facts={"marital_status": {"old": "single", "new": "married"}},
            response=raw,
        )

    assert "private-secret-token" not in str(exc_info.value)
    assert "freeform invalid completion" not in str(exc_info.value)


@pytest.mark.parametrize(
    "response",
    [
        {"disposition": "needs_manual_review", "patches": [], "unchanged_evidence": []},
        {
            "disposition": "needs_manual_review",
            "patches": [{"old_excerpt": "Før ændring", "new_excerpt": "x"}],
            "unchanged_evidence": [],
            "manual_review_reason": "ambiguous",
        },
        {
            "disposition": "needs_manual_review",
            "patches": [],
            "unchanged_evidence": [
                {"field": "marital_status", "kind": "fact_not_stated", "quote": ""}
            ],
            "manual_review_reason": "ambiguous",
        },
    ],
)
def test_needs_manual_review_rejects_patch_or_classifier_payloads(
    response: dict[str, object],
) -> None:
    """Keep manual review separate from patching and unchanged classification."""
    with pytest.raises(ProseReviewError):
        validate_prose_review(
            original_text=_persona_text("Før ændring"),
            changed_facts={"marital_status": {"old": "single", "new": "married"}},
            response=response,
        )


def test_needs_manual_review_requires_bounded_reason_and_no_edits() -> None:
    """Accept only a reason enum when the reviewer cannot make a safe decision."""
    original = _persona_text("Før ændring")

    result = validate_prose_review(
        original_text=original,
        changed_facts={"marital_status": {"old": "single", "new": "married"}},
        response={
            "disposition": "needs_manual_review",
            "patches": [],
            "unchanged_evidence": [],
            "manual_review_reason": "multiple_edits",
        },
    )

    assert result.disposition == "needs_manual_review"
    assert result.proposed_text == original
    assert result.changed_fraction == 0.0
    assert result.manual_review_reason == "multiple_edits"


def test_patched_review_applies_existing_patch_validator_and_is_immutable() -> None:
    """Apply only exact unique excerpts through the shared local patcher."""
    original = _persona_text("Før ændring")

    result = validate_prose_review(
        original_text=original,
        changed_facts={"marital_status": {"old": "single", "new": "married"}},
        response={
            "disposition": "patched",
            "patches": [{"old_excerpt": "Før ændring", "new_excerpt": "Efter ændring"}],
            "unchanged_evidence": [],
        },
    )

    assert result.disposition == "patched"
    assert result.original_text == original
    assert result.proposed_text == original.replace("Før ændring", "Efter ændring")
    assert result.changed_fraction == max(
        len("Før ændring"), len("Efter ændring")
    ) / len(original)
    assert result.patches[0].old_excerpt == "Før ændring"
    with pytest.raises(FrozenInstanceError):
        setattr(result, "proposed_text", original)


@pytest.mark.parametrize(
    "response",
    [
        {"disposition": "patched", "patches": [], "unchanged_evidence": []},
        {
            "disposition": "patched",
            "patches": [
                {"old_excerpt": "Før ændring", "new_excerpt": "Efter ændring"},
                {"old_excerpt": "syntetisk", "new_excerpt": "kunstigt"},
                {"old_excerpt": "person", "new_excerpt": "borger"},
            ],
            "unchanged_evidence": [],
        },
        {
            "disposition": "patched",
            "patches": [{"old_excerpt": "Før ændring", "new_excerpt": "Efter ændring"}],
            "unchanged_evidence": [
                {"field": "marital_status", "kind": "fact_not_stated", "quote": ""}
            ],
        },
    ],
)
def test_patched_review_fails_closed_for_non_patch_disposition_data(
    response: dict[str, object],
) -> None:
    """Reject empty, excessive, or mixed patch decisions locally."""
    with pytest.raises(ProseReviewError):
        validate_prose_review(
            original_text=_persona_text("Før ændring"),
            changed_facts={"marital_status": {"old": "single", "new": "married"}},
            response=response,
        )


def test_provider_schema_uses_exact_review_dispositions_and_basic_keywords() -> None:
    """Expose a strict proxy schema while keeping local-only constraints local."""
    schema = ProseReviewResponse.provider_json_schema()

    properties = _schema_object(schema["properties"])
    disposition = _schema_object(properties["disposition"])
    manual_review_reason = _schema_object(properties["manual_review_reason"])
    required = schema["required"]
    assert isinstance(required, list)

    assert disposition["enum"] == [
        "patched",
        "unchanged_consistent",
        "needs_manual_review",
    ]
    assert set(required) == set(properties)
    assert manual_review_reason["type"] == ["string", "null"]
    assert manual_review_reason["enum"] == [
        "ambiguous",
        "multiple_edits",
        "sensitive",
        "insufficient_evidence",
        None,
    ]
    assert _schema_keys(schema).isdisjoint(
        {
            "allOf",
            "anyOf",
            "maxItems",
            "maxLength",
            "minItems",
            "minLength",
            "oneOf",
            "pattern",
        }
    )


def _schema_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for child in value.values():
            keys.update(_schema_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(_schema_keys(child))
        return keys
    return set()


def _schema_object(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return t.cast(dict[str, object], value)


@pytest.mark.parametrize(
    "evidence",
    [
        [],
        [{"field": "age", "kind": "new_value_present", "quote": "42 år"}],
        [
            {"field": "age", "kind": "new_value_present", "quote": "43 år"},
            {"field": "job_title", "kind": "fact_not_stated", "quote": ""},
        ],
        [
            {"field": "age", "kind": "fact_not_stated", "quote": "42 år"},
            {"field": "job_title", "kind": "fact_not_stated", "quote": ""},
        ],
    ],
)
def test_unchanged_consistent_rejects_unbounded_or_non_exact_evidence(
    evidence: list[dict[str, str]],
) -> None:
    """Avoid treating unchanged-consistent as an independent semantic proof."""
    with pytest.raises(ProseReviewError):
        validate_prose_review(
            original_text=_persona_text("Personen er 42 år og har stabile rutiner"),
            changed_facts={
                "age": {"old": 41, "new": 42},
                "job_title": {"old": "analytiker", "new": "koordinator"},
            },
            response={
                "disposition": "unchanged_consistent",
                "patches": [],
                "unchanged_evidence": evidence,
            },
        )


def test_unchanged_consistent_requires_bounded_evidence_for_each_fact() -> None:
    """Accept explicit fact-not-stated and exact quote evidence only provisionally."""
    original = _persona_text("Personen er 42 år og har stabile rutiner")

    result = validate_prose_review(
        original_text=original,
        changed_facts={
            "age": {"old": 41, "new": 42},
            "job_title": {"old": "analytiker", "new": "koordinator"},
        },
        response=json.dumps(
            {
                "disposition": "unchanged_consistent",
                "patches": [],
                "unchanged_evidence": [
                    {
                        "field": "age",
                        "kind": "new_value_present",
                        "quote": "Personen er 42 år",
                    },
                    {"field": "job_title", "kind": "fact_not_stated", "quote": ""},
                ],
            }
        ),
    )

    assert result.disposition == "unchanged_consistent"
    assert result.proposed_text == original
    assert result.changed_fraction == 0.0
    assert result.unchanged_consistent_note == UNCHANGED_CONSISTENT_NOTE
    assert result.patches == ()


@pytest.mark.parametrize(
    ("target", "quote"),
    [
        ("skilt", "Hun er skilt"),
        ("skilt", "Hun er fraskilt"),
        ("enkestand", "Han lever i enkestand"),
        ("enkestand", "Hun er enke"),
        ("aldrig gift", "Han er aldrig gift"),
        ("aldrig gift", "Hun er ugift"),
    ],
)
def test_null_detail_context_accepts_conservative_danish_quote(
    target: str, quote: str
) -> None:
    """Accept null detail evidence only when verified Danish category text appears."""
    original = _persona_text(f"{quote} og har stabile rutiner")

    result = validate_prose_review(
        original_text=original,
        changed_facts={"legal_status_detail": {"old": "married", "new": None}},
        response={
            "disposition": "unchanged_consistent",
            "patches": [],
            "unchanged_evidence": [
                {
                    "field": "legal_status_detail",
                    "kind": "new_value_present",
                    "quote": quote,
                }
            ],
        },
        verified_context={
            "legal_status_detail_null_context": {"target_marital_category_da": target}
        },
    )

    assert result.disposition == "unchanged_consistent"
    assert result.unchanged_consistent_note == UNCHANGED_CONSISTENT_NOTE


@pytest.mark.parametrize(
    ("target", "quote"),
    [
        ("skilt", "Hun er separeret"),
        ("skilt", "Hun er ikke skilt"),
        ("enkestand", "Han er ikke enkemand"),
        ("aldrig gift", "Hun er ikke længere ugift"),
        ("aldrig gift", "Hun er tidligere ugift"),
        ("skilt", "Hun er divorced"),
    ],
)
def test_null_detail_context_rejects_wrong_or_unsafe_quotes(
    target: str, quote: str
) -> None:
    """Keep the null-detail exception narrow and Danish-only."""
    with pytest.raises(ProseReviewError):
        validate_prose_review(
            original_text=_persona_text(f"{quote} og har stabile rutiner"),
            changed_facts={"legal_status_detail": {"old": "married", "new": None}},
            response={
                "disposition": "unchanged_consistent",
                "patches": [],
                "unchanged_evidence": [
                    {
                        "field": "legal_status_detail",
                        "kind": "new_value_present",
                        "quote": quote,
                    }
                ],
            },
            verified_context={
                "legal_status_detail_null_context": {
                    "target_marital_category_da": target
                }
            },
        )


def test_null_detail_quote_without_verified_context_is_rejected() -> None:
    """Do not accept quoted evidence for None without runner-verified context."""
    with pytest.raises(ProseReviewError):
        validate_prose_review(
            original_text=_persona_text("Hun er skilt og har stabile rutiner"),
            changed_facts={"legal_status_detail": {"old": "married", "new": None}},
            response={
                "disposition": "unchanged_consistent",
                "patches": [],
                "unchanged_evidence": [
                    {
                        "field": "legal_status_detail",
                        "kind": "new_value_present",
                        "quote": "Hun er skilt",
                    }
                ],
            },
        )


def test_non_null_new_value_present_validation_is_unchanged() -> None:
    """Keep ordinary concrete new-value evidence on the existing path."""
    original = _persona_text("Personen arbejder som koordinator")

    result = validate_prose_review(
        original_text=original,
        changed_facts={"job_title": {"old": "analytiker", "new": "koordinator"}},
        response={
            "disposition": "unchanged_consistent",
            "patches": [],
            "unchanged_evidence": [
                {
                    "field": "job_title",
                    "kind": "new_value_present",
                    "quote": "arbejder som koordinator",
                }
            ],
        },
    )

    assert result.disposition == "unchanged_consistent"
