"""Offline verification of provisional prose patch proposals."""

from __future__ import annotations

import pytest

from danish_personas.release.prose_patch_verification import (
    ProsePatchSecondReview,
    ProsePatchVerificationError,
    verify_prose_patch_proposal,
)

_SHA = "a" * 64


def test_accepts_provisional_second_pass_with_exact_patch_evidence() -> None:
    """Accept only after exact local patch replay and exact quote evidence."""
    original = _persona_text("Hun er ugift og bruger aftenerne på at læse.")
    proposed = original.replace("Hun er ugift", "Hun er gift")

    result = verify_prose_patch_proposal(
        original_text=original,
        changed_facts={"marital_status": {"old": "ugift", "new": "gift"}},
        proposed_text=proposed,
        patches=[{"old_excerpt": "Hun er ugift", "new_excerpt": "Hun er gift"}],
        second_review={
            "verdict": "accept",
            "reasons": [],
            "fact_evidence": [
                {
                    "field": "marital_status",
                    "original_quote": "Hun er ugift",
                    "proposed_quote": "Hun er gift",
                }
            ],
        },
        original_checkpoint_sha256=_SHA,
    )

    assert result.accepted is True
    assert result.provisional is True
    assert result.requires_later_release_gate is True
    assert result.changed_characters == len("Hun er ugift")
    assert result.patch_count == 1
    assert result.original_checkpoint_sha256 == _SHA


def test_rejects_forged_proposed_text_with_extra_edits() -> None:
    """The proposed prose must equal reapplying only the exact patches."""
    original = _persona_text("Hun er ugift og bruger aftenerne på at læse.")
    proposed = original.replace("Hun er ugift", "Hun er gift").replace("læse", "bokse")

    with pytest.raises(ProsePatchVerificationError, match="exact patches"):
        verify_prose_patch_proposal(
            original_text=original,
            changed_facts={"marital_status": {"old": "ugift", "new": "gift"}},
            proposed_text=proposed,
            patches=[{"old_excerpt": "Hun er ugift", "new_excerpt": "Hun er gift"}],
            second_review={
                "verdict": "accept",
                "reasons": [],
                "fact_evidence": [
                    {
                        "field": "marital_status",
                        "original_quote": "Hun er ugift",
                        "proposed_quote": "Hun er gift",
                    }
                ],
            },
            original_checkpoint_sha256=_SHA,
        )


def test_rejects_accept_verdict_without_per_fact_evidence() -> None:
    """A reviewer saying yes is insufficient without exact evidence per fact."""
    original = _persona_text("Hun er ugift og bruger aftenerne på at læse.")
    proposed = original.replace("Hun er ugift", "Hun er gift")

    with pytest.raises(ProsePatchVerificationError, match="per-fact evidence"):
        verify_prose_patch_proposal(
            original_text=original,
            changed_facts={"marital_status": {"old": "ugift", "new": "gift"}},
            proposed_text=proposed,
            patches=[{"old_excerpt": "Hun er ugift", "new_excerpt": "Hun er gift"}],
            second_review={"verdict": "accept", "reasons": [], "fact_evidence": []},
            original_checkpoint_sha256=_SHA,
        )


def test_rejects_unsafe_identity_fields_without_reporting_raw_ids() -> None:
    """Identity sidecar and raw-id facts cannot enter patch acceptance reports."""
    raw_id = "persona-raw-12345"
    original = _persona_text("Hun er ugift og bruger aftenerne på at læse.")
    proposed = original.replace("Hun er ugift", "Hun er gift")

    with pytest.raises(ProsePatchVerificationError) as error:
        verify_prose_patch_proposal(
            original_text=original,
            changed_facts={"persona_id": {"old": raw_id, "new": "other-raw-id"}},
            proposed_text=proposed,
            patches=[{"old_excerpt": "Hun er ugift", "new_excerpt": "Hun er gift"}],
            second_review={
                "verdict": "reject",
                "reasons": ["privacy"],
                "fact_evidence": [],
            },
            original_checkpoint_sha256=_SHA,
        )

    assert raw_id not in str(error.value)


@pytest.mark.parametrize(
    "field", ["gender", "partner_gender", "same_sex_partner_target"]
)
def test_rejects_sidecar_and_forbidden_identity_fact_fields(field: str) -> None:
    """Do not second-pass audit patches for forbidden identity dimensions."""
    original = _persona_text("Hun er ugift og bruger aftenerne på at læse.")
    proposed = original.replace("Hun er ugift", "Hun er gift")

    with pytest.raises(ProsePatchVerificationError, match="forbidden field"):
        verify_prose_patch_proposal(
            original_text=original,
            changed_facts={field: {"old": "old", "new": "new"}},
            proposed_text=proposed,
            patches=[{"old_excerpt": "Hun er ugift", "new_excerpt": "Hun er gift"}],
            second_review={
                "verdict": "reject",
                "reasons": ["privacy"],
                "fact_evidence": [],
            },
            original_checkpoint_sha256=_SHA,
        )


def test_rejection_verdict_is_bounded_and_not_accepted() -> None:
    """Reject and manual-review verdicts remain separate from local acceptance."""
    original = _persona_text("Hun er ugift og bruger aftenerne på at læse.")
    proposed = original.replace("Hun er ugift", "Hun er gift")

    result = verify_prose_patch_proposal(
        original_text=original,
        changed_facts={"marital_status": {"old": "ugift", "new": "gift"}},
        proposed_text=proposed,
        patches=[{"old_excerpt": "Hun er ugift", "new_excerpt": "Hun er gift"}],
        second_review={
            "verdict": "reject",
            "reasons": ["fact_mismatch"],
            "fact_evidence": [
                {
                    "field": "marital_status",
                    "original_quote": "Hun er ugift",
                    "proposed_quote": "Hun er gift",
                }
            ],
        },
        original_checkpoint_sha256=_SHA,
    )

    assert result.accepted is False
    assert result.review_verdict == "reject"
    assert result.reasons == ["fact_mismatch"]


def test_provider_schema_uses_basic_proxy_compatible_keywords() -> None:
    """Expose a basic strict schema while keeping local rules out of the schema."""
    schema = ProsePatchSecondReview.provider_json_schema()

    assert schema["additionalProperties"] is False
    assert schema["required"] == ["verdict", "reasons", "fact_evidence"]
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert properties["verdict"] == {
        "type": "string",
        "enum": ["accept", "reject", "needs_manual_review"],
    }
    assert "fact_mismatch" in properties["reasons"]["items"]["enum"]


def _persona_text(sentence: str) -> str:
    filler = (
        " Personen beskrives med rolige hverdagsrutiner, lokale fritidsvaner og "
        "forsigtige planer for de kommende måneder."
    )
    return f"{sentence}{filler * 4}"
