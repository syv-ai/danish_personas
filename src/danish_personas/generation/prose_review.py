"""Local validation for bounded prose-review decisions."""

from __future__ import annotations

import json
import re
import typing as t
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import Field, ValidationError

from ..models import StrictModel
from .prose_patch import ProsePatch, ProsePatchError, apply_patches

ReviewDisposition: t.TypeAlias = t.Literal[
    "patched", "unchanged_consistent", "needs_manual_review"
]
ManualReviewReason: t.TypeAlias = t.Literal[
    "ambiguous", "multiple_edits", "sensitive", "insufficient_evidence"
]
EvidenceKind: t.TypeAlias = t.Literal["fact_not_stated", "new_value_present"]
VerifiedContext: t.TypeAlias = Mapping[str, object]

REVIEW_MODEL = "gpt-6-luna"
_DISPOSITIONS: tuple[str, ...] = (
    "patched",
    "unchanged_consistent",
    "needs_manual_review",
)
_MANUAL_REVIEW_REASONS: tuple[str, ...] = (
    "ambiguous",
    "multiple_edits",
    "sensitive",
    "insufficient_evidence",
)
_EVIDENCE_KINDS: tuple[str, ...] = ("fact_not_stated", "new_value_present")
_MAX_FACTS = 20
_MAX_FIELD_LENGTH = 80
_MAX_FACT_VALUE_LENGTH = 120
_MAX_QUOTE_LENGTH = 160
_NULL_DETAIL_CONTEXT_KEY = "legal_status_detail_null_context"
_NULL_DETAIL_TARGET_KEY = "target_marital_category_da"
_NULL_DETAIL_TARGET_SYNONYMS: dict[str, tuple[str, ...]] = {
    "skilt": ("skilt", "fraskilt"),
    "enkestand": ("enke", "enkemand", "enkestand"),
    "aldrig gift": ("aldrig gift", "ugift"),
}
_NEGATED_CONTEXT_RE = re.compile(
    r"(?:^|\W)(?:ikke|ingen|hverken|aldrig)\W*$", re.IGNORECASE
)
_NEVER_MARRIED_REJECT_RE = re.compile(
    r"(?:^|\W)(?:ikke|længere|tidligere|før|førhen|var|blev)\W+ugift(?:\W|$)",
    re.IGNORECASE,
)
UNCHANGED_CONSISTENT_NOTE = "provisional_classifier_not_independently_certified"


class ProseReviewEvidence(StrictModel):
    """Bounded evidence for an unchanged-consistent review decision."""

    field: str = Field(min_length=1, max_length=_MAX_FIELD_LENGTH)
    kind: EvidenceKind
    quote: str = Field(max_length=_MAX_QUOTE_LENGTH)


class ProseReviewResponse(StrictModel):
    """Provider response for local prose-review validation."""

    disposition: ReviewDisposition
    patches: list[ProsePatch] = Field(default_factory=list, max_length=2)
    unchanged_evidence: list[ProseReviewEvidence] = Field(
        default_factory=list, max_length=_MAX_FACTS
    )
    manual_review_reason: ManualReviewReason | None = None

    @staticmethod
    def provider_json_schema() -> dict[str, object]:
        """Return a strict proxy-compatible schema with only basic keywords.

        Returns:
            JSON schema for the provider response format. Length, count, edit-budget,
            and cross-field rules are deliberately enforced locally instead.
        """
        return {
            "type": "object",
            "properties": {
                "disposition": {"type": "string", "enum": list(_DISPOSITIONS)},
                "patches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_excerpt": {"type": "string"},
                            "new_excerpt": {"type": "string"},
                        },
                        "required": ["old_excerpt", "new_excerpt"],
                        "additionalProperties": False,
                    },
                },
                "unchanged_evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "kind": {"type": "string", "enum": list(_EVIDENCE_KINDS)},
                            "quote": {"type": "string"},
                        },
                        "required": ["field", "kind", "quote"],
                        "additionalProperties": False,
                    },
                },
                "manual_review_reason": {
                    "type": ["string", "null"],
                    "enum": [*_MANUAL_REVIEW_REASONS, None],
                },
            },
            "required": [
                "disposition",
                "patches",
                "unchanged_evidence",
                "manual_review_reason",
            ],
            "additionalProperties": False,
        }


def _parse_response(response: str | Mapping[str, object]) -> ProseReviewResponse:
    try:
        parsed = (
            ProseReviewResponse.model_validate_json(response)
            if isinstance(response, str)
            else ProseReviewResponse.model_validate(response)
        )
    except ValidationError, ValueError, TypeError, json.JSONDecodeError:
        raise ProseReviewError("Invalid prose-review response") from None
    return parsed


class ProseReviewError(ValueError):
    """Raised when a prose-review response fails local validation."""


def _validate_changed_facts(
    changed_facts: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    if not changed_facts or len(changed_facts) > _MAX_FACTS:
        raise ProseReviewError("Changed facts must be a non-empty bounded mapping")
    facts: dict[str, dict[str, object]] = {}
    for field, pair in changed_facts.items():
        if not isinstance(field, str) or not 1 <= len(field) <= _MAX_FIELD_LENGTH:
            raise ProseReviewError("Changed fact field is outside the local bounds")
        if not isinstance(pair, Mapping) or set(pair) != {"old", "new"}:
            raise ProseReviewError("Each changed fact must contain old and new")
        old_value = pair["old"]
        new_value = pair["new"]
        if old_value == new_value:
            raise ProseReviewError("Changed facts must actually change")
        if not _safe_fact_value(old_value) or not _safe_fact_value(new_value):
            raise ProseReviewError("Changed fact value is outside the local bounds")
        facts[field] = {"old": old_value, "new": new_value}
    return facts


def _safe_fact_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return abs(value) <= 150
    if isinstance(value, str):
        return len(value) <= _MAX_FACT_VALUE_LENGTH
    if isinstance(value, list):
        return len(value) <= _MAX_FACTS and all(
            isinstance(item, str) and len(item) <= _MAX_FACT_VALUE_LENGTH
            for item in value
        )
    return False


def _validate_verified_context(
    *,
    verified_context: VerifiedContext | None,
    changed_facts: dict[str, dict[str, object]],
) -> dict[str, str]:
    if verified_context is None:
        return {}
    if set(verified_context) != {_NULL_DETAIL_CONTEXT_KEY}:
        raise ProseReviewError("Verified context is outside the local allowlist")
    detail_change = changed_facts.get("legal_status_detail")
    if detail_change is None or detail_change.get("new") is not None:
        raise ProseReviewError("Verified context requires a null legal detail change")
    detail_context = verified_context[_NULL_DETAIL_CONTEXT_KEY]
    if not isinstance(detail_context, Mapping) or set(detail_context) != {
        _NULL_DETAIL_TARGET_KEY
    }:
        raise ProseReviewError("Verified context is malformed")
    target = detail_context[_NULL_DETAIL_TARGET_KEY]
    if not isinstance(target, str) or target not in _NULL_DETAIL_TARGET_SYNONYMS:
        raise ProseReviewError("Verified context target is outside the allowlist")
    return {"legal_status_detail": target}


@dataclass(frozen=True)
class ProseReviewPatchResult:
    """Immutable accepted patch evidence."""

    old_excerpt: str
    new_excerpt: str


def _changed_fraction(old_text: str, patches: list[ProsePatch]) -> float:
    return sum(
        max(len(patch.old_excerpt), len(patch.new_excerpt)) for patch in patches
    ) / len(old_text)


@dataclass(frozen=True)
class ProseReviewEvidenceResult:
    """Immutable accepted unchanged-consistent evidence."""

    field: str
    kind: EvidenceKind
    quote: str


@dataclass(frozen=True)
class ProseReviewResult:
    """Immutable local review result with no retained invalid completion text."""

    disposition: ReviewDisposition
    original_text: str
    proposed_text: str
    changed_fraction: float
    patches: tuple[ProseReviewPatchResult, ...]
    unchanged_evidence: tuple[ProseReviewEvidenceResult, ...]
    manual_review_reason: ManualReviewReason | None
    unchanged_consistent_note: str | None


def validate_prose_review(
    *,
    original_text: str,
    changed_facts: Mapping[str, Mapping[str, object]],
    response: str | Mapping[str, object],
    verified_context: VerifiedContext | None = None,
) -> ProseReviewResult:
    """Validate one gpt-6-luna prose-review response locally.

    Args:
        original_text:
            Original Mistral persona text.
        changed_facts:
            Verified source/candidate fact deltas keyed by field. Each value must
            contain ``old`` and ``new``.
        response:
            Provider JSON text or decoded object. Invalid responses are rejected
            without returning or embedding the raw completion text.
        verified_context (optional):
            Local runner-verified contextual evidence for allowlisted transitions that
            are not represented by a concrete new fact value. Defaults to None.

    Returns:
        Immutable result containing the original text, proposed text, and changed
        fraction. ``unchanged_consistent`` is only a provisional classifier.
    """
    facts = _validate_changed_facts(changed_facts)
    context = _validate_verified_context(
        verified_context=verified_context, changed_facts=facts
    )
    review = _parse_response(response)
    if review.disposition == "patched":
        return _validated_patched(original_text=original_text, review=review)
    if review.disposition == "unchanged_consistent":
        return _validated_unchanged(
            original_text=original_text,
            changed_facts=facts,
            review=review,
            verified_context=context,
        )
    return _validated_manual_review(original_text=original_text, review=review)


def _validated_manual_review(
    *, original_text: str, review: ProseReviewResponse
) -> ProseReviewResult:
    if review.disposition != "needs_manual_review":
        raise ProseReviewError("Unknown prose-review disposition")
    if review.patches:
        raise ProseReviewError("Manual review must not include patches")
    if review.unchanged_evidence:
        raise ProseReviewError("Manual review must not include unchanged evidence")
    if review.manual_review_reason is None:
        raise ProseReviewError("Manual review requires a bounded reason")
    return ProseReviewResult(
        disposition="needs_manual_review",
        original_text=original_text,
        proposed_text=original_text,
        changed_fraction=0.0,
        patches=(),
        unchanged_evidence=(),
        manual_review_reason=review.manual_review_reason,
        unchanged_consistent_note=None,
    )


def _validated_patched(
    *, original_text: str, review: ProseReviewResponse
) -> ProseReviewResult:
    if review.unchanged_evidence:
        raise ProseReviewError("Patched review must not include unchanged evidence")
    if review.manual_review_reason is not None:
        raise ProseReviewError("Patched review must not include a manual reason")
    patch_response = {"patches": [patch.model_dump() for patch in review.patches]}
    try:
        proposed_text = apply_patches(original_text, patch_response)
    except ProsePatchError as error:
        raise ProseReviewError("Patch failed local prose validation") from error
    changed_fraction = _changed_fraction(original_text, review.patches)
    return ProseReviewResult(
        disposition="patched",
        original_text=original_text,
        proposed_text=proposed_text,
        changed_fraction=changed_fraction,
        patches=tuple(
            ProseReviewPatchResult(
                old_excerpt=patch.old_excerpt, new_excerpt=patch.new_excerpt
            )
            for patch in review.patches
        ),
        unchanged_evidence=(),
        manual_review_reason=None,
        unchanged_consistent_note=None,
    )


def _validated_unchanged(
    *,
    original_text: str,
    changed_facts: dict[str, dict[str, object]],
    review: ProseReviewResponse,
    verified_context: dict[str, str],
) -> ProseReviewResult:
    if review.patches:
        raise ProseReviewError("Unchanged-consistent review must not include patches")
    if review.manual_review_reason is not None:
        raise ProseReviewError("Unchanged-consistent review must not include a reason")
    _validate_unchanged_evidence(
        original_text=original_text,
        changed_facts=changed_facts,
        evidence=review.unchanged_evidence,
        verified_context=verified_context,
    )
    return ProseReviewResult(
        disposition="unchanged_consistent",
        original_text=original_text,
        proposed_text=original_text,
        changed_fraction=0.0,
        patches=(),
        unchanged_evidence=tuple(
            ProseReviewEvidenceResult(
                field=item.field, kind=item.kind, quote=item.quote
            )
            for item in review.unchanged_evidence
        ),
        manual_review_reason=None,
        unchanged_consistent_note=UNCHANGED_CONSISTENT_NOTE,
    )


def _validate_unchanged_evidence(
    *,
    original_text: str,
    changed_facts: dict[str, dict[str, object]],
    evidence: list[ProseReviewEvidence],
    verified_context: dict[str, str],
) -> None:
    if not evidence:
        raise ProseReviewError("Unchanged-consistent review requires evidence")
    fields = [item.field for item in evidence]
    if len(fields) != len(set(fields)):
        raise ProseReviewError("Unchanged evidence fields must be unique")
    if set(fields) != set(changed_facts):
        raise ProseReviewError("Unchanged evidence must cover each changed fact")
    for item in evidence:
        new_value = changed_facts[item.field]["new"]
        if item.kind == "fact_not_stated":
            if item.quote:
                raise ProseReviewError("Fact-not-stated evidence must not quote text")
            continue
        if not item.quote or item.quote not in original_text:
            raise ProseReviewError("New-value evidence must quote the original text")
        if not _quote_shows_new_value(
            quote=item.quote, new_value=new_value
        ) and not _quote_shows_verified_null_detail(
            item=item, new_value=new_value, verified_context=verified_context
        ):
            raise ProseReviewError("Quoted evidence must include the new value")


def _quote_shows_verified_null_detail(
    *, item: ProseReviewEvidence, new_value: object, verified_context: dict[str, str]
) -> bool:
    if item.field != "legal_status_detail" or new_value is not None:
        return False
    target = verified_context.get(item.field)
    if target is None:
        return False
    return _quote_contains_contextual_category(quote=item.quote, target=target)


def _quote_shows_new_value(*, quote: str, new_value: object) -> bool:
    normalised_quote = quote.casefold()
    if isinstance(new_value, int) and not isinstance(new_value, bool):
        return str(new_value) in normalised_quote
    if isinstance(new_value, str):
        return new_value.casefold() in normalised_quote
    if isinstance(new_value, list):
        return any(
            isinstance(item, str) and item.casefold() in normalised_quote
            for item in new_value
        )
    return False


def _quote_contains_contextual_category(*, quote: str, target: str) -> bool:
    normalised_quote = quote.casefold()
    if target == "skilt" and re.search(
        r"(?:^|\W)separeret(?:\W|$)", normalised_quote
    ):
        return False
    if target == "aldrig gift" and _NEVER_MARRIED_REJECT_RE.search(
        normalised_quote
    ):
        return False
    return any(
        _quote_contains_unnegated_synonym(
            normalised_quote=normalised_quote, synonym=synonym
        )
        for synonym in _NULL_DETAIL_TARGET_SYNONYMS[target]
    )


def _quote_contains_unnegated_synonym(*, normalised_quote: str, synonym: str) -> bool:
    pattern = re.compile(rf"(?:^|\W){re.escape(synonym.casefold())}(?:\W|$)")
    return any(
        not _NEGATED_CONTEXT_RE.search(
            normalised_quote[max(0, match.start() - 24) : match.start()]
        )
        for match in pattern.finditer(normalised_quote)
    )


__all__ = [
    "UNCHANGED_CONSISTENT_NOTE",
    "ProseReviewError",
    "ProseReviewEvidence",
    "ProseReviewEvidenceResult",
    "ProseReviewPatchResult",
    "ProseReviewResponse",
    "ProseReviewResult",
    "REVIEW_MODEL",
    "validate_prose_review",
]
