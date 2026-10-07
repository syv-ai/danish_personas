"""Offline second-pass verification for provisional prose patch proposals."""

from __future__ import annotations

import collections.abc as c
import json
import re
import typing as t

from pydantic import Field, ValidationError

from ..models import StrictModel

VerificationVerdict: t.TypeAlias = t.Literal["accept", "reject", "needs_manual_review"]
VerificationReason: t.TypeAlias = t.Literal[
    "unrelated_content_change",
    "fact_mismatch",
    "unsupported_claim",
    "ambiguity",
    "privacy",
    "missing_fact_evidence",
    "mechanical_mismatch",
    "patch_budget",
    "unsafe_field",
    "invalid_quote_evidence",
]
FactScalar: t.TypeAlias = str | int | float | bool | None
FactValue: t.TypeAlias = FactScalar | list[FactScalar]

_VERDICTS: tuple[str, ...] = ("accept", "reject", "needs_manual_review")
_REASONS: tuple[str, ...] = (
    "unrelated_content_change",
    "fact_mismatch",
    "unsupported_claim",
    "ambiguity",
    "privacy",
    "missing_fact_evidence",
    "mechanical_mismatch",
    "patch_budget",
    "unsafe_field",
    "invalid_quote_evidence",
)
_FORBIDDEN_CHANGED_FACT_FIELDS = frozenset(
    {
        "persona_id",
        "id",
        "gender",
        "partner_gender",
        "sexual_orientation",
        "partner_sexual_orientation",
        "transgender",
        "partner_transgender",
        "variation_in_sex_characteristics",
        "partner_variation_in_sex_characteristics",
        "same_sex_partner_target",
        "partner_same_sex_target",
        "identity_sidecar",
    }
)
_MAX_FACTS = 20
_MAX_FIELD_LENGTH = 80
_MAX_FACT_VALUE_LENGTH = 120
_MAX_QUOTE_LENGTH = 200
_MAX_PATCH_LENGTH = 120
_MAX_PATCHES = 2
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


class ProsePatchVerificationError(ValueError):
    """Raised when a prose patch proposal fails local verification."""


class ProsePatchExcerpt(StrictModel):
    """One exact replacement proposed by the first-pass patcher."""

    old_excerpt: str = Field(min_length=1, max_length=_MAX_PATCH_LENGTH)
    new_excerpt: str = Field(min_length=1, max_length=_MAX_PATCH_LENGTH)


class ProsePatchFactEvidence(StrictModel):
    """Exact quote evidence for one changed structured fact."""

    field: str = Field(min_length=1, max_length=_MAX_FIELD_LENGTH)
    original_quote: str | None = Field(default=None, max_length=_MAX_QUOTE_LENGTH)
    proposed_quote: str | None = Field(default=None, max_length=_MAX_QUOTE_LENGTH)


class ProsePatchSecondReview(StrictModel):
    """Second AI reviewer verdict, validated again before use."""

    verdict: VerificationVerdict
    reasons: list[VerificationReason] = Field(default_factory=list, max_length=10)
    fact_evidence: list[ProsePatchFactEvidence] = Field(
        default_factory=list, max_length=_MAX_FACTS
    )

    @staticmethod
    def provider_json_schema() -> dict[str, object]:
        """Return a strict proxy-compatible schema using basic JSON keywords.

        Returns:
            JSON schema for a second-pass review response. Cross-field rules,
            quote exactness, and evidence completeness are enforced locally.
        """
        return {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": list(_VERDICTS)},
                "reasons": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(_REASONS)},
                },
                "fact_evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "original_quote": {"type": ["string", "null"]},
                            "proposed_quote": {"type": ["string", "null"]},
                        },
                        "required": ["field", "original_quote", "proposed_quote"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["verdict", "reasons", "fact_evidence"],
            "additionalProperties": False,
        }


class ProsePatchVerificationResult(StrictModel):
    """Local verification result without row identifiers or provider metadata."""

    accepted: bool
    review_verdict: VerificationVerdict
    reasons: list[VerificationReason]
    changed_fraction: float
    changed_characters: int
    patch_count: int
    original_checkpoint_sha256: str
    provisional: bool = True
    requires_later_release_gate: bool = True
    fact_evidence: list[ProsePatchFactEvidence] = Field(default_factory=list)


def verify_prose_patch_proposal(
    *,
    original_text: str,
    changed_facts: c.Mapping[str, c.Mapping[str, object]],
    proposed_text: str,
    patches: c.Sequence[ProsePatchExcerpt | c.Mapping[str, object]],
    second_review: str | c.Mapping[str, object] | ProsePatchSecondReview,
    original_checkpoint_sha256: str | None = None,
    original_checkpoint_sha: str | None = None,
) -> ProsePatchVerificationResult:
    """Verify a first-pass patched proposal with a second-pass review.

    The function is offline and performs no provider, ledger, file-system, or CLI
    activity. A returned acceptance is only provisional: a later privacy, source,
    and semantic release gate is still required.

    Args:
        original_text:
            Original Mistral persona prose.
        changed_facts:
            Candidate structured fact deltas keyed by field. Each value must contain
            exactly ``old`` and ``new``.
        proposed_text:
            Candidate prose after applying the first-pass patches.
        patches:
            Exact first-pass replacements, at most two.
        second_review:
            Provider JSON text, decoded object, or validated second-pass verdict.
        original_checkpoint_sha256 (optional):
            SHA-256 of the original patch checkpoint. Defaults to None.
        original_checkpoint_sha (optional):
            Backwards-compatible alias for ``original_checkpoint_sha256``. Defaults
            to None.

    Returns:
        Verification result. ``accepted=True`` never means release-ready.
    """
    checkpoint_sha = _checkpoint_sha(
        original_checkpoint_sha256=original_checkpoint_sha256,
        original_checkpoint_sha=original_checkpoint_sha,
    )
    facts = _validate_changed_facts(changed_facts=changed_facts)
    parsed_patches = _parse_patches(patches=patches)
    changed_characters = _verify_reapplied_text(
        original_text=original_text, proposed_text=proposed_text, patches=parsed_patches
    )
    changed_fraction = changed_characters / len(original_text)
    review = _parse_second_review(second_review=second_review)
    evidence = _validate_quote_evidence(
        original_text=original_text,
        proposed_text=proposed_text,
        changed_facts=facts,
        evidence=review.fact_evidence,
    )
    accepted = _accepts_review(review=review, changed_facts=facts, evidence=evidence)
    return ProsePatchVerificationResult(
        accepted=accepted,
        review_verdict=review.verdict,
        reasons=list(review.reasons),
        changed_fraction=changed_fraction,
        changed_characters=changed_characters,
        patch_count=len(parsed_patches),
        original_checkpoint_sha256=checkpoint_sha,
        fact_evidence=list(evidence),
    )


def verify_prose_patch(
    *,
    original_text: str,
    changed_facts: c.Mapping[str, c.Mapping[str, object]],
    proposed_text: str,
    patches: c.Sequence[ProsePatchExcerpt | c.Mapping[str, object]],
    second_review: str | c.Mapping[str, object] | ProsePatchSecondReview,
    original_checkpoint_sha256: str | None = None,
    original_checkpoint_sha: str | None = None,
) -> ProsePatchVerificationResult:
    """Alias for ``verify_prose_patch_proposal``.

    Returns:
        Verification result. ``accepted=True`` never means release-ready.
    """
    return verify_prose_patch_proposal(
        original_text=original_text,
        changed_facts=changed_facts,
        proposed_text=proposed_text,
        patches=patches,
        second_review=second_review,
        original_checkpoint_sha256=original_checkpoint_sha256,
        original_checkpoint_sha=original_checkpoint_sha,
    )


def validate_prose_patch_verification(
    *,
    original_text: str,
    changed_facts: c.Mapping[str, c.Mapping[str, object]],
    proposed_text: str,
    patches: c.Sequence[ProsePatchExcerpt | c.Mapping[str, object]],
    second_review: str | c.Mapping[str, object] | ProsePatchSecondReview,
    original_checkpoint_sha256: str | None = None,
    original_checkpoint_sha: str | None = None,
) -> ProsePatchVerificationResult:
    """Alias for callers that use validation terminology.

    Returns:
        Verification result. ``accepted=True`` never means release-ready.
    """
    return verify_prose_patch_proposal(
        original_text=original_text,
        changed_facts=changed_facts,
        proposed_text=proposed_text,
        patches=patches,
        second_review=second_review,
        original_checkpoint_sha256=original_checkpoint_sha256,
        original_checkpoint_sha=original_checkpoint_sha,
    )


def _checkpoint_sha(
    *, original_checkpoint_sha256: str | None, original_checkpoint_sha: str | None
) -> str:
    if original_checkpoint_sha256 is not None and original_checkpoint_sha is not None:
        raise ProsePatchVerificationError("Checkpoint SHA must be supplied once")
    checkpoint_sha = original_checkpoint_sha256 or original_checkpoint_sha
    if checkpoint_sha is None or not _SHA256_RE.fullmatch(checkpoint_sha):
        raise ProsePatchVerificationError("Checkpoint SHA must be a SHA-256 digest")
    return checkpoint_sha.lower()


def _validate_changed_facts(
    *, changed_facts: c.Mapping[str, c.Mapping[str, object]]
) -> dict[str, dict[str, object]]:
    if not changed_facts or len(changed_facts) > _MAX_FACTS:
        raise ProsePatchVerificationError("Changed facts must be non-empty and bounded")
    facts: dict[str, dict[str, object]] = {}
    for field, pair in changed_facts.items():
        if not isinstance(field, str) or not 1 <= len(field) <= _MAX_FIELD_LENGTH:
            raise ProsePatchVerificationError("Changed fact field is outside bounds")
        if field in _FORBIDDEN_CHANGED_FACT_FIELDS:
            raise ProsePatchVerificationError("Changed facts contain a forbidden field")
        if field.endswith("_id") or field.endswith("_sidecar"):
            raise ProsePatchVerificationError("Changed facts contain a forbidden field")
        if not isinstance(pair, c.Mapping) or set(pair) != {"old", "new"}:
            raise ProsePatchVerificationError(
                "Each changed fact must contain old and new"
            )
        old_value = pair["old"]
        new_value = pair["new"]
        if old_value == new_value:
            raise ProsePatchVerificationError("Changed facts must actually change")
        if not _safe_fact_value(old_value) or not _safe_fact_value(new_value):
            raise ProsePatchVerificationError("Changed fact value is outside bounds")
        facts[field] = {"old": old_value, "new": new_value}
    return facts


def _parse_patches(
    *, patches: c.Sequence[ProsePatchExcerpt | c.Mapping[str, object]]
) -> list[ProsePatchExcerpt]:
    if not 1 <= len(patches) <= _MAX_PATCHES:
        raise ProsePatchVerificationError(
            "Patched proposals must contain one or two patches"
        )
    parsed: list[ProsePatchExcerpt] = []
    try:
        for patch in patches:
            parsed.append(
                patch
                if isinstance(patch, ProsePatchExcerpt)
                else ProsePatchExcerpt.model_validate(patch)
            )
    except (ValidationError, TypeError, ValueError) as error:
        raise ProsePatchVerificationError("Invalid exact prose patch") from error
    return parsed


def _verify_reapplied_text(
    *, original_text: str, proposed_text: str, patches: list[ProsePatchExcerpt]
) -> int:
    if not original_text:
        raise ProsePatchVerificationError("Original text must be non-empty")
    spans = _find_patch_spans(original_text=original_text, patches=patches)
    changed_characters = _changed_characters(spans=spans)
    if (
        changed_characters > _MAX_PATCH_LENGTH
        or changed_characters > len(original_text) * 0.2
    ):
        raise ProsePatchVerificationError("Patch exceeds the changed-text budget")
    reapplied = _apply_spans(original_text=original_text, spans=spans)
    if reapplied != proposed_text:
        raise ProsePatchVerificationError("Proposed text does not match exact patches")
    return changed_characters


def _parse_second_review(
    *, second_review: str | c.Mapping[str, object] | ProsePatchSecondReview
) -> ProsePatchSecondReview:
    if isinstance(second_review, ProsePatchSecondReview):
        return second_review
    try:
        return (
            ProsePatchSecondReview.model_validate_json(second_review)
            if isinstance(second_review, str)
            else ProsePatchSecondReview.model_validate(second_review)
        )
    except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise ProsePatchVerificationError(
            "Invalid second-pass review response"
        ) from error


def _validate_quote_evidence(
    *,
    original_text: str,
    proposed_text: str,
    changed_facts: dict[str, dict[str, object]],
    evidence: list[ProsePatchFactEvidence],
) -> tuple[ProsePatchFactEvidence, ...]:
    seen: set[str] = set()
    validated: list[ProsePatchFactEvidence] = []
    for item in evidence:
        if item.field in seen:
            raise ProsePatchVerificationError("Duplicate fact evidence is not allowed")
        seen.add(item.field)
        if item.field not in changed_facts:
            raise ProsePatchVerificationError(
                "Fact evidence references an unknown field"
            )
        if item.original_quote is None and item.proposed_quote is None:
            raise ProsePatchVerificationError("Fact evidence must include a quote")
        if item.original_quote == "" or item.proposed_quote == "":
            raise ProsePatchVerificationError("Fact evidence quote must be non-empty")
        if item.original_quote is not None and item.original_quote not in original_text:
            raise ProsePatchVerificationError("Original quote evidence is not exact")
        if item.proposed_quote is not None and item.proposed_quote not in proposed_text:
            raise ProsePatchVerificationError("Proposed quote evidence is not exact")
        validated.append(item)
    return tuple(validated)


def _accepts_review(
    *,
    review: ProsePatchSecondReview,
    changed_facts: dict[str, dict[str, object]],
    evidence: tuple[ProsePatchFactEvidence, ...],
) -> bool:
    if review.verdict != "accept":
        if not review.reasons:
            raise ProsePatchVerificationError("Non-accept verdicts require a reason")
        return False
    if review.reasons:
        raise ProsePatchVerificationError("Accepted verdicts must not include reasons")
    evidence_by_field = {item.field: item for item in evidence}
    if set(evidence_by_field) != set(changed_facts):
        raise ProsePatchVerificationError("Accepted verdict lacks per-fact evidence")
    if any(item.proposed_quote is None for item in evidence_by_field.values()):
        raise ProsePatchVerificationError(
            "Accepted verdict lacks proposed quote evidence"
        )
    return True


def _find_patch_spans(
    *, original_text: str, patches: list[ProsePatchExcerpt]
) -> list[tuple[int, int, ProsePatchExcerpt]]:
    spans: list[tuple[int, int, ProsePatchExcerpt]] = []
    seen: set[str] = set()
    for patch in patches:
        if patch.old_excerpt == patch.new_excerpt:
            raise ProsePatchVerificationError("Identical replacements are not allowed")
        if patch.old_excerpt in seen:
            raise ProsePatchVerificationError("Duplicate old excerpts are not allowed")
        seen.add(patch.old_excerpt)
        occurrences = _occurrences(text=original_text, needle=patch.old_excerpt)
        if len(occurrences) != 1:
            raise ProsePatchVerificationError(
                "Each old excerpt must occur exactly once"
            )
        start = occurrences[0]
        spans.append((start, start + len(patch.old_excerpt), patch))
    spans.sort(key=lambda span: span[0])
    if any(current[0] < previous[1] for previous, current in zip(spans, spans[1:])):
        raise ProsePatchVerificationError("Overlapping patches are not allowed")
    return spans


def _changed_characters(*, spans: list[tuple[int, int, ProsePatchExcerpt]]) -> int:
    return sum(max(end - start, len(patch.new_excerpt)) for start, end, patch in spans)


def _apply_spans(
    *, original_text: str, spans: list[tuple[int, int, ProsePatchExcerpt]]
) -> str:
    parts: list[str] = []
    cursor = 0
    for start, end, patch in spans:
        parts.append(original_text[cursor:start])
        parts.append(patch.new_excerpt)
        cursor = end
    parts.append(original_text[cursor:])
    return "".join(parts)


def _occurrences(*, text: str, needle: str) -> list[int]:
    starts: list[int] = []
    position = text.find(needle)
    while position != -1:
        starts.append(position)
        position = text.find(needle, position + 1)
    return starts


def _safe_fact_value(value: object) -> bool:
    if value is None or isinstance(value, bool | int | float):
        return True
    if isinstance(value, str):
        return len(value) <= _MAX_FACT_VALUE_LENGTH
    if isinstance(value, list):
        return all(
            isinstance(item, str) and len(item) <= _MAX_FACT_VALUE_LENGTH
            for item in value
        )
    return False
