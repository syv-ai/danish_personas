"""Strict, local-only patching for generated persona prose."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping

from pydantic import Field, ValidationError

from ..models import StrictModel


class ProsePatch(StrictModel):
    """One bounded replacement in existing prose."""

    old_excerpt: str = Field(min_length=1, max_length=120)
    new_excerpt: str = Field(min_length=1, max_length=120)


class ProsePatchResponse(StrictModel):
    """Provider response containing a small set of prose edits."""

    patches: list[ProsePatch] = Field(max_length=2)

    @staticmethod
    def provider_json_schema() -> dict[str, object]:
        """Return a proxy-compatible schema without provider-fragile constraints."""
        return {
            "type": "object",
            "properties": {
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
                }
            },
            "required": ["patches"],
            "additionalProperties": False,
        }


def apply_patches(old_text: str, response: str | Mapping[str, object]) -> str:
    """Apply validated local replacements while preserving untouched text.

    ``response`` is JSON text from a provider or its decoded object. Ambiguity or
    any policy violation fails closed.

    Returns:
        The patched prose.

    Raises:
        ProsePatchError: If parsing, uniqueness, boundary, edit-budget, paragraph,
            overlap, or final-length validation fails.
    """
    parsed = _parse_response(response)
    spans = _find_patch_spans(old_text, parsed.patches)
    _validate_spans(old_text, spans)
    result = _apply_spans(old_text, spans)
    if not 300 <= len(result) <= 900:
        raise ProsePatchError("Patched persona text must be 300–900 characters")
    return result


class ProsePatchError(ValueError):
    """Raised when a proposed prose patch cannot be applied safely."""


def _apply_spans(old_text: str, spans: list[tuple[int, int, ProsePatch]]) -> str:
    """Apply spans from right to left to preserve original offsets.

    Returns:
        The patched text.
    """
    result = old_text
    for start, end, patch in reversed(spans):
        result = result[:start] + patch.new_excerpt + result[end:]
    return result


def _find_patch_spans(
    old_text: str, patches: list[ProsePatch]
) -> list[tuple[int, int, ProsePatch]]:
    """Resolve each unique excerpt to its original text span.

    Returns:
        Patch spans in provider order.

    Raises:
        ProsePatchError: If an excerpt is duplicate, ambiguous, or unsafe.
    """
    spans: list[tuple[int, int, ProsePatch]] = []
    seen: set[str] = set()
    for patch in patches:
        excerpt = patch.old_excerpt
        if excerpt in seen:
            raise ProsePatchError("Duplicate old excerpts are not allowed")
        seen.add(excerpt)
        occurrences = _occurrences(old_text, excerpt)
        if len(occurrences) != 1:
            raise ProsePatchError("Each old excerpt must occur exactly once")
        start = occurrences[0]
        end = start + len(excerpt)
        if not _has_word_boundaries(old_text, start, end, excerpt):
            raise ProsePatchError("Old excerpt must have Unicode word boundaries")
        if excerpt == patch.new_excerpt:
            raise ProsePatchError("Identical replacements are not allowed")
        if _replaces_whole_paragraph(old_text, start, end):
            raise ProsePatchError("Whole-paragraph replacements are not allowed")
        spans.append((start, end, patch))
    return spans


def _has_word_boundaries(text: str, start: int, end: int, excerpt: str) -> bool:
    """Ensure the excerpt does not begin or end inside a Unicode word.

    Returns:
        Whether both excerpt edges align with word boundaries.
    """
    begins_inside_word = (
        start > 0
        and _is_word_character(text[start - 1])
        and _is_word_character(excerpt[0])
    )
    ends_inside_word = (
        end < len(text)
        and _is_word_character(text[end])
        and _is_word_character(excerpt[-1])
    )
    return not begins_inside_word and not ends_inside_word


def _is_word_character(character: str) -> bool:
    """Treat Unicode alphanumerics and underscore as word characters.

    Returns:
        Whether the character belongs to a word.
    """
    return character == "_" or unicodedata.category(character)[0] in {"L", "N"}


def _occurrences(text: str, excerpt: str) -> list[int]:
    """Find all occurrences, including overlapping ones.

    Returns:
        Starting offsets for every match.
    """
    found: list[int] = []
    offset = 0
    while (index := text.find(excerpt, offset)) != -1:
        found.append(index)
        offset = index + 1
    return found


def _replaces_whole_paragraph(text: str, start: int, end: int) -> bool:
    """Reject edits that consume all non-whitespace content in a paragraph.

    Returns:
        Whether the edit spans the entire paragraph content.
    """
    paragraph_start = text.rfind("\n\n", 0, start) + 2
    separator = text.find("\n\n", end)
    paragraph_end = len(text) if separator == -1 else separator
    return (
        not text[paragraph_start:start].strip() and not text[end:paragraph_end].strip()
    )


def _parse_response(response: str | Mapping[str, object]) -> ProsePatchResponse:
    """Parse provider text or a decoded response object.

    Returns:
        A validated prose-patch response.

    Raises:
        ProsePatchError: If the response does not satisfy its schema.
    """
    try:
        parsed = (
            ProsePatchResponse.model_validate_json(response)
            if isinstance(response, str)
            else ProsePatchResponse.model_validate(response)
        )
    except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise ProsePatchError("Invalid prose patch response") from error
    if not parsed.patches:
        raise ProsePatchError("At least one prose patch is required")
    return parsed


def _validate_spans(old_text: str, spans: list[tuple[int, int, ProsePatch]]) -> None:
    """Reject overlapping edits and edits beyond the strict character budget.

    Raises:
        ProsePatchError: If edits overlap or exceed 20%/120 characters.
    """
    spans.sort(key=lambda span: span[0])
    if any(current[0] < previous[1] for previous, current in zip(spans, spans[1:])):
        raise ProsePatchError("Overlapping patches are not allowed")
    changed_length = sum(
        max(end - start, len(patch.new_excerpt)) for start, end, patch in spans
    )
    if changed_length > 120 or changed_length > len(old_text) * 0.2:
        raise ProsePatchError("Changed text exceeds the patch budget")
