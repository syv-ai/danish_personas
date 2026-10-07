"""Strict, local-only patching for generated persona prose."""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping
from typing import Any

from pydantic import Field, ValidationError

from ..models import StrictModel


class ProsePatch(StrictModel):
    """One bounded replacement in existing prose."""

    old_excerpt: str = Field(min_length=1, max_length=120)
    new_excerpt: str = Field(min_length=1, max_length=140)


class ProsePatchResponse(StrictModel):
    """Provider response containing a small set of prose edits."""

    patches: list[ProsePatch] = Field(max_length=2)

    @classmethod
    def provider_json_schema(cls) -> dict[str, Any]:
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


class ProsePatchError(ValueError):
    """Raised when a proposed prose patch cannot be applied safely."""


def apply_patches(old_text: str, response: str | Mapping[str, Any]) -> str:
    """Apply validated, local replacements while preserving all untouched text.

    ``response`` is either JSON text from a provider or its already-decoded object.
    Any ambiguity or violation fails closed with :class:`ProsePatchError`.
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

    spans: list[tuple[int, int, ProsePatch]] = []
    seen: set[str] = set()
    for patch in parsed.patches:
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

    spans.sort(key=lambda span: span[0])
    if any(current[0] < previous[1] for previous, current in zip(spans, spans[1:])):
        raise ProsePatchError("Overlapping patches are not allowed")

    changed_length = sum(end - start for start, end, _ in spans)
    if changed_length > 120 or changed_length > len(old_text) * 0.2:
        raise ProsePatchError("Changed text exceeds the patch budget")

    result = old_text
    for start, end, patch in reversed(spans):
        result = result[:start] + patch.new_excerpt + result[end:]
    if not 300 <= len(result) <= 900:
        raise ProsePatchError("Patched persona text must be 300–900 characters")
    return result


def _occurrences(text: str, excerpt: str) -> list[int]:
    """Find all occurrences, including overlapping ones."""
    found: list[int] = []
    offset = 0
    while (index := text.find(excerpt, offset)) != -1:
        found.append(index)
        offset = index + 1
    return found


def _is_word_character(character: str) -> bool:
    """Treat Unicode alphanumerics and underscore as word characters."""
    return character == "_" or unicodedata.category(character)[0] in {"L", "N"}


def _has_word_boundaries(text: str, start: int, end: int, excerpt: str) -> bool:
    """Ensure the excerpt does not begin or end inside a Unicode word."""
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


def _replaces_whole_paragraph(text: str, start: int, end: int) -> bool:
    """Reject edits that consume all non-whitespace content in a paragraph."""
    paragraph_start = text.rfind("\n\n", 0, start) + 2
    separator = text.find("\n\n", end)
    paragraph_end = len(text) if separator == -1 else separator
    return (
        not text[paragraph_start:start].strip()
        and not text[end:paragraph_end].strip()
    )
