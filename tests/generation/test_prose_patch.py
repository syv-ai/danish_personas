"""Tests for fail-closed persona prose patching."""

import json

import pytest

from danish_personas.generation.prose_patch import (
    ProsePatchError,
    ProsePatchResponse,
    apply_patches,
)

TEXT = (
    "Maja er rolig og nysgerrig. Hun holder af lange gåture ved kysten og bruger "
    "weekenderne på at besøge familie og venner. På arbejdet er hun grundig, "
    "hjælpsom og glad for at samarbejde med kolleger. Hun tager gerne imod nye "
    "opgaver, men sørger for at planlægge dem i et tempo, der giver plads til "
    "fordybelse. Når hun har fri, læser hun ofte en roman eller laver mad med "
    "sæsonens grøntsager. Hun sætter pris på nærværende samtaler "
    "og små udflugter."
)


def response(*patches: tuple[str, str]) -> str:
    """Serialise patch pairs as a provider response.

    Returns:
        JSON text containing the requested patches.
    """
    return json.dumps(
        {"patches": [{"old_excerpt": old, "new_excerpt": new} for old, new in patches]},
        ensure_ascii=False,
    )


def test_apply_preserves_untouched_unicode_text_exactly() -> None:
    """Preserve all text around a Unicode replacement exactly."""
    original = TEXT.replace("kysten", "Øresundskysten")
    output = apply_patches(original, response(("Øresundskysten", "Øresund")))

    assert output == original.replace("Øresundskysten", "Øresund", 1)
    assert output.startswith("Maja er rolig og nysgerrig.")
    assert output.endswith("små udflugter.")
    assert len(output) == len(original) - len("skysten")


def test_two_non_overlapping_patches_apply_from_original_offsets() -> None:
    """Apply two edits using offsets from the unchanged source text."""
    output = apply_patches(
        TEXT, response(("rolig", "venlig"), ("grundig", "omhyggelig"))
    )

    assert "venlig og nysgerrig" in output
    assert "omhyggelig, hjælpsom" in output
    expected_delta = (len("venlig") - len("rolig")) + (
        len("omhyggelig") - len("grundig")
    )
    assert len(output) == len(TEXT) + expected_delta


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '{"patches": []}',
        '{"patches": [{"old_excerpt": "rolig", "new_excerpt": "venlig", "extra": 1}]}',
        '{"patches": [{"old_excerpt": "", "new_excerpt": "venlig"}]}',
        '{"patches": [{"old_excerpt": "x"}], "extra": true}',
    ],
)
def test_rejects_malformed_empty_or_extra_field_responses(raw: str) -> None:
    """Reject malformed payloads and fields outside the response schema."""
    with pytest.raises(ProsePatchError):
        apply_patches(TEXT, raw)


def test_rejects_duplicate_and_non_unique_excerpt() -> None:
    """Require each distinct old excerpt to occur exactly once."""
    with pytest.raises(ProsePatchError, match="exactly once"):
        apply_patches(TEXT + " rolig", response(("rolig", "venlig")))
    with pytest.raises(ProsePatchError, match="Duplicate"):
        apply_patches(TEXT, response(("rolig", "venlig"), ("rolig", "mild")))


def test_rejects_excerpt_without_unicode_word_boundaries() -> None:
    """Reject excerpt matches that begin inside a Unicode word."""
    with pytest.raises(ProsePatchError, match="word boundaries"):
        apply_patches(TEXT.replace("rolig", "urolig"), response(("rolig", "venlig")))


def test_rejects_overlapping_and_identical_patches() -> None:
    """Reject overlapping spans and replacements with no change."""
    with pytest.raises(ProsePatchError, match="Overlapping"):
        apply_patches(
            TEXT, response(("rolig og", "venlig og"), ("og nysgerrig", "og åben"))
        )
    with pytest.raises(ProsePatchError, match="Identical"):
        apply_patches(TEXT, response(("rolig", "rolig")))


def test_rejects_wide_edit_and_whole_paragraph_rewrite() -> None:
    """Enforce the edit budget and prohibit complete paragraph replacement."""
    wide_excerpt = "Z" * 110
    wide_text = "A" * 150 + " " + wide_excerpt + " " + "B" * 250
    with pytest.raises(ProsePatchError, match="budget"):
        apply_patches(wide_text, response((wide_excerpt, "Y" * 110)))

    paragraph = "Særligt langt afsnit med tekst."
    original = paragraph + "\n\n" + TEXT
    with pytest.raises(ProsePatchError, match="paragraph"):
        apply_patches(original, response((paragraph, "Et andet afsnit.")))


def test_rejects_out_of_range_result_and_enforces_local_limits() -> None:
    """Reject invalid final lengths and excerpts outside local-edit limits."""
    short_text = "Maja er nysgerrig. " + "Hun læser gerne bøger. " * 7
    with pytest.raises(ProsePatchError, match="300–900"):
        apply_patches(short_text, response(("nysgerrig", "åben")))
    with pytest.raises(ProsePatchError):
        apply_patches(TEXT, response(("rolig", "x" * 141)))
    with pytest.raises(ProsePatchError):
        apply_patches(TEXT, response(("x" * 121, "y")))


def test_patch_is_deterministic_and_reapplication_fails_closed() -> None:
    """Ensure patching is deterministic and cannot be reapplied."""
    patch = response(("rolig", "venlig"))
    first = apply_patches(TEXT, patch)
    assert apply_patches(TEXT, patch) == first
    with pytest.raises(ProsePatchError):
        apply_patches(first, patch)


def test_provider_schema_omits_field_length_constraints() -> None:
    """Expose a proxy-compatible schema without provider-fragile constraints."""
    schema = ProsePatchResponse.provider_json_schema()
    assert schema["required"] == ["patches"]
    assert schema["additionalProperties"] is False
    item = schema["properties"]["patches"]["items"]
    assert item["required"] == ["old_excerpt", "new_excerpt"]
    assert item["properties"]["old_excerpt"] == {"type": "string"}
