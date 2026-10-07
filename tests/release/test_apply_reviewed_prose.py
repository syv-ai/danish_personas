"""Offline contracts for provisional reviewed-prose previews."""

from __future__ import annotations

import polars as pl
import pytest

from danish_personas.io import sha256_text
from danish_personas.release.apply_reviewed_prose import apply_reviewed_prose

_SHA = "a" * 64
_SECOND_SHA = "b" * 64


def test_accepted_patch_updates_only_persona_and_reports_preview() -> None:
    """Accepted, linked second-pass reviews build only a provisional preview."""
    original, v4, proposed = _frames()

    preview, report = apply_reviewed_prose(
        original_published_frame=original,
        v4_structured_frame=v4,
        first_pass_records=[_first_record(proposed=proposed)],
        second_pass_records=[_second_record()],
        source_metadata=_metadata(),
    )

    assert preview.get_column("persona").to_list() == [proposed, _other_persona()]
    assert preview.get_column("marital_status").to_list() == ["gift", "gift"]
    assert original.get_column("persona").to_list()[0] == _original_persona()
    assert v4.get_column("persona").to_list()[0] == _original_persona()
    assert report["label"] == "PROVISIONAL PREVIEW"
    assert report["release_ready"] is False
    assert report["approves_proposals"] is False
    assert report["changed_rows"] == 1
    assert report["changed_fraction"] == 0.5
    assert report["accepted_second_pass_rows"] == 1
    assert len(report["preview_sha256"]) == 64


def test_forged_second_pass_checkpoint_link_is_rejected() -> None:
    """A second-pass acceptance cannot be attached to another first checkpoint."""
    original, v4, proposed = _frames()
    second = _second_record()
    second["original_checkpoint_sha256"] = _SECOND_SHA

    with pytest.raises(ValueError, match="different first checkpoint"):
        apply_reviewed_prose(
            original_published_frame=original,
            v4_structured_frame=v4,
            first_pass_records=[_first_record(proposed=proposed)],
            second_pass_records=[second],
            source_metadata=_metadata(),
        )


def test_unchanged_and_rejected_status_records_are_skipped() -> None:
    """Status-only decisions remain original prose and unresolved in the report."""
    original, v4, _ = _frames()

    preview, report = apply_reviewed_prose(
        original_published_frame=original,
        v4_structured_frame=v4,
        first_pass_records=[
            {
                "persona_id": "p1",
                "persona_hash": sha256_text("p1"),
                "disposition": "unchanged_consistent",
            }
        ],
        second_pass_records=[
            {
                "persona_id": "p1",
                "persona_hash": sha256_text("p1"),
                "status": "rejected",
                "accepted": False,
                "original_checkpoint_sha256": _SHA,
            }
        ],
        source_metadata=_metadata(),
    )

    assert preview.get_column("persona").to_list() == [
        _original_persona(),
        _other_persona(),
    ]
    assert report["changed_rows"] == 0
    assert report["skipped_first_pass_rows"] == 1
    assert report["skipped_second_pass_rows"] == 1
    assert report["unresolved_count"] == 2
    assert report["review_status_counts"] == {"rejected": 1, "unchanged_consistent": 1}


def test_duplicate_and_missing_ids_fail_closed() -> None:
    """Published and review inputs must identify rows exactly once."""
    original, v4, proposed = _frames()
    duplicate = pl.concat([original, original.slice(0, 1)])

    with pytest.raises(ValueError, match="duplicate persona IDs"):
        apply_reviewed_prose(
            original_published_frame=duplicate,
            v4_structured_frame=duplicate,
            first_pass_records=[],
            second_pass_records=[],
            source_metadata=_metadata(),
        )

    missing_record = {**_first_record(proposed=proposed), "persona_id": "missing"}
    with pytest.raises(ValueError, match="unknown persona ID"):
        apply_reviewed_prose(
            original_published_frame=original,
            v4_structured_frame=v4,
            first_pass_records=[missing_record],
            second_pass_records=[_second_record()],
            source_metadata=_metadata(),
        )


def test_structured_columns_are_preserved_from_v4() -> None:
    """The preview never rewrites source-backed structured columns."""
    original, v4, proposed = _frames()
    v4 = v4.with_columns(pl.lit("reviewed").alias("source_version"))

    preview, _ = apply_reviewed_prose(
        original_published_frame=original,
        v4_structured_frame=v4,
        first_pass_records=[_first_record(proposed=proposed)],
        second_pass_records=[_second_record()],
        source_metadata=_metadata(),
    )

    assert preview.drop("persona").to_dicts() == v4.drop("persona").to_dicts()


def test_rejects_blank_proposals_and_unpinned_metadata() -> None:
    """Blank prose and metadata without immutable SHA pins are never previewed."""
    original, v4, proposed = _frames()
    blank = {**_first_record(proposed=proposed), "proposed_text": " "}

    with pytest.raises(ValueError, match="SHA-256 pin"):
        apply_reviewed_prose(
            original_published_frame=original,
            v4_structured_frame=v4,
            first_pass_records=[_first_record(proposed=proposed)],
            second_pass_records=[_second_record()],
            source_metadata={"manifest": "unpinned"},
        )

    with pytest.raises(ValueError, match="blank"):
        apply_reviewed_prose(
            original_published_frame=original,
            v4_structured_frame=v4,
            first_pass_records=[blank],
            second_pass_records=[_second_record()],
            source_metadata=_metadata(),
        )


def _frames() -> tuple[pl.DataFrame, pl.DataFrame, str]:
    original_persona = _original_persona()
    proposed = original_persona.replace("Hun er ugift", "Hun er gift")
    original = pl.DataFrame(
        {
            "persona_id": ["p1", "p2"],
            "persona": [original_persona, _other_persona()],
            "marital_status": ["ugift", "gift"],
            "age": [42, 50],
        }
    )
    v4 = pl.DataFrame(
        {
            "persona_id": ["p1", "p2"],
            "persona": [original_persona, _other_persona()],
            "marital_status": ["gift", "gift"],
            "age": [42, 50],
        }
    )
    return original, v4, proposed


def _first_record(*, proposed: str) -> dict[str, object]:
    return {
        "persona_id": "p1",
        "persona_hash": sha256_text("p1"),
        "disposition": "patched",
        "checkpoint_sha256": _SHA,
        "original_persona_sha256": sha256_text(_original_persona()),
        "proposed_text_sha256": sha256_text(proposed),
        "proposed_text": proposed,
        "patches": [{"old_excerpt": "Hun er ugift", "new_excerpt": "Hun er gift"}],
        "changed_facts": {"marital_status": {"old": "ugift", "new": "gift"}},
    }


def _second_record() -> dict[str, object]:
    return {
        "persona_id": "p1",
        "persona_hash": sha256_text("p1"),
        "accepted": True,
        "review_verdict": "accept",
        "reasons": [],
        "original_checkpoint_sha256": _SHA,
        "fact_evidence": [
            {
                "field": "marital_status",
                "status": "corrected",
                "original_quote": "Hun er ugift",
                "proposed_quote": "Hun er gift",
            }
        ],
    }


def _metadata() -> dict[str, object]:
    return {
        "original_manifest_sha256": "c" * 64,
        "v4_manifest_sha256": "d" * 64,
        "second_pass_status_sha256": "e" * 64,
    }


def _original_persona() -> str:
    return (
        "Hun er ugift og arbejder med planlægning i kommunen. Hun cykler til "
        "biblioteket, hjælper naboer og holder af rolige aftener med bøger."
    )


def _other_persona() -> str:
    return (
        "Han er gift og arbejder på et værksted. Han laver mad til familien, "
        "går ture ved havnen og følger lokale sportskampe i weekenden."
    )
