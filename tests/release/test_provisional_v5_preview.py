"""Pure merge contracts for the provisional v5 preview."""

from __future__ import annotations

import polars as pl

from danish_personas.io import sha256_text
from danish_personas.release.provisional_v5_preview import merge_provisional_v5_preview


def test_merge_excludes_stale_v4_overlap_and_preserves_original_h90_prose() -> None:
    """H90 rows start from original prose before verified H90 patches apply."""
    overlap_original = (
        "Jeg er rådgiver i en rolig syntetisk testpersona, som beskriver "
        "hverdagsrutiner uden private detaljer, og jeg er 41 år."
    )
    overlap_h90 = overlap_original.replace("rådgiver", "analytiker")
    original = pl.DataFrame(
        [
            _row("overlap", overlap_original, "rådgiver"),
            _row("h90-blocked", "Jeg er lærer og er 50 år.", "lærer"),
            _row("v4-only", "Jeg er mekaniker og er 60 år.", "mekaniker"),
        ]
    )
    v4_structured = original.clone()
    v5_h90 = pl.DataFrame(
        [
            _row("overlap", overlap_original, "analytiker"),
            _row("h90-blocked", "Jeg er lærer og er 50 år.", "konsulent"),
            _row("v4-only", "Jeg er mekaniker og er 60 år.", "mekaniker"),
        ]
    )
    v4_preview = pl.DataFrame(
        [
            _row("overlap", overlap_original.replace("41 år", "42 år"), "rådgiver"),
            _row("h90-blocked", "Jeg er lærer og er 51 år.", "lærer"),
            _row("v4-only", "Jeg er smed og er 60 år.", "mekaniker"),
        ]
    )
    h90_first = [
        _first_record(
            persona_id="overlap",
            original_text=overlap_original,
            proposed_text=overlap_h90,
            old="rådgiver",
            new="analytiker",
            checkpoint_sha="a" * 64,
        )
    ]
    h90_second = [
        _second_record(
            persona_id="overlap",
            original_checkpoint_sha="a" * 64,
            old="rådgiver",
            new="analytiker",
        )
    ]

    preview, report = merge_provisional_v5_preview(
        original_frame=original,
        v4_structured_frame=v4_structured,
        v5_h90_frame=v5_h90,
        v4_prose_preview_frame=v4_preview,
        h90_first_pass_records=h90_first,
        h90_second_pass_records=h90_second,
        h90_changed_hashes=[sha256_text("overlap"), sha256_text("h90-blocked")],
        h90_source_metadata={"h90_manifest_sha256": "b" * 64},
        expected_row_count=3,
    )

    personas = {row["persona_id"]: row["persona"] for row in preview.to_dicts()}
    assert personas["overlap"] == overlap_h90
    assert personas["h90-blocked"] == "Jeg er lærer og er 50 år."
    assert personas["v4-only"] == "Jeg er smed og er 60 år."
    assert report["v4_excluded_overlap_count"] == 2
    assert report["v4_retained_non_h90_count"] == 1
    assert report["h90_accepted_count"] == 1
    assert preview.drop("persona").to_dicts() == v5_h90.drop("persona").to_dicts()


def _first_record(
    *,
    persona_id: str,
    original_text: str,
    proposed_text: str,
    old: str,
    new: str,
    checkpoint_sha: str,
) -> dict[str, object]:
    return {
        "persona_id": persona_id,
        "persona_hash": sha256_text(persona_id),
        "disposition": "patched",
        "checkpoint_sha256": checkpoint_sha,
        "original_persona_sha256": sha256_text(original_text),
        "proposed_text_sha256": sha256_text(proposed_text),
        "proposed_text": proposed_text,
        "patches": [{"old_excerpt": old, "new_excerpt": new}],
        "changed_facts": {"job_title": {"old": old, "new": new}},
    }


def _row(persona_id: str, persona: str, job_title: str) -> dict[str, object]:
    return {
        "persona_id": persona_id,
        "persona": persona,
        "job_title": job_title,
        "age": 41,
    }


def _second_record(
    *, persona_id: str, original_checkpoint_sha: str, old: str, new: str
) -> dict[str, object]:
    return {
        "persona_id": persona_id,
        "persona_hash": sha256_text(persona_id),
        "accepted": True,
        "status": "accepted",
        "review_verdict": "accept",
        "original_checkpoint_sha256": original_checkpoint_sha,
        "reasons": [],
        "fact_evidence": [
            {
                "field": "job_title",
                "status": "corrected",
                "original_quote": old,
                "proposed_quote": new,
            }
        ],
    }
