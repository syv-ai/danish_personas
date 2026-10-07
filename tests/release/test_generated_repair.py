"""Offline contracts for deterministic generated-attribute repairs."""

import polars as pl

from danish_personas.generation.job_titles import load_job_title_mapping
from danish_personas.release.generated_repair import repair_generated_attributes


def _row(persona_id: str, **updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "persona_id": persona_id,
        "marital_status": "never_married",
        "detailed_status_code": "130",
        "legal_status_detail": "married",
        "current_relationship_status": "not_partnered",
        "partner_gender": "female",
        "same_sex_partner_target": True,
        "job_function_code": None,
        "job_title": "unrelated title",
        "skills_and_expertise": ["Samarbejde!", "samarbejde", "Planlægning."],
        "hobbies_and_interests": ["Musik!", "musik", "Natur."],
        "sex": "female",
        "origin_country": "Danmark",
        "ocean_openness": 0.5,
        "persona": "Uændret prosatekst.",
        "cultural_context": "Uændret kulturel kontekst.",
    }
    row.update(updates)
    return row


def test_repairs_are_deterministic_idempotent_and_keep_every_record() -> None:
    """Keep every row and flag rather than fabricate duplicate-list replacements."""
    mapping = load_job_title_mapping()
    code = next(iter(mapping.job_functions))
    title = mapping.job_functions[code].titles[0]
    rows = [
        _row(
            "a",
            detailed_status_code="15",
            job_function_code=code,
            job_title=title.swapcase(),
        ),
        _row(
            "b",
            marital_status="married_or_separated",
            legal_status_detail="married",
            current_relationship_status="partnered",
            partner_gender="male",
        ),
        _row(
            "c",
            marital_status="married_or_separated",
            legal_status_detail="married",
            current_relationship_status="not_partnered",
        ),
    ]
    frame = pl.DataFrame(rows)
    repaired, report = repair_generated_attributes(frame, mapping)
    repeated, second_report = repair_generated_attributes(repaired, mapping)

    assert repaired.height == frame.height
    assert repeated.equals(repaired)
    assert second_report["changed"] == {}
    assert repaired["persona_id"].to_list() == frame["persona_id"].to_list()
    assert repaired["sex"].to_list() == frame["sex"].to_list()
    assert repaired["origin_country"].to_list() == frame["origin_country"].to_list()
    assert repaired["ocean_openness"].to_list() == frame["ocean_openness"].to_list()
    assert repaired["persona"].to_list() == frame["persona"].to_list()
    assert repaired["cultural_context"].to_list() == frame["cultural_context"].to_list()
    assert repaired.filter(pl.col("persona_id") == "a")["job_title"].item() == title
    assert repaired["legal_status_detail"].to_list() == [None, "married", "separated"]
    assert repaired["partner_gender"].to_list() == [None, "male", None]
    for field in ("skills_and_expertise", "hobbies_and_interests"):
        assert all(3 <= len(values) <= 6 for values in repaired[field].to_list())
        assert field in report["unresolved"]["a"]
    assert repaired["hobbies_and_interests"][0].to_list() == ["musik", "musik", "natur"]


def test_allowed_title_retained_and_missing_or_unlisted_title_is_stable() -> None:
    """Choose only a reviewed title for eligible employee rows."""
    mapping = load_job_title_mapping()
    code = next(iter(mapping.job_functions))
    title = mapping.job_functions[code].titles[0]
    frame = pl.DataFrame(
        [
            _row(
                "allowed",
                detailed_status_code="15",
                job_function_code=code,
                job_title=title,
            ),
            _row(
                "missing",
                detailed_status_code="15",
                job_function_code=code,
                job_title=None,
            ),
            _row(
                "unlisted",
                detailed_status_code="15",
                job_function_code=code,
                job_title="ukendt",
            ),
            _row("ineligible", job_function_code=None, job_title=title),
        ]
    )
    repaired, _ = repair_generated_attributes(frame, mapping)
    titles = dict(
        zip(
            repaired["persona_id"].to_list(),
            repaired["job_title"].to_list(),
            strict=True,
        )
    )
    assert titles["allowed"] == title
    assert titles["missing"] == titles["unlisted"]
    assert titles["ineligible"] is None
    again, _ = repair_generated_attributes(frame, mapping)
    assert again["job_title"].to_list() == repaired["job_title"].to_list()
