"""Offline contracts for deterministic generated-attribute repairs."""

import polars as pl

from danish_personas.generation.job_titles import load_job_title_mapping
from danish_personas.release.generated_repair import repair_generated_attributes


def _row(persona_id: str, **updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "persona_id": persona_id,
        "marital_status": "single",
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
    mapping = load_job_title_mapping()
    code = next(iter(mapping.job_functions))
    title = mapping.job_functions[code].titles[0]
    rows = [
        _row("a", job_function_code=code, job_title=title.swapcase()),
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
        assert all(len({value.casefold() for value in values}) == len(values)
                   for values in repaired[field].to_list())
    assert "same_sex_partner_target" in report["unresolved"]["a"]


def test_allowed_title_retained_and_missing_or_unlisted_title_is_stable() -> None:
    mapping = load_job_title_mapping()
    code = next(iter(mapping.job_functions))
    title = mapping.job_functions[code].titles[0]
    frame = pl.DataFrame(
        [
            _row("allowed", job_function_code=code, job_title=title),
            _row("missing", job_function_code=code, job_title=None),
            _row("unlisted", job_function_code=code, job_title="ukendt"),
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
