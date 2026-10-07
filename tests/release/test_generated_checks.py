"""Offline contracts for generated-field checks and text-review diagnostics."""

import polars as pl

from danish_personas.generation.job_titles import load_job_title_mapping
from danish_personas.release.generated_checks import check_generated_fields


def _row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "persona_id": "p-1",
        "age": 37,
        "sex": "female",
        "origin_country_da": "Rumænien",
        "municipality": "Aarhus",
        "job_function_code": None,
        "job_title": None,
        "marital_status": "never_married",
        "legal_status_detail": None,
        "current_relationship_status": "not_partnered",
        "partner_gender": None,
        "same_sex_partner_target": False,
        "skills_and_expertise": ["planlægning", "samarbejde", "formidling"],
        "hobbies_and_interests": ["keramik", "kajakture", "havearbejde"],
        "persona": (
            "Hun er vokset op med flere sprog og har fundet sin egen vej i livet. "
            "I Aarhus bruger hun sin fritid på keramik og kajakture, og hun har "
            "glæde af at planlægge små projekter. Hun arbejder omhyggeligt sammen "
            "med andre og holder af at lære nye teknikker. Rumænske familietraditioner "
            "har givet hende gode minder, uden at de bestemmer hendes hverdag. "
            "Hun har aldrig været gift og trives med sine selvstændige planer."
        ),
    }
    row.update(updates)
    return row


def test_valid_paraphrase_is_not_a_hard_failure() -> None:
    """Paraphrased text does not trigger objective failures."""
    report = check_generated_fields(pl.DataFrame([_row()]))
    assert report.hard_failures == ()
    assert report.row_count == 1
    # Literal absence is advisory; paraphrase only produces review flags.


def test_relationship_and_list_conflicts_are_hard_failures() -> None:
    """Objective relationship and list inconsistencies are rejected."""
    report = check_generated_fields(
        pl.DataFrame(
            [
                _row(
                    marital_status="married_or_separated",
                    legal_status_detail="married",
                    current_relationship_status="not_partnered",
                    partner_gender="female",
                    hobbies_and_interests=["Keramik", "keramik", "havearbejde."],
                )
            ]
        )
    )
    assert {item.check for item in report.hard_failures} >= {
        "married_requires_partnered",
        "partner_gender_presence",
        "interest_format",
        "hobbies_and_interests_unique_items",
    }
    assert report.hard_failure_persona_ids == ("p-1",)


def test_each_marital_status_has_expected_detail_contract() -> None:
    """Legal detail is present exactly for married-or-separated status."""
    rows = [
        _row(persona_id="never", marital_status="never_married"),
        _row(
            persona_id="married",
            marital_status="married_or_separated",
            legal_status_detail="married",
            current_relationship_status="partnered",
            partner_gender="male",
        ),
        _row(
            persona_id="separated",
            marital_status="married_or_separated",
            legal_status_detail="separated",
        ),
        _row(persona_id="divorced", marital_status="divorced"),
        _row(persona_id="widowed", marital_status="widowed"),
        _row(persona_id="missing", marital_status="married_or_separated"),
        _row(
            persona_id="unexpected",
            marital_status="divorced",
            legal_status_detail="separated",
        ),
    ]
    report = check_generated_fields(pl.DataFrame(rows))
    assert {
        x.persona_id
        for x in report.hard_failures
        if x.check == "legal_status_detail_presence"
    } == {"missing", "unexpected"}


def test_same_sex_target_and_exact_job_title_allowlist() -> None:
    """Partner targets and case-only title deviations are rejected."""
    mapping = load_job_title_mapping()
    code = next(iter(mapping.job_functions))
    exact_title = mapping.job_functions[code].titles[0]
    changed_first = next(
        index for index, character in enumerate(exact_title) if character.isalpha()
    )
    case_only_title = (
        exact_title[:changed_first]
        + exact_title[changed_first].swapcase()
        + exact_title[changed_first + 1 :]
    )
    base = _row(
        job_function_code=code,
        job_title=exact_title,
        current_relationship_status="partnered",
        partner_gender="female",
        same_sex_partner_target=True,
    )
    wrong = _row(
        persona_id="wrong",
        job_function_code=code,
        job_title=case_only_title,
        current_relationship_status="partnered",
        partner_gender="male",
        same_sex_partner_target=True,
    )
    report = check_generated_fields(
        pl.DataFrame([base, wrong]), job_title_mapping=mapping
    )
    assert {
        item.check for item in report.hard_failures if item.persona_id == "wrong"
    } == {"job_title_case_only", "partner_target_gender"}


def test_text_absence_is_advisory_and_conflicting_age_is_review_flag() -> None:
    """Text mismatch is advisory and an explicit age conflict is flagged."""
    row = _row(age=37, persona="Hun er 42 år og holder af at være ude i naturen.")
    report = check_generated_fields(pl.DataFrame([row]))
    assert report.hard_failures == ()
    assert "age_conflict" in report.review_flag_counts
    assert "origin_not_literal" in report.review_flag_counts
