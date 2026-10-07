"""Objective checks and review-only diagnostics for generated persona releases.

Text diagnostics are deliberately advisory: a missing literal phrase does not prove
that a generated persona omits the corresponding idea.
"""

from __future__ import annotations

import re
import typing as t
from collections import Counter
from dataclasses import dataclass

import polars as pl

from ..generation.job_titles import JobFunctionTitleMapping
from ..generation.models import GenerationConfig
from ..generation.partner_target import required_partner_gender, same_sex_partner_target


@dataclass(frozen=True)
class Finding:
    """One field-level finding, without copying generated text into the report."""

    persona_id: str
    check: str


@dataclass(frozen=True)
class GeneratedChecksReport:
    """Aggregated hard failures and advisory text-review flags."""

    row_count: int
    hard_failure_counts: dict[str, int]
    review_flag_counts: dict[str, int]
    hard_failures: tuple[Finding, ...]
    review_flags: tuple[Finding, ...]

    @property
    def hard_failure_persona_ids(self) -> tuple[str, ...]:
        """Unique persona IDs with objective failures."""
        return tuple(sorted({finding.persona_id for finding in self.hard_failures}))

    @property
    def review_flag_persona_ids(self) -> tuple[str, ...]:
        """Unique persona IDs requiring human text review."""
        return tuple(sorted({finding.persona_id for finding in self.review_flags}))


def check_generated_fields(
    frame: pl.DataFrame,
    *,
    generation_config: GenerationConfig | None = None,
    job_title_mapping: JobFunctionTitleMapping | None = None,
) -> GeneratedChecksReport:
    """Check objective generated-field contracts and flag text for human review.

    ``same_sex_partner_target`` is read from the row when present. Otherwise, an
    effective config derives it deterministically from persona ID and its configured
    probability. Text diagnostics never assert that absence of a phrase is definitive.

    Returns:
        Aggregated hard-failure and text-review counts and findings.

    Raises:
        ValueError: If required input columns are absent.
    """
    required = {
        "persona_id",
        "current_relationship_status",
        "partner_gender",
        "legal_status_detail",
        "marital_status",
        "skills_and_expertise",
        "hobbies_and_interests",
        "persona",
    }
    absent = required - set(frame.columns)
    if absent:
        raise ValueError(
            f"Generated frame is missing required columns: {sorted(absent)}"
        )

    failures: list[Finding] = []
    flags: list[Finding] = []
    for row in frame.iter_rows(named=True):
        persona_id = str(row["persona_id"])
        _check_row(
            row, persona_id, failures, flags, generation_config, job_title_mapping
        )
    return GeneratedChecksReport(
        row_count=frame.height,
        hard_failure_counts=dict(sorted(Counter(x.check for x in failures).items())),
        review_flag_counts=dict(sorted(Counter(x.check for x in flags).items())),
        hard_failures=tuple(failures),
        review_flags=tuple(flags),
    )


def _check_row(
    row: dict[str, t.Any],
    persona_id: str,
    failures: list[Finding],
    flags: list[Finding],
    config: GenerationConfig | None,
    mapping: JobFunctionTitleMapping | None,
) -> None:
    def fail(name: str) -> None:
        failures.append(Finding(persona_id, name))

    def flag(name: str) -> None:
        flags.append(Finding(persona_id, name))

    marital = row["marital_status"]
    detail = row["legal_status_detail"]
    relationship = row["current_relationship_status"]
    partner_gender = row["partner_gender"]
    if (marital == "married_or_separated") != (detail is not None):
        fail("legal_status_detail_presence")
    if detail == "married" and relationship != "partnered":
        fail("married_requires_partnered")
    if (relationship == "partnered") != (partner_gender is not None):
        fail("partner_gender_presence")

    target = row.get("same_sex_partner_target")
    if target is None and config is not None:
        target = same_sex_partner_target(
            persona_id=persona_id, probability=config.same_sex_partner_probability
        )
    if target is not None and relationship == "partnered":
        try:
            expected = required_partner_gender(sex=row["sex"], same_sex_target=target)
        except KeyError, ValueError:
            fail("partner_target_input")
        else:
            if partner_gender != expected:
                fail("partner_target_gender")

    if mapping is not None:
        code, title = row.get("job_function_code"), row.get("job_title")
        allowed = (
            mapping.job_functions.get(code).titles
            if code in mapping.job_functions
            else []
        )
        if title is None:
            if allowed:
                fail("job_title_missing")
        elif title not in allowed:
            if any(title.casefold() == candidate.casefold() for candidate in allowed):
                fail("job_title_case_only")
            else:
                fail("job_title_not_allowed")

    for field in ("skills_and_expertise", "hobbies_and_interests"):
        values = row[field]
        if not isinstance(values, (list, tuple)) or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            fail(f"{field}_nonempty_items")
        elif len({value.casefold() for value in values}) != len(values):
            fail(f"{field}_unique_items")
    interests = row["hobbies_and_interests"]
    if isinstance(interests, (list, tuple)):
        for value in interests:
            if (
                isinstance(value, str)
                and value
                and (value != value.lower() or value[-1] in ".!?;:,")
            ):
                fail("interest_format")

    text = row["persona"]
    if not isinstance(text, str):
        flag("persona_text_unavailable")
        return
    _text_review(row, text, flag)


def _text_review(
    row: dict[str, t.Any], text: str, flag: t.Callable[[str], None]
) -> None:
    """Apply intentionally conservative literal-pattern review diagnostics."""
    folded = text.casefold()
    if not 300 <= len(text) <= 900:
        flag("persona_length")
    if not re.search(r"\b(?:18|19|[2-9]\d|1[01]\d|12[0-5])\s*[- ]?år", folded):
        flag("age_not_literal")
    for field, check in (
        ("origin_country_da", "origin_not_literal"),
        ("municipality", "municipality_not_literal"),
        ("job_title", "job_title_not_literal"),
    ):
        value = row.get(field)
        if value and str(value).casefold() not in folded:
            flag(check)
    if row.get("current_relationship_status") == "partnered" and not re.search(
        r"\b(partner|kæreste|ægtefælle|mand|kone)\b", folded
    ):
        flag("partner_term_not_literal")
    unsafe = (
        r"\bvedkommende\b",
        r"\bpersonen\b",
        r"\bkan\s+\w+\s+være\b",
        r"\b(?:seksuel orientering|cpr(?:-?nummer)?|telefonnummer)\b",
        r"https?://|www\.",
        r"\b\d{10}\b",
    )
    if any(re.search(pattern, folded) for pattern in unsafe):
        flag("unsafe_or_forbidden_pattern")
    # Explicit numeric age disagreement is review-worthy, never automatically rejected.
    age = row.get("age")
    if age is not None and re.search(r"\b(\d{2,3})\s*[- ]?år", folded):
        if any(
            int(match) != age for match in re.findall(r"\b(\d{2,3})\s*[- ]?år", folded)
        ):
            flag("age_conflict")
    for field, check in (
        ("origin_country_da", "origin_conflict"),
        ("municipality", "municipality_conflict"),
        ("job_title", "job_title_conflict"),
    ):
        value = row.get(field)
        if (
            value
            and value.casefold() in folded
            and re.search(r"\b(?:ikke|ikke fra|bor ikke i|arbejder ikke som)\b", folded)
        ):
            flag(check)
    if row.get("current_relationship_status") == "not_partnered" and re.search(
        r"\b(?:min|sin|hendes|hans)\s+(?:partner|kæreste|ægtefælle)\b", folded
    ):
        flag("partner_conflict")
    detail = row.get("legal_status_detail")
    if detail:
        words = {"married": r"gift|ægtefælle", "separated": r"separeret"}
        if not re.search(words[detail], folded):
            flag("legal_status_not_literal")
    statuses = {
        "never_married": r"aldrig været gift",
        "divorced": r"skilt",
        "widowed": r"enke|enkemand",
    }
    status = row.get("marital_status")
    if status in statuses and not re.search(statuses[status], folded):
        flag("marital_status_not_literal")
