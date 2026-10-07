"""Objective checks and review-only diagnostics for generated persona releases.

Text diagnostics are deliberately advisory: a missing literal phrase does not prove
that a generated persona omits the corresponding idea.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
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

    The partner target is derived only from the effective generation config and
    persona ID; a row-level target is not trusted as a config binding. Without config,
    partnered rows receive an advisory unable-to-check finding. Text diagnostics are
    advisory pattern matches: absence of a phrase is not proof that an idea is absent,
    and a detected pattern is not proof of a contradiction.

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


def _check_partner_target(
    row: dict[str, object],
    persona_id: str,
    config: GenerationConfig | None,
    partner_gender: object,
    fail: Callable[[str], None],
    flag: Callable[[str], None],
) -> None:
    """Check partner gender only against the effective generation configuration."""
    if config is None:
        flag("partner_target_unavailable")
        return
    sex = row.get("sex")
    if not isinstance(sex, str):
        fail("partner_target_input")
        return
    target = same_sex_partner_target(
        persona_id=persona_id, probability=config.same_sex_partner_probability
    )
    try:
        expected = required_partner_gender(sex=sex, same_sex_target=target)
    except KeyError, ValueError:
        fail("partner_target_input")
        return
    if partner_gender != expected:
        fail("partner_target_gender")


def _check_job_title(
    row: dict[str, object],
    mapping: JobFunctionTitleMapping,
    fail: Callable[[str], None],
) -> None:
    """Check a generated job title against its exact configured allowlist."""
    code, title = row.get("job_function_code"), row.get("job_title")
    function = mapping.job_functions.get(code) if isinstance(code, str) else None
    allowed = function.titles if function is not None else []
    if title is None:
        if allowed:
            fail("job_title_missing")
    elif isinstance(title, str) and title not in allowed:
        check = (
            "job_title_case_only"
            if any(title.casefold() == item.casefold() for item in allowed)
            else "job_title_not_allowed"
        )
        fail(check)
    elif not isinstance(title, str):
        fail("job_title_not_allowed")


def _check_lists(row: dict[str, object], fail: Callable[[str], None]) -> None:
    """Check list-valued generated fields for valid, unique items and formatting."""
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


def _check_row(
    row: dict[str, object],
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
    if marital != "married_or_separated" and detail is not None:
        fail("legal_status_detail_presence")
    elif marital == "married_or_separated" and detail not in {"married", "separated"}:
        fail("legal_status_detail_presence")
    if detail == "married" and relationship != "partnered":
        fail("married_requires_partnered")
    if (relationship == "partnered") != (partner_gender is not None):
        fail("partner_gender_presence")

    if relationship == "partnered":
        _check_partner_target(row, persona_id, config, partner_gender, fail, flag)

    if mapping is not None:
        _check_job_title(row, mapping, fail)

    _check_lists(row, fail)

    text = row["persona"]
    if not isinstance(text, str):
        flag("persona_text_unavailable")
        return
    _text_review(row, text, flag)


def _text_review(
    row: dict[str, object], text: str, flag: Callable[[str], None]
) -> None:
    """Emit advisory pattern matches that require human interpretation."""
    folded = text.casefold()
    if not 300 <= len(text) <= 900:
        flag("persona_length")
    _review_literal_fields(row, folded, flag)
    _review_relationship(row, folded, flag)
    _review_safety(folded, flag)
    _review_age(row, folded, flag)
    _review_conflicts(row, folded, flag)
    _review_legal_status(row, folded, flag)


def _review_literal_fields(
    row: dict[str, object], folded: str, flag: Callable[[str], None]
) -> None:
    """Flag literal phrases not detected; paraphrases may still satisfy intent."""
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


def _review_relationship(
    row: dict[str, object], folded: str, flag: Callable[[str], None]
) -> None:
    """Flag relationship wording patterns for human review."""
    if row.get("current_relationship_status") == "partnered" and not re.search(
        r"\b(partner|kæreste|ægtefælle|mand|kone)\b", folded
    ):
        flag("partner_term_not_literal")
    if row.get("current_relationship_status") == "not_partnered" and re.search(
        r"\b(?:min|sin|hendes|hans)\s+(?:partner|kæreste|ægtefælle)\b", folded
    ):
        flag("partner_conflict")


def _review_safety(folded: str, flag: Callable[[str], None]) -> None:
    """Flag literal safety patterns without deciding whether text is harmful."""
    unsafe_patterns = (
        r"\bvedkommende\b",
        r"\bpersonen\b",
        r"\bkan\s+\w+\s+være\b",
        r"\b(?:seksuel orientering|cpr(?:-?nummer)?|telefonnummer)\b",
        r"https?://|www\.",
        r"\b\d{10}\b",
    )
    if any(re.search(pattern, folded) for pattern in unsafe_patterns):
        flag("unsafe_or_forbidden_pattern")


def _review_age(
    row: dict[str, object], folded: str, flag: Callable[[str], None]
) -> None:
    """Flag explicit age disagreement as an advisory review item."""
    age = row.get("age")
    if isinstance(age, int) and any(
        int(match) != age for match in re.findall(r"\b(\d{2,3})\s*[- ]?år", folded)
    ):
        flag("age_conflict")


def _review_conflicts(
    row: dict[str, object], folded: str, flag: Callable[[str], None]
) -> None:
    """Flag possible contradictions; negation scope is not inferred."""
    negation = r"\b(?:ikke|ikke fra|bor ikke i|arbejder ikke som)\b"
    for field, check in (
        ("origin_country_da", "origin_conflict"),
        ("municipality", "municipality_conflict"),
        ("job_title", "job_title_conflict"),
    ):
        value = row.get(field)
        if (
            isinstance(value, str)
            and value.casefold() in folded
            and re.search(negation, folded)
        ):
            flag(check)


def _review_legal_status(
    row: dict[str, object], folded: str, flag: Callable[[str], None]
) -> None:
    """Flag absent literal legal-status terms for review, not as proof of omission."""
    detail_patterns = {"married": r"gift|ægtefælle", "separated": r"separeret"}
    detail = row.get("legal_status_detail")
    if isinstance(detail, str) and detail in detail_patterns:
        if not re.search(detail_patterns[detail], folded):
            flag("legal_status_not_literal")
    status_patterns = {
        "never_married": r"aldrig været gift",
        "divorced": r"skilt",
        "widowed": r"enke|enkemand",
    }
    status = row.get("marital_status")
    if isinstance(status, str) and status in status_patterns:
        if not re.search(status_patterns[status], folded):
            flag("marital_status_not_literal")
