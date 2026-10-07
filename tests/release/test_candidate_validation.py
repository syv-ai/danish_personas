"""Offline release-candidate diagnostics for full-data publication gates."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from pytest import MonkeyPatch

from danish_personas.io import sha256_text
from danish_personas.release import candidate_validation as module
from danish_personas.release.candidate_validation import (
    REPORT_LABEL,
    validate_release_candidate,
)
from danish_personas.release.generated_checks import (
    GeneratedChecksReport,
    check_generated_fields,
)


def test_validate_release_candidate_accepts_unchanged_supported_frame(
    tmp_path: Path,
) -> None:
    """An unchanged candidate with source support has no hard failures."""
    rows = _base_rows()
    candidate_path, original_path = _write_candidate_pair(tmp_path=tmp_path, rows=rows)
    bundle_dir = _write_bundle(tmp_path=tmp_path, rows=rows)

    report = validate_release_candidate(
        candidate_path=candidate_path,
        original_path=original_path,
        bundle_dir=bundle_dir,
        expected_row_count=len(rows),
    )

    assert report.label == REPORT_LABEL
    assert report.passes_hard_gates is True
    assert report.hard_failure_counts == {}
    assert len(report.candidate_sha256) == 64
    assert len(report.original_sha256) == 64
    assert report.identity.ordered_ids_match is True
    assert report.source_support.unsupported_rows == {
        "folk1a": 0,
        "ras209": 0,
        "ras202": 0,
    }
    assert "source-distribution pass/fail" in report.source_distribution_claim


def _base_rows() -> list[dict[str, object]]:
    prose = (
        "Hun er 30 år og bor i Aarhus med en rolig hverdag. Hun nævner Rumænien "
        "som en del af sin baggrund, men beskriver først og fremmest hverdagsliv, "
        "arbejde og fritid. Hun har aldrig været gift og bruger tid på læsning, "
        "cykling og madlavning sammen med venner."
    )
    return [
        {
            "persona_id": "p-001",
            "country": "Danmark",
            "origin_country_da": "Rumænien",
            "age": 30,
            "age_band": "30-34",
            "sex": "female",
            "marital_status": "never_married",
            "municipality_code": "101",
            "municipality": "Aarhus",
            "education_level": "secondary_or_vocational",
            "education_source_code": "H20-H35",
            "labour_market_status": "employed",
            "detailed_status_code": "A",
            "job_function_code": "123",
            "job_function": "kontorarbejde",
            "cultural_context": "Rumænsk-dansk hverdagskontekst",
            "skills_and_expertise": ["planlægning", "samarbejde", "formidling"],
            "hobbies_and_interests": ["læsning", "cykling", "madlavning"],
            "career_goals_and_ambitions": "Vil gerne lære mere om koordinering.",
            "job_title": None,
            "current_relationship_status": "not_partnered",
            "partner_gender": None,
            "legal_status_detail": None,
            "persona": prose,
        },
        {
            "persona_id": "p-002",
            "country": "Danmark",
            "origin_country_da": "Polen",
            "age": 31,
            "age_band": "30-34",
            "sex": "male",
            "marital_status": "divorced",
            "municipality_code": "101",
            "municipality": "Aarhus",
            "education_level": "secondary_or_vocational",
            "education_source_code": "H20-H35",
            "labour_market_status": "employed",
            "detailed_status_code": "A",
            "job_function_code": "123",
            "job_function": "kontorarbejde",
            "cultural_context": "Polsk-dansk hverdagskontekst",
            "skills_and_expertise": ["overblik", "service", "samarbejde"],
            "hobbies_and_interests": ["løb", "musik", "brætspil"],
            "career_goals_and_ambitions": "Vil gerne styrke sine rutiner.",
            "job_title": None,
            "current_relationship_status": "not_partnered",
            "partner_gender": None,
            "legal_status_detail": None,
            "persona": prose.replace("Rumænien", "Polen").replace("Hun", "Han"),
        },
        {
            "persona_id": "p-003",
            "country": "Danmark",
            "origin_country_da": "Norge",
            "age": 30,
            "age_band": "30-34",
            "sex": "female",
            "marital_status": "never_married",
            "municipality_code": "102",
            "municipality": "København",
            "education_level": "secondary_or_vocational",
            "education_source_code": "H20-H35",
            "labour_market_status": "employed",
            "detailed_status_code": "A",
            "job_function_code": "123",
            "job_function": "kontorarbejde",
            "cultural_context": "Norsk-dansk hverdagskontekst",
            "skills_and_expertise": ["struktur", "formidling", "samarbejde"],
            "hobbies_and_interests": ["svømning", "film", "bagning"],
            "career_goals_and_ambitions": "Vil gerne prøve nye opgaver.",
            "job_title": None,
            "current_relationship_status": "not_partnered",
            "partner_gender": None,
            "legal_status_detail": None,
            "persona": prose.replace("Rumænien", "Norge").replace(
                "Aarhus", "København"
            ),
        },
    ]


def _write_bundle(*, tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    bundle_dir = tmp_path / "bundle"
    normalized = bundle_dir / "normalized"
    normalized.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "age": [row["age"] for row in rows],
            "sex": [row["sex"] for row in rows],
            "municipality_code": [row["municipality_code"] for row in rows],
            "marital_status": [row["marital_status"] for row in rows],
            "count": [1] * len(rows),
            "suppressed": [False] * len(rows),
        }
    ).write_parquet(normalized / "folk1a_base_unpooled.parquet")
    pl.DataFrame(
        {
            "age_band": [row["age_band"] for row in rows],
            "municipality_code": [row["municipality_code"] for row in rows],
            "sex": [row["sex"] for row in rows],
            "education_source_code": ["H20"] * len(rows),
            "labour_market_status": [row["labour_market_status"] for row in rows],
            "count": [1] * len(rows),
            "suppressed": [False] * len(rows),
        }
    ).write_parquet(normalized / "ras209_joint_unpooled.parquet")
    pl.DataFrame(
        {
            "age_band": [row["age_band"] for row in rows],
            "age_key": [str(row["age"]) for row in rows],
            "sex": [row["sex"] for row in rows],
            "labour_market_status": [row["labour_market_status"] for row in rows],
            "detailed_status_code": [row["detailed_status_code"] for row in rows],
            "count": [1] * len(rows),
            "suppressed": [False] * len(rows),
        }
    ).write_parquet(normalized / "ras202_detail_unpooled.parquet")
    return bundle_dir


def _write_candidate_pair(
    *,
    tmp_path: Path,
    rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]] | None = None,
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    candidate_path = tmp_path / "candidate.parquet"
    original_path = tmp_path / "original.parquet"
    pl.DataFrame(candidate_rows if candidate_rows is not None else rows).write_parquet(
        candidate_path
    )
    pl.DataFrame(rows).write_parquet(original_path)
    return candidate_path, original_path


def test_validate_release_candidate_computes_marginal_tv_without_threshold(
    tmp_path: Path,
) -> None:
    """Candidate-vs-original marginal drift is reported as a metric only."""
    rows = _base_rows()
    candidate_rows = [dict(row) for row in rows]
    candidate_rows[1]["sex"] = "female"
    candidate_path, original_path = _write_candidate_pair(
        tmp_path=tmp_path, rows=rows, candidate_rows=candidate_rows
    )
    bundle_dir = _write_bundle(tmp_path=tmp_path, rows=candidate_rows)

    report = validate_release_candidate(
        candidate_path=candidate_path,
        original_path=original_path,
        bundle_dir=bundle_dir,
        expected_row_count=len(rows),
    )

    sex_metric = next(
        metric for metric in report.marginal_total_variation if metric.field == "sex"
    )
    assert sex_metric.total_variation == pytest.approx(1 / 3)
    assert report.passes_hard_gates is True
    assert "no retrospective thresholds" in report.source_distribution_claim


def test_validate_release_candidate_fails_loudly_on_missing_columns_and_counts(
    tmp_path: Path,
) -> None:
    """Missing columns and row-count mismatches raise errors instead of reports."""
    rows = _base_rows()
    candidate_path, original_path = _write_candidate_pair(tmp_path=tmp_path, rows=rows)
    bundle_dir = _write_bundle(tmp_path=tmp_path, rows=rows)

    with pytest.raises(ValueError, match="expected release size"):
        validate_release_candidate(
            candidate_path=candidate_path,
            original_path=original_path,
            bundle_dir=bundle_dir,
        )

    short_path = tmp_path / "short-candidate.parquet"
    pl.DataFrame(rows[:-1]).write_parquet(short_path)
    with pytest.raises(ValueError, match="row counts differ"):
        validate_release_candidate(
            candidate_path=short_path,
            original_path=original_path,
            bundle_dir=bundle_dir,
            expected_row_count=len(rows),
        )

    missing_path = tmp_path / "missing.parquet"
    pl.DataFrame(rows).drop("persona").write_parquet(missing_path)
    with pytest.raises(ValueError, match="missing required columns"):
        validate_release_candidate(
            candidate_path=missing_path,
            original_path=original_path,
            bundle_dir=bundle_dir,
            expected_row_count=len(rows),
        )


def test_validate_release_candidate_is_offline_and_uses_no_provider_config(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Generated checks are run locally without provider configuration."""
    rows = _base_rows()
    candidate_path, original_path = _write_candidate_pair(tmp_path=tmp_path, rows=rows)
    bundle_dir = _write_bundle(tmp_path=tmp_path, rows=rows)
    seen = False

    def local_check(frame: pl.DataFrame) -> GeneratedChecksReport:
        nonlocal seen
        seen = True
        return check_generated_fields(frame)

    monkeypatch.setattr(module, "check_generated_fields", local_check)

    report = validate_release_candidate(
        candidate_path=candidate_path,
        original_path=original_path,
        bundle_dir=bundle_dir,
        expected_row_count=len(rows),
    )

    assert seen is True
    assert report.generated_fields.hard_failure_counts == {}
    assert isinstance(report.generated_fields.advisory_prose_flag_counts, dict)


def test_validate_release_candidate_reports_id_and_prose_violations(
    tmp_path: Path,
) -> None:
    """Duplicate IDs, row-order drift, and unvetted prose changes fail."""
    rows = _base_rows()
    bundle_dir = _write_bundle(tmp_path=tmp_path, rows=rows)

    duplicate_rows = [dict(row) for row in rows]
    duplicate_rows[1]["persona_id"] = duplicate_rows[0]["persona_id"]
    duplicate_candidate, duplicate_original = _write_candidate_pair(
        tmp_path=tmp_path / "duplicates", rows=rows, candidate_rows=duplicate_rows
    )
    duplicate_report = validate_release_candidate(
        candidate_path=duplicate_candidate,
        original_path=duplicate_original,
        bundle_dir=bundle_dir,
        expected_row_count=len(rows),
    )
    assert duplicate_report.passes_hard_gates is False
    assert duplicate_report.identity.candidate_duplicate_id_rows == 1
    assert duplicate_report.collision_blank.candidate_duplicate_id_values == 1

    reordered_rows = [dict(rows[1]), dict(rows[0]), dict(rows[2])]
    reordered_candidate, reordered_original = _write_candidate_pair(
        tmp_path=tmp_path / "reordered", rows=rows, candidate_rows=reordered_rows
    )
    reordered_report = validate_release_candidate(
        candidate_path=reordered_candidate,
        original_path=reordered_original,
        bundle_dir=bundle_dir,
        expected_row_count=len(rows),
    )
    assert reordered_report.passes_hard_gates is False
    assert reordered_report.identity.ordered_ids_match is False
    assert reordered_report.hard_failure_counts["ordered_id_mismatch"] == 1

    changed_rows = [dict(row) for row in rows]
    changed_rows[0]["persona"] = f"{changed_rows[0]['persona']} Et rettet afsnit."
    changed_candidate, changed_original = _write_candidate_pair(
        tmp_path=tmp_path / "changed", rows=rows, candidate_rows=changed_rows
    )
    changed_report = validate_release_candidate(
        candidate_path=changed_candidate,
        original_path=changed_original,
        bundle_dir=bundle_dir,
        expected_row_count=len(rows),
    )
    assert changed_report.passes_hard_gates is False
    assert changed_report.prose.unvetted_changed_rows == 1

    manifest_path = tmp_path / "vetted-prose-patches.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "status": "vetted",
                "allowed_persona_id_hashes": [sha256_text("p-001")],
            }
        ),
        encoding="utf-8",
    )
    vetted_report = validate_release_candidate(
        candidate_path=changed_candidate,
        original_path=changed_original,
        bundle_dir=bundle_dir,
        vetted_patch_manifest_path=manifest_path,
        expected_row_count=len(rows),
    )
    assert vetted_report.passes_hard_gates is True
    assert vetted_report.prose.allowed_changed_rows == 1
    assert vetted_report.prose.vetted_patch_manifest_sha256 is not None


def test_validate_release_candidate_reports_unsupported_source_cells(
    tmp_path: Path,
) -> None:
    """Unsupported FOLK1A/RAS209 cells are hard failures."""
    rows = _base_rows()
    candidate_rows = [dict(row) for row in rows]
    candidate_rows[0]["municipality_code"] = "999"
    candidate_path, original_path = _write_candidate_pair(
        tmp_path=tmp_path, rows=rows, candidate_rows=candidate_rows
    )
    bundle_dir = _write_bundle(tmp_path=tmp_path, rows=rows)

    report = validate_release_candidate(
        candidate_path=candidate_path,
        original_path=original_path,
        bundle_dir=bundle_dir,
        expected_row_count=len(rows),
    )

    assert report.passes_hard_gates is False
    assert report.source_support.unsupported_rows["folk1a"] == 1
    assert report.source_support.unsupported_rows["ras209"] == 1
    assert report.source_support.unsupported_rows["ras202"] == 0
    assert report.hard_failure_counts["folk1a_unsupported_rows"] == 1
