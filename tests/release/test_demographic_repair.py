"""Offline orchestration tests for demographic repair."""

from pathlib import Path

import polars as pl

from danish_personas.release.demographic_repair import repair_demographics


def test_repair_preserves_identity_and_protected_columns_on_missing_sources(
    tmp_path: Path,
) -> None:
    """Missing offline inputs block repair without mutating the release frame."""
    frame = pl.DataFrame(
        {
            "persona_id": ["p1", "p2"],
            "age": [25, 31],
            "municipality_code": [101, 101],
            "region": ["Capital", "Capital"],
            "sex": ["female", "male"],
            "origin_country": ["DK", "DK"],
            "ocean_openness": [3, 4],
            "marital_status": ["never_married", "married_or_separated"],
            "education_source_code": ["H10", "H40-H80"],
            "education_level": ["primary", "higher_education"],
            "labour_market_status": ["employed", "student"],
            "detailed_status_code": ["05", "130"],
        }
    )
    repaired, report = repair_demographics(frame, tmp_path)
    assert repaired.equals(frame)
    assert repaired.get_column("persona_id").to_list() == ["p1", "p2"]
    assert repaired.select(
        "age", "municipality_code", "region", "sex", "origin_country", "ocean_openness"
    ).equals(
        frame.select(
            "age",
            "municipality_code",
            "region",
            "sex",
            "origin_country",
            "ocean_openness",
        )
    )
    assert len(report["stages"]) == 3
    assert all(stage["changed_count"] == 0 for stage in report["stages"])
    assert all(stage["blocker"] for stage in report["stages"])
    assert report["blocking_infeasibility"]


def test_newly_eligible_row_gets_source_weighted_job_function(tmp_path: Path) -> None:
    """Newly eligible status gets a valid sex-conditioned source allocation."""
    frame = _release_frame().with_columns(
        pl.Series("detailed_status_code", ["130", "15"]),
        pl.Series("detailed_status", ["Student", "Employee"]),
        pl.Series("job_function_code", [None, "11"]),
        pl.Series("job_function", [None, "Administration"]),
        pl.Series("job_function_resolution", ["not_applicable", "lons20_sex_marginal"]),
        pl.Series("labour_market_status", ["employed", "employed"]),
    )
    _write_folk1a(tmp_path)
    _write_ras209(tmp_path, ["employed", "employed"])
    _write_ras202(tmp_path, ["employed", "employed"], employee_code="15")
    _write_job_function_marginal(tmp_path)

    repaired, report = repair_demographics(frame, tmp_path)

    assert not report["blocking_infeasibility"]
    assert repaired["job_function_code"][0] in {"11", "12"}
    assert repaired["job_function"][0] in {"Administration", "Management"}
    assert repaired["job_function_resolution"][0] == "lons20_sex_marginal"
    assert repaired["job_function_code"][1] == "11"
    assert repaired["job_title"].to_list() == [None, "Accountant"]
    detail = next(
        stage for stage in report["stages"] if stage["name"] == "ras202_detail"
    )
    assert detail["diagnostics"]["newly_eligible_job_functions_assigned"] == 1


def test_newly_eligible_without_job_function_source_rolls_back(tmp_path: Path) -> None:
    """Missing job-function source blocks the complete repair atomically."""
    frame = _release_frame().with_columns(
        pl.Series("detailed_status_code", ["130", "15"]),
        pl.Series("detailed_status", ["Student", "Employee"]),
        pl.Series("job_function_code", [None, "11"]),
        pl.Series("job_function", [None, "Administration"]),
        pl.Series("job_function_resolution", ["not_applicable", "lons20_sex_marginal"]),
        pl.Series("labour_market_status", ["employed", "employed"]),
    )
    _write_folk1a(tmp_path)
    _write_ras209(tmp_path, ["employed", "employed"])
    _write_ras202(tmp_path, ["employed", "employed"], employee_code="15")

    repaired, report = repair_demographics(frame, tmp_path)

    assert report["blocking_infeasibility"]
    assert repaired.equals(frame)


def test_broad_status_crossing_with_infeasible_detail_rolls_back(
    tmp_path: Path,
) -> None:
    """An unavailable exact RAS202 quota never leaves a broad-status-only edit."""
    frame = _release_frame()
    _write_folk1a(tmp_path)
    _write_ras209(tmp_path, ["employed", "student"])
    _write_ras202(tmp_path, ["employed", "employed"])
    repaired, report = repair_demographics(frame, tmp_path)
    assert repaired.equals(frame)
    assert report["blocking_infeasibility"]
    assert any(stage["name"] == "ras202_detail" for stage in report["stages"])


def test_broad_status_crossing_updates_detail_and_paired_fields(tmp_path: Path) -> None:
    """A feasible repair keeps detailed labels and job fields consistent."""
    frame = _release_frame()
    _write_folk1a(tmp_path)
    _write_ras209(tmp_path, ["employed", "student"])
    _write_ras202(tmp_path, ["employed", "student"])
    repaired, report = repair_demographics(frame, tmp_path)
    assert not report["blocking_infeasibility"]
    assert repaired["labour_market_status"].to_list() == ["employed", "student"]
    assert repaired["detailed_status_code"].to_list() == ["05", "130"]
    assert repaired["detailed_status"].to_list() == ["Employee", "Student"]
    assert repaired["detailed_status_resolution"].to_list() == [
        "status",
        "age_band_sex_status",
    ]
    assert repaired["job_function_code"].to_list() == ["11", None]
    assert repaired["job_function"].to_list() == ["Administration", None]
    assert repaired["job_function_resolution"].to_list() == [
        "lons20_sex_marginal",
        "not_applicable",
    ]
    assert repaired["job_title"].to_list() == ["Engineer", None]
    assert repaired["persona_id"].to_list() == frame["persona_id"].to_list()


def _release_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "persona_id": ["p1", "p2"],
            "age": [25, 31],
            "municipality_code": [101, 101],
            "sex": ["female", "male"],
            "marital_status": ["never_married", "married_or_separated"],
            "education_source_code": ["H10", "H10"],
            "education_level": ["primary", "primary"],
            "labour_market_status": ["employed", "employed"],
            "detailed_status_code": ["05", "05"],
            "detailed_status": ["Employee", "Employee"],
            "detailed_status_resolution": ["status", "status"],
            "job_function_code": ["11", "11"],
            "job_function": ["Administration", "Administration"],
            "job_function_resolution": ["lons20_sex_marginal", "lons20_sex_marginal"],
            "job_title": ["Engineer", "Accountant"],
        }
    )


def _write_folk1a(tmp_path: Path) -> None:
    path = tmp_path / "normalized" / "folk1a_base_unpooled.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "age": [25, 31],
            "municipality_code": [101, 101],
            "sex": ["female", "male"],
            "marital_status": ["never_married", "married_or_separated"],
            "count": [1, 1],
            "suppressed": [False, False],
        }
    ).write_parquet(path)


def _write_ras209(tmp_path: Path, statuses: list[str]) -> None:
    path = tmp_path / "normalized" / "ras209_joint_unpooled.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "age_band": ["25-29", "30-34"],
            "municipality_code": [101, 101],
            "sex": ["female", "male"],
            "education_source_code": ["H10", "H10"],
            "education_level": ["primary", "primary"],
            "labour_market_status": statuses,
            "count": [1, 1],
            "suppressed": [False, False],
        }
    ).write_parquet(path)


def _write_job_function_marginal(tmp_path: Path) -> None:
    path = tmp_path / "normalized" / "job_function_sex_marginal.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "job_function_code": ["11", "12", "11", "12"],
            "job_function": [
                "Administration",
                "Management",
                "Administration",
                "Management",
            ],
            "sex": ["female", "female", "male", "male"],
            "count": [3, 1, 1, 3],
        }
    ).write_parquet(path)


def _write_ras202(
    tmp_path: Path, statuses: list[str], employee_code: str = "05"
) -> None:
    path = tmp_path / "normalized" / "ras202_detail_unpooled.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    codes = ["130" if status == "student" else employee_code for status in statuses]
    labels = ["Student" if code == "130" else "Employee" for code in codes]
    pl.DataFrame(
        {
            "age_key": ["25", "31"],
            "sex": ["female", "male"],
            "labour_market_status": statuses,
            "detailed_status_code": codes,
            "detailed_status": labels,
            "count": [1, 1],
            "suppressed": [False, False],
        }
    ).write_parquet(path)
