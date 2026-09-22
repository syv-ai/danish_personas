"""Offline contracts for the standalone persona dashboard."""

import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from click.testing import CliRunner

from scripts.build_persona_dashboard import (
    _generated_distribution,
    build_dashboard,
    load_dst_targets,
    main,
    persona_embedding,
)


def test_cli_translates_invalid_parquet_to_click_error(tmp_path: Path) -> None:
    """Malformed input produces a normal Click failure rather than a traceback."""
    input_path = tmp_path / "invalid.parquet"
    bundle_path = tmp_path / "bundle"
    input_path.write_text("not parquet", encoding="utf-8")
    bundle_path.mkdir()

    result = CliRunner().invoke(
        main,
        [
            "--input",
            str(input_path),
            "--bundle",
            str(bundle_path),
            "--output",
            str(tmp_path / "dashboard.html"),
        ],
    )

    assert result.exit_code != 0
    assert result.exception is not None
    assert "Error:" in result.output
    assert "Traceback" not in result.output


def test_dashboard_handles_frame_without_colour_fields(tmp_path: Path) -> None:
    """Minimal valid records use one ungrouped embedding trace."""
    frame = pl.DataFrame(
        {
            "persona_id": ["p-1", "p-2"],
            "persona": ["En rolig hverdag.", "En travl hverdag."],
        }
    )

    document = build_dashboard(frame=frame, bundle_path=tmp_path)

    assert "All personas" in document
    assert "Persona text embedding" in document


def test_dashboard_is_one_inline_plotly_html(tmp_path: Path) -> None:
    """The CLI writes one file with no external Plotly script."""
    input_path = tmp_path / "generated-personas.parquet"
    bundle_path = tmp_path / "bundle" / "normalized"
    output_path = tmp_path / "dashboard.html"
    bundle_path.mkdir(parents=True)
    _personas().write_parquet(input_path)
    pl.DataFrame({"age": [30, 40], "count": [1, 3]}).write_parquet(
        bundle_path / "folk_age_sampling.parquet"
    )

    result = CliRunner().invoke(
        main,
        [
            "--input",
            str(input_path),
            "--bundle",
            str(bundle_path.parent),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    document = output_path.read_text(encoding="utf-8")
    assert document.startswith("<!doctype html>")
    assert document.count("<html") == 1
    assert "Plotly.newPlot" in document
    assert "cdn.plot.ly" not in document
    assert "Anna beskriver en rolig hverdag" in document
    assert "Clara møder venner" in document


def _personas() -> pl.DataFrame:
    """Return a tiny complete dashboard fixture."""
    return pl.DataFrame(
        {
            "persona_id": ["p-1", "p-2", "p-3"],
            "first_name": ["Anna", "Bo", "Clara"],
            "persona": [
                "Anna beskriver en rolig hverdag med bøger og cykling.",
                "Bo arbejder med planlægning og holder af musik i byen.",
                "Clara møder venner og lærer nye færdigheder i fritiden.",
            ],
            "age": [30, 40, 40],
            "age_band": ["30-49", "30-49", "30-49"],
            "sex": ["female", "male", "female"],
            "marital_status": ["single", "married", "single"],
            "education_level": ["higher_education", "higher_education", "primary"],
            "labour_market_status": ["employed", "employed", "retired"],
            "region": ["North", "North", "South"],
            "municipality": ["Aarhus", "Aarhus", "Odense"],
            "origin_country_da": ["Danmark", "Sverige", "Danmark"],
            "job_function": ["Care", "Care", None],
            "job_title": ["Planlægger", "Rådgiver", None],
            "current_relationship_status": [
                "not_partnered",
                "partnered",
                "not_partnered",
            ],
            "partner_gender": [None, "female", None],
            "openness_score": [40.0, 50.0, 60.0],
            "conscientiousness_score": [40.0, 50.0, 60.0],
            "extraversion_score": [40.0, 50.0, 60.0],
            "agreeableness_score": [40.0, 50.0, 60.0],
            "neuroticism_score": [40.0, 50.0, 60.0],
        }
    )


def test_dst_targets_are_normalised_and_aggregated(tmp_path: Path) -> None:
    """Prepared target counts become proportions by semantic value."""
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    pl.DataFrame(
        {"age": [30, 30, 40], "count": [2, 3, 5], "sex": ["f", "f", "m"]}
    ).write_parquet(normalized / "folk_age_sampling.parquet")

    targets = load_dst_targets(bundle_path=tmp_path)

    assert targets["age"] == {"30": 0.5, "40": 0.5}
    assert targets["sex"] == {"f": 0.5, "m": 0.5}


def test_education_target_uses_generated_pooling_contract(tmp_path: Path) -> None:
    """Real RAS209 columns are pooled to the generated education labels."""
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    pl.DataFrame(
        {
            "municipality_code": ["101", "101", "101"],
            "municipality": ["København", "København", "København"],
            "region_code": ["084", "084", "084"],
            "region": ["Region Hovedstaden"] * 3,
            "age_band": ["30-34", "30-34", "30-34"],
            "sex": ["female", "female", "female"],
            "education_source_code": ["H10", "H20", "H70"],
            "education_level": ["primary", "upper_secondary", "masters"],
            "labour_market_status": ["employed", "employed", "employed"],
            "count": [1, 2, 3],
            "suppressed": [False, False, False],
        }
    ).write_parquet(normalized / "ras209_joint_unpooled.parquet")

    generated = _generated_distribution(
        frame=pl.DataFrame(
            {
                "education_level": [
                    "primary",
                    "secondary_or_vocational",
                    "higher_education",
                ]
            }
        ),
        field="education_level",
    )
    target = load_dst_targets(bundle_path=tmp_path)["education_level"]

    assert set(target) == set(generated)
    assert sum(target.values()) == pytest.approx(1.0)


def test_identical_prose_embedding_is_exactly_repeatable(tmp_path: Path) -> None:
    """Rank-deficient identical prose is stable across calls and processes."""
    frame = pl.DataFrame(
        {
            "persona": ["Samme rolige tekst."] * 4,
            "persona_id": ["p-1", "p-2", "p-3", "p-4"],
        }
    )
    input_path = tmp_path / "identical.parquet"
    frame.write_parquet(input_path)

    first = persona_embedding(frame=frame)
    second = persona_embedding(frame=frame)
    code = (
        "import json; import polars as pl; "
        "from scripts.build_persona_dashboard import persona_embedding; "
        f"frame = pl.read_parquet({str(input_path)!r}); "
        "print(json.dumps(persona_embedding(frame=frame)))"
    )
    process = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )

    assert first == second
    assert first == [tuple(row) for row in json.loads(process.stdout)]
    assert {coordinate[1] for coordinate in first} == {0.0}


def test_job_function_series_use_equivalent_eligible_denominators(
    tmp_path: Path,
) -> None:
    """Job-function generated and conditioned target series each sum to one."""
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    pl.DataFrame(
        {
            "job_function_code": ["11", "21", "11", "21"],
            "job_function": ["Management", "Science", "Management", "Science"],
            "sex": ["female", "female", "male", "male"],
            "count": [90, 10, 20, 80],
        }
    ).write_parquet(normalized / "job_function_sex_marginal.parquet")
    frame = pl.DataFrame(
        {
            "job_function": ["Management", "Science", None],
            "sex": ["female", "male", "female"],
        }
    )

    generated = _generated_distribution(frame=frame, field="job_function")
    target = load_dst_targets(bundle_path=tmp_path, frame=frame)["job_function"]

    assert generated == {"Management": 0.5, "Science": 0.5}
    assert sum(generated.values()) == pytest.approx(1.0)
    assert target == pytest.approx({"Management": 0.55, "Science": 0.45})
    assert sum(target.values()) == pytest.approx(1.0)


def test_job_function_target_is_omitted_without_conditioning_fields(
    tmp_path: Path,
) -> None:
    """No target overlay is claimed when generated sex conditioning is absent."""
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    pl.DataFrame(
        {
            "job_function": ["Management", "Science"],
            "sex": ["female", "female"],
            "count": [1, 1],
        }
    ).write_parquet(normalized / "job_function_sex_marginal.parquet")

    targets = load_dst_targets(
        bundle_path=tmp_path, frame=pl.DataFrame({"job_function": ["Management"]})
    )

    assert "job_function" not in targets


def test_rank_deficient_embedding_is_exactly_repeatable() -> None:
    """Duplicate-document rank deficiency does not change repeated coordinates."""
    frame = pl.DataFrame(
        {
            "persona": ["rolig cykeltur", "rolig cykeltur", "travl arbejdsdag"] * 2,
            "persona_id": [f"p-{index}" for index in range(6)],
        }
    )

    assert persona_embedding(frame=frame) == persona_embedding(frame=frame)


def test_tied_singular_embedding_is_exactly_repeatable_across_processes(
    tmp_path: Path,
) -> None:
    """Tied singular subspaces have identical coordinates in fresh processes."""
    frame = pl.DataFrame(
        {
            "persona": [
                "alpha beta",
                "alpha beta",
                "gamma delta",
                "gamma delta",
                "epsilon zeta",
                "epsilon zeta",
                "theta iota",
                "theta iota",
            ],
            "persona_id": [f"p-{index}" for index in range(8)],
        }
    )
    input_path = tmp_path / "tied-singular-values.parquet"
    frame.write_parquet(input_path)
    code = (
        "import json; import polars as pl; "
        "from scripts.build_persona_dashboard import persona_embedding; "
        f"frame = pl.read_parquet({str(input_path)!r}); "
        "print(json.dumps(persona_embedding(frame=frame), separators=(',', ':')))"
    )

    outputs = [
        subprocess.run(
            [sys.executable, "-c", code], check=True, capture_output=True, text=True
        ).stdout
        for _ in range(4)
    ]

    assert outputs[0] == outputs[1]
