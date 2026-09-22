"""Offline contracts for the standalone persona dashboard."""

from pathlib import Path

import polars as pl
from click.testing import CliRunner

from scripts.build_persona_dashboard import load_dst_targets, main


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
            "education_level": ["higher", "higher", "basic"],
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
