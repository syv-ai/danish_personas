"""Offline contracts for the standalone persona dashboard."""

import json
import subprocess
import sys
from pathlib import Path

import httpx
import polars as pl
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

import scripts.build_persona_dashboard as dashboard
from scripts.build_persona_dashboard import (
    _distribution_chart,
    _generated_distribution,
    _relationship_pairs,
    build_dashboard,
    load_dst_targets,
    persona_embedding,
)

ROOT = Path(__file__).parents[2]


@pytest.fixture
def embedding_client(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[httpx.Client, list[dict[str, object]]]:
    """Return an offline OpenAI-compatible transport and captured requests."""
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": index, "embedding": [float(index), 1.0, 2.0]}
                    for index, _ in enumerate(payload["input"])
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(dashboard.httpx, "Client", lambda **_: client)
    return client, requests


def test_age_chart_is_sorted_numerically() -> None:
    """Age categories increase numerically rather than by frequency or text."""
    frame = pl.DataFrame({"age": [100, 19, 19, 18]})

    chart = _distribution_chart(
        frame=frame,
        field="age",
        title="Ages",
        source="test",
        target={"100": 0.2, "18": 0.3, "19": 0.5},
    )

    assert '"x":["18","19","100"]' in chart


def test_cli_translates_invalid_parquet_to_hydra_error(tmp_path: Path) -> None:
    """Malformed input produces a normal Hydra failure rather than a traceback."""
    input_path = tmp_path / "invalid.parquet"
    bundle_path = tmp_path / "bundle"
    input_path.write_text("not parquet", encoding="utf-8")
    bundle_path.mkdir()

    result = _run_dashboard(
        input_path=input_path,
        bundle_path=bundle_path,
        output_path=tmp_path / "dashboard.html",
        cwd=tmp_path,
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert "ERROR parquet: File out of specification" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / ".hydra").exists()


def _run_dashboard(
    *, input_path: Path, bundle_path: Path, output_path: Path, cwd: Path
) -> subprocess.CompletedProcess[str]:
    """Run the public dashboard script with Hydra overrides.

    Returns:
        The completed offline subprocess.
    """
    script = ROOT / "src/scripts/build_persona_dashboard.py"
    return subprocess.run(
        [
            sys.executable,
            str(script),
            f"persona_dashboard.input={input_path}",
            f"persona_dashboard.bundle={bundle_path}",
            f"persona_dashboard.output={output_path}",
        ],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def _dashboard_config(*, overrides: list[str]) -> DictConfig:
    """Compose dashboard settings from the canonical Hydra entry point.

    Returns:
        Composed configuration with the supplied overrides.
    """
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        return compose(config_name="config", overrides=overrides)


def test_dashboard_handles_frame_without_colour_fields(
    tmp_path: Path, embedding_client: tuple[httpx.Client, list[dict[str, object]]]
) -> None:
    """Minimal valid records use one ungrouped embedding trace."""
    frame = pl.DataFrame(
        {
            "persona_id": ["p-1", "p-2"],
            "persona": ["En rolig hverdag.", "En travl hverdag."],
        }
    )

    document = build_dashboard(
        frame=frame, bundle_path=tmp_path, embedding_client=embedding_client[0]
    )

    assert "All personas" in document
    assert "Persona text embedding" in document


def test_dashboard_is_one_inline_plotly_html(
    tmp_path: Path,
    embedding_client: tuple[httpx.Client, list[dict[str, object]]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI writes one file with no external Plotly script."""
    input_path = tmp_path / "generated-personas.parquet"
    bundle_path = tmp_path / "bundle" / "normalized"
    output_path = tmp_path / "dashboard.html"
    bundle_path.mkdir(parents=True)
    _personas().write_parquet(input_path)
    pl.DataFrame({"age": [30, 40], "count": [1, 3]}).write_parquet(
        bundle_path / "folk_age_sampling.parquet"
    )
    config = _dashboard_config(
        overrides=[
            f"persona_dashboard.input={input_path}",
            f"persona_dashboard.bundle={bundle_path.parent}",
            f"persona_dashboard.output={output_path}",
            "persona_dashboard.embedding_base_url=http://test",
        ]
    )

    dashboard._run(config=config)

    assert capsys.readouterr().out == f"{output_path}\n"
    document = output_path.read_text(encoding="utf-8")
    assert document.startswith("<!doctype html>")
    assert document.count("<html") == 1
    assert "Plotly.newPlot" in document
    assert "cdn.plot.ly" not in document
    assert "Unnamed persona" in document
    assert "first_name" not in document
    assert "Provenance and semantics" not in document
    assert "TF-IDF" not in document
    assert embedding_client[1]


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


def test_embedding_http_errors_are_not_silenced() -> None:
    """Provider failures stop dashboard construction rather than fabricating points."""
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(503, json={"error": "unavailable"})
        )
    )
    frame = pl.DataFrame({"persona": ["tekst"]})

    with pytest.raises(httpx.HTTPStatusError):
        persona_embedding(frame=frame, http_client=client)


def test_embedding_requests_are_batched_and_configured(
    embedding_client: tuple[httpx.Client, list[dict[str, object]]],
) -> None:
    """The local client sends bounded batches with the selected model alias."""
    frame = pl.DataFrame({"persona": [f"tekst {index}" for index in range(5)]})

    persona_embedding(
        frame=frame,
        base_url="http://embedding.test/v1",
        model="test-model",
        batch_size=2,
        http_client=embedding_client[0],
    )

    inputs = [request["input"] for request in embedding_client[1]]
    assert all(isinstance(value, list) for value in inputs)
    assert [len(value) for value in inputs if isinstance(value, list)] == [2, 2, 1]
    assert all(request["model"] == "test-model" for request in embedding_client[1])


def test_horizontal_layout_has_card_title_only() -> None:
    """Long labels use horizontal bars without a duplicated Plotly title."""
    chart = _distribution_chart(
        frame=pl.DataFrame({"municipality": ["A very long municipality"]}),
        field="municipality",
        title="Municipalities",
        source="RAS209 municipality marginal",
        target=None,
    )

    assert '"orientation":"h"' in chart
    assert '"title":{"text":"Municipalities"}' not in chart
    assert chart.count("<h2>Municipalities</h2>") == 1


def test_identical_prose_embedding_is_exactly_repeatable(
    embedding_client: tuple[httpx.Client, list[dict[str, object]]],
) -> None:
    """Rank-deficient identical prose is stable across calls and processes."""
    frame = pl.DataFrame(
        {
            "persona": ["Samme rolige tekst."] * 4,
            "persona_id": ["p-1", "p-2", "p-3", "p-4"],
        }
    )
    first = persona_embedding(frame=frame, http_client=embedding_client[0])
    second = persona_embedding(frame=frame, http_client=embedding_client[0])

    assert first == second
    assert {coordinate[1] for coordinate in first} == {0.0}
    assert len(embedding_client[1]) == 2


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


def test_municipality_target_comes_from_ras209_marginal(tmp_path: Path) -> None:
    """Municipality overlays use RAS209 rather than the age-sampling table."""
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    pl.DataFrame({"municipality": ["Folk", "Folk"], "count": [100, 100]}).write_parquet(
        normalized / "folk_age_sampling.parquet"
    )
    pl.DataFrame(
        {
            "municipality": ["Ras", "Ras", "Other"],
            "education_level": ["primary", "primary", "primary"],
            "labour_market_status": ["employed", "employed", "employed"],
            "count": [1, 2, 1],
        }
    ).write_parquet(normalized / "ras209_joint_unpooled.parquet")

    assert load_dst_targets(bundle_path=tmp_path)["municipality"] == {
        "Ras": 0.75,
        "Other": 0.25,
    }


def test_not_stated_is_removed_before_distribution_normalisation(
    tmp_path: Path,
) -> None:
    """Education charts omit undisclosed values from both generated and target pools."""
    frame = pl.DataFrame({"education_level": ["primary", "not_stated", "primary"]})
    assert _generated_distribution(frame=frame, field="education_level") == {
        "primary": 1.0
    }
    normalized = tmp_path / "normalized"
    normalized.mkdir()
    pl.DataFrame(
        {
            "education_level": ["primary", "not_stated"],
            "labour_market_status": ["employed", "employed"],
            "count": [2, 8],
        }
    ).write_parquet(normalized / "ras209_joint_unpooled.parquet")
    assert load_dst_targets(bundle_path=tmp_path)["education_level"] == {"primary": 1.0}


def test_rank_deficient_embedding_is_exactly_repeatable(
    embedding_client: tuple[httpx.Client, list[dict[str, object]]],
) -> None:
    """Duplicate-document rank deficiency does not change repeated coordinates."""
    frame = pl.DataFrame(
        {
            "persona": ["rolig cykeltur", "rolig cykeltur", "travl arbejdsdag"] * 2,
            "persona_id": [f"p-{index}" for index in range(6)],
        }
    )

    assert persona_embedding(
        frame=frame, http_client=embedding_client[0]
    ) == persona_embedding(frame=frame, http_client=embedding_client[0])


def test_relationship_pair_chart_uses_partnered_gender_pairs() -> None:
    """The derived chart has only the three labelled partnered combinations."""
    frame = pl.DataFrame(
        {
            "sex": ["male", "male", "female", "female"],
            "partner_gender": ["female", "male", "female", "male"],
            "current_relationship_status": [
                "partnered",
                "partnered",
                "partnered",
                "not_partnered",
            ],
        }
    )

    assert _relationship_pairs(frame=frame) == ["man-woman", "man-man", "woman-woman"]
    chart = _distribution_chart(
        frame=frame,
        field="relationship_pair",
        title="Partner relationship pair",
        source="synthetic",
        target=None,
    )
    assert all(label in chart for label in ("man-woman", "man-man", "woman-woman"))
    assert "Partner gender" not in chart


def test_tied_singular_embedding_is_exactly_repeatable(
    embedding_client: tuple[httpx.Client, list[dict[str, object]]],
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
    outputs = [
        persona_embedding(frame=frame, http_client=embedding_client[0])
        for _ in range(4)
    ]

    assert outputs[0] == outputs[1]
