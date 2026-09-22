"""Tests for the importable deterministic input workflow."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from danish_personas import workflows
from danish_personas.models import SamplingConfig


def test_standard_sample_runs_all_gates_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The standard workflow validates smoke output before statistical output."""
    order: list[str] = []
    sampling = SamplingConfig(
        version=4,
        seed=1,
        smoke_rows=2,
        statistical_rows=3,
        country="Danmark",
        minimum_age=18,
        maximum_age=125,
        publication_geography="municipality",
        smoothing=0.0,
        ocean={
            "mean": 50,
            "standard_deviation": 10,
            "minimum": 20,
            "maximum": 80,
            "label_boundaries": [35, 45, 55, 65],
            "labels": ["very_low", "low", "average", "high", "very_high"],
        },
    )
    raw_parent = tmp_path / "data"
    monkeypatch.setattr(workflows, "load_yaml_model", lambda **_: sampling)
    monkeypatch.setattr(workflows, "restore_raw_sources", lambda **_: 1)
    monkeypatch.setattr(
        workflows,
        "prepare_bundle",
        lambda **_: order.append("prepare") or Path("bundle"),
    )
    monkeypatch.setattr(
        workflows,
        "validate_sources",
        lambda **_: order.append("sources") or SimpleNamespace(passed=True),
    )
    runs = iter((Path("smoke"), Path("statistical")))
    monkeypatch.setattr(
        workflows,
        "generate_records",
        lambda **_: order.append("generate") or next(runs),
    )
    monkeypatch.setattr(
        workflows,
        "validate_demographics",
        lambda **_: order.append("demographics") or SimpleNamespace(passed=True),
    )
    sample = tmp_path / "sample.parquet"
    monkeypatch.setattr(
        workflows,
        "freeze_sample",
        lambda **kwargs: order.append("freeze") or kwargs["output"],
    )

    result = workflows.prepare_standard_sample(
        archive_path=tmp_path / "archive.tar.zst",
        raw_parent=raw_parent,
        sample_path=sample,
    )

    assert result == (sample, sample.with_suffix(".manifest.json"))
    assert order == [
        "prepare",
        "sources",
        "generate",
        "demographics",
        "generate",
        "demographics",
        "freeze",
    ]


@pytest.mark.parametrize("existing_artifact", ["sample", "manifest"])
def test_standard_sample_rebuilds_invalid_existing_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    existing_artifact: str,
) -> None:
    """An invalid derived sample artefact is rebuilt after upstream gates pass."""
    sampling = SamplingConfig(
        version=4,
        seed=1,
        smoke_rows=2,
        statistical_rows=3,
        country="Danmark",
        minimum_age=18,
        maximum_age=125,
        publication_geography="municipality",
        smoothing=0.0,
        ocean={
            "mean": 50,
            "standard_deviation": 10,
            "minimum": 20,
            "maximum": 80,
            "label_boundaries": [35, 45, 55, 65],
            "labels": ["very_low", "low", "average", "high", "very_high"],
        },
    )
    monkeypatch.setattr(workflows, "load_yaml_model", lambda **_: sampling)
    monkeypatch.setattr(workflows, "restore_raw_sources", lambda **_: 1)
    monkeypatch.setattr(workflows, "prepare_bundle", lambda **_: Path("bundle"))
    monkeypatch.setattr(
        workflows, "validate_sources", lambda **_: SimpleNamespace(passed=True)
    )
    runs = iter((Path("smoke"), Path("statistical")))
    monkeypatch.setattr(workflows, "generate_records", lambda **_: next(runs))
    monkeypatch.setattr(
        workflows, "validate_demographics", lambda **_: SimpleNamespace(passed=True)
    )
    monkeypatch.setattr(workflows, "_run_id", lambda **_: "statistical-run")

    sample = tmp_path / "sample.parquet"
    manifest = sample.with_suffix(".manifest.json")
    if existing_artifact == "sample":
        sample.write_bytes(b"stale sample")
    else:
        manifest.write_text("{}", encoding="utf-8")
    rebuilt: list[Path] = []
    monkeypatch.setattr(
        workflows,
        "freeze_sample",
        lambda **kwargs: rebuilt.append(kwargs["output"]) or kwargs["output"],
    )

    caplog.set_level("WARNING", logger=workflows.LOGGER.name)
    assert workflows.prepare_standard_sample(
        archive_path=tmp_path / "archive.tar.zst",
        raw_parent=tmp_path / "data",
        sample_path=sample,
    ) == (sample, manifest)

    assert rebuilt == [sample]
    assert "stale" in caplog.text
    assert "rebuilding sample" in caplog.text


def test_standard_sample_stops_at_failed_source_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed source report prevents demographic generation."""
    generated = False
    monkeypatch.setattr(workflows, "restore_raw_sources", lambda **_: 1)
    monkeypatch.setattr(workflows, "prepare_bundle", lambda **_: Path("bundle"))
    monkeypatch.setattr(
        workflows, "validate_sources", lambda **_: SimpleNamespace(passed=False)
    )

    def generate(**_: object) -> Path:
        nonlocal generated
        generated = True
        return Path("run")

    monkeypatch.setattr(workflows, "generate_records", generate)

    with pytest.raises(ValueError, match="Source validation failed"):
        workflows.prepare_standard_sample(raw_parent=tmp_path / "data")
    assert not generated
