"""Tests for local workflow orchestration helpers."""

import json
import os
from pathlib import Path

import polars as pl
import pytest

from danish_personas.io import load_env_file, sha256_file, write_json
from danish_personas.models import RunManifest
from danish_personas.workflow import (
    export_record,
    load_run_frame,
    read_pointer,
    select_fields,
    summarise_run,
    write_pointer,
)


def test_env_file_fills_gaps_without_overriding_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local environment file sets missing variables only."""
    monkeypatch.setenv("ALREADY_SET", "from-shell")
    monkeypatch.delenv("FROM_FILE", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        '# comment\nexport FROM_FILE="value"\nALREADY_SET=from-file\n',
        encoding="utf-8",
        newline="\n",
    )
    assert load_env_file(env_file) == ["FROM_FILE"]
    assert os.environ["FROM_FILE"] == "value"
    assert os.environ["ALREADY_SET"] == "from-shell"
    assert load_env_file(tmp_path / "absent") == []


def test_export_separates_inputs_from_generated_text(tmp_path: Path) -> None:
    """An exported record keeps its statistical inputs beside generated text."""
    record: dict[str, object] = {
        "persona_id": "id-0",
        "age": 30,
        "detailed_status": "Old-age pension",
        "hobbies_and_interests": ["kor", "løb"],
        "persona": "Dansk beskrivelse.",
    }
    json_path, markdown_path = export_record(
        record=record, run_dir=tmp_path / "run", output_dir=tmp_path / "exports"
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["statistical_inputs"]["detailed_status"] == "Old-age pension"
    assert payload["generated"]["hobbies_and_interests"] == ["kor", "løb"]
    assert "persona" not in payload["statistical_inputs"]
    assert "Dansk beskrivelse." in markdown_path.read_text(encoding="utf-8")


def test_load_run_frame_reads_and_reports_missing_data(tmp_path: Path) -> None:
    """A run is read through its manifest, and an empty directory is rejected."""
    build_run(tmp_path / "run")
    assert load_run_frame(tmp_path / "run").height == 2
    with pytest.raises(ValueError, match="No run manifest"):
        load_run_frame(tmp_path / "empty")


def build_run(run_dir: Path, rows: int = 2) -> pl.DataFrame:
    """Write a minimal demographic run to a directory.

    Args:
        run_dir:
            Destination directory.
        rows:
            Number of records to write.

    Returns:
        The written records.
    """
    frame = pl.DataFrame(
        {
            "persona_id": [f"id-{index}" for index in range(rows)],
            "age": [30 + index for index in range(rows)],
            "sex": ["female", "male"][:rows],
            "detailed_status": ["Employees - upper level", "Old-age pension"][:rows],
            **{
                f"{trait}_score": [50.0 + index for index in range(rows)]
                for trait in (
                    "openness",
                    "conscientiousness",
                    "extraversion",
                    "agreeableness",
                    "neuroticism",
                )
            },
        }
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    data_path = run_dir / "structured-records.parquet"
    frame.write_parquet(data_path)
    manifest = RunManifest(
        run_id="fixture-run",
        created_at="2026-09-15T00:00:00+00:00",
        bundle_id="fixture-bundle",
        bundle_manifest_sha256="0" * 64,
        sampling_config_sha256="1" * 64,
        rows=rows,
        seed=42,
        data_file=Path(data_path.name),
        data_sha256=sha256_file(data_path),
        logical_content_sha256="2" * 64,
        llm_calls=0,
    )
    write_json(path=run_dir / "run-manifest.json", payload=manifest)
    return frame


def test_pointers_record_and_report_stage_paths(tmp_path: Path) -> None:
    """A recorded pointer round-trips, and a missing pointer reports nothing."""
    pointer = tmp_path / "run"
    assert read_pointer(pointer) is None
    write_pointer(pointer, Path("data/runs/local/abc"))
    assert read_pointer(pointer) == Path("data/runs/local/abc")


def test_select_fields_keeps_the_identifier_and_rejects_unknown_fields() -> None:
    """Field selection always keeps the identifier and validates field names."""
    record: dict[str, object] = {"persona_id": "id-0", "age": 30, "sex": "female"}
    assert select_fields(record=record, fields=["sex"]) == {
        "persona_id": "id-0",
        "sex": "female",
    }
    assert select_fields(record=record, fields=[]) == record
    with pytest.raises(ValueError, match="Unknown fields"):
        select_fields(record=record, fields=["income"])


def test_summary_reports_shares_for_a_demographic_run(tmp_path: Path) -> None:
    """A summary reports the record count, breakdowns, and personality means."""
    lines = summarise_run(frame=build_run(tmp_path / "run"), top=3)
    report = "\n".join(lines)
    assert "Records: 2" in report
    assert "Content: demographic briefs" in report
    assert "Old-age pension" in report
    assert "openness" in report
