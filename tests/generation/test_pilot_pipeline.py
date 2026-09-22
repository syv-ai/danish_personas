"""Pilot identity and mechanical merge tests."""

import json
from pathlib import Path

import polars as pl
import pytest
from generation_test_helpers import MockGenerationClient, write_generation_inputs

from danish_personas.generation.pilot import run_pilot


def test_pilot_identity_changes_when_prompt_context_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing an effective prompt creates a new pilot artefact directory."""
    paths = write_generation_inputs(root=tmp_path)
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", MockGenerationClient
    )
    MockGenerationClient.requests = 0
    _run_test_pilot(paths=paths, output_dir=tmp_path / "pilot", rows=1)
    paths["prompt"].write_text("En ændret dansk prompt", encoding="utf-8")
    _run_test_pilot(paths=paths, output_dir=tmp_path / "pilot", rows=1)

    pilot_dirs = list((tmp_path / "pilot").iterdir())
    manifests = [
        json.loads((pilot_dir / "pilot-manifest.json").read_text(encoding="utf-8"))
        for pilot_dir in pilot_dirs
    ]
    assert len(pilot_dirs) == 2
    assert manifests[0]["pilot_id"] != manifests[1]["pilot_id"]
    assert len({manifest["generation_context_sha256"] for manifest in manifests}) == 2


def _run_test_pilot(*, paths: dict[str, Path], output_dir: Path, rows: int) -> Path:
    """Run a bounded pilot from focused flat generation fixtures.

    Returns:
        Completed pilot directory.
    """
    return run_pilot(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=output_dir,
        rows=rows,
        batch_size=2,
        concurrency=1,
        delay_between_batches=0.0,
        maximum_total_requests=10,
        input_price_per_million=0.3,
        output_price_per_million=1.2,
    )


def test_pilot_merges_schema_parsed_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pilot runner merges each bounded invocation exactly once."""
    paths = write_generation_inputs(root=tmp_path)
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", MockGenerationClient
    )
    MockGenerationClient.requests = 0

    pilot_dir = _run_test_pilot(paths=paths, output_dir=tmp_path / "pilot", rows=2)

    output = pl.read_parquet(pilot_dir / "generated-personas.parquet")
    assert output.get_column("persona_id").to_list() == ["persona-1", "persona-2"]
    assert MockGenerationClient.requests == 2
    assert not (pilot_dir / "pilot-validation-report.json").exists()
    assert not list((pilot_dir / "batches").glob("*/validation-report.json"))
