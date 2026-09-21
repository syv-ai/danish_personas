"""Public pilot-service lifecycle and scheduling tests."""

import json
from pathlib import Path

import polars as pl
import pytest
from generation_test_helpers import MockGenerationClient, write_generation_inputs

from danish_personas.generation.pilot import run_pilot
from danish_personas.io import write_json


def test_pilot_service_preserves_order_enforces_budget_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public pilot service orders shards, accounts for requests, and resumes."""
    paths = write_generation_inputs(root=tmp_path)
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", MockGenerationClient
    )
    MockGenerationClient.requests = 0

    pilot_kwargs = {
        "input_path": paths["sample"],
        "sample_manifest_path": paths["sample_manifest"],
        "config_path": paths["config"],
        "output_dir": tmp_path / "pilot",
        "rows": 2,
        "batch_size": 1,
        "concurrency": 1,
        "delay_between_batches": 0.0,
        "maximum_total_requests": 10,
        "input_price_per_million": 0.3,
        "output_price_per_million": 1.2,
    }
    progress: list[int] = []
    pilot_dir = run_pilot(**pilot_kwargs, progress_callback=progress.append)
    output = pl.read_parquet(pilot_dir / "generated-personas.parquet")
    assert output.get_column("persona_id").to_list() == ["persona-1", "persona-2"]
    assert MockGenerationClient.requests == 2
    assert progress == [1, 1]

    progress.clear()
    run_pilot(**pilot_kwargs, progress_callback=progress.append)
    assert MockGenerationClient.requests == 2
    assert progress == [1, 1]

    with pytest.raises(ValueError, match="Worst-case pilot requests"):
        run_pilot(**{**pilot_kwargs, "maximum_total_requests": 1})

    tampered = json.loads(paths["sample_manifest"].read_text(encoding="utf-8"))
    tampered["source_run_id"] = "different-upstream"
    write_json(path=paths["sample_manifest"], payload=tampered)
    with pytest.raises(ValueError, match="different upstream run"):
        run_pilot(**{**pilot_kwargs, "output_dir": tmp_path / "provenance-pilot"})


@pytest.mark.parametrize("concurrency", [0, 9])
def test_pilot_service_rejects_invalid_concurrency_before_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, concurrency: int
) -> None:
    """Invalid concurrency limits make no provider calls or output files."""
    paths = write_generation_inputs(root=tmp_path)
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", MockGenerationClient
    )
    MockGenerationClient.requests = 0
    output_dir = tmp_path / f"invalid-concurrency-{concurrency}"

    with pytest.raises(ValueError, match="between 1 and 8"):
        run_pilot(
            input_path=paths["sample"],
            sample_manifest_path=paths["sample_manifest"],
            config_path=paths["config"],
            output_dir=output_dir,
            rows=1,
            batch_size=1,
            concurrency=concurrency,
            delay_between_batches=0.0,
            maximum_total_requests=10,
            input_price_per_million=0.3,
            output_price_per_million=1.2,
        )

    assert MockGenerationClient.requests == 0
    assert not output_dir.exists()


def test_pilot_service_stops_after_early_shard_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed shard prevents later merging and cancels pending work."""
    paths = write_generation_inputs(root=tmp_path)

    def fail_generation(**kwargs: object) -> Path:
        raise ValueError("synthetic shard failure")

    monkeypatch.setattr(
        "danish_personas.generation.pilot.generate_personas", fail_generation
    )
    progress: list[int] = []
    with pytest.raises(ValueError, match="synthetic shard failure"):
        run_pilot(
            input_path=paths["sample"],
            sample_manifest_path=paths["sample_manifest"],
            config_path=paths["config"],
            output_dir=tmp_path / "failed-pilot",
            rows=2,
            batch_size=1,
            concurrency=1,
            delay_between_batches=0.0,
            maximum_total_requests=10,
            input_price_per_million=0.3,
            output_price_per_million=1.2,
            progress_callback=progress.append,
        )
    assert progress == []
    assert not list((tmp_path / "failed-pilot").glob("*/pilot-manifest.json"))
