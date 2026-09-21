"""Command-level generation workflow tests."""

# Test functions intentionally omit repetitive docstrings.
# ruff: noqa: D103

from pathlib import Path

import pytest
from click.testing import CliRunner
from generation_test_helpers import write_generation_inputs

from danish_personas import cli
from danish_personas.sampling.freeze import freeze_sample


def test_workflow_sample_immediately_plans_committed_v2_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An in-run frozen sample passes offline shard planning without credentials."""
    paths = write_generation_inputs(root=tmp_path)
    run_dir = paths["sample"].parent
    sample = freeze_sample(
        run_dir=run_dir, rows=1, output=run_dir / "text-development-seeds.parquet"
    )
    output_dir = tmp_path / "persona-runs"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def reject_provider(*args: object, **kwargs: object) -> None:
        pytest.fail(f"Dry-run planning constructed a provider: {args}, {kwargs}")

    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", reject_provider
    )
    arguments = [
        "personas",
        "shard",
        "--input",
        str(sample),
        "--sample-manifest",
        str(sample.with_suffix(".manifest.json")),
        "--config",
        "config/generation.yaml",
        "--output-dir",
        str(output_dir),
        "--rows",
        "1",
    ]
    first = CliRunner().invoke(cli.main, arguments)
    second = CliRunner().invoke(cli.main, arguments)

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert first.output == second.output
    assert Path(first.output.strip()).parent == output_dir
    assert not output_dir.exists()
