"""Direct behavioural tests for the public sample and pilot services."""

import json
from pathlib import Path

import polars as pl
import pytest
from click.testing import CliRunner
from test_pipeline import _MockClient, _write_inputs

from danish_personas import cli
from danish_personas.generation.pilot import run_pilot
from danish_personas.io import sha256_file, write_json
from danish_personas.sampling.freeze import SampleSizeError, freeze_sample
from scripts.freeze_demographic_sample import main as freeze_main


def test_freeze_service_is_deterministic_and_bounds_size(tmp_path: Path) -> None:
    """The public freezer preserves round-robin output and rejects oversized input."""
    paths = _write_inputs(root=tmp_path)
    run_dir = paths["sample"].parent
    first = run_dir / "first.parquet"
    second = run_dir / "second.parquet"

    freeze_sample(run_dir=run_dir, rows=2, output=first)
    freeze_sample(run_dir=run_dir, rows=2, output=second)

    assert sha256_file(first) == sha256_file(second)
    assert pl.read_parquet(first).get_column("persona_id").to_list() == [
        "persona-1",
        "persona-2",
    ]
    assert first.parent == run_dir
    assert first.with_suffix(".manifest.json").parent == run_dir
    with pytest.raises(SampleSizeError, match="exceeds the run row count"):
        freeze_sample(run_dir=run_dir, rows=3, output=first)


def test_freeze_service_rejects_dangling_output_symlink(tmp_path: Path) -> None:
    """A dangling output link is rejected before sample generation."""
    paths = _write_inputs(root=tmp_path)
    run_dir = paths["sample"].parent
    output = run_dir / "dangling.parquet"
    _make_symlink(path=output, target=tmp_path / "missing.parquet")

    with pytest.raises(ValueError, match="symlink"):
        freeze_sample(run_dir=run_dir, rows=1, output=output)

    assert output.is_symlink()


def _make_symlink(*, path: Path, target: Path, directory: bool = False) -> None:
    """Create a symlink or skip where the platform disallows symlinks."""
    try:
        path.symlink_to(target, target_is_directory=directory)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")


def test_freeze_service_rejects_manifest_symlink(tmp_path: Path) -> None:
    """A manifest link is rejected before the corresponding sample is written."""
    paths = _write_inputs(root=tmp_path)
    run_dir = paths["sample"].parent
    output = run_dir / "manifest-link.parquet"
    external = tmp_path / "external-manifest.json"
    external.write_text("protected", encoding="utf-8")
    _make_symlink(path=output.with_suffix(".manifest.json"), target=external)

    with pytest.raises(ValueError, match="symlink"):
        freeze_sample(run_dir=run_dir, rows=1, output=output)

    assert not output.exists()
    assert external.read_text(encoding="utf-8") == "protected"


def test_freeze_service_rejects_output_symlink_to_external_file(tmp_path: Path) -> None:
    """An output link cannot redirect a freeze to an external file."""
    paths = _write_inputs(root=tmp_path)
    run_dir = paths["sample"].parent
    external = tmp_path / "external.parquet"
    external.write_bytes(b"protected")
    output = run_dir / "external-output.parquet"
    _make_symlink(path=output, target=external)

    with pytest.raises(ValueError, match="symlink"):
        freeze_sample(run_dir=run_dir, rows=1, output=output)

    assert external.read_bytes() == b"protected"
    assert output.is_symlink()


def test_freeze_service_rejects_relocation_from_run_manifest(tmp_path: Path) -> None:
    """A frozen sample cannot be detached from its validated upstream evidence."""
    paths = _write_inputs(root=tmp_path)

    with pytest.raises(ValueError, match="inside its validated run directory"):
        freeze_sample(
            run_dir=paths["sample"].parent,
            rows=1,
            output=tmp_path / "relocated.parquet",
        )


def test_freeze_service_rejects_symlinked_parent(tmp_path: Path) -> None:
    """A path through a linked parent is not a direct run child."""
    paths = _write_inputs(root=tmp_path)
    run_dir = paths["sample"].parent
    real_parent = run_dir / "real-parent"
    real_parent.mkdir()
    linked_parent = run_dir / "linked-parent"
    _make_symlink(path=linked_parent, target=real_parent, directory=True)

    with pytest.raises(ValueError, match="inside its validated run directory"):
        freeze_sample(run_dir=run_dir, rows=1, output=linked_parent / "sample.parquet")


def test_freeze_service_rejects_traversal_alias(tmp_path: Path) -> None:
    """Lexical traversal aliases are rejected even when they normalise in-run."""
    paths = _write_inputs(root=tmp_path)
    run_dir = paths["sample"].parent
    output = run_dir / "nested" / ".." / "traversal.parquet"

    with pytest.raises(ValueError, match="traversal"):
        freeze_sample(run_dir=run_dir, rows=1, output=output)


def test_legacy_freeze_script_remains_compatible(tmp_path: Path) -> None:
    """The legacy freezer keeps its Click options and output contract."""
    paths = _write_inputs(root=tmp_path)
    output = paths["sample"].parent / "legacy.parquet"

    result = CliRunner().invoke(
        freeze_main,
        ["--run", str(paths["sample"].parent), "--rows", "2", "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    assert output.with_suffix(".manifest.json").exists()


def test_pilot_service_preserves_order_enforces_budget_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public pilot service orders shards, accounts for requests, and resumes."""
    paths = _write_inputs(root=tmp_path)
    monkeypatch.setattr("danish_personas.generation.pipeline.OpenAIClient", _MockClient)
    _MockClient.requests = 0

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
    pilot_dir = run_pilot(**pilot_kwargs)
    output = pl.read_parquet(pilot_dir / "generated-personas.parquet")
    assert output.get_column("persona_id").to_list() == ["persona-1", "persona-2"]
    assert _MockClient.requests == 4

    run_pilot(**pilot_kwargs)
    assert _MockClient.requests == 4

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
    paths = _write_inputs(root=tmp_path)
    monkeypatch.setattr("danish_personas.generation.pipeline.OpenAIClient", _MockClient)
    _MockClient.requests = 0
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

    assert _MockClient.requests == 0
    assert not output_dir.exists()


def test_pilot_service_stops_after_early_shard_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed shard prevents later merging and cancels pending work."""
    paths = _write_inputs(root=tmp_path)

    def fail_generation(**kwargs: object) -> Path:
        raise ValueError("synthetic shard failure")

    monkeypatch.setattr(
        "danish_personas.generation.pilot.generate_personas", fail_generation
    )
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
        )
    assert not list((tmp_path / "failed-pilot").glob("*/pilot-manifest.json"))


def test_workflow_sample_immediately_plans_committed_v2_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An in-run frozen sample passes offline shard planning without credentials."""
    paths = _write_inputs(root=tmp_path)
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
