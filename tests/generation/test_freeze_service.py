"""Public frozen-sample service and filesystem-safety tests."""

from pathlib import Path

import polars as pl
import pytest
from generation_test_helpers import write_generation_inputs

from danish_personas.io import sha256_file
from danish_personas.sampling.freeze import SampleSizeError, freeze_sample


def test_freeze_service_is_deterministic_and_bounds_size(tmp_path: Path) -> None:
    """The public freezer preserves round-robin output and rejects oversized input."""
    paths = write_generation_inputs(root=tmp_path)
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
    paths = write_generation_inputs(root=tmp_path)
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
    paths = write_generation_inputs(root=tmp_path)
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
    paths = write_generation_inputs(root=tmp_path)
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
    paths = write_generation_inputs(root=tmp_path)

    with pytest.raises(ValueError, match="inside its validated run directory"):
        freeze_sample(
            run_dir=paths["sample"].parent,
            rows=1,
            output=tmp_path / "relocated.parquet",
        )


def test_freeze_service_rejects_symlinked_parent(tmp_path: Path) -> None:
    """A path through a linked parent is not a direct run child."""
    paths = write_generation_inputs(root=tmp_path)
    run_dir = paths["sample"].parent
    real_parent = run_dir / "real-parent"
    real_parent.mkdir()
    linked_parent = run_dir / "linked-parent"
    _make_symlink(path=linked_parent, target=real_parent, directory=True)

    with pytest.raises(ValueError, match="inside its validated run directory"):
        freeze_sample(run_dir=run_dir, rows=1, output=linked_parent / "sample.parquet")


def test_freeze_service_rejects_traversal_alias(tmp_path: Path) -> None:
    """Lexical traversal aliases are rejected even when they normalise in-run."""
    paths = write_generation_inputs(root=tmp_path)
    run_dir = paths["sample"].parent
    output = run_dir / "nested" / ".." / "traversal.parquet"

    with pytest.raises(ValueError, match="traversal"):
        freeze_sample(run_dir=run_dir, rows=1, output=output)
