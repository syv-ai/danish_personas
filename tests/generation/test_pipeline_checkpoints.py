"""Checkpoint, resume, and schema-parsing tests."""

import json
from pathlib import Path

import pytest
from generation_test_helpers import (
    InterruptingGenerationClient,
    MockGenerationClient,
    RejectingGenerationClient,
    write_generation_inputs,
)
from pydantic import ValidationError

from danish_personas.generation.client import RequestBudgetExceeded
from danish_personas.generation.pipeline import generate_personas


def test_interrupted_request_is_counted_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transport interruption remains counted when generation resumes."""
    paths = write_generation_inputs(root=tmp_path)
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", InterruptingGenerationClient
    )
    InterruptingGenerationClient.requests = 0
    InterruptingGenerationClient.interrupt_once = True
    with pytest.raises(RequestBudgetExceeded):
        _generate(paths=paths, output_dir=tmp_path / "outputs")

    run_dir = _generate(paths=paths, output_dir=tmp_path / "outputs")

    manifest = json.loads(
        (run_dir / "generation-manifest.json").read_text(encoding="utf-8")
    )
    assert InterruptingGenerationClient.requests == 2
    assert manifest["requests"] == 2
    assert manifest["retries"] == 1


def _generate(*, paths: dict[str, Path], output_dir: Path) -> Path:
    """Generate one row from shared offline fixtures.

    Returns:
        Generation run directory.
    """
    return generate_personas(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=output_dir,
        rows=1,
    )


def test_schema_invalid_response_fails_without_semantic_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed provider output is parsed once and is not checkpointed."""
    paths = write_generation_inputs(root=tmp_path)
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", RejectingGenerationClient
    )
    RejectingGenerationClient.requests = 0
    RejectingGenerationClient.reject_once = True

    with pytest.raises(ValidationError):
        _generate(paths=paths, output_dir=tmp_path / "outputs")

    assert RejectingGenerationClient.requests == 1
    assert not list((tmp_path / "outputs").glob("*/checkpoints/*.json"))


def test_schema_parsed_checkpoint_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint with unchanged generation inputs avoids another provider call."""
    paths = write_generation_inputs(root=tmp_path)
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", MockGenerationClient
    )
    MockGenerationClient.requests = 0

    first = _generate(paths=paths, output_dir=tmp_path / "outputs")
    second = _generate(paths=paths, output_dir=tmp_path / "outputs")

    assert first == second
    assert MockGenerationClient.requests == 1
    assert (first / "generated-personas.parquet").exists()
    assert not (first / "validation-report.json").exists()
