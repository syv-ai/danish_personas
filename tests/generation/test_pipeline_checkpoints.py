"""Checkpoint, resume, and generation-accounting integrity tests."""

import json
from pathlib import Path

import polars as pl
import pytest
from generation_test_helpers import (
    InterruptingGenerationClient,
    MockGenerationClient,
    RejectingGenerationClient,
    write_generation_inputs,
)

from danish_personas.generation.client import RequestBudgetExceeded
from danish_personas.generation.models import FrozenSampleManifest
from danish_personas.generation.pipeline import generate_personas
from danish_personas.generation.report import validate_persona_run
from danish_personas.io import sha256_file, write_json


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_checkpoint",
        "malformed_checkpoint",
        "accounting",
        "accepted_content",
        "trailing_response",
        "attempts",
        "ledger_context",
        "llm_flag",
        "run_id",
        "offset",
    ],
)
def test_persona_validation_rejects_independent_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    """Each checkpoint, accounting, and identity tamper fails from a valid run."""
    paths = write_generation_inputs(root=tmp_path)
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", MockGenerationClient
    )
    MockGenerationClient.requests = 0
    run_dir = generate_personas(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "outputs",
        rows=1,
        live=True,
    )
    assert validate_persona_run(run_dir=run_dir).passed
    manifest_path = run_dir / "generation-manifest.json"
    checkpoint_path = next((run_dir / "checkpoints").glob("*.json"))
    ledger_path = run_dir / "request-ledger.json"

    if tamper == "missing_checkpoint":
        checkpoint_path.unlink()
    elif tamper == "malformed_checkpoint":
        checkpoint_path.write_text("{")
    elif tamper == "accounting":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["prompt_tokens"] += 1
        write_json(path=manifest_path, payload=manifest)
    elif tamper in {"accepted_content", "trailing_response", "attempts"}:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if tamper == "accepted_content":
            accepted = json.loads(checkpoint["responses"][0]["content"])
            accepted["cultural_context"] = (
                "Personen har en anden dansk hverdag og deltager i lokale fællesskaber."
            )
            checkpoint["responses"][0]["content"] = json.dumps(
                accepted, ensure_ascii=False
            )
        elif tamper == "trailing_response":
            checkpoint["responses"].append(checkpoint["responses"][-1])
            checkpoint["attempts"] += 1
        else:
            checkpoint["attempts"] += 1
        write_json(path=checkpoint_path, payload=checkpoint)
    elif tamper == "ledger_context":
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger["generation_context_sha256"] = "f" * 64
        write_json(path=ledger_path, payload=ledger)
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if tamper == "llm_flag":
            manifest["llm_generation"] = False
        elif tamper == "run_id":
            manifest["run_id"] = "forged-run-id"
        else:
            manifest["offset"] = 1
        write_json(path=manifest_path, payload=manifest)

    assert not validate_persona_run(run_dir=run_dir).passed


def test_pipeline_rejects_tampering_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validated rows generate once; forged rows and stale checkpoints fail."""
    paths = write_generation_inputs(root=tmp_path)
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", MockGenerationClient
    )
    MockGenerationClient.requests = 0
    run_dir = generate_personas(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "outputs",
        rows=1,
        live=True,
    )
    assert MockGenerationClient.requests == 2
    assert (run_dir / "generated-personas.parquet").exists()
    assert validate_persona_run(run_dir=run_dir).passed

    generate_personas(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "outputs",
        rows=1,
        live=True,
    )
    assert MockGenerationClient.requests == 2

    checkpoint_path = next((run_dir / "checkpoints").glob("*.json"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["generation_context_sha256"] = "f" * 64
    checkpoint_path.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match="Stale generation context"):
        generate_personas(
            input_path=paths["sample"],
            sample_manifest_path=paths["sample_manifest"],
            config_path=paths["config"],
            output_dir=tmp_path / "outputs",
            rows=1,
            live=True,
        )

    sample = pl.read_parquet(paths["sample"]).with_columns(pl.lit(99).alias("age"))
    sample.write_parquet(paths["sample"])
    manifest = FrozenSampleManifest.model_validate_json(
        paths["sample_manifest"].read_text(encoding="utf-8")
    ).model_copy(update={"sha256": sha256_file(paths["sample"])})
    write_json(path=paths["sample_manifest"], payload=manifest)
    with pytest.raises(ValueError, match="absent from validated"):
        generate_personas(
            input_path=paths["sample"],
            sample_manifest_path=paths["sample_manifest"],
            config_path=paths["config"],
            output_dir=tmp_path / "outputs",
            rows=1,
            live=True,
        )


def test_rejected_completion_text_is_not_checkpointed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rejected text is removed while its usage and response hash remain auditable."""
    paths = write_generation_inputs(root=tmp_path)
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", RejectingGenerationClient
    )
    RejectingGenerationClient.requests = 0
    RejectingGenerationClient.reject_once = True
    run_dir = generate_personas(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "outputs",
        rows=1,
        live=True,
    )
    checkpoint_path = next((run_dir / "checkpoints").glob("*.json"))
    responses = json.loads(checkpoint_path.read_text(encoding="utf-8"))["responses"]
    assert responses[0]["content"] == ""
    assert responses[0]["raw_response_sha256"] == "0" * 64
    assert len(responses) == 3


def test_stage_checkpoint_avoids_repeating_attributes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A descriptions-stage interruption preserves the billable attributes stage."""
    paths = write_generation_inputs(root=tmp_path)
    monkeypatch.setattr(
        "danish_personas.generation.pipeline.OpenAIClient", InterruptingGenerationClient
    )
    InterruptingGenerationClient.requests = 0
    InterruptingGenerationClient.interrupt_once = True
    with pytest.raises(RequestBudgetExceeded):
        generate_personas(
            input_path=paths["sample"],
            sample_manifest_path=paths["sample_manifest"],
            config_path=paths["config"],
            output_dir=tmp_path / "outputs",
            rows=1,
            live=True,
        )
    assert InterruptingGenerationClient.requests == 2
    assert (
        len(list((tmp_path / "outputs").glob("*/checkpoints/*.attributes.json"))) == 1
    )

    run_dir = generate_personas(
        input_path=paths["sample"],
        sample_manifest_path=paths["sample_manifest"],
        config_path=paths["config"],
        output_dir=tmp_path / "outputs",
        rows=1,
        live=True,
    )
    assert InterruptingGenerationClient.requests == 3
    manifest = json.loads(
        (run_dir / "generation-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["requests"] == 3
    assert manifest["retries"] == 1
    assert manifest["estimated_cost_usd"] == 0.002
    assert manifest["inference_providers"] == ["mock-provider"]
