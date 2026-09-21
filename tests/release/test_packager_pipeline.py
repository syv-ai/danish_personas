"""Tests for the public offline release packager."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from support import ReleaseCase, _clean_provenance, _package

from danish_personas.generation.models import GenerationManifest, PilotBatchReference
from danish_personas.generation.validation import VALIDATOR_VERSION
from danish_personas.io import sha256_file
from danish_personas.release import packager
from danish_personas.release.packager import ReleasePackagingError


def test_package_captures_root_generation_config(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The root Hydra config is captured as the effective generation config."""
    result = _package(release_case, monkeypatch)
    assert (result.path / "provenance/config/config.yaml").read_bytes() == (
        release_case.repository / "config.yaml"
    ).read_bytes()


@pytest.mark.parametrize(
    "mutation",
    ["restore_checkpoint", "add_checkpoint", "remove_ledger", "add_unrelated"],
    ids=["restore-checkpoint", "add-checkpoint", "remove-ledger", "add-unrelated"],
)
def test_package_release_rejects_pilot_tree_membership_changes(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """Pilot checkpoint and ledger membership cannot change during staging."""
    shard = release_case.pilot / "shard"
    checkpoints = shard / "checkpoints"
    ledger = shard / "request-ledger.json"
    if mutation == "add_checkpoint":
        checkpoints.mkdir(parents=True)
        (checkpoints / "existing.json").write_text("{}", encoding="utf-8")
    elif mutation == "remove_ledger":
        shard.mkdir()
        ledger.write_text("{}", encoding="utf-8")

    original = packager._install_files

    def mutate(**kwargs: object) -> None:
        original(**kwargs)
        if mutation in {"restore_checkpoint", "add_checkpoint"}:
            checkpoints.mkdir(parents=True, exist_ok=True)
            (checkpoints / "restored.json").write_text("{}", encoding="utf-8")
        elif mutation == "remove_ledger":
            ledger.unlink()
        else:
            (release_case.pilot / "unexpected.txt").write_text(
                "unexpected", encoding="utf-8"
            )

    monkeypatch.setattr(packager, "_install_files", mutate)
    with pytest.raises(ReleasePackagingError, match="Pilot tree membership changed"):
        _package(release_case, monkeypatch)
    assert not list(release_case.output_parent.glob("*"))


def test_package_release_rejects_prompt_hash_mismatch(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prompt files are public inputs whose manifest hashes cannot drift."""
    prompt = release_case.repository / "config/prompts/attributes-da.md"
    prompt.write_text(prompt.read_text(encoding="utf-8") + "changed", encoding="utf-8")
    monkeypatch.setattr(packager, "_git_provenance", _clean_provenance)
    with pytest.raises(ReleasePackagingError, match="Prompt checksum"):
        _package(release_case, monkeypatch)


def test_package_release_rejects_source_mutation_before_install(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source changed during staging is caught by the final inventory check."""
    original = packager._install_files

    def mutate(**kwargs: object) -> None:
        original(**kwargs)
        release_case.card.write_text("changed", encoding="utf-8")

    monkeypatch.setattr(packager, "_install_files", mutate)
    with pytest.raises(ReleasePackagingError, match="Consumed file changed"):
        _package(release_case, monkeypatch)
    assert not list(release_case.output_parent.glob("*"))


def test_package_release_rejects_wrong_licence_and_policy_metadata(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Licence and policy bindings are checked before installation."""
    release_case.licence.write_text("wrong licence", encoding="utf-8")
    with pytest.raises(ReleasePackagingError, match="licence"):
        _package(release_case, monkeypatch)


def test_package_uses_captured_bytes_during_copy_time_replacement(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replacement cannot alter bytes selected for the package."""
    captured_card = release_case.card.read_bytes()
    original = packager._install_files
    observed: list[bytes] = []

    def replace_before_copy(**kwargs: object) -> None:
        release_case.card.write_bytes(b"replacement")
        original(**kwargs)
        stage = kwargs["stage"]
        assert isinstance(stage, Path)
        observed.append((stage / "README.md").read_bytes())

    monkeypatch.setattr(packager, "_install_files", replace_before_copy)
    with pytest.raises(ReleasePackagingError, match="Consumed file changed"):
        _package(release_case, monkeypatch)
    assert observed == [captured_card]


def test_real_small_shard_accounting_derivation_needs_no_shard_fanout(
    release_case: ReleaseCase,
) -> None:
    """A real checkpoint/shard derivation remains cheap and additive."""
    small_output = release_case.pilot / "small-output.parquet"
    pl.read_parquet(release_case.output).head(2).write_parquet(small_output)
    shard_dir = release_case.pilot / "small-shard"
    shard_dir.mkdir()
    shard_output = shard_dir / "output.parquet"
    small_output.replace(shard_output)
    shard_manifest = GenerationManifest(
        run_id="small-shard",
        upstream_run_id="upstream-run",
        input_file=Path("sample.parquet"),
        sample_manifest_file=Path("sample-manifest.json"),
        input_sha256="a" * 64,
        ordered_persona_ids_sha256="b" * 64,
        generation_config_file=Path("config.yaml"),
        generation_config_sha256=release_case.manifest.generation_config_sha256,
        generation_context_sha256=release_case.manifest.generation_context_sha256,
        origin_label_contract_file=release_case.manifest.origin_label_contract_file,
        origin_label_contract_sha256=release_case.manifest.origin_label_contract_sha256,
        origin_label_contract_version=release_case.manifest.origin_label_contract_version,
        origin_label_contract_content=release_case.manifest.origin_label_contract_content,
        validator_version=VALIDATOR_VERSION,
        job_title_mapping_file=Path("config/job-function-titles.yaml"),
        job_title_mapping_sha256=release_case.manifest.job_title_mapping_sha256,
        job_title_mapping_version=release_case.manifest.job_title_mapping_version,
        attributes_prompt_sha256=release_case.manifest.attributes_prompt_sha256,
        personas_prompt_sha256=release_case.manifest.personas_prompt_sha256,
        model=release_case.manifest.model,
        base_url="https://llm.example/v1",
        rows=2,
        offset=0,
        requests=2,
        retries=0,
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        estimated_cost_usd=0,
        inference_providers=["test-provider"],
        output_file=Path("output.parquet"),
        output_sha256=sha256_file(shard_output),
        llm_generation=True,
    )
    shard_manifest_path = shard_dir / "manifest.json"
    shard_manifest_path.write_text(
        json.dumps(shard_manifest.model_dump(mode="json")), encoding="utf-8"
    )
    report_path = shard_dir / "report.json"
    report_path.write_text("{}", encoding="utf-8")
    (release_case.pilot / "pilot-validation-report.json").write_text(
        json.dumps(release_case.report.model_dump(mode="json")), encoding="utf-8"
    )
    reference = PilotBatchReference(
        offset=0,
        rows=2,
        run_id="small-shard",
        manifest_file=Path("small-shard/manifest.json"),
        manifest_sha256=sha256_file(shard_manifest_path),
        validation_report_file=Path("small-shard/report.json"),
        validation_report_sha256=sha256_file(report_path),
        origin_label_contract_file=release_case.manifest.origin_label_contract_file,
        origin_label_contract_sha256=release_case.manifest.origin_label_contract_sha256,
        origin_label_contract_version=release_case.manifest.origin_label_contract_version,
        origin_label_contract_content=release_case.manifest.origin_label_contract_content,
    )
    manifest = release_case.manifest.model_copy(
        update={
            "rows": 2,
            "batch_runs": [reference],
            "batches": 1,
            "requests": 2,
            "output_file": Path("small-shard/output.parquet"),
            "output_sha256": sha256_file(shard_output),
        }
    )
    evidence = packager._derive_evidence(
        pilot_dir=release_case.pilot,
        manifest=manifest,
        policy_path=release_case.policy,
        attestation_path=release_case.attestation,
        licence_path=release_case.licence,
        repository_root=release_case.repository,
    )
    assert evidence.shards[0].rows == 2
    assert evidence.accounting.requests == 2
    assert evidence.accounting.retries == 0
