"""Tests for independent, relocation-safe release verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from support import _refresh_artifact, _refresh_manifest

from danish_personas.generation.job_titles import load_job_title_mapping
from danish_personas.generation.models import GenerationConfig
from danish_personas.generation.pipeline import generation_context_sha256
from danish_personas.io import load_yaml_model, sha256_file
from danish_personas.release.verifier import ReleaseVerificationError, verify_release


def test_verify_release_rejects_joint_config_tamper_with_stale_context(
    verifier_package: tuple[Path, str],
) -> None:
    """Re-signing config and evidence cannot bypass their context binding."""
    release, _ = verifier_package
    config_path = release / "provenance/config/config.yaml"
    config_path.write_bytes(
        config_path.read_bytes().replace(b"TEST_TOKEN", b"NEW_TOKEN")
    )
    evidence_path = release / "provenance/evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    config_hash = sha256_file(config_path)
    evidence["config_hashes"]["config.yaml"] = config_hash
    evidence["generation_config_sha256"] = config_hash
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = json.loads(
        (release / "release-manifest.json").read_text(encoding="utf-8")
    )
    manifest["evidence_sha256"] = sha256_file(evidence_path)
    for artifact in manifest["artifacts"]:
        if artifact["path"] == "provenance/evidence.json":
            artifact["sha256"] = sha256_file(evidence_path)
            artifact["size"] = evidence_path.stat().st_size
        elif artifact["path"] == "provenance/config/config.yaml":
            artifact["sha256"] = config_hash
            artifact["size"] = config_path.stat().st_size
    digest = _refresh_manifest(release, **manifest)
    with pytest.raises(ReleaseVerificationError, match="context"):
        verify_release(release_dir=release, expected_manifest_sha256=digest)


def test_verify_release_rejects_model_and_review_binding_errors(
    verifier_package: tuple[Path, str],
) -> None:
    """Manifest model and attested review identity cannot be changed independently."""
    release, _ = verifier_package
    manifest = json.loads(
        (release / "release-manifest.json").read_text(encoding="utf-8")
    )
    manifest["model"] = "other/model"
    digest = _refresh_manifest(release, **manifest)
    with pytest.raises(ReleaseVerificationError, match="model|ID|approval"):
        verify_release(release_dir=release, expected_manifest_sha256=digest)


@pytest.mark.parametrize(
    ("relative", "payload"),
    [
        ("provenance/release-policy.yaml", b"version: 1\nenabled: false\n"),
        ("attestations/human-review.json", b"{}\n"),
        ("LICENSE.txt", b"different licence\n"),
    ],
    ids=["policy", "attestation", "licence"],
)
def test_verify_release_rejects_policy_attestation_and_licence_errors(
    verifier_package: tuple[Path, str], relative: str, payload: bytes
) -> None:
    """Publication metadata must remain strict and bound to the evidence."""
    release, _ = verifier_package
    (release / relative).write_bytes(payload)
    digest = _refresh_artifact(release, relative)
    with pytest.raises(ReleaseVerificationError):
        verify_release(release_dir=release, expected_manifest_sha256=digest)


def test_verify_release_rejects_prompt_hash_mismatch(
    verifier_package: tuple[Path, str],
) -> None:
    """Prompt bytes remain bound through portable evidence hashes."""
    release, _ = verifier_package
    prompt = release / "provenance/prompts/personas-da.md"
    prompt.write_bytes(prompt.read_bytes() + b"drift")
    digest = _refresh_artifact(release, prompt.relative_to(release).as_posix())
    with pytest.raises(ReleaseVerificationError, match="prompt|checksum"):
        verify_release(release_dir=release, expected_manifest_sha256=digest)


@pytest.mark.parametrize(
    ("relative", "replacement"),
    [
        ("README.md", b"local file /Users/example/secret.txt\n"),
        ("LICENSE.txt", b"file:///tmp/private-token\n"),
        ("provenance/prompts/attributes-da.md", b"Authorization Bearer secret\n"),
        (
            "provenance/docs/source-register.md",
            b"https://user:password@example.invalid\n",
        ),
    ],
    ids=["local-path", "file-url", "bearer-token", "basic-auth"],
)
def test_verify_release_rejects_secret_and_local_path_text(
    verifier_package: tuple[Path, str], relative: str, replacement: bytes
) -> None:
    """Public text cannot expose local paths, credentials, or bearer tokens."""
    release, _ = verifier_package
    target = release / relative
    target.write_bytes(replacement)
    digest = _refresh_artifact(release, relative)
    with pytest.raises(ReleaseVerificationError):
        verify_release(release_dir=release, expected_manifest_sha256=digest)


def test_verify_release_rejects_title_map_config_path_substitution(
    verifier_package: tuple[Path, str],
) -> None:
    """A re-signed config with a different title-map path cannot bypass binding."""
    release, _ = verifier_package
    config_path = release / "provenance/config/config.yaml"
    config_bytes = config_path.read_bytes()
    posix_path = b"config/job-function-titles.yaml"
    windows_path = b"config\\job-function-titles.yaml"
    original_path = posix_path if posix_path in config_bytes else windows_path
    assert original_path in config_bytes
    config_path.write_bytes(
        config_bytes.replace(original_path, b"config/other-titles.yaml")
    )
    config_sha = sha256_file(config_path)
    evidence_path = release / "provenance/evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["config_hashes"]["config.yaml"] = config_sha
    evidence["generation_config_sha256"] = config_sha
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = _refresh_artifact(release, "provenance/config/config.yaml")
    digest = _refresh_artifact(release, "provenance/evidence.json")
    manifest = json.loads(
        (release / "release-manifest.json").read_text(encoding="utf-8")
    )
    manifest["evidence_sha256"] = sha256_file(evidence_path)
    digest = _refresh_manifest(release, **manifest)
    with pytest.raises(ReleaseVerificationError, match="path binding"):
        verify_release(release_dir=release, expected_manifest_sha256=digest)


def test_verify_release_rejects_title_map_substitution_even_with_recalculated_bindings(
    verifier_package: tuple[Path, str],
) -> None:
    """A valid substituted map cannot authorise a title from another mapping."""
    release, _ = verifier_package
    map_path = release / "provenance/config/job-function-titles.yaml"
    map_path.write_bytes(
        map_path.read_bytes().replace(b"forretningsspecialist", b"topchef")
    )
    map_sha = sha256_file(map_path)
    config_path = release / "provenance/config/config.yaml"
    config = load_yaml_model(path=config_path, model=GenerationConfig)
    context = generation_context_sha256(
        config=config,
        attributes_prompt=(release / "provenance/prompts/attributes-da.md").read_text(
            encoding="utf-8"
        ),
        personas_prompt=(release / "provenance/prompts/personas-da.md").read_text(
            encoding="utf-8"
        ),
        job_title_mapping=load_job_title_mapping(map_path),
        job_title_mapping_sha256=map_sha,
    )
    evidence_path = release / "provenance/evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["config_hashes"]["job-function-titles.yaml"] = map_sha
    evidence["generation_context_sha256"] = context
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = _refresh_artifact(release, "provenance/config/job-function-titles.yaml")
    digest = _refresh_artifact(release, "provenance/evidence.json")
    manifest = json.loads(
        (release / "release-manifest.json").read_text(encoding="utf-8")
    )
    manifest["evidence_sha256"] = sha256_file(evidence_path)
    digest = _refresh_manifest(release, **manifest)
    with pytest.raises(ReleaseVerificationError, match="contextual"):
        verify_release(release_dir=release, expected_manifest_sha256=digest)
