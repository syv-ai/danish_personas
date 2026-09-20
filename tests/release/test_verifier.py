"""Tests for independent, relocation-safe release verification."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from danish_personas.generation.job_titles import load_job_title_mapping
from danish_personas.generation.models import GenerationConfig
from danish_personas.generation.pipeline import generation_context_sha256
from danish_personas.io import load_yaml_model, sha256_file
from danish_personas.release.common import release_id
from danish_personas.release.verifier import ReleaseVerificationError, verify_release


@pytest.mark.parametrize(
    "relative", ["data/personas.parquet", "provenance/evidence.json"]
)
def test_recalculated_manifest_digest_does_not_bypass_internal_bindings(
    verifier_package: tuple[Path, str], relative: str
) -> None:
    """Re-signing the outer manifest cannot authorise output or evidence tampering."""
    release, _ = verifier_package
    path = release / relative
    if relative.endswith("parquet"):
        path.write_bytes(path.read_bytes() + b"tampered")
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["accounting"]["requests"] += 1
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    digest = _refresh_artifact(release, relative)
    if relative.endswith("evidence"):
        manifest = json.loads((release / "release-manifest.json").read_text())
        manifest["evidence_sha256"] = sha256_file(path)
        digest = _refresh_manifest(release, **manifest)
    with pytest.raises(ReleaseVerificationError):
        verify_release(release_dir=release, expected_manifest_sha256=digest)


def _refresh_artifact(release: Path, relative: str) -> str:
    """Refresh one manifest artifact and the manifest sidecar.

    Args:
        release:
            Release directory.
        relative:
            Manifest-relative artifact path.

    Returns:
        The refreshed manifest digest.

    Raises:
        AssertionError:
            If the artifact is not listed in the manifest.
    """
    manifest = json.loads(
        (release / "release-manifest.json").read_text(encoding="utf-8")
    )
    for artifact in manifest["artifacts"]:
        if artifact["path"] == relative:
            artifact["sha256"] = sha256_file(release / relative)
            artifact["size"] = (release / relative).stat().st_size
            break
    else:
        raise AssertionError(relative)
    return _refresh_manifest(release, artifacts=manifest["artifacts"])


def _refresh_manifest(release: Path, **changes: object) -> str:
    """Apply manifest changes and return its new externally retained digest.

    Returns:
        The refreshed manifest digest.
    """
    path = release / "release-manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(changes)
    path.write_bytes(
        (
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    )
    digest = sha256_file(path)
    (release / "release-manifest.sha256").write_bytes(
        f"{digest}  release-manifest.json\n".encode("ascii")
    )
    return digest


def test_refresh_manifest_preserves_utf8_origin_labels(
    verifier_package: tuple[Path, str],
) -> None:
    """Refreshing a manifest preserves canonical Unicode label values."""
    release, _ = verifier_package
    manifest_path = release / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    digest = _refresh_manifest(release, **manifest)

    refreshed = manifest_path.read_bytes()
    assert "Grækenland".encode() in refreshed
    assert b"Gr\\u00e6kenland" not in refreshed
    verify_release(release_dir=release, expected_manifest_sha256=digest)


def test_validation_diagnostic_does_not_echo_contract_input(
    verifier_package: tuple[Path, str],
) -> None:
    """Contract diagnostics expose a field and type, but not hostile input."""
    release, _ = verifier_package
    manifest = json.loads(
        (release / "release-manifest.json").read_text(encoding="utf-8")
    )
    labels = manifest["origin_label_contract_content"]["labels_da"]
    first_code = next(iter(labels))
    labels.pop(first_code)
    marker = "SECRET-MARKER"
    labels[marker] = 123
    digest = _refresh_manifest(release, **manifest)

    with pytest.raises(ReleaseVerificationError) as exc_info:
        verify_release(release_dir=release, expected_manifest_sha256=digest)

    diagnostic = str(exc_info.value)
    assert "origin_label_contract_content: string_type" in diagnostic
    assert marker not in diagnostic


def test_verify_release_accepts_external_digest_and_relocation(
    verifier_package: tuple[Path, str], tmp_path: Path
) -> None:
    """A release verifies after moving to another parent directory."""
    release, digest = verifier_package
    relocated_parent = tmp_path / "relocated"
    relocated_parent.mkdir()
    relocated = shutil.move(str(release), relocated_parent / release.name)
    manifest = verify_release(
        release_dir=Path(relocated), expected_manifest_sha256=digest
    )
    assert manifest.release_id == Path(relocated).name


def test_verify_release_rejects_an_injected_release_directory(
    verifier_package: tuple[Path, str],
) -> None:
    """Verification refuses a release directory that is itself a symlink."""
    release, digest = verifier_package
    link = release.parent / "release-link"
    link.symlink_to(release, target_is_directory=True)
    with pytest.raises(ReleaseVerificationError, match="directory"):
        verify_release(release_dir=link, expected_manifest_sha256=digest)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "case", "traversal"])
def test_verify_release_rejects_filesystem_and_manifest_path_attacks(
    verifier_package: tuple[Path, str], kind: str, tmp_path: Path
) -> None:
    """Symlinks, links, collisions, and traversal-shaped manifest entries fail."""
    release, digest = verifier_package
    if kind == "symlink":
        (release / "README.md").unlink()
        (release / "README.md").symlink_to(tmp_path / "outside")
    elif kind == "hardlink":
        (release / "README.md").unlink()
        os.link(release / "LICENSE.txt", release / "README.md")
    elif kind == "case":
        (release / "README.MD").write_text("collision", encoding="utf-8")
    else:
        manifest = json.loads((release / "release-manifest.json").read_text())
        manifest["artifacts"][0]["path"] = "../escape.txt"
        digest = _refresh_manifest(release, **manifest)
    with pytest.raises(
        ReleaseVerificationError,
        match="non-regular|hard link|colliding|layout|checksum",
    ):
        verify_release(release_dir=release, expected_manifest_sha256=digest)


def test_verify_release_rejects_forbidden_parquet_string(
    verifier_package: tuple[Path, str],
) -> None:
    """The same secret scan covers strings embedded in Parquet columns."""
    release, _ = verifier_package
    output = pl.read_parquet(release / "data/personas.parquet")
    output = output.with_columns(
        pl.when(pl.int_range(output.height) == 0)
        .then(pl.lit("/Users/private/secret.txt"))
        .otherwise(pl.col("cultural_context"))
        .alias("cultural_context")
    )
    output.write_parquet(release / "data/personas.parquet")
    manifest = json.loads(
        (release / "release-manifest.json").read_text(encoding="utf-8")
    )
    output_hash = sha256_file(release / "data/personas.parquet")
    for artifact in manifest["artifacts"]:
        if artifact["path"] == "data/personas.parquet":
            artifact.update(
                sha256=output_hash,
                size=(release / "data/personas.parquet").stat().st_size,
            )
    evidence_path = release / "provenance/evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["output_sha256"] = output_hash
    evidence["shards"][0]["output_sha256"] = output_hash
    attestation_path = release / "attestations/human-review.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["output_sha256"] = output_hash
    attestation_path.write_text(
        json.dumps(attestation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    attestation_hash = sha256_file(attestation_path)
    evidence["attestation_sha256"] = attestation_hash
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evidence_hash = sha256_file(evidence_path)
    for artifact in manifest["artifacts"]:
        if artifact["path"] == "attestations/human-review.json":
            artifact.update(
                sha256=attestation_hash, size=attestation_path.stat().st_size
            )
        if artifact["path"] == "provenance/evidence.json":
            artifact.update(sha256=evidence_hash, size=evidence_path.stat().st_size)
    manifest["evidence_sha256"] = evidence_hash
    manifest["release_id"] = release_id(
        pilot_id=manifest["pilot_id"],
        output_sha256=output_hash,
        reviewed_at=datetime.fromisoformat(manifest["created_at"]),
        git_head=manifest["git_head"],
        origin_url=manifest["origin_url"],
    )
    digest = _refresh_manifest(release, **manifest)
    release = release.rename(release.parent / manifest["release_id"])
    with pytest.raises(ReleaseVerificationError):
        verify_release(release_dir=release, expected_manifest_sha256=digest)


def test_verify_release_rejects_joint_config_tamper_with_stale_context(
    verifier_package: tuple[Path, str],
) -> None:
    """Re-signing config and evidence cannot bypass their context binding."""
    release, _ = verifier_package
    config_path = release / "provenance/config/generation.yaml"
    config_path.write_bytes(
        config_path.read_bytes().replace(b"TEST_TOKEN", b"NEW_TOKEN")
    )
    evidence_path = release / "provenance/evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    config_hash = sha256_file(config_path)
    evidence["config_hashes"]["generation.yaml"] = config_hash
    evidence["generation_config_sha256"] = config_hash
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = json.loads((release / "release-manifest.json").read_text())
    manifest["evidence_sha256"] = sha256_file(evidence_path)
    for artifact in manifest["artifacts"]:
        if artifact["path"] == "provenance/evidence.json":
            artifact["sha256"] = sha256_file(evidence_path)
            artifact["size"] = evidence_path.stat().st_size
        elif artifact["path"] == "provenance/config/generation.yaml":
            artifact["sha256"] = config_hash
            artifact["size"] = config_path.stat().st_size
    digest = _refresh_manifest(release, **manifest)
    with pytest.raises(ReleaseVerificationError, match="context"):
        verify_release(release_dir=release, expected_manifest_sha256=digest)


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "empty", "title-map-missing", "title-map-extra"]
)
def test_verify_release_rejects_missing_extra_and_empty_paths(
    verifier_package: tuple[Path, str], mutation: str
) -> None:
    """The verifier accepts only the fixed complete layout."""
    release, digest = verifier_package
    if mutation == "missing":
        (release / "README.md").unlink()
    elif mutation == "extra":
        (release / "unexpected.txt").write_text("extra", encoding="utf-8")
    elif mutation == "title-map-missing":
        (release / "provenance/config/job-function-titles.yaml").unlink()
    elif mutation == "title-map-extra":
        map_path = release / "provenance/config/job-function-titles.yaml"
        map_path.rename(map_path.with_name("job-function-titles-extra.yaml"))
    else:
        (release / "empty").mkdir()
    with pytest.raises(ReleaseVerificationError, match="missing|extra|empty|paths"):
        verify_release(release_dir=release, expected_manifest_sha256=digest)


def test_verify_release_rejects_missing_manifest_sidecar(
    verifier_package: tuple[Path, str],
) -> None:
    """The sidecar is checked after the externally supplied digest."""
    release, digest = verifier_package
    (release / "release-manifest.sha256").unlink()
    with pytest.raises(ReleaseVerificationError, match="sidecar|missing|empty"):
        verify_release(release_dir=release, expected_manifest_sha256=digest)


def test_verify_release_rejects_model_and_review_binding_errors(
    verifier_package: tuple[Path, str],
) -> None:
    """Manifest model and attested review identity cannot be changed independently."""
    release, _ = verifier_package
    manifest = json.loads((release / "release-manifest.json").read_text())
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
    config_path = release / "provenance/config/generation.yaml"
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
    evidence["config_hashes"]["generation.yaml"] = config_sha
    evidence["generation_config_sha256"] = config_sha
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = _refresh_artifact(release, "provenance/config/generation.yaml")
    digest = _refresh_artifact(release, "provenance/evidence.json")
    manifest = json.loads((release / "release-manifest.json").read_text())
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
    config_path = release / "provenance/config/generation.yaml"
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
    manifest = json.loads((release / "release-manifest.json").read_text())
    manifest["evidence_sha256"] = sha256_file(evidence_path)
    digest = _refresh_manifest(release, **manifest)
    with pytest.raises(ReleaseVerificationError, match="contextual"):
        verify_release(release_dir=release, expected_manifest_sha256=digest)


@pytest.mark.parametrize("expected", ["", "not-a-digest", "a" * 63, "g" * 64])
def test_verify_release_requires_correct_expected_digest(
    verifier_package: tuple[Path, str], expected: str
) -> None:
    """An absent or malformed external digest never reaches content trust."""
    release, _ = verifier_package
    with pytest.raises(ReleaseVerificationError, match="mandatory|mismatch"):
        verify_release(release_dir=release, expected_manifest_sha256=expected)
