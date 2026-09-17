"""Adversarial tests for release boundaries and public text scanning."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest
from conftest import ReleaseCase
from test_packager import _package
from test_verifier import _refresh_artifact

from danish_personas.release import packager
from danish_personas.release.packager import ReleasePackagingError
from danish_personas.release.verifier import verify_release


@pytest.mark.parametrize(
    "content",
    [
        b"Read more at https://example.invalid/docs.\n",
        b"The source is https://example.invalid/path?a=1&b=2.\n",
        b"A plain URL https://example.invalid is public metadata.\n",
    ],
)
def test_normal_https_documentation_is_not_a_secret(
    verifier_package: tuple[Path, str], content: bytes
) -> None:
    """Ordinary HTTPS links remain valid public provenance."""
    release, _ = verifier_package
    card = release / "README.md"
    card.write_bytes(content)
    digest = _refresh_artifact(release, "README.md")
    assert verify_release(
        release_dir=release, expected_manifest_sha256=digest
    ).release_id


def test_package_has_no_network_dependency(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Packaging remains offline even if socket creation is denied."""

    def denied(*_: object, **__: object) -> socket.socket:
        raise AssertionError("network access")

    monkeypatch.setattr(socket, "socket", denied)
    result = _package(release_case, monkeypatch)
    assert result.path.is_dir()


def test_package_rejects_concurrent_lock(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lock prevents concurrent staging into one output parent."""
    release_case.output_parent.mkdir()
    (release_case.output_parent / ".release-package.lock").write_text("busy")
    with pytest.raises(ReleasePackagingError, match="active"):
        _package(release_case, monkeypatch)


def test_package_rejects_git_provenance_change_during_install(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A changed HEAD or origin cannot race the final installation boundary."""
    original = packager._git_provenance
    calls = 0

    def changed(root: Path) -> tuple[str, str]:
        nonlocal calls
        calls += 1
        value = original(root)
        return value if calls == 1 else ("f" * 40, value[1])

    monkeypatch.setattr(packager, "_git_provenance", changed)
    with pytest.raises(ReleasePackagingError, match="provenance changed"):
        _package(release_case, monkeypatch)
    assert not release_case.output_parent.exists()


def test_package_rejects_linked_input_files(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Untrusted input symlinks and hard links are never copied."""
    original = release_case.licence
    original.unlink()
    original.symlink_to(tmp_path / "licence-target")
    (tmp_path / "licence-target").write_bytes(
        b"Creative Commons Attribution 4.0 International\n"
    )
    with pytest.raises(ReleasePackagingError, match="regular non-linked"):
        _package(release_case, monkeypatch)

    original.unlink()
    original.write_bytes(b"Creative Commons Attribution 4.0 International\n")
    target = tmp_path / "card-target"
    target.write_text("card", encoding="utf-8")
    release_case.card.unlink()
    release_case.card.hardlink_to(target)
    with pytest.raises(ReleasePackagingError, match="regular non-linked"):
        _package(release_case, monkeypatch)


@pytest.mark.parametrize(
    "content",
    [
        b"/private/local/output.parquet\n",
        b"C:\\Users\\person\\secret.txt\n",
        b"https://example.invalid?api_key=secret\n",
        b"https://example.invalid/#token\n",
    ],
)
def test_package_rejects_local_and_secret_card_content(
    release_case: ReleaseCase, monkeypatch: pytest.MonkeyPatch, content: bytes
) -> None:
    """The packager scans the dataset card before any release is installed."""
    release_case.card.write_bytes(content)
    with pytest.raises(ReleasePackagingError, match="path or secret"):
        _package(release_case, monkeypatch)
    assert not release_case.output_parent.exists()
