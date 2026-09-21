"""Tests for the public offline release packager."""

from __future__ import annotations

import shutil
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from danish_personas.release import packager
from danish_personas.release.packager import ReleasePackagingError


def test_aba_replacement_cannot_change_validation_snapshot(tmp_path: Path) -> None:
    """A valid replacement at the original path cannot repair captured input."""
    repository = tmp_path / "repository"
    pilot = tmp_path / "pilot"
    repository.mkdir()
    pilot.mkdir()
    checkpoint = pilot / "checkpoint.json"
    checkpoint.write_bytes(b"invalid")
    inventory = packager._snapshot_inventory(paths=[checkpoint])
    snapshot = packager._materialise_snapshot(
        inventory=inventory, pilot_dir=pilot, repository_root=repository
    )
    try:
        checkpoint.write_bytes(b"valid")
        checkpoint.write_bytes(b"invalid")
        assert (snapshot / "pilot/checkpoint.json").read_bytes() == b"invalid"
    finally:
        shutil.rmtree(snapshot)


def test_capture_accepts_windows_path_and_descriptor_identity_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows identity fields may differ between path and descriptor stats."""
    source = tmp_path / "source.bin"
    content = b"captured"
    source.write_bytes(content)
    actual = source.stat()
    path_calls = 0
    fd_calls = 0

    def stat_view(offset: int, *, path_observer: bool) -> SimpleNamespace:
        return SimpleNamespace(
            st_mode=actual.st_mode ^ (0o1 if path_observer else 0),
            st_nlink=actual.st_nlink,
            st_size=actual.st_size,
            st_dev=actual.st_dev + offset,
            st_ino=actual.st_ino + offset,
            st_mtime_ns=actual.st_mtime_ns + (1000 if path_observer else 0),
            st_ctime_ns=actual.st_ctime_ns + offset,
        )

    def fake_lstat(_: object) -> SimpleNamespace:
        nonlocal path_calls
        path_calls += 1
        return stat_view(path_calls, path_observer=True)

    def fake_fstat(_: int) -> SimpleNamespace:
        nonlocal fd_calls
        fd_calls += 1
        return stat_view(100 + fd_calls, path_observer=False)

    monkeypatch.setattr(packager, "_WINDOWS", True)
    monkeypatch.setattr(packager, "_WINDOWS_NATIVE", False)
    monkeypatch.setattr(packager.os, "lstat", fake_lstat)
    monkeypatch.setattr(packager.os, "fstat", fake_fstat)

    item = packager._capture_file(source)

    assert item.content == content
    assert item.sha256 == packager.sha256_bytes(content)


def test_capture_inventory_keeps_captured_bytes_after_replacement(
    tmp_path: Path,
) -> None:
    """Packaging bytes remain bound to the descriptor capture."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"captured")
    inventory = packager._snapshot_inventory(paths=[source])
    source.write_bytes(b"replacement")
    assert packager._captured_bytes(inventory, source) == b"captured"
    with pytest.raises(ReleasePackagingError, match="Consumed file changed"):
        packager._recheck_inventory(inventory)


def test_capture_rejects_windows_size_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows capture rejects a path observation with a changed size."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"captured")
    actual = source.stat()
    path_calls = 0

    def fake_lstat(_: object) -> SimpleNamespace:
        nonlocal path_calls
        path_calls += 1
        return SimpleNamespace(
            st_mode=actual.st_mode,
            st_nlink=actual.st_nlink,
            st_size=actual.st_size + (path_calls > 1),
            st_dev=actual.st_dev,
            st_ino=actual.st_ino,
            st_mtime_ns=actual.st_mtime_ns + path_calls,
            st_ctime_ns=actual.st_ctime_ns,
        )

    monkeypatch.setattr(packager, "_WINDOWS", True)
    monkeypatch.setattr(packager, "_WINDOWS_NATIVE", False)
    monkeypatch.setattr(packager.os, "lstat", fake_lstat)

    with pytest.raises(ReleasePackagingError, match="Input metadata changed"):
        packager._capture_file(source)


def test_capture_uses_binary_flag_and_preserves_binary_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Descriptor capture preserves binary bytes and requests binary mode."""
    source = tmp_path / "source.bin"
    content = b"prefix\r\n\x1a\r\nparquet-bytes\x1a\r\n"
    source.write_bytes(content)
    if packager._WINDOWS_NATIVE:
        # Native capture uses CreateFileW and an explicit reparse-point denial
        # contract instead of the POSIX os.open flags below.
        assert (
            packager._WINDOWS_FINAL_FLAGS
            & packager._WINDOWS_FILE_FLAG_OPEN_REPARSE_POINT
        )
        assert packager._WINDOWS_FINAL_SHARE_MODE == packager._WINDOWS_FILE_SHARE_READ
        item = packager._capture_file(source)
    else:
        binary_flag = getattr(packager.os, "O_BINARY", 0)
        captured_flags: list[int] = []
        original_open = packager.os.open

        def capture_open(path: Path, flags: int, *args: int) -> int:
            captured_flags.append(flags)
            return original_open(path, flags, *args)

        monkeypatch.setattr(packager.os, "open", capture_open)
        item = packager._capture_file(source)

        assert captured_flags
        assert binary_flag == 0 or captured_flags[0] & binary_flag == binary_flag

    assert item.content == content
    assert item.size == len(content)
    assert item.sha256 == packager.sha256_bytes(content)


def test_capture_uses_separate_windows_observer_stability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows descriptor mtime drift does not mask path stability checks."""
    source = tmp_path / "source.bin"
    content = b"captured"
    source.write_bytes(content)
    actual = source.stat()
    fd_calls = 0

    def fake_lstat(_: object) -> SimpleNamespace:
        return SimpleNamespace(
            st_mode=actual.st_mode,
            st_nlink=actual.st_nlink,
            st_size=actual.st_size,
            st_dev=actual.st_dev + 1,
            st_ino=actual.st_ino + 1,
            st_mtime_ns=actual.st_mtime_ns,
            st_ctime_ns=actual.st_ctime_ns + 1,
        )

    def fake_fstat(_: int) -> SimpleNamespace:
        nonlocal fd_calls
        fd_calls += 1
        return SimpleNamespace(
            st_mode=actual.st_mode,
            st_nlink=actual.st_nlink,
            st_size=actual.st_size,
            st_dev=actual.st_dev + 100 + fd_calls,
            st_ino=actual.st_ino + 100 + fd_calls,
            st_mtime_ns=actual.st_mtime_ns + fd_calls,
            st_ctime_ns=actual.st_ctime_ns + 100 + fd_calls,
        )

    monkeypatch.setattr(packager, "_WINDOWS", True)
    monkeypatch.setattr(packager, "_WINDOWS_NATIVE", False)
    monkeypatch.setattr(packager.os, "lstat", fake_lstat)
    monkeypatch.setattr(packager.os, "fstat", fake_fstat)

    item = packager._capture_file(source)

    assert item.content == content


def test_snapshot_materialisation_preserves_repository_and_pilot_paths(
    tmp_path: Path,
) -> None:
    """Captured files are copied without links and with relative semantics."""
    repository = tmp_path / "repository"
    pilot = tmp_path / "pilot"
    repository.mkdir()
    pilot.mkdir()
    repository_file = repository / "config" / "generation.yaml"
    pilot_file = pilot / "checkpoints" / "persona.json"
    repository_file.parent.mkdir()
    pilot_file.parent.mkdir()
    repository_file.write_bytes(b"repository")
    pilot_file.write_bytes(b"pilot")
    inventory = packager._snapshot_inventory(paths=[repository_file, pilot_file])
    snapshot = packager._materialise_snapshot(
        inventory=inventory, pilot_dir=pilot, repository_root=repository
    )
    try:
        assert (
            snapshot / "repository/config/generation.yaml"
        ).read_bytes() == b"repository"
        assert (snapshot / "pilot/checkpoints/persona.json").read_bytes() == b"pilot"
        assert not (snapshot / "repository/config/generation.yaml").is_symlink()
        assert not (snapshot / "pilot/checkpoints/persona.json").is_symlink()
    finally:
        shutil.rmtree(snapshot)


def test_supplied_path_with_symlink_parent_is_rejected(tmp_path: Path) -> None:
    """Path checks inspect the first relative component and every parent."""
    real = tmp_path / "real"
    real.mkdir()
    (real / "input.txt").write_text("input", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ReleasePackagingError, match="symlink"):
        packager._require_regular_file(link / "input.txt")


def test_windows_capture_contract_uses_no_follow_and_stable_identity() -> None:
    """Native file and parent handles deny mutation sharing."""
    first = packager._WindowsFileInfo(
        attributes=0,
        volume_serial=7,
        file_index=2**40,
        size=17,
        number_of_links=1,
        write_time=19,
    )
    same = replace(first)
    changed = replace(first, file_index=2**40 + 1)

    assert (
        packager._WINDOWS_FINAL_FLAGS & packager._WINDOWS_FILE_FLAG_OPEN_REPARSE_POINT
    )
    assert packager._WINDOWS_FINAL_SHARE_MODE == packager._WINDOWS_FILE_SHARE_READ
    assert not packager._WINDOWS_FINAL_SHARE_MODE & packager._WINDOWS_FILE_SHARE_WRITE
    assert not packager._WINDOWS_FINAL_SHARE_MODE & packager._WINDOWS_FILE_SHARE_DELETE
    assert packager._WINDOWS_DIRECTORY_SHARE_MODE == packager._WINDOWS_FILE_SHARE_READ
    assert not (
        packager._WINDOWS_DIRECTORY_SHARE_MODE & packager._WINDOWS_FILE_SHARE_WRITE
    )
    assert not packager._WINDOWS_DIRECTORY_SHARE_MODE & (
        packager._WINDOWS_FILE_SHARE_DELETE
    )
    assert packager._windows_observations_match(first, same)
    assert not packager._windows_observations_match(first, changed)


@pytest.mark.parametrize(
    "kind",
    ["symlink", "hardlink", "nonregular"],
    ids=["symlink", "hardlink", "nonregular"],
)
@pytest.mark.parametrize(
    "observer", ["path", "fd"], ids=["path-observer", "fd-observer"]
)
def test_windows_capture_rejects_unsafe_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, observer: str
) -> None:
    """Windows checks reject unsafe path and descriptor observations."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"captured")
    actual = source.stat()
    unsafe = _unsafe_stat(kind=kind, actual=actual)

    def fake_lstat(_: object) -> object:
        return unsafe if observer == "path" else actual

    def fake_fstat(_: int) -> object:
        return unsafe if observer == "fd" else actual

    monkeypatch.setattr(packager, "_WINDOWS", True)
    monkeypatch.setattr(packager, "_WINDOWS_NATIVE", False)
    monkeypatch.setattr(packager.os, "lstat", fake_lstat)
    monkeypatch.setattr(packager.os, "fstat", fake_fstat)

    with pytest.raises(ReleasePackagingError):
        packager._capture_file(source)


def _unsafe_stat(kind: str, actual: object) -> SimpleNamespace:
    if kind == "symlink":
        mode = stat.S_IFLNK | 0o777
        nlink = 1
    elif kind == "hardlink":
        mode = getattr(actual, "st_mode")
        nlink = 2
    else:
        mode = stat.S_IFDIR | 0o755
        nlink = 1
    return SimpleNamespace(
        st_mode=mode,
        st_nlink=nlink,
        st_size=getattr(actual, "st_size"),
        st_dev=getattr(actual, "st_dev"),
        st_ino=getattr(actual, "st_ino"),
        st_mtime_ns=getattr(actual, "st_mtime_ns"),
        st_ctime_ns=getattr(actual, "st_ctime_ns"),
    )


@pytest.mark.skipif(not packager._WINDOWS_NATIVE, reason="Windows-only mutation test")
def test_windows_parent_handles_block_directory_rename_during_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retained parent handles block renames until capture finishes."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "source.bin"
    source.write_bytes(b"captured")
    renamed = tmp_path / "renamed"
    original_read = packager._read_capture_descriptor

    def read_with_mutation(descriptor: int, *, path: Path) -> bytes:
        with pytest.raises(OSError):
            source_dir.rename(renamed)
        return original_read(descriptor, path=path)

    monkeypatch.setattr(packager, "_read_capture_descriptor", read_with_mutation)
    item = packager._capture_file(source)

    assert item.content == b"captured"
    source_dir.rename(renamed)
    assert (renamed / source.name).read_bytes() == b"captured"


def test_windows_recheck_ignores_unreliable_identity_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows rechecks accept stable files with changed identity fields."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"captured")
    monkeypatch.setattr(packager, "_WINDOWS", True)
    monkeypatch.setattr(packager, "_WINDOWS_NATIVE", False)
    inventory = packager._snapshot_inventory(paths=[source])
    item = inventory[0]
    changed_identity = replace(
        item,
        mode=item.mode ^ 0o1,
        device=item.device + 1,
        inode=item.inode + 1,
        mtime_ns=item.mtime_ns + 1,
        ctime_ns=item.ctime_ns + 1,
    )
    monkeypatch.setattr(packager, "_capture_file", lambda _: changed_identity)

    packager._recheck_inventory(inventory)


@pytest.mark.parametrize(
    "field", ["size", "sha256"], ids=["size-change", "hash-change"]
)
def test_windows_recheck_rejects_size_or_hash_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """Windows rechecks reject changed size or content hashes."""
    source = tmp_path / "source.bin"
    source.write_bytes(b"captured")
    monkeypatch.setattr(packager, "_WINDOWS", True)
    monkeypatch.setattr(packager, "_WINDOWS_NATIVE", False)
    inventory = packager._snapshot_inventory(paths=[source])
    item = inventory[0]
    value = "0" * 64 if field == "sha256" else getattr(item, field) + 1
    changed = replace(item, **{field: value})
    monkeypatch.setattr(packager, "_capture_file", lambda _: changed)

    with pytest.raises(ReleasePackagingError, match="Consumed file changed"):
        packager._recheck_inventory(inventory)
