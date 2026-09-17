"""Integration tests for the committed raw source archive."""

import io
import tarfile
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from danish_personas.sources.prepare import prepare_bundle
from danish_personas.validation.checks import validate_sources
from scripts.build_raw_archive import main as pack
from scripts.restore_raw_sources import RAW_DIRECTORY, main

PROJECT_ROOT = Path(__file__).parents[1]
ARCHIVE_PATH = PROJECT_ROOT / "data" / f"{RAW_DIRECTORY}.tar.zst"


def test_committed_archive_restores_a_valid_source_bundle(tmp_path: Path) -> None:
    """The committed archive is sufficient for offline source preparation."""
    output_dir = tmp_path / "data"
    result = _restore(archive_path=ARCHIVE_PATH, output_dir=output_dir)
    assert result.exit_code == 0, result.output
    assert len(_restored_files(output_dir=output_dir)) == 39

    bundle_dir = prepare_bundle(
        lock_path=PROJECT_ROOT / "config" / "sources.lock.yaml",
        categories_path=PROJECT_ROOT / "config" / "categories.yaml",
        raw_dir=output_dir / RAW_DIRECTORY,
        output_dir=tmp_path / "prepared",
    )
    assert bundle_dir.name == "a637a48f35aa2fad"
    assert validate_sources(bundle_dir=bundle_dir).passed


def _restore(archive_path: Path, output_dir: Path, force: bool = False) -> Result:
    arguments = ["--archive", str(archive_path), "--output-dir", str(output_dir)]
    if force:
        arguments.append("--force")
    return CliRunner().invoke(main, arguments)


def _restored_files(output_dir: Path) -> list[Path]:
    return [path for path in (output_dir / RAW_DIRECTORY).rglob("*") if path.is_file()]


def test_force_replaces_target_symlink_without_following_it(tmp_path: Path) -> None:
    """Forced restoration replaces a destination symlink rather than following it."""
    output_dir = tmp_path / "data"
    output_dir.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    marker = external / "marker.txt"
    marker.write_text("untouched")
    target = output_dir / RAW_DIRECTORY
    target.symlink_to(external, target_is_directory=True)

    refused = _restore(archive_path=ARCHIVE_PATH, output_dir=output_dir)
    assert refused.exit_code != 0
    assert target.is_symlink()

    restored = _restore(archive_path=ARCHIVE_PATH, output_dir=output_dir, force=True)
    assert restored.exit_code == 0, restored.output
    assert target.is_dir() and not target.is_symlink()
    assert marker.read_text(encoding="utf-8") == "untouched"


def test_packing_is_byte_stable(tmp_path: Path) -> None:
    """Repacking unchanged snapshots reproduces identical archive bytes."""
    raw_dir = tmp_path / RAW_DIRECTORY
    (raw_dir / "folk1a" / "abc").mkdir(parents=True)
    (raw_dir / "folk1a" / "abc" / "data.csv").write_text("a;b\n1;2\n", encoding="utf-8")
    (raw_dir / "classifications").mkdir()
    (raw_dir / "classifications" / "data.csv").write_text("x;y\n", encoding="utf-8")

    first = tmp_path / "first.tar.zst"
    second = tmp_path / "second.tar.zst"
    for archive in (first, second):
        result = CliRunner().invoke(
            pack, ["--raw-dir", str(raw_dir), "--archive", str(archive)]
        )
        assert result.exit_code == 0, result.output

    assert first.read_bytes() == second.read_bytes()


def test_packing_rejects_symlink_to_content_outside_raw_tree(tmp_path: Path) -> None:
    """A symlink cannot make the archive include content outside the raw tree."""
    raw_dir = tmp_path / RAW_DIRECTORY
    raw_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    (raw_dir / "escaped.txt").symlink_to(outside)
    archive_path = tmp_path / "archive.tar.zst"

    result = CliRunner().invoke(
        pack, ["--raw-dir", str(raw_dir), "--archive", str(archive_path)]
    )

    assert result.exit_code != 0
    assert "non-regular" in result.output
    assert not archive_path.exists()


def test_restore_refuses_existing_target_without_force(tmp_path: Path) -> None:
    """Restoration cannot silently merge with stale source files."""
    output_dir = tmp_path / "data"
    target = output_dir / RAW_DIRECTORY
    target.mkdir(parents=True)
    stale = target / "stale.txt"
    stale.write_text("preserve on refusal")

    result = _restore(archive_path=ARCHIVE_PATH, output_dir=output_dir)
    assert result.exit_code != 0
    assert stale.read_text(encoding="utf-8") == "preserve on refusal"

    result = _restore(archive_path=ARCHIVE_PATH, output_dir=output_dir, force=True)
    assert result.exit_code == 0, result.output
    assert not stale.exists()
    assert len(_restored_files(output_dir=output_dir)) == 39


@pytest.mark.parametrize(
    ("name", "member_type"),
    [
        (f"{RAW_DIRECTORY}/../../escape.txt", tarfile.REGTYPE),
        ("/tmp/danish-personas-absolute.txt", tarfile.REGTYPE),
        (f"{RAW_DIRECTORY}/link", tarfile.SYMTYPE),
        (f"{RAW_DIRECTORY}/hardlink", tarfile.LNKTYPE),
    ],
)
def test_restore_rejects_unsafe_members(
    tmp_path: Path, name: str, member_type: bytes
) -> None:
    """Traversal, absolute paths, symlinks, and hardlinks are rejected."""
    archive_path = tmp_path / "unsafe.tar.zst"
    _write_archive(path=archive_path, name=name, member_type=member_type)
    output_dir = tmp_path / "data"

    result = _restore(archive_path=archive_path, output_dir=output_dir)
    assert result.exit_code != 0
    assert not (tmp_path / "escape.txt").exists()
    assert not Path("/tmp/danish-personas-absolute.txt").exists()
    assert not (output_dir / RAW_DIRECTORY).exists()


def _write_archive(path: Path, name: str, member_type: bytes) -> None:
    payload = b"unsafe"
    member = tarfile.TarInfo(name=name)
    member.type = member_type
    if member_type == tarfile.REGTYPE:
        member.size = len(payload)
    else:
        member.linkname = "../../outside"
    with tarfile.open(path, mode="w:zst") as archive:
        archive.addfile(member, fileobj=io.BytesIO(payload))
