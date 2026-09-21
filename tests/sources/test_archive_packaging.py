"""Raw archive packing safety and reproducibility tests."""

from pathlib import Path

import pytest

from danish_personas.sources import archive as archive_service
from danish_personas.sources.archive import RAW_DIRECTORY
from danish_personas.sources.exceptions import SourceArchiveError


def test_packing_is_byte_stable(tmp_path: Path) -> None:
    """Repacking unchanged snapshots reproduces identical archive bytes."""
    raw_dir = tmp_path / RAW_DIRECTORY
    (raw_dir / "folk1a" / "abc").mkdir(parents=True)
    (raw_dir / "folk1a" / "abc" / "data.csv").write_text(
        "a;b\n1;2\n", encoding="utf-8"
    )
    (raw_dir / "classifications").mkdir()
    (raw_dir / "classifications" / "data.csv").write_text("x;y\n", encoding="utf-8")

    first = tmp_path / "first.tar.zst"
    second = tmp_path / "second.tar.zst"
    for archive in (first, second):
        archive_service.build_raw_archive(raw_dir=raw_dir, archive_path=archive)

    assert first.read_bytes() == second.read_bytes()


def test_packing_rejects_symlink_to_content_outside_raw_tree(tmp_path: Path) -> None:
    """A symlink cannot make the archive include content outside the raw tree."""
    raw_dir = tmp_path / RAW_DIRECTORY
    raw_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    (raw_dir / "escaped.txt").symlink_to(outside)
    archive_path = tmp_path / "archive.tar.zst"

    with pytest.raises(SourceArchiveError, match="non-regular"):
        archive_service.build_raw_archive(raw_dir=raw_dir, archive_path=archive_path)
    assert not archive_path.exists()


def test_packing_reports_enumeration_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Enumeration failures remain visible at the service boundary."""
    raw_dir = tmp_path / RAW_DIRECTORY
    raw_dir.mkdir()
    archive_path = tmp_path / "archive.tar.zst"

    def fail_enumeration(raw_dir: Path) -> list[Path]:
        raise PermissionError("enumeration denied")

    monkeypatch.setattr(archive_service, "_regular_files", fail_enumeration)

    with pytest.raises(SourceArchiveError, match="enumeration denied"):
        archive_service.build_raw_archive(raw_dir=raw_dir, archive_path=archive_path)
    assert not archive_path.exists()
