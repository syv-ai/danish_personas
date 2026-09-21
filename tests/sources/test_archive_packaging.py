"""Raw archive packing safety and reproducibility tests."""

from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from danish_personas.sources import archive as archive_service
from danish_personas.sources.exceptions import SourceArchiveError
from scripts.build_raw_archive import main as pack
from scripts.restore_raw_sources import RAW_DIRECTORY, main


def test_archive_command_help_retains_legacy_wording() -> None:
    """Archive command help retains the original descriptions."""
    runner = CliRunner()

    pack_help = runner.invoke(pack, ["--help"])
    restore_help = runner.invoke(main, ["--help"])

    assert pack_help.exit_code == 0
    assert restore_help.exit_code == 0
    compact_pack_help = " ".join(pack_help.output.split())
    compact_restore_help = " ".join(restore_help.output.split())
    assert "Members are sorted by archive path" in compact_pack_help
    assert "Optional directory holding the immutable raw snapshots" in compact_pack_help
    assert "Optional destination archive" in compact_pack_help
    assert (
        "If the archive is missing, unsafe, corrupt, or the target already exists."
        in compact_restore_help
    )


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


def test_packing_translates_enumeration_failure_to_click_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Enumeration OSErrors cross the service and command error boundaries."""
    raw_dir = tmp_path / RAW_DIRECTORY
    raw_dir.mkdir()
    archive_path = tmp_path / "archive.tar.zst"

    def fail_enumeration(raw_dir: Path) -> list[Path]:
        raise PermissionError("enumeration denied")

    monkeypatch.setattr(archive_service, "_regular_files", fail_enumeration)

    with pytest.raises(click.ClickException) as raised:
        CliRunner().invoke(
            pack,
            ["--raw-dir", str(raw_dir), "--archive", str(archive_path)],
            catch_exceptions=False,
            standalone_mode=False,
        )

    assert isinstance(raised.value.__cause__, SourceArchiveError)
    assert "enumeration denied" in str(raised.value)
    assert not archive_path.exists()
