"""Raw archive restoration and offline preparation tests."""

import io
import tarfile
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from danish_personas.io import sha256_file, write_json
from danish_personas.models import SnapshotManifest
from danish_personas.sampling.generator import generate_records
from danish_personas.sources.prepare import prepare_bundle
from danish_personas.validation.checks import validate_demographics, validate_sources
from scripts.restore_raw_sources import RAW_DIRECTORY, main

PROJECT_ROOT = Path(__file__).parents[2]
ARCHIVE_PATH = PROJECT_ROOT / "data" / f"{RAW_DIRECTORY}.tar.zst"


def test_committed_archive_restores_a_valid_source_bundle(tmp_path: Path) -> None:
    """The committed archive is sufficient for offline source preparation."""
    output_dir = tmp_path / "data"
    result = _restore(archive_path=ARCHIVE_PATH, output_dir=output_dir)
    assert result.exit_code == 0, result.output
    assert len(_restored_files(output_dir=output_dir)) == 45

    bundle_dir = prepare_bundle(
        lock_path=PROJECT_ROOT / "config" / "sources.lock.yaml",
        categories_path=PROJECT_ROOT / "config" / "categories.yaml",
        raw_dir=output_dir / RAW_DIRECTORY,
        output_dir=tmp_path / "prepared",
    )
    assert validate_sources(bundle_dir=bundle_dir).passed
    report_path = bundle_dir / "validation-report.json"
    manifest_path = bundle_dir / "bundle-manifest.json"
    report_bytes = report_path.read_bytes()
    report_sha256 = sha256_file(report_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = sha256_file(manifest_path)

    assert (
        prepare_bundle(
            lock_path=PROJECT_ROOT / "config" / "sources.lock.yaml",
            categories_path=PROJECT_ROOT / "config" / "categories.yaml",
            raw_dir=output_dir / RAW_DIRECTORY,
            output_dir=tmp_path / "prepared",
        )
        == bundle_dir
    )

    run_dir = generate_records(
        bundle_dir=bundle_dir,
        sampling_config_path=PROJECT_ROOT / "config" / "sampling.yaml",
        output_dir=tmp_path / "runs",
        rows=2_000,
        seed=20260914,
    )
    assert validate_sources(bundle_dir=bundle_dir).passed
    assert report_path.read_bytes() == report_bytes
    assert sha256_file(report_path) == report_sha256
    assert manifest_path.read_bytes() == manifest_bytes
    assert sha256_file(manifest_path) == manifest_sha256

    report = validate_demographics(
        run_dir=run_dir,
        bundle_dir=bundle_dir,
        validation_config_path=PROJECT_ROOT / "config" / "validation.yaml",
        categories_path=PROJECT_ROOT / "config" / "categories.yaml",
    )
    assert report.passed


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


def test_preparation_rejects_unexpected_missing_origin_code(tmp_path: Path) -> None:
    """Preparation rejects a selected origin code removed from a valid snapshot."""
    output_dir = tmp_path / "data"
    result = _restore(archive_path=ARCHIVE_PATH, output_dir=output_dir)
    assert result.exit_code == 0, result.output

    snapshot_dir = next((output_dir / RAW_DIRECTORY).glob("folk2/*"))
    data_path = snapshot_dir / "data.csv"
    rows = data_path.read_text(encoding="utf-8-sig").splitlines(keepends=True)
    data_path.write_text(
        rows[0]
        + "".join(
            row
            for row in rows[1:]
            if row.split(";", maxsplit=5)[4].split(" ", maxsplit=1)[0] != "5100"
        ),
        encoding="utf-8",
        newline="",
    )
    manifest_path = snapshot_dir / "snapshot-manifest.json"
    snapshot = SnapshotManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    write_json(
        path=manifest_path,
        payload=snapshot.model_copy(
            update={
                "data_sha256": sha256_file(data_path),
                "data_bytes": data_path.stat().st_size,
            }
        ),
    )

    with pytest.raises(ValueError, match="Unexpected missing FOLK2 IELAND codes"):
        prepare_bundle(
            lock_path=PROJECT_ROOT / "config" / "sources.lock.yaml",
            categories_path=PROJECT_ROOT / "config" / "categories.yaml",
            raw_dir=output_dir / RAW_DIRECTORY,
            output_dir=tmp_path / "prepared",
        )


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
    assert len(_restored_files(output_dir=output_dir)) == 45


@pytest.mark.parametrize(
    ("name", "member_type"),
    [
        (f"{RAW_DIRECTORY}/../../escape.txt", tarfile.REGTYPE),
        ("/tmp/danish-personas-absolute.txt", tarfile.REGTYPE),
        (f"{RAW_DIRECTORY}/link", tarfile.SYMTYPE),
        (f"{RAW_DIRECTORY}/hardlink", tarfile.LNKTYPE),
    ],
    ids=["traversal", "absolute-path", "symlink", "hardlink"],
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
    assert not output_dir.exists()


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
