"""Direct tests for public source archive and acquisition services."""

from pathlib import Path

import pytest

from danish_personas.models import SourceLock
from danish_personas.sources import acquisition, classification, statbank
from danish_personas.sources.archive import build_raw_archive, restore_raw_sources
from danish_personas.sources.exceptions import SourceAcquisitionError


def test_archive_services_round_trip_raw_files(tmp_path: Path) -> None:
    """The package archive services preserve files through a safe round trip."""
    raw_dir = tmp_path / "raw-hardened-20260917"
    source_file = raw_dir / "folk1a" / "query-hash" / "data.csv"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("value\n1\n", encoding="utf-8", newline="\n")
    archive_path = tmp_path / "sources.tar.zst"

    assert build_raw_archive(raw_dir=raw_dir, archive_path=archive_path) == 1
    restored_dir = tmp_path / "restored"
    assert restore_raw_sources(archive_path=archive_path, output_dir=restored_dir) == 1
    restored_file = restored_dir / raw_dir.name / "folk1a" / "query-hash" / "data.csv"
    assert restored_file.read_text(encoding="utf-8", newline="") == "value\n1\n"


def test_fetch_service_exposes_domain_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Adapter failures are translated to the package domain exception."""
    lock = SourceLock.model_construct(classifications=[])

    def fail_fetch(lock: SourceLock, raw_dir: Path) -> list[object]:
        raise OSError("offline")

    monkeypatch.setattr(statbank, "fetch_sources", fail_fetch)

    with pytest.raises(SourceAcquisitionError, match="offline"):
        acquisition.fetch_sources(lock=lock, raw_dir=tmp_path)


def test_fetch_service_orchestrates_tables_and_classifications(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fetching uses both source adapters behind one explicit service boundary."""
    lock = SourceLock.model_construct(classifications=[])
    table_manifests = [object()]
    classification_manifests = [object()]

    def fetch_tables(lock: SourceLock, raw_dir: Path) -> list[object]:
        assert lock is lock_fixture
        assert raw_dir == tmp_path
        return table_manifests

    def fetch_classifications(
        classifications: list[object], raw_dir: Path
    ) -> list[object]:
        assert classifications == []
        assert raw_dir == tmp_path
        return classification_manifests

    lock_fixture = lock
    monkeypatch.setattr(statbank, "fetch_sources", fetch_tables)
    monkeypatch.setattr(classification, "fetch_classifications", fetch_classifications)

    result = acquisition.fetch_sources(lock=lock, raw_dir=tmp_path)

    assert result.table_manifests == table_manifests
    assert result.classification_manifests == classification_manifests
