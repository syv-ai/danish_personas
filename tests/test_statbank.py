"""Tests for StatBank query sizing and format guards."""

import typing as t
from pathlib import Path

import pytest

from danish_personas.io import load_yaml_model
from danish_personas.models import (
    SourceDefinition,
    SourceLock,
    SourcesConfig,
    SourceSelection,
    StatBankMetadata,
    StatBankValue,
    StatBankVariable,
)
from danish_personas.sources import statbank
from danish_personas.sources.statbank import (
    MAX_CELLS,
    estimate_query_cells,
    resolve_sources,
)


def test_locked_non_bulk_sources_remain_below_api_limit() -> None:
    """Resolved non-streaming sources fit the official cell limit."""
    lock = load_yaml_model(path=Path("config/sources.lock.yaml"), model=SourceLock)

    assert all(
        source.estimated_cells <= MAX_CELLS
        for source in lock.sources
        if source.format != "BULK"
    )


def test_over_limit_bulk_query_is_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """BULK is the explicitly exempt streaming format for large queries."""
    source = _source(format="BULK", value_count=1_000)
    _stub_metadata(monkeypatch=monkeypatch, source=source)

    lock = resolve_sources(
        config=_config(source=source), lock_path=tmp_path / "sources.lock.yaml"
    )

    assert lock.sources[0].estimated_cells == 4_000_000


def _config(source: SourceDefinition) -> SourcesConfig:
    return SourcesConfig(
        version=2,
        language="en",
        release_rows=100,
        minimum_source_count=0,
        minimum_expected_release_count=0,
        sources=[source],
        classifications=[],
    )


def _source(format: t.Literal["CSV", "BULK"], value_count: int) -> SourceDefinition:
    return SourceDefinition(
        table_id="TEST",
        role="test",
        period="2025",
        format=format,
        dimensions={
            "A": SourceSelection(values=[str(value) for value in range(value_count)]),
            "B": SourceSelection(values=[str(value) for value in range(value_count)]),
            "Tid": SourceSelection(values=["2025"]),
        },
    )


def _stub_metadata(monkeypatch: pytest.MonkeyPatch, source: SourceDefinition) -> None:
    metadata = StatBankMetadata(
        id=source.table_id,
        text="Test",
        description="Test table",
        unit="Number",
        updated="2026-01-01T00:00:00",
        variables=[
            StatBankVariable(
                id="A",
                text="A",
                values=[
                    StatBankValue(id=str(value), text=str(value))
                    for value in range(len(source.dimensions["A"].values or []))
                ],
            ),
            StatBankVariable(
                id="B",
                text="B",
                values=[
                    StatBankValue(id=str(value), text=str(value))
                    for value in range(len(source.dimensions["B"].values or []))
                ],
            ),
            StatBankVariable(
                id="Tid", text="Time", values=[StatBankValue(id="2025", text="2025")]
            ),
        ],
    )

    def fake_get_metadata(
        client: object, table_id: str, language: str
    ) -> tuple[StatBankMetadata, bytes]:
        del client, table_id, language
        return metadata, b"{}"

    monkeypatch.setattr(statbank, "_get_metadata", fake_get_metadata)


def test_over_limit_csv_query_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-streaming query over the actual cell limit is rejected."""
    source = _source(format="CSV", value_count=1_000)
    _stub_metadata(monkeypatch=monkeypatch, source=source)

    with pytest.raises(ValueError, match="above the 1,000,000-cell API limit"):
        resolve_sources(
            config=_config(source=source), lock_path=tmp_path / "sources.lock.yaml"
        )


def test_query_cells_include_returned_dimension_and_value_columns() -> None:
    """Cell estimates include one column for every dimension and the value."""
    assert estimate_query_cells(dimensions={"age": ["18", "19"], "Tid": ["2025"]}) == 6


def test_resolved_lock_carries_expected_zero_codes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resolution copies the reviewed omission set into the lock."""
    source = _source(format="BULK", value_count=1).model_copy(
        update={"expected_zero_codes": ["0"]}
    )
    _stub_metadata(monkeypatch=monkeypatch, source=source)

    lock = resolve_sources(
        config=_config(source=source), lock_path=tmp_path / "sources.lock.yaml"
    )

    assert lock.sources[0].expected_zero_codes == ["0"]


def test_under_limit_csv_query_is_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A CSV query whose actual cells fit the limit resolves successfully."""
    source = _source(format="CSV", value_count=10)
    _stub_metadata(monkeypatch=monkeypatch, source=source)

    lock = resolve_sources(
        config=_config(source=source), lock_path=tmp_path / "sources.lock.yaml"
    )

    assert lock.sources[0].estimated_cells == 400
