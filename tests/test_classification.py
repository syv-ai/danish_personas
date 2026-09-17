"""Tests for the official geography classification."""

from pathlib import Path

import pytest

from danish_personas.sources.prepare import (
    _geography_metrics,
    read_geography_classification,
)

HEADER = "SEKVENS;KODE;NIVEAU;TITEL;GENERELLE_NOTER\n"


def test_cross_check_fails_when_statbank_disagrees(tmp_path: Path) -> None:
    """A region mismatch against StatBank metadata must fail the bundle."""
    csv_path = _write_classification(
        tmp_path=tmp_path,
        body=(
            '1;"084";1;Region Hovedstaden;\n'
            '2;"01";2;Landsdel Byen København;\n'
            '3;"101";3;København;\n'
        ),
    )
    geography = read_geography_classification(csv_path=csv_path)

    agreeing = _geography_metrics(
        geography=geography, statbank_region_map={"101": ("084", "Region Hovedstaden")}
    )
    assert agreeing["passed"]

    disagreeing = _geography_metrics(
        geography=geography, statbank_region_map={"101": ("085", "Region Sjælland")}
    )
    assert not disagreeing["passed"]
    assert disagreeing["statbank_disagreements"] == ["101"]

    incomplete = _geography_metrics(
        geography=geography,
        statbank_region_map={
            "101": ("084", "Region Hovedstaden"),
            "400": ("084", "Region Hovedstaden"),
        },
    )
    assert not incomplete["passed"]
    assert incomplete["missing_from_classification"] == ["400"]


def _write_classification(tmp_path: Path, body: str) -> Path:
    """Write a classification attachment fixture with the official header.

    Args:
        tmp_path:
            Temporary directory for the fixture.
        body:
            Semicolon-delimited rows following the header.

    Returns:
        Path to the written attachment.
    """
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(HEADER + body, encoding="utf-8")
    return csv_path


def test_reader_rejects_a_classification_without_municipalities(tmp_path: Path) -> None:
    """A truncated attachment must not produce an empty hierarchy."""
    csv_path = _write_classification(
        tmp_path=tmp_path, body='1;"084";1;Region Hovedstaden;\n'
    )

    with pytest.raises(ValueError, match="No municipalities"):
        read_geography_classification(csv_path=csv_path)


def test_reader_rejects_a_municipality_before_its_parents(tmp_path: Path) -> None:
    """An orphaned municipality is a parsing failure, not a silent null."""
    csv_path = _write_classification(tmp_path=tmp_path, body='1;"101";3;København;\n')

    with pytest.raises(ValueError, match="before its parents"):
        read_geography_classification(csv_path=csv_path)


def test_reader_resolves_each_municipality_to_its_parents(tmp_path: Path) -> None:
    """Municipalities inherit the landsdel and region above them."""
    csv_path = _write_classification(
        tmp_path=tmp_path,
        body=(
            '1;"084";1;Region Hovedstaden;\n'
            '2;"01";2;Landsdel Byen København;\n'
            '3;"101";3;København;"A note\n\nspanning lines."\n'
            '4;"04";2;Landsdel Bornholm;\n'
            '5;"400";3;Bornholm;\n'
            '6;"085";1;Region Sjælland;\n'
            '7;"02";2;Landsdel Østsjælland;\n'
            '8;"265";3;Roskilde;\n'
        ),
    )

    frame = read_geography_classification(csv_path=csv_path)

    assert frame.height == 3
    rows = {row["municipality_code"]: row for row in frame.iter_rows(named=True)}
    assert rows["101"]["region_code"] == "084"
    assert rows["101"]["landsdel"] == "Landsdel Byen København"
    assert rows["400"]["landsdel_code"] == "04"
    assert rows["400"]["region_code"] == "084"
    assert rows["265"]["region"] == "Region Sjælland"
