"""Offline tests for the FOLK2 origin marginal preparation."""

from pathlib import Path

import polars as pl

from danish_personas.io import load_yaml_model
from danish_personas.models import SourceLock
from danish_personas.sources.prepare import (
    _origin_country_marginal,
    _origin_country_metrics,
)


def test_folk2_lock_freezes_the_adult_official_partition() -> None:
    """The offline lock preserves the reviewed FOLK2 selection."""
    lock = load_yaml_model(path=Path("config/sources.lock.yaml"), model=SourceLock)
    source = next(source for source in lock.sources if source.table_id == "FOLK2")

    assert source.period == "2025"
    assert source.format == "BULK"
    assert source.estimated_cells == 312336
    assert source.dimensions["ALDER"] == [str(age) for age in range(18, 126)]
    assert source.dimensions["KØN"] == ["M", "K"]
    assert source.dimensions["HERKOMST"] == ["5", "4", "3"]
    assert source.dimensions["STATSB"] == ["DANSK", "UDLAND"]
    assert len(source.dimensions["IELAND"]) == 241
    assert source.dimensions["Tid"] == ["2025"]


def test_origin_marginal_preserves_official_labels_and_weights() -> None:
    """Aggregation retains unequal official counts and zero categories."""
    raw = pl.DataFrame(
        {
            "IELAND": ["5100", "5100", "5103"],
            "count": [100, 7, 2],
            "suppressed": [False, False, False],
        }
    )
    labels = {"5100": "Denmark", "5103": "Stateless", "5999": "Not stated"}

    marginal = _origin_country_marginal(raw_frame=raw, official_labels=labels)

    assert marginal.to_dicts() == [
        {"origin_country_code": "5100", "origin_country": "Denmark", "count": 107},
        {"origin_country_code": "5103", "origin_country": "Stateless", "count": 2},
        {"origin_country_code": "5999", "origin_country": "Not stated", "count": 0},
    ]


def test_origin_metrics_require_the_expected_partition() -> None:
    """The source checks reject an unhandled selected origin code."""
    raw = pl.DataFrame(
        {"IELAND": ["5100", "9999"], "count": [100, 3], "suppressed": [False, False]}
    )
    labels = {"5100": "Denmark"}
    marginal = _origin_country_marginal(raw_frame=raw, official_labels=labels)

    metrics = _origin_country_metrics(
        raw_frame=raw,
        prepared_frame=marginal,
        official_labels=labels,
        selected_codes=["5100"],
    )

    assert not metrics["passed"]
    unhandled_values = metrics["unhandled_values"]
    assert isinstance(unhandled_values, dict)
    assert unhandled_values["values"] == ["9999"]
