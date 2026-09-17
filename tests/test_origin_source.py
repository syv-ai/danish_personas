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
    assert source.estimated_cells == 2186352
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


def test_origin_metrics_fail_when_a_selected_code_is_missing_raw() -> None:
    """The selected raw partition must contain every selected origin code."""
    raw = _origin_raw(codes=["5100"], suppressed=[False])
    labels = {"5100": "Denmark", "5103": "Stateless"}
    marginal = _origin_country_marginal(raw_frame=raw, official_labels=labels)

    metrics = _origin_country_metrics(
        raw_frame=raw,
        prepared_frame=marginal,
        official_labels=labels,
        selected_codes=["5100", "5103"],
    )

    assert not metrics["passed"]
    partition = _nested_metric(metrics=metrics, name="expected_partition")
    assert not partition["passed"]
    assert partition["missing_raw"] == ["5103"]


def _nested_metric(metrics: dict[str, object], name: str) -> dict[str, object]:
    metric = metrics[name]
    assert isinstance(metric, dict)
    return metric


def _origin_raw(codes: list[str], suppressed: list[bool]) -> pl.DataFrame:
    return pl.DataFrame(
        {"IELAND": codes, "count": [100] * len(codes), "suppressed": suppressed}
    )


def test_origin_metrics_reject_changed_official_label() -> None:
    """The prepared mapping must retain every official label verbatim."""
    raw = _origin_raw(codes=["5100", "5103"], suppressed=[False, False])
    labels = {"5100": "Denmark", "5103": "Stateless"}
    marginal = _origin_country_marginal(raw_frame=raw, official_labels=labels)
    marginal = marginal.with_columns(
        pl.when(pl.col("origin_country_code") == "5103")
        .then(pl.lit("Changed"))
        .otherwise(pl.col("origin_country"))
        .alias("origin_country")
    )

    metrics = _origin_country_metrics(
        raw_frame=raw,
        prepared_frame=marginal,
        official_labels=labels,
        selected_codes=["5100", "5103"],
    )

    assert not metrics["passed"]
    mapping = _nested_metric(metrics=metrics, name="metadata_mapping")
    assert not mapping["passed"]
    assert mapping["mismatches"] == ["5103"]


def test_origin_metrics_reject_suppressed_cells() -> None:
    """A suppressed raw FOLK2 cell fails the origin integrity gate."""
    raw = _origin_raw(codes=["5100", "5103"], suppressed=[False, True])
    labels = {"5100": "Denmark", "5103": "Stateless"}
    marginal = _origin_country_marginal(raw_frame=raw, official_labels=labels)

    metrics = _origin_country_metrics(
        raw_frame=raw,
        prepared_frame=marginal,
        official_labels=labels,
        selected_codes=["5100", "5103"],
    )

    assert not metrics["passed"]
    suppression = _nested_metric(metrics=metrics, name="zero_suppression")
    assert not suppression["passed"]


def test_origin_metrics_require_the_expected_partition() -> None:
    """The source checks reject an unhandled origin code."""
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


def test_origin_metrics_require_unique_labels() -> None:
    """Two origin codes cannot share a prepared label."""
    raw = _origin_raw(codes=["5100", "5103"], suppressed=[False, False])
    labels = {"5100": "Denmark", "5103": "Stateless"}
    marginal = _origin_country_marginal(raw_frame=raw, official_labels=labels)
    marginal = marginal.with_columns(
        pl.when(pl.col("origin_country_code") == "5103")
        .then(pl.lit("Denmark"))
        .otherwise(pl.col("origin_country"))
        .alias("origin_country")
    )

    metrics = _origin_country_metrics(
        raw_frame=raw,
        prepared_frame=marginal,
        official_labels=labels,
        selected_codes=["5100", "5103"],
    )

    assert not metrics["passed"]
    labels_metric = _nested_metric(metrics=metrics, name="label_uniqueness")
    assert not labels_metric["passed"]
