"""Offline tests for deterministic FOLK2 origin-country sampling."""

import numpy as np
import polars as pl
import pytest

from danish_personas.sampling.generator import _origin_quota_sample


def test_origin_sampling_is_deterministic_and_preserves_unequal_weights() -> None:
    """Repeated seeded quota sampling has exact largest-remainder counts."""
    first = _origin_quota_sample(
        frame=_marginal(), rows=10, rng=np.random.default_rng(42)
    )
    second = _origin_quota_sample(
        frame=_marginal(), rows=10, rng=np.random.default_rng(42)
    )

    assert first.equals(second)
    assert first.group_by("origin_country_code").len().sort("origin_country_code").rows(
        named=True
    ) == [
        {"origin_country_code": "5100", "len": 8},
        {"origin_country_code": "5103", "len": 2},
    ]
    assert set(first.get_column("origin_country")) == {"Denmark", "Stateless"}


def _marginal() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "origin_country_code": ["5100", "5103", "5999"],
            "origin_country": ["Denmark", "Stateless", "Not stated"],
            "count": [7, 2, 0],
        }
    )


def test_origin_sampling_rejects_duplicate_code_label_mapping() -> None:
    """A code-to-label partition must be one-to-one."""
    malformed = _marginal().with_columns(
        pl.when(pl.col("origin_country_code") == "5103")
        .then(pl.lit("Denmark"))
        .otherwise(pl.col("origin_country"))
        .alias("origin_country")
    )

    with pytest.raises(ValueError, match="duplicate origin labels"):
        _origin_quota_sample(frame=malformed, rows=10, rng=np.random.default_rng(42))


def test_origin_sampling_rejects_malformed_distributions() -> None:
    """Malformed official partitions fail before a record can be emitted."""
    cases = [
        _marginal().with_columns(pl.lit(-1).alias("count")),
        _marginal().with_columns(pl.lit(1.5).alias("count")),
        _marginal().with_columns(pl.lit(None).alias("origin_country")),
        _marginal().vstack(_marginal().head(1)),
        _marginal().with_columns(pl.lit(0).alias("count")),
    ]

    for frame in cases:
        with pytest.raises(ValueError):
            _origin_quota_sample(frame=frame, rows=10, rng=np.random.default_rng(42))
