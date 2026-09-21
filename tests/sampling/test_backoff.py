"""Tests for sparse-cell back-off in the demographic sampler."""

import typing as t

import numpy as np
import polars as pl
import pytest

from danish_personas.ladders import SAMPLED_ATTRIBUTES, levels
from danish_personas.models import DemographicRecord
from danish_personas.sampling.generator import (
    LadderIndex,
    _distribution_index,
    _draw,
    _ladder_index,
)

VALUES = {
    "municipality_code": "101",
    "age_band": "30-49",
    "sex": "female",
    "labour_market_status": "employed",
}
MARITAL_LADDER = SAMPLED_ATTRIBUTES[1].ladder
DETAIL_LADDER = SAMPLED_ATTRIBUTES[2].ladder


def test_back_off_consumes_one_draw_regardless_of_level() -> None:
    """Sparsity must not shift the random stream."""
    exact, backed_off = np.random.default_rng(7), np.random.default_rng(7)
    _draw(ladder_index=_marital_index(), values=VALUES, rng=exact)
    _draw(
        ladder_index=_marital_index(), values={**VALUES, "sex": "male"}, rng=backed_off
    )

    assert exact.integers(0, 1_000_000) == backed_off.integers(0, 1_000_000)


def _marital_index(municipality_code: str = VALUES["municipality_code"]) -> LadderIndex:
    """Build a marital ladder index over one municipality's counts.

    Args:
        municipality_code:
            Municipality the counts belong to. A different municipality leaves
            every municipality-bound cell unpopulated.

    Returns:
        The built marital ladder.
    """
    return _ladder_index(
        frame=_marital_frame(municipality_code=municipality_code),
        ladder=MARITAL_LADDER,
        payload_columns=["marital_status"],
        smoothing=0.0,
    )


def _marital_frame(municipality_code: str) -> pl.DataFrame:
    """Build a two-category marital frame for one municipality.

    Args:
        municipality_code:
            Municipality the counts belong to.

    Returns:
        Source counts with the columns the marital ladder addresses.
    """
    return pl.DataFrame(
        {
            "municipality_code": [municipality_code, municipality_code],
            "age_band": ["30-49", "30-49"],
            "sex": ["female", "female"],
            "marital_status": ["married_or_separated", "never_married"],
            "count": [900, 100],
            "suppressed": [False, False],
        }
    )


def test_detailed_status_never_backs_off_past_its_broad_status() -> None:
    """RAS202 may only refine the broad status already sampled."""
    frame = pl.DataFrame(
        {
            "age_band": ["30-49"],
            "sex": ["male"],
            "labour_market_status": ["retired"],
            "detailed_status_code": ["135"],
            "detailed_status": ["pensioner"],
            "count": [10],
            "suppressed": [False],
        }
    )
    index = _ladder_index(
        frame=frame,
        ladder=DETAIL_LADDER,
        payload_columns=["detailed_status_code", "detailed_status"],
        smoothing=0.0,
    )

    with pytest.raises(ValueError, match="No prepared distribution"):
        _draw(ladder_index=index, values=VALUES, rng=np.random.default_rng(0))


def test_draw_fails_when_the_municipality_is_absent() -> None:
    """Back-off must never cross from one municipality into another."""
    with pytest.raises(ValueError, match="No prepared distribution"):
        _draw(
            ladder_index=_marital_index(municipality_code="999"),
            values=VALUES,
            rng=np.random.default_rng(0),
        )


def test_draw_uses_the_most_specific_populated_cell() -> None:
    """An exact cell is preferred over any coarser one."""
    _, level = _draw(
        ladder_index=_marital_index(), values=VALUES, rng=np.random.default_rng(0)
    )

    assert level == "municipality_age_band_sex"


def test_smoothing_only_reweights_surviving_cells() -> None:
    """Smoothing must not resurrect a structurally absent category."""
    # Counts 900 and 100 give 0.9/0.1 unsmoothed and 1000/1200, 200/1200 at 100.
    frame = _marital_frame(municipality_code=VALUES["municipality_code"])
    key = (VALUES["municipality_code"], VALUES["age_band"], VALUES["sex"])
    marital_key_columns = ["municipality_code", "age_band", "sex"]
    plain = _distribution_index(
        frame=frame,
        key_columns=marital_key_columns,
        payload_columns=["marital_status"],
        smoothing=0.0,
    )
    smoothed = _distribution_index(
        frame=frame,
        key_columns=marital_key_columns,
        payload_columns=["marital_status"],
        smoothing=100.0,
    )
    payloads, probabilities = smoothed[key]

    # Smoothing reweights the categories that survived the structural filter,
    # pulling them towards uniform without introducing an absent category.
    assert {payload["marital_status"] for payload in payloads} == {
        "married_or_separated",
        "never_married",
    }
    assert probabilities.sum() == pytest.approx(1.0)
    assert plain[key][1].min() == pytest.approx(0.1)
    assert probabilities.min() == pytest.approx(200 / 1200)


def test_the_schema_accepts_exactly_the_ladder_levels() -> None:
    """Every level a ladder can report must be a value the record allows."""
    for attribute in SAMPLED_ATTRIBUTES:
        field = DemographicRecord.model_fields[attribute.resolution_column]

        assert t.get_args(field.annotation) == levels(attribute.ladder)
