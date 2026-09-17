"""Tests for sparse-cell back-off in the demographic sampler."""

import numpy as np
import polars as pl
import pytest

from danish_personas.sampling.generator import (
    DETAIL_LADDER,
    MARITAL_LADDER,
    _draw,
    _ladder_index,
)

VALUES = {
    "age_band": "30-49",
    "sex": "female",
    "region_code": "084",
    "labour_market_status": "employed",
}


def test_back_off_consumes_one_draw_regardless_of_level() -> None:
    """Sparsity must not shift the random stream."""
    exact = _ladder_index(
        frame=_marital_frame(region_code="084"),
        ladder=MARITAL_LADDER,
        payload_columns=["marital_status"],
        smoothing=0.0,
    )
    coarse = _ladder_index(
        frame=_marital_frame(region_code="999"),
        ladder=MARITAL_LADDER,
        payload_columns=["marital_status"],
        smoothing=0.0,
    )

    states = []
    for index in (exact, coarse):
        rng = np.random.default_rng(7)
        _draw(ladder_index=index, values=VALUES, rng=rng)
        states.append(rng.integers(0, 1_000_000))

    assert states[0] == states[1]


def _marital_frame(region_code: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "region_code": [region_code, region_code],
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


def test_draw_backs_off_when_the_exact_cell_is_absent() -> None:
    """A missing region cell falls back to the age-and-sex cell."""
    index = _ladder_index(
        frame=_marital_frame(region_code="999"),
        ladder=MARITAL_LADDER,
        payload_columns=["marital_status"],
        smoothing=0.0,
    )

    payload, level = _draw(
        ladder_index=index, values=VALUES, rng=np.random.default_rng(0)
    )

    assert level == "age_band_sex"
    assert payload["marital_status"] in {"married_or_separated", "never_married"}


def test_draw_uses_the_most_specific_populated_cell() -> None:
    """An exact cell is preferred over any coarser one."""
    index = _ladder_index(
        frame=_marital_frame(region_code="084"),
        ladder=MARITAL_LADDER,
        payload_columns=["marital_status"],
        smoothing=0.0,
    )

    _, level = _draw(ladder_index=index, values=VALUES, rng=np.random.default_rng(0))

    assert level == "region_age_band_sex"


def test_smoothing_only_reweights_surviving_cells() -> None:
    """Smoothing must not resurrect a structurally absent category."""
    frame = _marital_frame(region_code="084")
    key = ("084", "30-49", "female")
    plain = _ladder_index(
        frame=frame,
        ladder=MARITAL_LADDER,
        payload_columns=["marital_status"],
        smoothing=0.0,
    )
    smoothed = _ladder_index(
        frame=frame,
        ladder=MARITAL_LADDER,
        payload_columns=["marital_status"],
        smoothing=100.0,
    )
    payloads, probabilities = smoothed[0][2][key]
    _, unsmoothed = plain[0][2][key]

    # Smoothing reweights the categories that survived the structural filter,
    # pulling them towards uniform without introducing an absent category.
    assert {payload["marital_status"] for payload in payloads} == {
        "married_or_separated",
        "never_married",
    }
    assert probabilities.sum() == pytest.approx(1.0)
    assert unsmoothed.min() == pytest.approx(0.1)
    assert probabilities.min() == pytest.approx(200 / 1200)
    assert probabilities.min() > unsmoothed.min()
