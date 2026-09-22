"""Offline tests for origin validation against sampling eligibility."""

from pathlib import Path

import polars as pl

from danish_personas.validation.checks import _origin_mapping_metrics
from tests.support.bundles import _write_bundle


def test_official_but_ineligible_origin_code_fails_validation(tmp_path: Path) -> None:
    """A valid official label triple cannot bypass the eligibility gate."""
    bundle_dir, _, _, _ = _write_bundle(root=tmp_path)
    origin = pl.read_parquet(
        bundle_dir / "normalized" / "folk2_origin_country_marginal.parquet"
    )
    emitted = origin.filter(pl.col("origin_country_code") == "5103").select(
        "origin_country_code", "origin_country", "origin_country_da"
    )

    metrics = _origin_mapping_metrics(frame=emitted, bundle_dir=bundle_dir)
    eligible = next(
        metric for metric in metrics if metric.name == "origin_country_eligible"
    )

    assert not eligible.passed
    assert eligible.value == 1
