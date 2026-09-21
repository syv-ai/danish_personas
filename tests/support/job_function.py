"""Shared LONS20 fixture data for source and sampler tests."""

import polars as pl

from danish_personas.models import DISCO_TWO_DIGIT_CODES
from danish_personas.sources.prepare import _job_function_sex_marginal


def prepared_job_function_marginal() -> pl.DataFrame:
    """Return the prepared LONS20 fixture table."""
    return _job_function_sex_marginal(
        raw_frame=raw_job_function_marginal(),
        official_labels=job_function_labels(),
        selected_codes=list(DISCO_TWO_DIGIT_CODES),
        sex_mapping={"M": "male", "K": "female"},
    )


def job_function_labels() -> dict[str, str]:
    """Return official-looking labels for every fixture job-function code."""
    return {code: f"{code} Official label" for code in DISCO_TWO_DIGIT_CODES}


def raw_job_function_marginal() -> pl.DataFrame:
    """Return a complete unsuppressed LONS20 fixture table."""
    return pl.DataFrame(
        [
            {
                "ARBF": code,
                "ARBF__label": f"{code} Official label",
                "SEKTOR": "1000",
                "AFLOEN": "TIFA",
                "LONGRP": "LTOT",
                "LØNMÅL": "ANTAL",
                "KØN": sex,
                "Tid": "2024",
                "count": index + (1 if sex == "M" else 2),
                "suppressed": False,
            }
            for index, code in enumerate(DISCO_TWO_DIGIT_CODES)
            for sex in ("M", "K")
        ]
    )
