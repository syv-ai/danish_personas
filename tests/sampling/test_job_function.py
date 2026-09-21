"""Job-function sampling tests."""

import numpy as np

from danish_personas.sampling.generator import _attach_job_functions
from tests.support.job_function import prepared_job_function_marginal


def test_job_function_allocation_is_eligible_and_deterministic() -> None:
    """Only employee codes receive deterministic sex-conditional allocations."""
    records: list[dict[str, object]] = [
        {"sex": "female", "detailed_status_code": "15"},
        {"sex": "female", "detailed_status_code": "05"},
        {"sex": "male", "detailed_status_code": "40"},
        {"sex": "male", "detailed_status_code": "10"},
    ]
    first = [dict(record) for record in records]
    second = [dict(record) for record in records]

    _attach_job_functions(
        records=first,
        marginal=prepared_job_function_marginal(),
        rng=np.random.default_rng(42),
    )
    _attach_job_functions(
        records=second,
        marginal=prepared_job_function_marginal(),
        rng=np.random.default_rng(42),
    )

    assert first == second
    assert first[0]["job_function_resolution"] == "lons20_sex_marginal"
    assert first[2]["job_function_resolution"] == "lons20_sex_marginal"
    for index in (1, 3):
        assert first[index]["job_function_code"] is None
        assert first[index]["job_function"] is None
        assert first[index]["job_function_resolution"] == "not_applicable"
