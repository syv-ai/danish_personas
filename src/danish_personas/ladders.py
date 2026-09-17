"""Sparse-cell back-off ladders for the demographic sampler.

A ladder lists the cells a drawn field may fall back to, most specific first.
Each one ends at the most general cell that is still structurally valid rather
than at a national distribution, so backing off cannot cross an invariant: an
age stays inside its band, and a detailed status stays inside the broad RAS209
status it refines. A cell missing at a ladder's final level is a structural
zero rather than sparsity, and must fail rather than widen further.

The ladders live in code rather than configuration for that reason: a terminal
level encodes an invariant, not a tunable, and no configuration schema could
reject a level that widened past it.
"""

from dataclasses import dataclass

Ladder = tuple[tuple[str, tuple[str, ...]], ...]

AGE_LADDER: Ladder = (
    ("age_band_sex", ("age_band", "sex")),
    ("age_band", ("age_band",)),
)
MARITAL_LADDER: Ladder = (
    ("region_age_band_sex", ("region_code", "age_band", "sex")),
    ("age_band_sex", ("age_band", "sex")),
    ("age_band", ("age_band",)),
)
DETAIL_LADDER: Ladder = (
    ("age_band_sex_status", ("age_band", "sex", "labour_market_status")),
    ("sex_status", ("sex", "labour_market_status")),
    ("status", ("labour_market_status",)),
)


@dataclass(frozen=True)
class SampledAttribute:
    """One field the sampler draws, and everything needed to draw it."""

    prepared_file: str
    ladder: Ladder
    payload_columns: tuple[str, ...]
    resolution_column: str


SAMPLED_ATTRIBUTES: tuple[SampledAttribute, ...] = (
    SampledAttribute(
        prepared_file="folk_age_sampling.parquet",
        ladder=AGE_LADDER,
        payload_columns=("age",),
        resolution_column="age_resolution",
    ),
    SampledAttribute(
        prepared_file="folk_marital_sampling.parquet",
        ladder=MARITAL_LADDER,
        payload_columns=("marital_status",),
        resolution_column="marital_resolution",
    ),
    SampledAttribute(
        prepared_file="ras202_sampling.parquet",
        ladder=DETAIL_LADDER,
        payload_columns=("detailed_status_code", "detailed_status"),
        resolution_column="detailed_status_resolution",
    ),
)

# The finest level of each ladder, keyed by the column that records it. The
# back-off gate measures against these, and the generation stage withholds the
# columns from prompts, so the level names are written down exactly once.
MOST_SPECIFIC_RESOLUTION: dict[str, str] = {
    attribute.resolution_column: attribute.ladder[0][0]
    for attribute in SAMPLED_ATTRIBUTES
}


def levels(ladder: Ladder) -> tuple[str, ...]:
    """Return every level name of a ladder, most specific first.

    Args:
        ladder:
            Ordered back-off levels.

    Returns:
        The level names in ladder order.

    Examples:
        >>> levels(AGE_LADDER)
        ('age_band_sex', 'age_band')
    """
    return tuple(name for name, _ in ladder)
