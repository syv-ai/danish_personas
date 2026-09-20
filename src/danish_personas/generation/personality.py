"""OCEAN personality terminology shared by generation stages."""

import collections.abc as c

OCEAN_TERMS: dict[str, dict[str, tuple[str, ...]]] = {
    "openness": {
        "high": ("nysgerrig", "kreativ", "åben for nye ideer"),
        "low": ("praktisk", "jordnær", "glad for det velkendte"),
    },
    "conscientiousness": {
        "high": ("struktureret", "omhyggelig", "planlagt"),
        "low": ("fleksibel", "spontan"),
    },
    "extraversion": {
        "high": ("social", "udadvendt", "snakkesalig"),
        "low": ("rolig", "eftertænksom", "reserveret"),
    },
    "agreeableness": {
        "high": ("samarbejdende", "hensynsfuld", "venlig"),
        "low": ("selvstændig", "direkte"),
    },
    "neuroticism": {
        "high": ("opmærksom", "varsom", "følsom"),
        "low": ("rolig", "afbalanceret"),
    },
}


def all_personality_tendencies() -> tuple[str, ...]:
    """Return every distinct literal term in the reviewed OCEAN lexicon."""
    return tuple(
        dict.fromkeys(
            term
            for levels in OCEAN_TERMS.values()
            for terms in levels.values()
            for term in terms
        )
    )


def allowed_personality_tendencies(
    *, context: c.Mapping[str, object]
) -> tuple[str, ...]:
    """Return terms compatible with one row's OCEAN labels and scores.

    Average scores intentionally allow both the high and low term sets. Terms are
    returned in lexicon order with duplicates removed, so overlapping terms such
    as ``rolig`` are supplied exactly once.

    Args:
        context:
            Row-like mapping containing the OCEAN ``*_label`` and ``*_score``
            fields.

    Returns:
        The deduplicated literal terms that may be used for the row.
    """
    terms: list[str] = []
    for trait, labels in OCEAN_TERMS.items():
        label = str(context.get(f"{trait}_label", "")).casefold()
        raw_score = context.get(f"{trait}_score", 50)
        score = float(raw_score) if isinstance(raw_score, (int, float)) else 50.0
        level = (
            "high"
            if label == "high" or score >= 60
            else "low"
            if label == "low" or score <= 40
            else "average"
        )
        levels = ("high", "low") if level == "average" else (level,)
        for selected in levels:
            terms.extend(labels[selected])
    return tuple(dict.fromkeys(terms))
