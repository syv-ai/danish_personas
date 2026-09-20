"""OCEAN personality terminology shared by generation stages."""

import collections.abc as c

OCEAN_TERMS: dict[str, dict[str, tuple[str, ...]]] = {
    "openness": {
        "high": ("nysgerrig", "kreativ", "åben for nye ideer"),
        "low": ("praktisk", "jordnær", "glad for det velkendte"),
    },
    "conscientiousness": {
        "high": ("struktureret", "omhyggelig", "målrettet"),
        "low": ("fleksibel", "spontan"),
    },
    "extraversion": {
        "high": ("social", "udadvendt", "snakkesalig"),
        "low": ("rolig", "eftertænksom", "reserveret"),
    },
    "agreeableness": {
        "high": ("samarbejdende", "hensynsfuld", "venlig"),
        "low": ("ligefrem", "direkte"),
    },
    "neuroticism": {
        "high": ("opmærksom", "varsom", "følsom"),
        "low": ("rolig", "afbalanceret"),
    },
}

# These are deliberately closed phrases rather than terms that a caller can combine
# with an arbitrary hedge.  The wording is shared by the prompt and the validator.
PERSONALITY_PHRASES = {
    term: f"kan være {term}"
    for levels in OCEAN_TERMS.values()
    for terms in levels.values()
    for term in terms
}


def all_personality_phrases() -> tuple[str, ...]:
    """Return every distinct complete phrase in lexicon order."""
    return tuple(PERSONALITY_PHRASES[term] for term in all_personality_tendencies())


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
    """Return deduplicated complete phrases compatible with one row's OCEAN labels.

    Average scores intentionally allow both the high and low term sets. Phrases are
    returned in lexicon order with duplicates removed, so overlapping terms such as
    ``rolig`` are supplied exactly once.

    Args:
        context:
            Row-like mapping containing the OCEAN ``*_label`` and ``*_score``
            fields.

    Returns:
        The complete Danish phrases that may be copied for the row.
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
        terms.extend(
            PERSONALITY_PHRASES[term]
            for selected in levels
            for term in labels[selected]
        )
    return tuple(dict.fromkeys(terms))
