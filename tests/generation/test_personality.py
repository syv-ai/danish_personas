"""Tests for the shared OCEAN tendency API."""

import pytest

from danish_personas.generation.personality import (
    all_personality_phrases,
    all_personality_tendencies,
    allowed_personality_tendencies,
)


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (
            {
                trait: "high"
                for trait in (
                    "openness",
                    "conscientiousness",
                    "extraversion",
                    "agreeableness",
                    "neuroticism",
                )
            },
            (
                "kan være nysgerrig",
                "kan være kreativ",
                "kan være åben for nye ideer",
                "kan være struktureret",
                "kan være omhyggelig",
                "kan være målrettet",
                "kan være social",
                "kan være udadvendt",
                "kan være snakkesalig",
                "kan være samarbejdende",
                "kan være hensynsfuld",
                "kan være venlig",
                "kan være opmærksom",
                "kan være varsom",
                "kan være følsom",
            ),
        ),
        (
            {
                trait: "low"
                for trait in (
                    "openness",
                    "conscientiousness",
                    "extraversion",
                    "agreeableness",
                    "neuroticism",
                )
            },
            (
                "kan være praktisk",
                "kan være jordnær",
                "kan være glad for det velkendte",
                "kan være fleksibel",
                "kan være spontan",
                "kan være rolig",
                "kan være eftertænksom",
                "kan være reserveret",
                "kan være ligefrem",
                "kan være direkte",
                "kan være afbalanceret",
            ),
        ),
        (
            {
                trait: "average"
                for trait in (
                    "openness",
                    "conscientiousness",
                    "extraversion",
                    "agreeableness",
                    "neuroticism",
                )
            },
            (
                "kan være nysgerrig",
                "kan være kreativ",
                "kan være åben for nye ideer",
                "kan være praktisk",
                "kan være jordnær",
                "kan være glad for det velkendte",
                "kan være struktureret",
                "kan være omhyggelig",
                "kan være målrettet",
                "kan være fleksibel",
                "kan være spontan",
                "kan være social",
                "kan være udadvendt",
                "kan være snakkesalig",
                "kan være rolig",
                "kan være eftertænksom",
                "kan være reserveret",
                "kan være samarbejdende",
                "kan være hensynsfuld",
                "kan være venlig",
                "kan være ligefrem",
                "kan være direkte",
                "kan være opmærksom",
                "kan være varsom",
                "kan være følsom",
                "kan være afbalanceret",
            ),
        ),
    ],
)
def test_allowed_personality_tendencies_are_exact_and_deduplicated(
    labels: dict[str, str], expected: tuple[str, ...]
) -> None:
    """High, low, and average levels preserve order and remove overlap."""
    context = {
        **{f"{trait}_label": label for trait, label in labels.items()},
        **{f"{trait}_score": 50.0 for trait in labels},
    }

    assert allowed_personality_tendencies(context=context) == expected
    assert len(expected) == len(set(expected))


def test_every_personality_tendency_has_one_deduplicated_phrase() -> None:
    """The public lexicon maps every term to one complete hedge phrase."""
    terms = all_personality_tendencies()
    phrases = all_personality_phrases()

    assert len(terms) == len(phrases) == len(set(phrases))
    assert all(phrase == f"kan være {term}" for term, phrase in zip(terms, phrases))
    assert "selvstændig" not in terms


def test_scores_select_levels_when_labels_are_not_informative() -> None:
    """Scores provide the same level selection as labels."""
    context = {
        "openness_label": "average",
        "openness_score": 80.0,
        "conscientiousness_label": "average",
        "conscientiousness_score": 20.0,
    }

    terms = allowed_personality_tendencies(context=context)

    assert terms[:3] == (
        "kan være nysgerrig",
        "kan være kreativ",
        "kan være åben for nye ideer",
    )
    assert terms[3:5] == ("kan være fleksibel", "kan være spontan")
