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
                "har ofte tendens til at være nysgerrig",
                "har ofte tendens til at være kreativ",
                "har ofte tendens til at være åben for nye ideer",
                "har ofte tendens til at være struktureret",
                "har ofte tendens til at være omhyggelig",
                "har ofte tendens til at være målrettet",
                "har ofte tendens til at være social",
                "har ofte tendens til at være udadvendt",
                "har ofte tendens til at være snakkesalig",
                "har ofte tendens til at være samarbejdende",
                "har ofte tendens til at være hensynsfuld",
                "har ofte tendens til at være venlig",
                "har ofte tendens til at være opmærksom",
                "har ofte tendens til at være varsom",
                "har ofte tendens til at være følsom",
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
                "har ofte tendens til at være praktisk",
                "har ofte tendens til at være jordnær",
                "har ofte tendens til at være glad for det velkendte",
                "har ofte tendens til at være fleksibel",
                "har ofte tendens til at være spontan",
                "har ofte tendens til at være rolig",
                "har ofte tendens til at være eftertænksom",
                "har ofte tendens til at være reserveret",
                "har ofte tendens til at være ligefrem",
                "har ofte tendens til at være direkte",
                "har ofte tendens til at være afbalanceret",
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
                "har ofte tendens til at være nysgerrig",
                "har ofte tendens til at være kreativ",
                "har ofte tendens til at være åben for nye ideer",
                "har ofte tendens til at være praktisk",
                "har ofte tendens til at være jordnær",
                "har ofte tendens til at være glad for det velkendte",
                "har ofte tendens til at være struktureret",
                "har ofte tendens til at være omhyggelig",
                "har ofte tendens til at være målrettet",
                "har ofte tendens til at være fleksibel",
                "har ofte tendens til at være spontan",
                "har ofte tendens til at være social",
                "har ofte tendens til at være udadvendt",
                "har ofte tendens til at være snakkesalig",
                "har ofte tendens til at være rolig",
                "har ofte tendens til at være eftertænksom",
                "har ofte tendens til at være reserveret",
                "har ofte tendens til at være samarbejdende",
                "har ofte tendens til at være hensynsfuld",
                "har ofte tendens til at være venlig",
                "har ofte tendens til at være ligefrem",
                "har ofte tendens til at være direkte",
                "har ofte tendens til at være opmærksom",
                "har ofte tendens til at være varsom",
                "har ofte tendens til at være følsom",
                "har ofte tendens til at være afbalanceret",
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
    assert all(
        phrase == f"har ofte tendens til at være {term}"
        for term, phrase in zip(terms, phrases)
    )
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
        "har ofte tendens til at være nysgerrig",
        "har ofte tendens til at være kreativ",
        "har ofte tendens til at være åben for nye ideer",
    )
    assert terms[3:5] == (
        "har ofte tendens til at være fleksibel",
        "har ofte tendens til at være spontan",
    )
