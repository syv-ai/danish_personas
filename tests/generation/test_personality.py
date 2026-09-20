"""Tests for the shared OCEAN tendency API."""

import pytest

from danish_personas.generation.personality import allowed_personality_tendencies


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
                "nysgerrig",
                "kreativ",
                "åben for nye ideer",
                "struktureret",
                "omhyggelig",
                "planlagt",
                "social",
                "udadvendt",
                "snakkesalig",
                "samarbejdende",
                "hensynsfuld",
                "venlig",
                "opmærksom",
                "varsom",
                "følsom",
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
                "praktisk",
                "jordnær",
                "glad for det velkendte",
                "fleksibel",
                "spontan",
                "rolig",
                "eftertænksom",
                "reserveret",
                "selvstændig",
                "direkte",
                "afbalanceret",
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
                "nysgerrig",
                "kreativ",
                "åben for nye ideer",
                "praktisk",
                "jordnær",
                "glad for det velkendte",
                "struktureret",
                "omhyggelig",
                "planlagt",
                "fleksibel",
                "spontan",
                "social",
                "udadvendt",
                "snakkesalig",
                "rolig",
                "eftertænksom",
                "reserveret",
                "samarbejdende",
                "hensynsfuld",
                "venlig",
                "selvstændig",
                "direkte",
                "opmærksom",
                "varsom",
                "følsom",
                "afbalanceret",
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


def test_scores_select_levels_when_labels_are_not_informative() -> None:
    """Scores provide the same level selection as labels."""
    context = {
        "openness_label": "average",
        "openness_score": 80.0,
        "conscientiousness_label": "average",
        "conscientiousness_score": 20.0,
    }

    terms = allowed_personality_tendencies(context=context)

    assert terms[:3] == ("nysgerrig", "kreativ", "åben for nye ideer")
    assert terms[3:5] == ("fleksibel", "spontan")
