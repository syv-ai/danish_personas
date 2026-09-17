"""Tests for deterministic persona-content validation."""

import json

import pytest

from danish_personas.generation.validation import parse_attributes, parse_descriptions


def test_danish_abbreviations_are_not_domains() -> None:
    """Common dotted Danish abbreviations do not trigger the URL detector."""
    payload = {
        "cultural_context": (
            "Personen deltager f.eks. i foreninger med naboer m.fl. i Danmark."
        ),
        "skills_and_expertise": ["planlægning", "samarbejde", "formidling"],
        "hobbies_and_interests": ["at læse", "musik", "brætspil med venner"],
        "career_goals_and_ambitions": None,
    }
    assert parse_attributes(json.dumps(payload, ensure_ascii=False))


@pytest.mark.parametrize(
    "english_text",
    [
        "The person prefers a quiet walk and enjoys time outside with friends.",
        "The person prefers a quiet walk and enjoys music with friends ø.",
    ],
)
def test_english_description_fails_individually(english_text: str) -> None:
    """One English description cannot hide among otherwise Danish fields."""
    descriptions = _safe_descriptions()
    descriptions["sports_persona"] = english_text
    with pytest.raises(ValueError, match="natural Danish"):
        parse_descriptions(json.dumps(descriptions, ensure_ascii=False))


def _safe_descriptions() -> dict[str, str]:
    return {
        "professional_persona": (
            "På arbejdet er personen grundig og har fokus på et godt samarbejde "
            "med andre."
        ),
        "sports_persona": (
            "Motion er en måde at få frisk luft på, og personen går gerne ture "
            "i naturen."
        ),
        "arts_persona": (
            "Personen holder af at læse og er nysgerrig på fortællinger fra "
            "forskellige miljøer."
        ),
        "travel_persona": (
            "Rejser planlægges i god tid, og der er plads til både ro og nye "
            "oplevelser undervejs."
        ),
        "culinary_persona": (
            "I køkkenet prøver personen enkle retter og deler gerne et måltid "
            "med venner."
        ),
        "persona": (
            "Personen trives med en rolig hverdag, men er også åben for at lære "
            "nyt sammen med andre."
        ),
        "visual_persona": (
            "Personen vælger en blå trøje og et grønt tørklæde. "
            "Baggrunden er et roligt atelier."
        ),
    }


def test_english_skills_cannot_hide_behind_danish_hobbies() -> None:
    """Language validation treats each generated list as a separate field."""
    payload = {
        "cultural_context": (
            "Personen bor i Danmark og deltager gerne i lokale fællesskaber."
        ),
        "skills_and_expertise": [
            "project management",
            "written communication",
            "data analysis",
        ],
        "hobbies_and_interests": ["at læse", "gåture i naturen", "brætspil med venner"],
        "career_goals_and_ambitions": None,
    }
    with pytest.raises(ValueError, match="natural Danish"):
        parse_attributes(json.dumps(payload, ensure_ascii=False))


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("Personen kan kontaktes på test@example.dk og bor i Danmark.", "email"),
        (
            "Personen bor på Nørrebrogade 12 og deltager i lokale fællesskaber.",
            "address",
        ),
        ("Personen skriver på eksempel.eu og deltager i lokale fællesskaber.", "URL"),
        ("Personen deler nyt på x.dk og deltager i lokale fællesskaber.", "URL"),
        ("Personen deler nyt på t.co/tekst og deltager i fællesskaber.", "URL"),
        ("Personen deler nyt på eksempel.xn--p1ai og deltager i fællesskaber.", "URL"),
        ("Personen har diabetes og deltager også i lokale fællesskaber.", "sensitive"),
        (
            "Personen genopladеr batterierne og deltager i fællesskaber.",
            "foreign-script",
        ),
    ],
)
def test_prohibited_attribute_content_fails(text: str, error: str) -> None:
    """Identifying and sensitive details are rejected before release."""
    payload = {
        "cultural_context": text,
        "skills_and_expertise": ["planlægning", "samarbejde", "madlavning"],
        "hobbies_and_interests": ["at læse", "gåture", "brætspil"],
        "career_goals_and_ambitions": None,
    }
    with pytest.raises(ValueError, match=error):
        parse_attributes(json.dumps(payload, ensure_ascii=False))


def test_safe_danish_content_passes() -> None:
    """Natural non-identifying Danish attributes and descriptions pass."""
    attributes = {
        "cultural_context": (
            "Personen bor i Danmark og deltager gerne i lokale fællesskaber."
        ),
        "skills_and_expertise": ["praktisk planlægning", "samarbejde", "madlavning"],
        "hobbies_and_interests": ["at læse", "gåture i naturen", "brætspil med venner"],
        "career_goals_and_ambitions": (
            "Vil gerne udvikle sine færdigheder på en rolig måde."
        ),
    }
    assert parse_attributes(json.dumps(attributes, ensure_ascii=False))
    assert parse_descriptions(json.dumps(_safe_descriptions(), ensure_ascii=False))


def test_unsafe_visual_claim_fails() -> None:
    """Visual guidance cannot encode sensitive or identifying traits."""
    descriptions = _safe_descriptions()
    descriptions["visual_persona"] = (
        "Personen har mørk hud og en slank kropsbygning. "
        "Portrættet kan placeres i København."
    )
    with pytest.raises(ValueError, match="Visual persona"):
        parse_descriptions(json.dumps(descriptions, ensure_ascii=False))


def test_visual_persona_accepts_controlled_vocabulary() -> None:
    """Controlled clothing, accessory, colour, and background choices pass."""
    descriptions = _safe_descriptions()
    descriptions["visual_persona"] = (
        "Personen vælger en rød kjole og en sort taske. "
        "Baggrunden er en neutral flade. Lyset er diffust. Rammen er neutral."
    )
    parse_descriptions(json.dumps(descriptions, ensure_ascii=False))


@pytest.mark.parametrize("colour", ["blåt", "turkist"])
def test_visual_persona_accepts_neuter_colour_forms(colour: str) -> None:
    """Neuter clothing and accessories use the correct colour forms."""
    descriptions = _safe_descriptions()
    descriptions["visual_persona"] = (
        f"Personen vælger et {colour} tørklæde og et {colour} armbånd. "
        "Baggrunden er en neutral flade."
    )
    parse_descriptions(json.dumps(descriptions, ensure_ascii=False))


def test_visual_persona_duplicates_are_rejected() -> None:
    """Visual guidance participates in the generic duplicate gate."""
    descriptions = _safe_descriptions()
    descriptions["persona"] = descriptions["visual_persona"]
    with pytest.raises(ValueError, match="exact duplicates"):
        parse_descriptions(json.dumps(descriptions, ensure_ascii=False))


def test_visual_persona_is_required() -> None:
    """The second-stage schema requires visual portrait guidance."""
    descriptions = _safe_descriptions()
    del descriptions["visual_persona"]
    with pytest.raises(ValueError, match="visual_persona"):
        parse_descriptions(json.dumps(descriptions, ensure_ascii=False))


@pytest.mark.parametrize(
    "visual",
    [
        (
            "Personen er 72 år og vælger en blå trøje. "
            "Et lyst atelier giver en neutral portrætramme."
        ),
        (
            "Personen er en ældre kvinde med en blå trøje. "
            "Et lyst atelier giver en neutral portrætramme."
        ),
        (
            "Personen er svensk og taler svensk. "
            "Et lyst atelier giver en neutral portrætramme."
        ),
        (
            "Personen har blå øjne og naturligt blondt hår. "
            "Et lyst atelier giver en neutral portrætramme."
        ),
        (
            "Personen har høj åbenhed og en blå trøje. "
            "Et lyst atelier giver en neutral portrætramme."
        ),
        (
            "Personen er amerikansk og vælger en blå skjorte. "
            "Baggrunden er en neutral flade."
        ),
        (
            "Personen er spansk og vælger en blå skjorte. "
            "Baggrunden er en neutral flade."
        ),
        (
            "Personen er midaldrende og vælger en blå skjorte. "
            "Baggrunden er en neutral flade."
        ),
        (
            "Personen er rødhåret og vælger en blå skjorte. "
            "Baggrunden er en neutral flade."
        ),
        (
            "Personen har fregner og vælger en blå skjorte. "
            "Baggrunden er en neutral flade."
        ),
        (
            "Personen er jødisk og vælger en blå skjorte. "
            "Baggrunden er en neutral flade."
        ),
    ],
)
def test_visual_persona_rejects_demographic_and_physical_claims(visual: str) -> None:
    """Visual guidance cannot repeat demographic or immutable input."""
    descriptions = _safe_descriptions()
    descriptions["visual_persona"] = visual
    with pytest.raises(ValueError, match="Visual persona"):
        parse_descriptions(json.dumps(descriptions, ensure_ascii=False))


def test_visual_persona_rejects_free_form_visual_text() -> None:
    """Words outside the closed visual grammar are rejected."""
    descriptions = _safe_descriptions()
    descriptions["visual_persona"] = (
        "Personen vælger en blå skjorte og et grønt tørklæde. "
        "Et lyst atelier med høj kontrast giver en neutral portrætramme."
    )
    with pytest.raises(ValueError, match="controlled Danish format"):
        parse_descriptions(json.dumps(descriptions, ensure_ascii=False))


@pytest.mark.parametrize("colour", ["blå", "turkis"])
def test_visual_persona_rejects_malformed_neuter_colour_forms(colour: str) -> None:
    """Common-gender colour forms are rejected before neuter nouns."""
    descriptions = _safe_descriptions()
    descriptions["visual_persona"] = (
        f"Personen vælger et {colour} tørklæde og et {colour} armbånd. "
        "Baggrunden er en neutral flade."
    )
    with pytest.raises(ValueError, match="controlled Danish format"):
        parse_descriptions(json.dumps(descriptions, ensure_ascii=False))


@pytest.mark.parametrize(
    "unsafe_word",
    ["amerikansk", "spansk", "midaldrende", "rødhåret", "fregner", "jødisk"],
)
def test_visual_persona_rejects_reviewer_examples(unsafe_word: str) -> None:
    """Reviewer examples cannot be smuggled into a controlled sentence."""
    descriptions = _safe_descriptions()
    descriptions["visual_persona"] = (
        f"Personen vælger en {unsafe_word} skjorte og et grønt armbånd. "
        "Baggrunden er en neutral flade."
    )
    with pytest.raises(ValueError, match="controlled Danish format"):
        parse_descriptions(json.dumps(descriptions, ensure_ascii=False))


def test_visual_persona_sentence_count_is_checked() -> None:
    """Visual guidance has the requested two-to-four sentence form."""
    descriptions = _safe_descriptions()
    descriptions["visual_persona"] = (
        "Personen vælger en blå trøje og et enkelt tørklæde uden at knytte det "
        "til andre egenskaber."
    )
    with pytest.raises(ValueError, match="2-4 sentences"):
        parse_descriptions(json.dumps(descriptions, ensure_ascii=False))
