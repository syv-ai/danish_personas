"""Deterministic validation for generated Danish persona content."""

import re
import unicodedata

from lingua import Language, LanguageDetectorBuilder
from pydantic import ValidationError
from tldextract import TLDExtract

from .models import GeneratedAttributes, PersonaDescriptions

VALIDATOR_VERSION = "persona-safety-v6"
EMAIL = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b", re.IGNORECASE)
_DOMAIN_LABEL = r"[a-z0-9æøå](?:[a-z0-9æøå-]{0,61}[a-z0-9æøå])?"
EXPLICIT_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
DOMAIN_CANDIDATE = re.compile(
    rf"\b(?:{_DOMAIN_LABEL}\.)+[a-z0-9-]{{2,63}}\b(?:/\S*)?", re.IGNORECASE
)
TLD_EXTRACTOR = TLDExtract(suffix_list_urls=())
CPR = re.compile(r"\b\d{6}[- ]?\d{4}\b")
FOREIGN_SCRIPT = re.compile(r"[\u0400-\u04ff\u4e00-\u9fff]")
PHONE = re.compile(r"(?<!\d)(?:\+45[ -]?)?(?:\d[ -]?){8}(?!\d)")
_STREET = r"[\wæøå.-]+(?:gade|vej|allé|alle|boulevard|stræde|vænget|torv)"
ADDRESS = re.compile(
    rf"(?:\b{_STREET}\s+\d{{1,4}}[a-z]?\b|\b\d{{1,4}}\s+{_STREET}\b)", re.IGNORECASE
)
LANGUAGE_DETECTOR = LanguageDetectorBuilder.from_languages(
    Language.BOKMAL,
    Language.DANISH,
    Language.ENGLISH,
    Language.GERMAN,
    Language.NYNORSK,
    Language.SWEDISH,
).build()
# Visual guidance uses a closed vocabulary rather than a blacklist. Colours include
# the common, neuter, and plural forms needed by the sentence grammar.
VISUAL_COLOURS: dict[str, tuple[str, str, str]] = {
    "blå": ("blå", "blåt", "blå"),
    "brun": ("brun", "brunt", "brune"),
    "grå": ("grå", "gråt", "grå"),
    "grøn": ("grøn", "grønt", "grønne"),
    "hvid": ("hvid", "hvidt", "hvide"),
    "lilla": ("lilla", "lilla", "lilla"),
    "orange": ("orange", "orange", "orange"),
    "pink": ("pink", "pink", "pink"),
    "rød": ("rød", "rødt", "røde"),
    "sort": ("sort", "sort", "sorte"),
    "turkis": ("turkis", "turkist", "turkise"),
    "gul": ("gul", "gult", "gule"),
}
VISUAL_CLOTHING: dict[str, tuple[str, ...]] = {
    "en": (
        "bluse",
        "cardigan",
        "frakke",
        "jakke",
        "kjole",
        "nederdel",
        "skjorte",
        "sweater",
        "trøje",
        "vest",
    ),
    "et": ("halstørklæde", "tørklæde"),
}
VISUAL_ACCESSORIES: dict[str, tuple[str, ...]] = {
    "en": ("broche", "halskæde", "hat", "kasket", "paraply", "taske"),
    "et": ("armbånd", "bælte", "sjal", "slips", "tørklæde", "ur"),
}
VISUAL_BACKGROUNDS = (
    "en afdæmpet flade",
    "en enkel flade",
    "en ensfarvet flade",
    "en lys flade",
    "en neutral flade",
    "en rolig flade",
    "et afdæmpet studie",
    "et enkelt studie",
    "et lyst atelier",
    "et roligt atelier",
)
VISUAL_LIGHTING = ("blødt", "klart", "dæmpet", "diffust", "jævnt", "roligt")


def _visual_alternation(values: tuple[str, ...]) -> str:
    """Build an escaped, longest-first regular-expression alternation.

    Args:
        values:
            Allowed phrases to include in the alternation.

    Returns:
        An escaped regular-expression alternation.
    """
    ordered = sorted(values, key=len, reverse=True)
    return "(?:" + "|".join(re.escape(value) for value in ordered) + ")"


def _visual_item_phrases(items: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Return all allowed colour-and-item phrases for a Danish noun group."""
    phrases: list[str] = []
    for article, nouns in items.items():
        form_index = 0 if article == "en" else 1
        for colour_forms in VISUAL_COLOURS.values():
            for noun in nouns:
                phrases.append(f"{article} {colour_forms[form_index]} {noun}")
    return tuple(phrases)


_VISUAL_CLOTHING_PHRASES = _visual_item_phrases(items=VISUAL_CLOTHING)
_VISUAL_ACCESSORY_PHRASES = _visual_item_phrases(items=VISUAL_ACCESSORIES)
_VISUAL_FIRST_SENTENCE = re.compile(
    rf"personen vælger {_visual_alternation(_VISUAL_CLOTHING_PHRASES)} og "
    rf"{_visual_alternation(_VISUAL_ACCESSORY_PHRASES)}"
)
_VISUAL_BACKGROUND_SENTENCE = re.compile(
    rf"baggrunden er {_visual_alternation(VISUAL_BACKGROUNDS)}"
)
_VISUAL_LIGHTING_SENTENCE = re.compile(
    rf"lyset er {_visual_alternation(VISUAL_LIGHTING)}"
)
_VISUAL_FRAME_SENTENCE = re.compile(r"rammen er neutral")

SENSITIVE_TERMS = {
    "adhd",
    "angst",
    "autisme",
    "bipolar",
    "blind",
    "depression",
    "diabetes",
    "døv",
    "etnicitet",
    "handicap",
    "helbred",
    "heteroseksuel",
    "homoseksuel",
    "kræft",
    "kristen",
    "muslim",
    "politisk parti",
    "religion",
    "seksualitet",
    "skizofreni",
    "stemme på",
    "sygdom",
    "transkønnet",
}


def parse_attributes(content: str) -> GeneratedAttributes:
    """Parse and validate first-stage generated attributes.

    Args:
        content:
            JSON response text.

    Returns:
        Validated attributes.

    Raises:
        ValueError:
            If JSON, language, or safety validation fails.
    """
    try:
        attributes = GeneratedAttributes.model_validate_json(content)
    except ValidationError as error:
        raise ValueError(str(error)) from error
    _validate_text(text=attributes.cultural_context, require_danish=True)
    _validate_texts(texts=attributes.skills_and_expertise, require_each_danish=False)
    _validate_texts(texts=attributes.hobbies_and_interests, require_each_danish=False)
    if attributes.career_goals_and_ambitions:
        _validate_text(text=attributes.career_goals_and_ambitions, require_danish=True)
    return attributes


def _validate_text(text: str, require_danish: bool) -> None:
    normalized = _normalize(text=text)
    patterns = {
        "email": EMAIL,
        "CPR-like number": CPR,
        "foreign-script text": FOREIGN_SCRIPT,
        "phone-like number": PHONE,
        "exact-address pattern": ADDRESS,
    }
    for name, pattern in patterns.items():
        if pattern.search(text):
            message = f"Generated content contains a prohibited {name}"
            raise ValueError(message)
    if _contains_url(text=text):
        message = "Generated content contains a prohibited URL"
        raise ValueError(message)
    found_sensitive = sorted(term for term in SENSITIVE_TERMS if term in normalized)
    if found_sensitive:
        message = f"Generated content contains sensitive terms: {found_sensitive}"
        raise ValueError(message)
    if require_danish:
        _require_danish(text=text)


def _contains_url(text: str) -> bool:
    if EXPLICIT_URL.search(text):
        return True
    for match in DOMAIN_CANDIDATE.finditer(text):
        host = match.group().split("/", maxsplit=1)[0]
        extracted = TLD_EXTRACTOR(host)
        if extracted.domain and extracted.suffix:
            return True
    return False


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _require_danish(text: str) -> None:
    if LANGUAGE_DETECTOR.detect_language_of(text) != Language.DANISH:
        message = "Generated content does not appear to be natural Danish"
        raise ValueError(message)


def _validate_texts(texts: list[str], require_each_danish: bool) -> None:
    for text in texts:
        _validate_text(text=text, require_danish=require_each_danish)
    if not require_each_danish:
        _require_danish(text=" ".join(texts))


def parse_descriptions(content: str) -> PersonaDescriptions:
    """Parse and validate second-stage persona descriptions.

    Args:
        content:
            JSON response text.

    Returns:
        Validated descriptions.

    Raises:
        ValueError:
            If JSON, language, safety, or duplication validation fails.
    """
    try:
        descriptions = PersonaDescriptions.model_validate_json(content)
    except ValidationError as error:
        raise ValueError(str(error)) from error
    texts = list(descriptions.model_dump().values())
    _validate_texts(texts=texts, require_each_danish=True)
    _validate_visual_persona(text=descriptions.visual_persona)
    normalized = [_normalize(text=text) for text in texts]
    if len(set(normalized)) != len(normalized):
        message = "Persona descriptions must not be exact duplicates"
        raise ValueError(message)
    return descriptions


def _validate_visual_persona(text: str) -> None:
    """Require the closed Danish visual-persona grammar.

    Args:
        text:
            Validated Danish visual guidance.

    Raises:
        ValueError:
            If the guidance does not use 2-4 controlled sentences.
    """
    normalized = _normalize(text=text)
    sentences = [part.strip() for part in normalized.split(".") if part.strip()]
    if not 2 <= len(sentences) <= 4:
        message = "Visual persona must contain 2-4 sentences"
        raise ValueError(message)

    if (
        not normalized.endswith(".")
        or any(mark in normalized for mark in "!?")
        or ".." in normalized
    ):
        message = "Visual persona must use the controlled Danish format and vocabulary"
        raise ValueError(message)

    patterns = [_VISUAL_FIRST_SENTENCE, _VISUAL_BACKGROUND_SENTENCE]
    if len(sentences) >= 3:
        patterns.append(_VISUAL_LIGHTING_SENTENCE)
    if len(sentences) == 4:
        patterns.append(_VISUAL_FRAME_SENTENCE)
    if any(
        not pattern.fullmatch(sentence)
        for pattern, sentence in zip(patterns, sentences)
    ):
        message = "Visual persona must use the controlled Danish format and vocabulary"
        raise ValueError(message)
