"""Deterministic validation for generated Danish persona content."""

import re
import unicodedata

from lingua import Language, LanguageDetectorBuilder
from pydantic import ValidationError
from tldextract import TLDExtract

from .models import GeneratedAttributes, PersonaDescriptions

VALIDATOR_VERSION = "persona-safety-v4"
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
# These are complete words or phrases rather than stems. Word boundaries are
# applied below so permitted compounds such as ``lavendelfarvet`` remain valid.
VISUAL_PROHIBITED_TERMS = {
    # Demographic and socioeconomic attributes.
    "alder",
    "år",
    "årig",
    "barn",
    "børn",
    "civilstand",
    "dreng",
    "drenge",
    "enkemand",
    "enke",
    "gift",
    "gammel",
    "gamle",
    "gymnasium",
    "hovedstaden",
    "institution",
    "kvinde",
    "kvinder",
    "køn",
    "mand",
    "mænd",
    "pensionist",
    "pige",
    "piger",
    "region",
    "landsdel",
    "national",
    "senior",
    "single",
    "skilt",
    "studerende",
    "teenager",
    "uddannelse",
    "uddannet",
    "uddannelsesniveau",
    "ugift",
    "voksen",
    "voksne",
    "universitet",
    "yngre",
    "ældre",
    # Labour-market status and employment.
    "arbejde",
    "arbejder",
    "arbejdsgiver",
    "arbejdsplads",
    "arbejdsløs",
    "arbejdsmarkedsstatus",
    "beskæftigelse",
    "deltid",
    "firma",
    "fuldtid",
    "job",
    "karriere",
    "løn",
    # Region and place, including origin and nationality.
    "aalborg",
    "afrikansk",
    "afstamning",
    "arabisk",
    "asiatisk",
    "bornholm",
    "danmark",
    "dansk",
    "engelsk",
    "etnicitet",
    "etnisk",
    "europæisk",
    "fyn",
    "forfædre",
    "fransk",
    "herkomst",
    "japansk",
    "jylland",
    "kristen",
    "kultur",
    "københavn",
    "latinamerikansk",
    "mellemøstlig",
    "modersmål",
    "muslim",
    "nationalitet",
    "nordisk",
    "nordjylland",
    "norsk",
    "nørrebro",
    "odense",
    "oprindelse",
    "polsk",
    "politisk",
    "politik",
    "religion",
    "roma",
    "russisk",
    "sjælland",
    "skandinavisk",
    "skole",
    "somalisk",
    "sprog",
    "svensk",
    "svensker",
    "tysk",
    "vesterbro",
    # Physical, immutable, sensitive, and identifying attributes.
    "ansigt",
    "attraktiv",
    "blind",
    "blond",
    "blondt",
    "flot",
    "gade",
    "hår",
    "håret",
    "hårfarve",
    "handicap",
    "helbred",
    "hospital",
    "hud",
    "hudfarve",
    "hudtone",
    "højde",
    "højden",
    "kropsbygning",
    "kropsmål",
    "kørestol",
    "muskuløs",
    "protese",
    "proteser",
    "seksualitet",
    "sex",
    "sexet",
    "slank",
    "smuk",
    "sød",
    "synshandicap",
    "skøn",
    "tynd",
    "tiltrækningskraft",
    "udseende",
    "universitet",
    "virksomhed",
    "vægt",
    "øje",
    "øjenfarve",
    "øjne",
    "skaldet",
    "skæg",
    "rynker",
    # OCEAN and other generated/personality attributes.
    "åbenhed",
    "ekstrovert",
    "ekstroversion",
    "introvert",
    "neuroticisme",
    "neurotisk",
    "samvittighedsfuld",
    "udadvendt",
}
VISUAL_PROHIBITED_PATTERNS = (
    # Height is only prohibited when it is clearly a physical description; the
    # standalone words occur naturally in safe colour and environment phrases.
    re.compile(r"\b(?:lav|høj)\s+(?:af\s+)?statur\b"),
    re.compile(r"\b(?:lav|høj)\s+(?:person|kvinde|mand|voksen)\b"),
)
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
    """Reject visual claims that could encode sensitive or identifying traits.

    Args:
        text:
            Validated Danish visual guidance.

    Raises:
        ValueError:
            If the guidance contains a prohibited visual claim or sentence count.
    """
    normalized = _normalize(text=text)
    found_sensitive = sorted(
        term
        for term in VISUAL_PROHIBITED_TERMS
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized)
    )
    found_sensitive.extend(
        pattern.pattern
        for pattern in VISUAL_PROHIBITED_PATTERNS
        if pattern.search(normalized)
    )
    if found_sensitive:
        message = f"Visual persona contains prohibited terms: {found_sensitive}"
        raise ValueError(message)
    sentences = [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]
    if not 2 <= len(sentences) <= 4:
        message = "Visual persona must contain 2-4 sentences"
        raise ValueError(message)
