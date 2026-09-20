"""Context-aware validation for generated Danish persona content."""

import collections.abc as c
import re
import unicodedata

from lingua import Language, LanguageDetectorBuilder
from pydantic import ValidationError
from tldextract import TLDExtract

from ..models import DemographicRecord
from .grounding import EDUCATION_DANISH, build_persona_grounding_facts
from .job_titles import (
    DEFAULT_JOB_TITLE_MAPPING_PATH,
    JobFunctionTitleMapping,
    load_job_title_mapping,
)
from .models import GeneratedAttributes, PersonaDescriptions
from .personality import (
    all_personality_phrases,
    all_personality_tendencies,
    allowed_personality_tendencies,
)

VALIDATOR_VERSION = "persona-safety-v14"
__all__ = ["EDUCATION_DANISH"]
_ATTRIBUTE_FIELDS = frozenset(
    {
        "cultural_context",
        "skills_and_expertise",
        "hobbies_and_interests",
        "career_goals_and_ambitions",
        "job_title",
    }
)
_DESCRIPTION_FIELDS = frozenset(
    {
        "professional_persona",
        "sports_persona",
        "arts_persona",
        "travel_persona",
        "culinary_persona",
        "persona",
    }
)
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
SENSITIVE_PATTERNS = (
    r"adhd|angst|autisme|bipolar(?:itet)?|blind(?:e|hed)?|depress(?:ion|iv)\w*",
    r"diabet(?:es|iker)\w*|døv(?:hed|e)?|etnicitet|handicap(?:ped)?\w*",
    r"helbred|heteroseksuel\w*|homoseksuel\w*|kræft|kristen\w*",
    r"muslim\w*|politisk\s+parti|religion\w*|seksualitet|skizofreni",
    r"stemme\s+på|sygdom\w*|transkønnet\w*",
)
UNSUPPORTED_PATTERNS = (
    r"familie(?:n|r|rne|s)?|ægtefælle(?:n|r|rne|s)?|partner(?:en|e|ne|s)?",
    r"barn(?:et|ene|enes|s)?|børn(?:et|ene|enes|s)?|"
    r"forældre(?:ne|s)?|søskende(?:ne|s)?|husstand(?:en|e|ene|s)?|bor\s+sammen",
    r"diagnos(?:e|er|en|erede)\w*|hår(?:et|ene|enes)?|"
    r"øjne?\w*",
    r"ansigt(?:et|er|ene|enes|stræk(?:ket|kene)?)?|"
    r"højde|vægt|krop(?:pen)?|udseende|ser\s+ud",
    r"hud(?:en|ens|farve|farven|farves)?|kropsbygning",
)
ALLOWED_STATUS_TEN_PHRASE = "medarbejdende ægtefælle"
FORMER_WORK = re.compile(
    r"(?<![\w])(?:tidligere|førhen|før|arbejdede|har\s+arbejdet|"
    r"var\s+ansat|forhenværende|pensioneret\s+fra)(?![\w])"
)
LIST_FORM = re.compile(r"(?:^|\s)(?:[-*•]|\d+[.)])\s|[\[\]{};]", re.MULTILINE)
DETERMINISTIC_CLAIMS = re.compile(r"\b(?:altid|aldrig|helt sikkert|garanteret)\b")


def parse_attributes(
    content: str,
    demographic: DemographicRecord | c.Mapping[str, object],
    *,
    job_title_mapping: JobFunctionTitleMapping | None = None,
) -> GeneratedAttributes:
    """Parse first-stage output against its demographic input.

    Returns:
        Validated generated attributes.

    Raises:
        ValueError:
            If the JSON, safety rules, or title eligibility is invalid.
    """
    try:
        attributes = GeneratedAttributes.model_validate_json(content)
    except ValidationError as error:
        raise ValueError(
            _format_schema_error(error=error, fields=_ATTRIBUTE_FIELDS)
        ) from error
    context = _context_values(demographic)
    _validate_field(
        field="cultural_context",
        validator=lambda: _validate_text(
            text=attributes.cultural_context, require_danish=True
        ),
    )
    _validate_field(
        field="skills_and_expertise",
        validator=lambda: _validate_texts(
            texts=attributes.skills_and_expertise, require_each_danish=False
        ),
    )
    _validate_field(
        field="hobbies_and_interests",
        validator=lambda: _validate_hobbies_and_interests(
            texts=attributes.hobbies_and_interests
        ),
    )
    career_goals = attributes.career_goals_and_ambitions
    if career_goals:
        _validate_field(
            field="career_goals_and_ambitions",
            validator=lambda: _validate_text(text=career_goals, require_danish=True),
        )
    _validate_field(
        field="job_title",
        validator=lambda: _validate_job_title(
            title=attributes.job_title,
            context=context,
            mapping=job_title_mapping
            or load_job_title_mapping(DEFAULT_JOB_TITLE_MAPPING_PATH),
        ),
    )
    return attributes


def _context_values(
    demographic: DemographicRecord | c.Mapping[str, object],
) -> dict[str, object]:
    if isinstance(demographic, DemographicRecord):
        return demographic.model_dump(mode="python")
    return dict(demographic)


def _format_schema_error(*, error: ValidationError, fields: frozenset[str]) -> str:
    """Format Pydantic errors without exposing generated input values.

    Returns:
        Safe structural error messages.
    """
    messages: list[str] = []
    for issue in error.errors():
        location = _safe_schema_location(location=issue.get("loc", ()), fields=fields)
        message = str(issue.get("msg", "Schema validation failed"))
        messages.append(f"{location}: {message}")
    return "; ".join(messages) or "response: Schema validation failed"


def _safe_schema_location(*, location: object, fields: frozenset[str]) -> str:
    """Keep schema locations while excluding arbitrary input keys.

    Returns:
        A response location containing only known fields and list indexes.
    """
    if not isinstance(location, tuple) or not location:
        return "response"
    field = location[0]
    if not isinstance(field, str) or field not in fields:
        return "response"
    suffix = "".join(
        f"[{part}]" for part in location[1:] if isinstance(part, int) and part >= 0
    )
    return f"{field}{suffix}"


def _validate_field(*, field: str, validator: c.Callable[[], None]) -> None:
    """Attach a generated response field to a validation failure.

    Raises:
        ValueError:
            If the field validator rejects the generated response field.
    """
    try:
        validator()
    except ValueError as error:
        raise ValueError(f"{field}: {error}") from error


def _validate_hobbies_and_interests(*, texts: list[str]) -> None:
    """Require interests to describe activities or topics, not OCEAN language.

    Personality terms are reserved for the grounded persona's separately validated
    tendency phrases. Matching against the shared term and phrase APIs also rejects
    a tendency when it is embedded in a longer interest, while preserving ordinary
    words that merely contain the same letters.

    Raises:
        ValueError:
            If an interest contains a reviewed personality term or phrase.
    """
    _validate_texts(texts=texts, require_each_danish=False)
    personality_terms = (*all_personality_tendencies(), *all_personality_phrases())
    if any(
        _contains_term(text=interest, term=term)
        for interest in texts
        for term in personality_terms
    ):
        raise ValueError("Every interest must be an activity or topic")


def _contains_term(text: str, term: str) -> bool:
    """Match a canonical Unicode token sequence, never a substring.

    Returns:
        Whether the complete token sequence occurs at Unicode word boundaries.
    """
    return bool(_term_spans(text=text, term=term))


def _term_spans(*, text: str, term: str) -> list[tuple[int, int]]:
    """Return token-boundary spans for a canonical phrase in normalised text."""
    tokens = [token for token in _normalize(term).split(" ") if token]
    if not tokens:
        return []
    expression = r"\s+".join(re.escape(token) for token in tokens)
    return [
        match.span()
        for match in re.finditer(rf"(?<![\w]){expression}(?![\w])", _normalize(text))
    ]


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _validate_texts(texts: list[str], require_each_danish: bool) -> None:
    for text in texts:
        _validate_text(text=text, require_danish=require_each_danish)
    if not require_each_danish:
        _require_danish(text=" ".join(texts))


def _require_danish(text: str) -> None:
    if LANGUAGE_DETECTOR.detect_language_of(text) != Language.DANISH:
        raise ValueError("Generated content does not appear to be natural Danish")


def _validate_text(
    text: str, require_danish: bool, allowed_unsupported_terms: tuple[str, ...] = ()
) -> None:
    normalized = _normalize(text=text)
    for term in allowed_unsupported_terms:
        normalized = re.sub(rf"(?<![\w]){re.escape(term)}(?![\w])", " ", normalized)
    patterns = {
        "email": EMAIL,
        "CPR-like number": CPR,
        "foreign-script text": FOREIGN_SCRIPT,
        "phone-like number": PHONE,
        "exact-address pattern": ADDRESS,
    }
    for name, pattern in patterns.items():
        if pattern.search(text):
            raise ValueError(f"Generated content contains a prohibited {name}")
    if _contains_url(text=text):
        raise ValueError("Generated content contains a prohibited URL")
    if FORMER_WORK.search(normalized):
        raise ValueError("Generated content contains former or past-work wording")
    found_sensitive = sorted(
        pattern
        for pattern in SENSITIVE_PATTERNS
        if _contains_pattern(normalized, pattern)
    )
    if found_sensitive:
        raise ValueError(
            f"Generated content contains sensitive terms: {found_sensitive}"
        )
    found_unsupported = sorted(
        pattern
        for pattern in UNSUPPORTED_PATTERNS
        if _contains_pattern(normalized, pattern)
    )
    if found_unsupported:
        raise ValueError(
            f"Generated content contains unsupported claims: {found_unsupported}"
        )
    if require_danish:
        _require_danish(text=text)


def _contains_pattern(text: str, pattern: str) -> bool:
    return re.search(rf"(?<![\w])(?:{pattern})(?![\w])", text) is not None


def _contains_url(text: str) -> bool:
    if EXPLICIT_URL.search(text):
        return True
    for match in DOMAIN_CANDIDATE.finditer(text):
        host = match.group().split("/", maxsplit=1)[0]
        extracted = TLD_EXTRACTOR(host)
        if extracted.domain and extracted.suffix:
            return True
    return False


def _validate_job_title(
    *, title: str | None, context: dict[str, object], mapping: JobFunctionTitleMapping
) -> None:
    resolution = context.get("job_function_resolution")
    if resolution not in {"lons20_sex_marginal", "not_applicable"}:
        raise ValueError("Unknown job-function resolution")
    eligible = resolution == "lons20_sex_marginal"
    if eligible != (title is not None):
        expected = "a title" if eligible else "null job_title"
        raise ValueError(f"Eligible job-function context requires {expected}")
    if title is None:
        return
    code = str(context.get("job_function_code", ""))
    entry = mapping.job_functions.get(code)
    if entry is None:
        label = str(context.get("job_function", "")).strip()
        entry = next(
            (item for item in mapping.job_functions.values() if item.label == label),
            None,
        )
    if entry is None or title not in entry.titles:
        raise ValueError("job_title must equal an allowlisted reviewed title")
    _validate_text(text=title, require_danish=False)
    if LIST_FORM.search(title) or "\n" in title or "\r" in title:
        raise ValueError("job_title must be a single plain Danish line")


def parse_descriptions(
    content: str,
    demographic: DemographicRecord | c.Mapping[str, object],
    attributes: GeneratedAttributes | c.Mapping[str, object],
) -> PersonaDescriptions:
    """Parse second-stage output against demographics and stage-one attributes.

    Returns:
        Validated persona descriptions.

    Raises:
        ValueError:
            If the JSON, factual grounding, prose, or safety rules are invalid.
    """
    try:
        descriptions = PersonaDescriptions.model_validate_json(content)
    except ValidationError as error:
        raise ValueError(
            _format_schema_error(error=error, fields=_DESCRIPTION_FIELDS)
        ) from error
    try:
        generated = (
            attributes
            if isinstance(attributes, GeneratedAttributes)
            else GeneratedAttributes.model_validate(attributes)
        )
    except ValidationError as error:
        raise ValueError(
            _format_schema_error(error=error, fields=_ATTRIBUTE_FIELDS)
        ) from error
    context = _context_values(demographic)
    for field, text in descriptions.model_dump().items():
        allowed_terms = (
            (ALLOWED_STATUS_TEN_PHRASE,)
            if field == "persona" and str(context.get("detailed_status_code")) == "10"
            else ()
        )
        _validate_field(
            field=field,
            validator=lambda text=text, allowed_terms=allowed_terms: _validate_text(
                text=text, require_danish=True, allowed_unsupported_terms=allowed_terms
            ),
        )
    _validate_field(
        field="persona",
        validator=lambda: _validate_persona(
            text=descriptions.persona, context=context, attributes=generated
        ),
    )
    texts = list(descriptions.model_dump().values())
    normalized = [_normalize(text=text) for text in texts]
    if len(set(normalized)) != len(normalized):
        raise ValueError("persona: Persona descriptions must not be exact duplicates")
    return descriptions


def _validate_persona(
    text: str, context: dict[str, object], attributes: GeneratedAttributes
) -> None:
    normalized = _normalize(text=text)
    if DETERMINISTIC_CLAIMS.search(normalized):
        raise ValueError("Persona must use cautious, non-deterministic language")
    sentences = _persona_sentences(text=text)
    _validate_persona_facts(
        normalized=normalized, demographic=context, attributes=attributes
    )
    _validate_interests(normalized=normalized, attributes=attributes)
    _validate_personality(normalized=normalized, sentences=sentences, context=context)
    if FORMER_WORK.search(normalized):
        raise ValueError("Persona must not contain former or past-work claims")


def _persona_sentences(text: str) -> list[str]:
    if LIST_FORM.search(text):
        raise ValueError("Persona must be prose, not a list")
    sentences = [
        part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()
    ]
    if not 2 <= len(sentences) <= 4 or not text.rstrip().endswith((".", "!", "?")):
        raise ValueError("persona must contain 2-4 prose sentences")
    return sentences


def _validate_interests(*, normalized: str, attributes: GeneratedAttributes) -> None:
    """Require two or three complete, literal interests in the summary prose.

    Matching complete terms prevents a short interest such as ``art`` from being
    accepted merely because it occurs inside an unrelated word.  The generated
    value is normalised in the same way as the prose so capitalisation and
    incidental whitespace do not change the contract.

    Raises:
        ValueError:
            If the prose contains fewer than two or more than three interests.
    """
    interests = {
        _normalize(interest)
        for interest in attributes.hobbies_and_interests
        if _contains_term(normalized, _normalize(interest))
    }
    if len(interests) not in {2, 3}:
        raise ValueError("Persona must contain exactly 2-3 generated interests")


def _validate_persona_facts(
    *, normalized: str, demographic: dict[str, object], attributes: GeneratedAttributes
) -> None:
    facts = build_persona_grounding_facts(
        demographic=demographic, attributes=attributes
    )
    names = {
        "age": "age",
        "sex": "sex",
        "municipality": "municipality",
        "education_level": "education",
        "origin_country_da": "origin",
        "current_employment": "current work status",
    }
    for field, value in facts.model_dump().items():
        if not _contains_term(normalized, value):
            raise ValueError(f"Persona does not contain the exact {names[field]} fact")


def _validate_personality(
    *, normalized: str, sentences: list[str], context: dict[str, object]
) -> None:
    """Require one or two compatible, complete phrases copied from the API.

    ``sentences`` remains part of the signature because the persona prose contract
    validates sentence structure before this check. Personality matching itself is
    deliberately whole-persona based: a phrase may occur in any one sentence, but a
    bare term elsewhere must not satisfy the contract.

    Raises:
        ValueError:
            If the persona contains an incompatible or incomplete phrase set.
    """
    del sentences
    compatible = set(allowed_personality_tendencies(context=context))
    all_phrases = set(all_personality_phrases())
    incompatible = {
        phrase
        for phrase in all_phrases - compatible
        if _contains_term(normalized, phrase)
    }
    if incompatible:
        raise ValueError("Persona contains an incompatible personality tendency")

    matched_phrases = {
        phrase for phrase in compatible if _contains_term(normalized, phrase)
    }
    phrase_spans = [
        span
        for phrase in matched_phrases
        for span in _term_spans(text=normalized, term=phrase)
    ]
    if not 1 <= len(matched_phrases) <= 2:
        raise ValueError("Persona must contain 1-2 compatible personality tendencies")

    for term in all_personality_tendencies():
        for start, end in _term_spans(text=normalized, term=term):
            if not any(
                phrase_start <= start and end <= phrase_end
                for phrase_start, phrase_end in phrase_spans
            ):
                raise ValueError(
                    "Personality terms must occur inside supplied complete phrases"
                )
