"""Context-aware validation for generated Danish persona content."""

import collections.abc as c
import re
import unicodedata

from lingua import Language, LanguageDetectorBuilder
from pydantic import ValidationError
from tldextract import TLDExtract

from ..models import DemographicRecord
from .job_titles import (
    DEFAULT_JOB_TITLE_MAPPING_PATH,
    JobFunctionTitleMapping,
    load_job_title_mapping,
)
from .models import GeneratedAttributes, PersonaDescriptions
from .personality import all_personality_tendencies, allowed_personality_tendencies

VALIDATOR_VERSION = "persona-safety-v9"
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
HEDGE = re.compile(r"(?<![\w])(?:kan|ofte|muligvis|gerne|typisk)(?![\w])")
DETERMINISTIC_CLAIMS = re.compile(r"\b(?:altid|aldrig|helt sikkert|garanteret)\b")

EDUCATION_DANISH = {
    "primary": "grundskole",
    "secondary_or_vocational": "ungdomsuddannelse eller erhvervsuddannelse",
    "higher_education": "videregående uddannelse",
    "not_stated": "uddannelse ikke oplyst",
}
SEX_DANISH = {"male": "mand", "female": "kvinde", "m": "mand", "k": "kvinde"}
STATUS_DANISH = {
    "unemployed": "ledig",
    "student": "studerende",
    "retired": "pensionist",
    "other": "uden for arbejdsmarkedet",
    "outside_labour_force": "uden for arbejdsmarkedet",
    "not_applicable": "uden for arbejdsmarkedet",
}
DETAILED_STATUS_DANISH = {
    "05": "selvstændig",
    "10": "medarbejdende ægtefælle",
    "50": "ledig",
    "85": "ledig",
    "90": "ledig",
    "95": "ledig",
    "130": "studerende",
    "154": "studerende",
    "156": "studerende",
    "158": "studerende",
    "160": "studerende",
    "135": "pensionist",
    "138": "pensionist",
    "139": "pensionist",
    "140": "pensionist",
    "145": "pensionist",
    "150": "pensionist",
    "155": "pensionist",
}


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
        raise ValueError(str(error)) from error
    context = _context_values(demographic)
    _validate_text(text=attributes.cultural_context, require_danish=True)
    _validate_texts(texts=attributes.skills_and_expertise, require_each_danish=False)
    _validate_texts(texts=attributes.hobbies_and_interests, require_each_danish=False)
    if attributes.career_goals_and_ambitions:
        _validate_text(text=attributes.career_goals_and_ambitions, require_danish=True)
    _validate_job_title(
        title=attributes.job_title,
        context=context,
        mapping=job_title_mapping
        or load_job_title_mapping(DEFAULT_JOB_TITLE_MAPPING_PATH),
    )
    return attributes


def _context_values(
    demographic: DemographicRecord | c.Mapping[str, object],
) -> dict[str, object]:
    if isinstance(demographic, DemographicRecord):
        return demographic.model_dump(mode="python")
    return dict(demographic)


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
    _validate_text(text=title, require_danish=True)
    if LIST_FORM.search(title) or "\n" in title or "\r" in title:
        raise ValueError("job_title must be a single plain Danish line")


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


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _require_danish(text: str) -> None:
    if LANGUAGE_DETECTOR.detect_language_of(text) != Language.DANISH:
        raise ValueError("Generated content does not appear to be natural Danish")


def _validate_texts(texts: list[str], require_each_danish: bool) -> None:
    for text in texts:
        _validate_text(text=text, require_danish=require_each_danish)
    if not require_each_danish:
        _require_danish(text=" ".join(texts))


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
        generated = (
            attributes
            if isinstance(attributes, GeneratedAttributes)
            else GeneratedAttributes.model_validate(attributes)
        )
    except ValidationError as error:
        raise ValueError(str(error)) from error
    context = _context_values(demographic)
    for field, text in descriptions.model_dump().items():
        allowed_terms = (
            (ALLOWED_STATUS_TEN_PHRASE,)
            if field == "persona" and str(context.get("detailed_status_code")) == "10"
            else ()
        )
        _validate_text(
            text=text, require_danish=True, allowed_unsupported_terms=allowed_terms
        )
    _validate_persona(text=descriptions.persona, context=context, attributes=generated)
    texts = list(descriptions.model_dump().values())
    normalized = [_normalize(text=text) for text in texts]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Persona descriptions must not be exact duplicates")
    return descriptions


def _validate_persona(
    text: str, context: dict[str, object], attributes: GeneratedAttributes
) -> None:
    normalized = _normalize(text=text)
    if DETERMINISTIC_CLAIMS.search(normalized):
        raise ValueError("Persona must use cautious, non-deterministic language")
    sentences = _persona_sentences(text=text)
    _validate_persona_facts(normalized=normalized, context=context)
    _validate_current_status(
        normalized=normalized, context=context, attributes=attributes
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


def _validate_current_status(
    *, normalized: str, context: dict[str, object], attributes: GeneratedAttributes
) -> None:
    if attributes.job_title is not None:
        required_status = attributes.job_title
    else:
        detailed_code = str(context.get("detailed_status_code", ""))
        required_status = DETAILED_STATUS_DANISH.get(detailed_code)
        if detailed_code not in {"05", "10"}:
            status = str(context.get("labour_market_status")).casefold()
            required_status = (
                "lønmodtager" if status == "employed" else STATUS_DANISH.get(status)
            )
        if required_status is None:
            raise ValueError("Unknown current labour status")
    if not required_status or not _contains_term(normalized, required_status):
        raise ValueError("Persona does not contain its exact current work status")


def _contains_term(text: str, term: str) -> bool:
    """Match a canonical Unicode token sequence, never a substring.

    Returns:
        Whether the complete token sequence occurs at Unicode word boundaries.
    """
    tokens = [token for token in _normalize(term).split(" ") if token]
    if not tokens:
        return False
    expression = r"\s+".join(re.escape(token) for token in tokens)
    return re.search(rf"(?<![\w]){expression}(?![\w])", _normalize(text)) is not None


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


def _validate_persona_facts(*, normalized: str, context: dict[str, object]) -> None:
    required = (
        (str(context.get("age")), "age"),
        (
            SEX_DANISH.get(str(context.get("sex")).casefold(), str(context.get("sex"))),
            "sex",
        ),
        (str(context.get("municipality")), "municipality"),
        (str(context.get("origin_country")), "origin"),
        (_education_label(context), "education"),
    )
    for value, name in required:
        if not value or not _contains_term(normalized, value):
            raise ValueError(f"Persona does not contain the exact {name} fact")
    age = str(context.get("age"))
    if not re.search(rf"(?<!\d){re.escape(age)}\s+år\b", normalized):
        raise ValueError("Persona age must use the fixed '<age> år' form")


def _education_label(context: dict[str, object]) -> str:
    value = str(context.get("education_level", "")).casefold()
    if value not in EDUCATION_DANISH:
        raise ValueError("Unknown education mapping")
    return EDUCATION_DANISH[value]


def _validate_personality(
    *, normalized: str, sentences: list[str], context: dict[str, object]
) -> None:
    compatible = set(allowed_personality_tendencies(context=context))
    mentioned = {
        term
        for term in all_personality_tendencies()
        if _contains_term(normalized, term)
    }
    incompatible = mentioned - compatible
    if incompatible:
        raise ValueError("Persona contains an incompatible personality tendency")
    if not 1 <= len(mentioned) <= 2:
        raise ValueError("Persona must contain 1-2 compatible personality tendencies")
    for sentence in sentences:
        sentence_terms = [term for term in mentioned if _contains_term(sentence, term)]
        for term in sentence_terms:
            term_match = re.search(
                rf"(?<![\w]){re.escape(term)}(?![\w])", _normalize(sentence)
            )
            if term_match is None or not _nearby_hedge(sentence, term_match.start()):
                raise ValueError("Personality tendencies must be hedged nearby")


def _nearby_hedge(sentence: str, position: int) -> bool:
    clause = re.split(r"[,;:]", _normalize(sentence))
    offset = 0
    for part in clause:
        end = offset + len(part)
        if offset <= position <= end:
            return any(
                abs(match.start() - position) <= 48 for match in HEDGE.finditer(part)
            )
        offset = end + 1
    return False
