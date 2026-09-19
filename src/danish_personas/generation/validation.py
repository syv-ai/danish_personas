"""Context-aware validation for generated Danish persona content."""

import collections.abc as c
import re
import unicodedata

from lingua import Language, LanguageDetectorBuilder
from pydantic import ValidationError
from tldextract import TLDExtract

from ..models import DemographicRecord
from .models import GeneratedAttributes, PersonaDescriptions

VALIDATOR_VERSION = "persona-safety-v7"
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
UNSUPPORTED_CLAIMS = (
    "familie",
    "børn",
    "barn",
    "ægtefælle",
    "partner",
    "forældre",
    "søskende",
    "husstand",
    "bor sammen",
    "diagnose",
    "diagnosticeret",
    "hår",
    "øjne",
    "ansigt",
    "højde",
    "vægt",
    "krop",
    "udseende",
    "ser ud",
)
FORMER_WORK = re.compile(
    r"\b(?:tidligere|før|arbejdede|har arbejdet|var ansat|forhenværende)\b"
)
JOB_TITLE_SENIORITY = ("chef", "leder", "direktør", "senior", "junior", "ansvarlig")
JOB_TITLE_DUTY_WORDS = re.compile(
    r"\b(?:ansvar for|arbejder med|udfører|hjælper med|laver)\b"
)
JOB_ROLE_GROUPS = {
    "administr": ("administr", "business", "office", "management"),
    "software": ("software", "program", "developer", "it "),
    "sundhed": ("health", "nurs", "medical", "sygeplej", "læge"),
    "undervis": ("teach", "teacher", "education", "lærer", "undervis"),
    "økonomi": ("account", "finance", "econom", "økonom", "revisor"),
}
LIST_FORM = re.compile(r"(?:^|\s)(?:[-*•]|\d+[.)])\s|[\[\]{};]", re.MULTILINE)
HEDGE = re.compile(r"\b(?:kan|ofte|muligvis|gerne|typisk)\b")
DETERMINISTIC_CLAIMS = re.compile(r"\b(?:altid|aldrig|helt sikkert|garanteret)\b")

EDUCATION_DANISH = {
    "primary": "grundskole",
    "upper_secondary": "gymnasial uddannelse",
    "vocational": "erhvervsuddannelse",
    "qualifying_programme": "kvalificerende uddannelse",
    "short_cycle_higher": "kort videregående uddannelse",
    "professional_bachelor": "professionsbacheloruddannelse",
    "bachelor": "bacheloruddannelse",
    "masters": "kandidatuddannelse",
    "phd": "ph.d.-uddannelse",
    "not_stated": "uddannelse ikke oplyst",
}
EDUCATION_DANISH.update(
    {
        "h10": "grundskole",
        "h20": "gymnasial uddannelse",
        "h30": "erhvervsuddannelse",
        "h35": "kvalificerende uddannelse",
        "h40": "kort videregående uddannelse",
        "h50": "professionsbacheloruddannelse",
        "h60": "bacheloruddannelse",
        "h70": "kandidatuddannelse",
        "h80": "ph.d.-uddannelse",
        "h90": "uddannelse ikke oplyst",
    }
)
SEX_DANISH = {"male": "mand", "female": "kvinde", "m": "mand", "k": "kvinde"}
STATUS_DANISH = {
    "unemployed": "ledig",
    "student": "studerende",
    "retired": "pensionist",
    "other": "uden for arbejdsmarkedet",
    "outside_labour_force": "uden for arbejdsmarkedet",
    "not_applicable": "uden for arbejdsmarkedet",
}
OCEAN_TERMS = {
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


def parse_attributes(
    content: str, demographic: DemographicRecord | c.Mapping[str, object]
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
    _validate_job_title(title=attributes.job_title, context=context)
    return attributes


def _context_values(
    demographic: DemographicRecord | c.Mapping[str, object],
) -> dict[str, object]:
    if isinstance(demographic, DemographicRecord):
        return demographic.model_dump(mode="python")
    return dict(demographic)


def _validate_job_title(title: str | None, context: dict[str, object]) -> None:
    resolution = context.get("job_function_resolution")
    if resolution not in {"lons20_sex_marginal", "not_applicable"}:
        raise ValueError("Unknown job-function resolution")
    eligible = resolution == "lons20_sex_marginal"
    if eligible and not str(context.get("job_function", "")).strip():
        raise ValueError("Eligible job-function context requires its official label")
    if eligible != (title is not None):
        expected = "a title" if eligible else "null job_title"
        raise ValueError(f"Eligible job-function context requires {expected}")
    if title is None:
        return
    _validate_text(text=title, require_danish=True)
    if LIST_FORM.search(title) or "\n" in title or "\r" in title:
        raise ValueError("job_title must be a single plain Danish line")
    normalized = _normalize(title)
    if any(
        term in normalized for term in ("arbejdsplads", "virksomhed", "institution")
    ):
        raise ValueError("job_title must not invent an employer or institution")
    if any(
        term in normalized for term in JOB_TITLE_SENIORITY
    ) or JOB_TITLE_DUTY_WORDS.search(normalized):
        raise ValueError("job_title must not invent seniority or duties")
    label = _normalize(str(context.get("job_function", "")))
    for group, markers in JOB_ROLE_GROUPS.items():
        if any(marker in normalized for marker in markers) and not any(
            marker in label for marker in markers
        ):
            raise ValueError(
                f"job_title is not grounded in the job-function label: {group}"
            )


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


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
            raise ValueError(f"Generated content contains a prohibited {name}")
    if _contains_url(text=text):
        raise ValueError("Generated content contains a prohibited URL")
    found_sensitive = sorted(
        term for term in SENSITIVE_TERMS if _contains_term(normalized, term)
    )
    if found_sensitive:
        raise ValueError(
            f"Generated content contains sensitive terms: {found_sensitive}"
        )
    found_unsupported = sorted(
        term for term in UNSUPPORTED_CLAIMS if _contains_term(normalized, term)
    )
    if found_unsupported:
        raise ValueError(
            f"Generated content contains unsupported claims: {found_unsupported}"
        )
    if require_danish:
        _require_danish(text=text)


def _contains_term(text: str, term: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None


def _contains_url(text: str) -> bool:
    if EXPLICIT_URL.search(text):
        return True
    for match in DOMAIN_CANDIDATE.finditer(text):
        host = match.group().split("/", maxsplit=1)[0]
        extracted = TLD_EXTRACTOR(host)
        if extracted.domain and extracted.suffix:
            return True
    return False


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
    texts = list(descriptions.model_dump().values())
    _validate_texts(texts=texts, require_each_danish=True)
    _validate_persona(text=descriptions.persona, context=context, attributes=generated)
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
    if str(
        context.get("labour_market_status")
    ).casefold() != "employed" and FORMER_WORK.search(normalized):
        raise ValueError("Non-employees must not receive former-work claims")


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
        status = str(context.get("labour_market_status")).casefold()
        required_status = STATUS_DANISH.get(status)
        if required_status is None:
            raise ValueError("Unknown non-employee labour status")
    if not required_status or _normalize(required_status) not in normalized:
        raise ValueError("Persona does not contain its exact current work status")


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
        if not value or _normalize(value) not in normalized:
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
    allowed_terms = _compatible_ocean_terms(context=context)
    found_terms = [term for term in allowed_terms if _contains_term(normalized, term)]
    if not 1 <= len(found_terms) <= 2:
        raise ValueError("Persona must contain 1-2 compatible personality tendencies")
    for sentence in sentences:
        if any(
            _contains_term(sentence.casefold(), term) for term in found_terms
        ) and not HEDGE.search(sentence):
            raise ValueError("Personality tendencies must be hedged")


def _compatible_ocean_terms(context: dict[str, object]) -> tuple[str, ...]:
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
