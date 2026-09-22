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
from .models import (
    GeneratedAttributes,
    GeneratedPersona,
    GenerationConfig,
    PersonaDescriptions,
)
from .partner_target import required_partner_gender, same_sex_partner_target
from .personality import all_personality_phrases, all_personality_tendencies

VALIDATOR_VERSION = "persona-safety-v20"
__all__ = ["EDUCATION_DANISH"]
_ATTRIBUTE_FIELDS = frozenset(
    {
        "cultural_context",
        "skills_and_expertise",
        "hobbies_and_interests",
        "career_goals_and_ambitions",
        "job_title",
        "current_relationship_status",
        "partner_gender",
        "legal_status_detail",
    }
)
_DESCRIPTION_FIELDS = frozenset({"persona"})
_GENERATED_PERSONA_FIELDS = _ATTRIBUTE_FIELDS | _DESCRIPTION_FIELDS
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
    r"lesbisk\w*|biseksuel\w*|panseksuel\w*|aseksuel\w*|queer\w*|"
    r"bøsse\w*|gay\b|hetero\b|homofili\w*|seksuel\s+orientering",
)
UNSUPPORTED_PATTERNS = (
    r"diagnos(?:e|er|en|erede)\w*|hår(?:et|ene|enes)?|"
    r"øjne?\w*",
    r"ansigt(?:et|er|ene|enes|stræk(?:ket|kene)?)?|"
    r"højde|vægt|krop(?:pen)?|udseende|ser\s+ud",
    r"hud(?:en|ens|farve|farven|farves)?|kropsbygning",
)
ALLOWED_STATUS_TEN_PHRASE = "medarbejdende ægtefælle"
_PERSON_NAME_TOKEN = r"[A-ZÆØÅ][A-Za-zÆØÅæøå]+(?:[-'][A-ZÆØÅ][A-Za-zÆØÅæøå]+)*"
_NAME_WORD = r"[A-Za-zÆØÅæøå]+(?:[-'][A-Za-zÆØÅæøå]+)*"
_AGE_NAME_WORD = r"[A-Za-zÆØÅæøå]+(?:[-'’´][A-Za-zÆØÅæøå]+)*"
_PERSON_NAME_AGE_EXPRESSION = rf"{_PERSON_NAME_TOKEN}(?:\s+{_AGE_NAME_WORD}){{0,3}}"
_RELATIONSHIP_ROLE = (
    r"kæreste(?:n)?|partner(?:en)?|mand(?:en)?|kone(?:n)?|hustru(?:en)?|"
    r"ægtefælle(?:n)?|datter(?:en)?|søn(?:nen)?|mor(?:en)?|far(?:en)?|"
    r"søster(?:en)?|bror(?:en)?|broderen|barn(?:et)?|ven(?:nen)?|"
    r"veninde(?:n)?|kollega(?:en)?"
)
_RELATIONSHIP_POSSESSIVE = r"sin|min|din|hans|hendes|deres|vores|jeres"
_NAME_POSSESSIVE = (
    r"mit|dit|min|din|sin|hans|hendes|deres|vores|jeres|personaens|personens"
)
_NAME_LABEL_WORD = (
    r"(?!(?i:er|har|hans|hendes|sin|deres|vores|jeres|mit|dit)\b)"
    rf"{_NAME_WORD}"
)
_PERSON_NAME_LABEL_EXPRESSION = (
    rf"{_PERSON_NAME_TOKEN}(?:\s+{_NAME_LABEL_WORD}){{0,3}}?"
)
# Capture the first token so a name is found even when prose continues without
# punctuation. Grounded labels are checked separately against the full text span.
_PERSON_NAME_EXPRESSION = _PERSON_NAME_TOKEN
_PERSON_NAME_PATTERNS = (
    re.compile(
        rf"(?m)(?:^|[.!?]\s+)(?P<name>"
        rf"(?!(?i:han|hun|personaen|personen|{_RELATIONSHIP_POSSESSIVE})\b)"
        rf"(?P<bounded_name>{_PERSON_NAME_AGE_EXPRESSION}))\s+"
        rf"(?i:er)\s+\d+\s+(?i:år)\b"
    ),
    re.compile(
        rf"\b(?:(?i:jeg|han|hun|personaen|personen))\s+"
        rf"(?:(?i:hedder|kaldes|går under navnet|er\s+ved\s+navn))\s+"
        rf"(?P<name>{_PERSON_NAME_EXPRESSION})"
    ),
    re.compile(
        rf"\b(?:(?i:jeg|han|hun|personaen|personen))\s+(?:(?i:er))\s+"
        rf"(?P<name>{_PERSON_NAME_EXPRESSION})"
    ),
    re.compile(
        rf"\b(?:(?i:{_NAME_POSSESSIVE}))\s+navn\s+"
        rf"(?:(?i:er))\s+(?P<name>{_PERSON_NAME_EXPRESSION})"
    ),
    re.compile(
        rf"\b(?:(?i:navnet|navn))\s+(?:(?i:er))\s+"
        rf"(?P<name>{_PERSON_NAME_EXPRESSION})"
    ),
    re.compile(
        rf"\b(?:(?i:{_RELATIONSHIP_POSSESSIVE})\s+)?"
        rf"(?:(?i:{_RELATIONSHIP_ROLE}))\s+"
        rf"(?:(?i:hedder|kaldes))\s+"
        rf"(?P<name>{_PERSON_NAME_EXPRESSION})"
    ),
    re.compile(
        rf"\b(?:(?i:{_RELATIONSHIP_POSSESSIVE})\s+)?"
        rf"(?:(?i:{_RELATIONSHIP_ROLE}))\s+(?:(?i:er)\s+)?"
        rf"(?i:ved navn)\s+"
        rf"(?P<name>{_PERSON_NAME_EXPRESSION})"
    ),
    re.compile(
        rf"\b(?!(?i:{_RELATIONSHIP_POSSESSIVE})\b)"
        rf"(?P<name>{_PERSON_NAME_LABEL_EXPRESSION})"
        rf"(?P<possessive>(?i:s|['’´]s?))\s+(?:(?i:{_RELATIONSHIP_ROLE}))\b"
    ),
    re.compile(
        rf"\b(?:(?i:{_RELATIONSHIP_POSSESSIVE})\s+)?"
        rf"(?:(?i:{_RELATIONSHIP_ROLE}))(?:\s+(?i:er)\s+|[,:]?\s+)"
        rf"(?P<name>{_PERSON_NAME_EXPRESSION})"
    ),
    re.compile(
        rf"\b(?P<name>{_PERSON_NAME_EXPRESSION})\s+(?:(?i:er))\s+"
        rf"(?:(?i:{_RELATIONSHIP_POSSESSIVE})\s+)?"
        rf"(?:(?i:{_RELATIONSHIP_ROLE}))\b"
    ),
)
LIST_FORM = re.compile(r"(?:^|\s)(?:[-*•]|\d+[.)])\s|[\[\]{};]", re.MULTILINE)
DASH_TRANSLATION = str.maketrans(
    {
        "\u00ad": "-",
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "−": "-",
        "﹘": "-",
        "﹣": "-",
        "－": "-",
    }
)


def parse_generated_persona(
    content: str,
    demographic: DemographicRecord | c.Mapping[str, object],
    *,
    job_title_mapping: JobFunctionTitleMapping | None = None,
    generation_config: GenerationConfig | None = None,
) -> GeneratedPersona:
    """Parse and validate one combined attributes-and-persona response.

    Returns:
        Validated generated attributes and persona text.

    Raises:
        ValueError:
            If the JSON, grounding, specificity, or safety rules are invalid.
    """
    try:
        generated = GeneratedPersona.model_validate_json(content)
    except ValidationError as error:
        raise ValueError(
            _format_schema_error(error=error, fields=_GENERATED_PERSONA_FIELDS)
        ) from error
    attributes = parse_attributes(
        GeneratedAttributes.model_validate(
            {
                field: getattr(generated, field)
                for field in GeneratedAttributes.model_fields
            }
        ).model_dump_json(),
        demographic,
        job_title_mapping=job_title_mapping,
        generation_config=generation_config,
    )
    parse_descriptions(
        PersonaDescriptions(persona=generated.persona).model_dump_json(),
        demographic,
        attributes,
    )
    return generated


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


def parse_attributes(
    content: str,
    demographic: DemographicRecord | c.Mapping[str, object],
    *,
    job_title_mapping: JobFunctionTitleMapping | None = None,
    generation_config: GenerationConfig | None = None,
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
    _validate_field(
        field="relationship",
        validator=lambda: _validate_relationship_attributes(
            attributes=attributes,
            demographic=context,
            generation_config=generation_config,
        ),
    )
    for field, value in attributes.model_dump().items():
        texts = value if isinstance(value, list) else [value]
        for text in texts:
            if isinstance(text, str):
                _validate_field(
                    field=field,
                    validator=lambda text=text: _validate_no_person_names(
                        text=text, context=context
                    ),
                )
    return attributes


def _context_values(
    demographic: DemographicRecord | c.Mapping[str, object],
) -> dict[str, object]:
    if isinstance(demographic, DemographicRecord):
        return demographic.model_dump(mode="python")
    return dict(demographic)


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
    """Require lowercase activity or topic phrases, not OCEAN language.

    Personality terms are reserved for the grounded persona's separately validated
    tendency phrases. Matching against the shared term and phrase APIs also rejects
    a tendency when it is embedded in a longer interest, while preserving ordinary
    words that merely contain the same letters.

    Raises:
        ValueError:
            If an interest is not a lowercase phrase, ends in punctuation, or
            contains a reviewed personality term or phrase.
    """
    _validate_texts(texts=texts, require_each_danish=False)
    if any(_contains_uppercase_character(text=interest) for interest in texts):
        raise ValueError("Every interest must be a lowercase Danish phrase")
    if any(_ends_with_punctuation(text=interest) for interest in texts):
        raise ValueError("Interests must not end with punctuation")
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
    """Return case-folded text with equivalent dashes and spacing unified."""
    normalized = unicodedata.normalize("NFKC", text).translate(DASH_TRANSLATION)
    return " ".join(normalized.casefold().split())


def _contains_uppercase_character(*, text: str) -> bool:
    """Return whether any cased character in a phrase is uppercase."""
    return any(character.isupper() for character in text if character.isalpha())


def _ends_with_punctuation(*, text: str) -> bool:
    """Return whether a phrase has terminal Unicode punctuation."""
    stripped = text.rstrip()
    return bool(stripped) and unicodedata.category(stripped[-1]).startswith("P")


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


def _validate_relationship_attributes(
    *,
    attributes: GeneratedAttributes,
    demographic: dict[str, object],
    generation_config: GenerationConfig | None = None,
) -> None:
    """Validate relationship fields against legal and stable target semantics.

    Raises:
        ValueError:
            If relationship fields do not match the supplied legal status or target.
    """
    marital_status = str(demographic.get("marital_status", "")).casefold()
    if generation_config is not None:
        persona_id = demographic.get("persona_id")
        if not isinstance(persona_id, str):
            raise ValueError("Generation target validation requires persona_id")
        target = same_sex_partner_target(
            persona_id=persona_id,
            probability=generation_config.same_sex_partner_probability,
        )
        if attributes.current_relationship_status == "partnered":
            expected_gender = required_partner_gender(
                sex=str(demographic.get("sex", "")), same_sex_target=target
            )
            if attributes.partner_gender != expected_gender:
                raise ValueError(
                    "partner_gender does not match the stable relationship target"
                )
    detail = attributes.legal_status_detail
    if marital_status == "married_or_separated":
        if detail is None:
            raise ValueError("married_or_separated requires legal_status_detail")
        if (
            detail == "married"
            and attributes.current_relationship_status != "partnered"
        ):
            raise ValueError("married responses must be partnered")
    elif marital_status in {"never_married", "divorced", "widowed"}:
        if detail is not None:
            raise ValueError(f"{marital_status} must not include legal_status_detail")
    else:
        raise ValueError(f"Unknown marital_status: {marital_status}")


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
        validator=lambda: _validate_no_person_names(
            text=descriptions.persona, context=context
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


def _validate_no_person_names(*, text: str, context: dict[str, object]) -> None:
    """Reject explicit personal-name constructions without guessing from vocabulary.

    The generation contract permits required place and origin labels, so this check
    deliberately looks only at reviewed name-bearing constructions. It does not scan
    capitalised words: that would reject sentence starts and grounded proper nouns.

    Raises:
        ValueError:
            If a name appears after an own-name or relationship construction.
    """
    allowed_labels = {
        _normalize(text=str(context.get(field, "")))
        for field in ("municipality", "origin_country_da")
        if context.get(field)
    }
    for pattern in _PERSON_NAME_PATTERNS:
        for match in pattern.finditer(text):
            if not _is_allowed_grounding_label(
                text=text, match=match, allowed_labels=allowed_labels
            ):
                raise ValueError("Generated content contains a prohibited person name")


def _is_allowed_grounding_label(
    *, text: str, match: re.Match[str], allowed_labels: set[str]
) -> bool:
    """Return whether a captured construction contains a complete grounded label.

    Most name patterns capture only the first capitalised token so they also catch
    prose that continues without punctuation. Compare the complete text at that
    position to each grounded label rather than treating that token as the whole
    label.
    """
    start, _ = match.span("name")
    if start and not _is_grounding_boundary(text=text, index=start - 1):
        return False
    tail = _normalize(text=text[start:])
    for label in allowed_labels:
        normalised_label = _normalize(text=label)
        if not tail.startswith(normalised_label):
            continue
        if (
            "bounded_name" in match.re.groupindex
            and _normalize(text=match.group("bounded_name")) != normalised_label
        ):
            continue
        remainder = tail[len(normalised_label) :]
        if not remainder or _is_grounding_boundary(text=remainder, index=0):
            return True
        if "possessive" not in match.re.groupindex:
            continue
        boundary_end = match.end("possessive")
        captured = _normalize(text=text[start:boundary_end])
        possessive = _normalize(text=match.group("possessive"))
        if captured == f"{normalised_label}{possessive}" and _is_grounding_boundary(
            text=text, index=boundary_end
        ):
            return True
    return False


def _is_grounding_boundary(*, text: str, index: int) -> bool:
    """Return whether the character at ``index`` terminates a grounded label."""
    return index >= len(text) or not _is_grounding_continuation_character(text[index])


def _is_grounding_continuation_character(character: str) -> bool:
    """Return whether a character can extend a grounded label or name."""
    normalised = unicodedata.normalize("NFKC", character).translate(DASH_TRANSLATION)
    return (
        normalised.isalnum() or normalised == "_" or normalised in {"-", "'", "’", "´"}
    )


def _validate_persona(
    text: str, context: dict[str, object], attributes: GeneratedAttributes
) -> None:
    """Require source-backed persona facts while allowing natural elaboration."""
    _validate_persona_facts(text=text, demographic=context, attributes=attributes)


def _validate_persona_facts(
    *, text: str, demographic: dict[str, object], attributes: GeneratedAttributes
) -> None:
    """Require supplied facts without accepting a nearby negation.

    The checks use small reviewed vocabularies rather than requiring copied clauses,
    allowing ordinary Danish paraphrases while rejecting denial or contradiction.

    Raises:
        ValueError:
            If a supplied fact is missing, denied, or contradicted.
    """
    facts = build_persona_grounding_facts(
        demographic=demographic, attributes=attributes
    )
    normalized = _normalize(text=text)
    _validate_age(text=normalized, demographic=demographic)
    official_english_origin = str(demographic.get("origin_country", ""))
    official_danish_origin = str(demographic.get("origin_country_da", ""))
    if (
        official_english_origin.casefold() != official_danish_origin.casefold()
        and _contains_term(text=normalized, term=official_english_origin)
    ):
        raise ValueError("Persona contains an alternative official origin label")
    for field, value in (
        ("municipality", str(demographic.get("municipality", ""))),
        ("origin", official_danish_origin),
    ):
        _validate_grounding_label(text=normalized, field=field, value=value)
    _validate_education(text=normalized, demographic=demographic)
    _validate_employment(
        text=normalized, employment=facts.employment, demographic=demographic
    )
    _validate_relationship_prose(
        text=normalized, demographic=demographic, attributes=attributes
    )


def _validate_age(*, text: str, demographic: dict[str, object]) -> None:
    """Require the supplied age and pronoun without rejecting other people.

    Raises:
        ValueError:
            If the supplied age or pronoun is absent or negated.
    """
    pronoun = _expected_pronoun(context=demographic)
    age = str(demographic.get("age", ""))
    if not _contains_term(text, pronoun):
        raise ValueError("Persona does not preserve the supplied pronoun or age")
    expected = list(re.finditer(rf"(?<!\w){re.escape(age)}\s*år(?:ig)?\b", text))
    if not expected or any(
        _is_negated(text=text, span=match.span()) for match in expected
    ):
        raise ValueError("Persona does not preserve the supplied pronoun or age")


def _expected_pronoun(*, context: dict[str, object]) -> str:
    """Return the supplied Danish subject pronoun.

    Raises:
        ValueError:
            If the supplied sex has no recognised pronoun.
    """
    expected = {"male": "han", "m": "han", "female": "hun", "k": "hun"}.get(
        str(context.get("sex", "")).casefold()
    )
    if expected is None:
        raise ValueError("Persona has no recognised supplied pronoun")
    return expected


def _is_negated(*, text: str, span: tuple[int, int]) -> bool:
    """Return whether a supplied fact is denied in its local clause."""
    sentence_start = max(
        text.rfind(".", 0, span[0]),
        text.rfind("!", 0, span[0]),
        text.rfind("?", 0, span[0]),
        text.rfind(";", 0, span[0]),
        text.rfind(",", 0, span[0]),
    )
    sentence_end = len(text)
    for punctuation in ".!?;":
        boundary = text.find(punctuation, span[1])
        if boundary >= 0:
            sentence_end = min(sentence_end, boundary)
    prefix = text[sentence_start + 1 : span[0]]
    suffix = text[span[1] : sentence_end]
    prefix_words = re.findall(r"[\wæøå]+", prefix.casefold())[-8:]
    suffix_words = re.findall(r"[\wæøå]+", suffix.casefold())[:8]
    if any(word in {"ikke", "ingen", "aldrig", "hverken"} for word in prefix_words):
        return True
    return any(
        suffix_words[index : index + 2] in (["ikke", "længere"], ["ikke", "mere"])
        for index in range(len(suffix_words) - 1)
    )


def _validate_education(*, text: str, demographic: dict[str, object]) -> None:
    """Require the supplied education level without policing other people.

    Raises:
        ValueError:
            If the supplied education is absent, negated, or remains ambiguous.
    """
    education_terms = {
        "primary": ("folkeskolen", "grundskole", "grundskolen"),
        "secondary_or_vocational": ("ungdomsuddannelse", "erhvervsuddannelse"),
        "higher_education": ("videregående uddannelse",),
        "not_stated": ("uddannelse ikke oplyst", "uddannelsen er ikke oplyst"),
    }
    level = str(demographic.get("education_level", "")).casefold()
    matches = [
        span
        for term in education_terms.get(level, ())
        for span in _term_spans(text=text, term=term)
    ]
    if (
        level == "not_stated"
        and _contains_term(text, "uddannelse")
        and _contains_term(text, "oplyst")
    ):
        matches.append((0, 0))
    if not matches:
        raise ValueError("Persona does not preserve the supplied education level")
    if level != "not_stated" and any(
        _is_negated(text=text, span=span) for span in matches
    ):
        raise ValueError("Persona negates the supplied education level")
    if level == "secondary_or_vocational" and all(
        _contains_term(text, term)
        for term in ("ungdomsuddannelse", "erhvervsuddannelse")
    ):
        raise ValueError("Persona must choose one specific education branch")


def _validate_employment(
    *, text: str, employment: str, demographic: dict[str, object]
) -> None:
    """Require the supplied title or status without policing other people.

    Raises:
        ValueError:
            If the supplied current title or status is absent or negated.
    """
    del demographic
    value = employment.removeprefix("arbejder som ").removeprefix("er ")
    spans = _term_spans(text=text, term=value)
    if not spans:
        raise ValueError("Persona does not preserve the supplied work status")
    if any(_is_negated(text=text, span=span) for span in spans):
        raise ValueError("Persona negates the supplied current work status")


def _validate_grounding_label(*, text: str, field: str, value: str) -> None:
    """Require a supplied municipality or origin label without negation.

    Raises:
        ValueError:
            If the supplied label is absent or negated.
    """
    spans = _term_spans(text=text, term=value)
    if not spans:
        raise ValueError(f"Persona does not preserve the supplied {field}")
    if any(_is_negated(text=text, span=span) for span in spans):
        raise ValueError(f"Persona negates the supplied {field}")


def _validate_relationship_prose(
    *, text: str, demographic: dict[str, object], attributes: GeneratedAttributes
) -> None:
    """Require legal and current relationship fields in the persona prose.

    Raises:
        ValueError:
            If relationship or legal-status wording is missing.
    """
    relationship_terms = ("partner", "kæreste", "ægtefælle", "mand", "kone", "hustru")
    if attributes.current_relationship_status == "partnered":
        if not any(_contains_term(text=text, term=term) for term in relationship_terms):
            raise ValueError("Persona does not preserve the partnered status")
    elif not any(
        _contains_term(text=text, term=term)
        for term in ("single", "alene", "uden partner", "ikke i et forhold")
    ):
        raise ValueError("Persona does not preserve the not_partnered status")

    marital_status = str(demographic.get("marital_status", "")).casefold()
    detail = attributes.legal_status_detail
    legal_terms = {
        "married": ("gift",),
        "separated": ("separeret",),
        "never_married": ("aldrig været gift", "har aldrig været gift"),
        "divorced": ("skilt",),
        "widowed": ("enke", "enkemand"),
    }
    expected_terms = (
        legal_terms[detail]
        if marital_status == "married_or_separated" and detail is not None
        else legal_terms.get(marital_status, ())
    )
    if not any(_contains_term(text=text, term=term) for term in expected_terms):
        raise ValueError("Persona does not preserve the supplied legal marital status")
