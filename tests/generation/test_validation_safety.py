"""Safety, prose, and language-boundary tests for persona validation."""

# Test functions intentionally omit repetitive docstrings.
# ruff: noqa: D103

import json

import pytest
from validation_test_helpers import attributes, demographic, persona

from danish_personas.generation.validation import parse_descriptions


@pytest.mark.parametrize(
    "former_work",
    [
        "tidligere",
        "førhen",
        "før",
        "arbejdede",
        "har arbejdet",
        "var ansat",
        "forhenværende",
        "pensioneret fra",
    ],
)
def test_all_former_work_wordings_fail(former_work: str) -> None:
    context = demographic()
    text = persona(context=context)["persona"] + f" Hun er {former_work} ansat."
    with pytest.raises(ValueError, match="former|past-work"):
        parse_descriptions(json.dumps({"persona": text}), context, attributes())


@pytest.mark.parametrize(
    "prefix",
    ["- ", "* ", "• ", "1. ", "1) ", "[", ";"],
    ids=[
        "dash",
        "asterisk",
        "bullet",
        "numbered-dot",
        "numbered-paren",
        "bracket",
        "semicolon",
    ],
)
def test_all_list_forms_fail(prefix: str) -> None:
    context = demographic()
    text = prefix + persona(context=context)["persona"]
    with pytest.raises(ValueError, match="prose"):
        parse_descriptions(json.dumps({"persona": text}), context, attributes())


@pytest.mark.parametrize(
    "phrase",
    [
        "oprindelsesland",
        "oprindelsesetiket",
        "brede uddannelsesbaggrund",
        "uddannelsesniveau",
        "aktuelle arbejdsforhold",
    ],
)
def test_all_technical_grounding_wordings_fail(phrase: str) -> None:
    context = demographic()
    text = persona(context=context)["persona"] + f" {phrase} er kun et teknisk felt."
    with pytest.raises(ValueError, match="redundant or technical"):
        parse_descriptions(json.dumps({"persona": text}), context, attributes())


@pytest.mark.parametrize(
    "claim",
    [
        "ansigter",
        "ansigtstræk",
        "hårene",
        "hudfarve",
        "højde",
        "vægt",
        "kroppen",
        "kropsbygning",
        "udseende",
        "ser ud",
    ],
)
def test_all_unsupported_appearance_claims_fail(claim: str) -> None:
    context = demographic()
    text = persona(context=context)["persona"] + f" Hun nævner {claim}."
    with pytest.raises(ValueError, match="unsupported"):
        parse_descriptions(json.dumps({"persona": text}), context, attributes())


def test_alternative_official_origin_label_is_rejected_without_echo() -> None:
    context = demographic()
    text = persona(context=context)["persona"] + " Hun kommer fra Denmark."
    with pytest.raises(ValueError, match="origin") as error:
        parse_descriptions(json.dumps({"persona": text}), context, attributes())
    assert "Denmark" not in str(error.value)


@pytest.mark.parametrize(
    "term", ["vedkommende", "personen", "kan være", "ungdoms- eller erhvervsuddannelse"]
)
def test_banned_persona_terms_are_case_insensitive_and_token_bounded(term: str) -> None:
    context = demographic()
    text = persona(context=context)["persona"]
    payload = {"persona": text[:-1] + f" {term.upper()}."}
    with pytest.raises(ValueError, match="prohibited|contract"):
        parse_descriptions(json.dumps(payload), context, attributes())


def test_banned_terms_do_not_match_larger_words() -> None:
    context = demographic()
    text = persona(context=context)["persona"][:-1] + " Personenhed er ikke nævnt."
    # ``personen`` is bounded: a larger word is not a generic subject substitute.
    assert parse_descriptions(json.dumps({"persona": text}), context, attributes())


@pytest.mark.parametrize(
    "term", ["person", "personen", "personens", "personer", "personerne"]
)
def test_bare_person_paradigm_is_rejected(term: str) -> None:
    context = demographic()
    text = persona(context=context)["persona"] + f" {term} læser."
    with pytest.raises(ValueError, match="generic|contract"):
        parse_descriptions(json.dumps({"persona": text}), context, attributes())


def test_cautious_ocean_paraphrase_is_allowed() -> None:
    context = demographic()
    payload = persona(context=context)
    payload["persona"] = payload["persona"][:-1] + " Hun er ofte social."
    assert parse_descriptions(json.dumps(payload), context, attributes())


@pytest.mark.parametrize(
    "term",
    [
        "vedkommendes",
        "personen",
        "personens",
        "personer",
        "personerne",
        "kan måske være",
        "kan ofte godt være",
        "ungdoms - eller erhvervsuddannelse",
        "ungdoms–eller erhvervsuddannelse",
    ],
)
def test_contract_variants_are_rejected(term: str) -> None:
    context = demographic()
    payload = persona(context=context)
    payload["persona"] = payload["persona"][:-1] + f" {term}."
    with pytest.raises(ValueError, match="prohibited|contract"):
        parse_descriptions(json.dumps(payload), context, attributes())


@pytest.mark.parametrize(
    "phrase",
    [
        "kan i mange forskellige situationer være",
        "kan på flere forskellige måder godt være",
    ],
)
def test_long_intervening_kan_vaere_is_rejected(phrase: str) -> None:
    context = demographic()
    text = persona(context=context)["persona"] + f" Hun {phrase} social."
    with pytest.raises(ValueError, match="prohibited|contract"):
        parse_descriptions(json.dumps({"persona": text}), context, attributes())


def test_ocean_terms_are_checked_independently_and_hedged() -> None:
    context = demographic()
    payload = persona(context=context)
    payload["persona"] = payload["persona"][:-1] + " Hun er social."
    with pytest.raises(ValueError, match="cautious"):
        parse_descriptions(json.dumps(payload), context, attributes())

    context["openness_label"] = "high"
    payload = persona(context=context)
    payload["persona"] = payload["persona"][:-1] + " Hun er praktisk."
    with pytest.raises(ValueError, match="incompatible"):
        parse_descriptions(json.dumps(payload), context, attributes())


@pytest.mark.parametrize(
    ("sex", "opposing"),
    [
        ("female", "han"),
        ("female", "ham"),
        ("female", "hans"),
        ("male", "hun"),
        ("male", "hende"),
        ("male", "hendes"),
    ],
)
def test_opposing_pronoun_variants_are_rejected(sex: str, opposing: str) -> None:
    context = demographic(sex=sex)
    payload = persona(context=context)
    payload["persona"] = payload["persona"][:-1] + f" {opposing.capitalize()} læser."
    with pytest.raises(ValueError, match="pronoun"):
        parse_descriptions(json.dumps(payload), context, attributes())


@pytest.mark.parametrize(
    "assertion",
    ["Hun er social.", "Hun er en social person."],
    ids=["bare-adjective", "generic-person"],
)
def test_person_directed_social_assertion_requires_local_hedge(assertion: str) -> None:
    context = demographic()
    text = persona(context=context)["persona"] + " " + assertion
    with pytest.raises(ValueError):
        parse_descriptions(json.dumps({"persona": text}), context, attributes())


@pytest.mark.parametrize(
    "dash", ["-", "\u00ad", "\uff0d"], ids=["ascii", "soft", "fullwidth"]
)
def test_unicode_dash_variants_are_normalised_before_contract_matching(
    dash: str,
) -> None:
    context = demographic(education_level="secondary_or_vocational")
    text = persona(context=context)["persona"] + (
        f" Hun har en ungdoms{dash}eller erhvervsuddannelse."
    )
    with pytest.raises(ValueError, match="prohibited|contract"):
        parse_descriptions(json.dumps({"persona": text}), context, attributes())


def test_unrelated_kan_vaere_idiom_remains_allowed() -> None:
    context = demographic()
    text = persona(context=context)["persona"] + " Hun kan godt lide at være frivillig."
    assert parse_descriptions(json.dumps({"persona": text}), context, attributes())
