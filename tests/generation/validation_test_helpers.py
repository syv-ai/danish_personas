"""Reusable input builders for generation validation tests."""

from collections.abc import Mapping

JOB_TITLE = "forretningsspecialist"


def attributes(*, job_title: str | None = JOB_TITLE) -> dict[str, object]:
    """Build a valid generated-attributes payload for validation tests.

    Returns:
        A payload accepted by the attributes validator.
    """
    return {
        "cultural_context": "En almindelig dansk hverdag med plads til fællesskab.",
        "skills_and_expertise": ["planlægning", "samarbejde", "formidling"],
        "hobbies_and_interests": ["at læse", "musik", "brætspil"],
        "career_goals_and_ambitions": None,
        "job_title": job_title,
    }


def persona(
    *,
    context: Mapping[str, object] | None = None,
    education: str | None = None,
    employment: str | None = None,
    pronoun: str | None = None,
    extra: str = "Den syntetiske hverdag rummer rolige rutiner og små oplevelser.",
) -> dict[str, str]:
    """Build a grounded Danish persona payload.

    Returns:
        A JSON-compatible persona response.
    """
    context = context or demographic()
    pronoun = pronoun or ("hun" if context["sex"] == "female" else "han")
    education = (
        education
        or {
            "primary": "har gået i grundskolen",
            "secondary_or_vocational": (
                "har en ungdomsuddannelse eller erhvervsuddannelse"
            ),
            "higher_education": "har en videregående uddannelse",
            "not_stated": "har en uddannelse, der ikke er oplyst",
        }[str(context["education_level"])]
    )
    employment = employment or (
        f"arbejder som {JOB_TITLE}" if context["job_title"] else "er pensionist"
    )
    text = (
        f"{pronoun.capitalize()} er {context['age']} år og bor i "
        f"{context['municipality']}. {pronoun.capitalize()} kommer fra "
        f"{context['origin_country_da']} og {education}. {pronoun.capitalize()} "
        f"{employment}, og {extra[0].lower() + extra[1:]}"
    )
    return {"persona": text}


def demographic(
    *,
    education_level: str = "higher_education",
    sex: str = "female",
    status: str = "employed",
    job_title: str | None = JOB_TITLE,
) -> dict[str, object]:
    """Build demographic facts used as the validator context.

    Returns:
        Demographic facts for a synthetic person.
    """
    eligible = status == "employed" and job_title is not None
    return {
        "persona_id": "persona-1",
        "age": 35,
        "sex": sex,
        "municipality": "København",
        "origin_country_code": "5100",
        "origin_country": "Denmark",
        "origin_country_da": "Danmark",
        "education_level": education_level,
        "labour_market_status": status,
        "job_function": "24 Business and administration professionals"
        if eligible
        else None,
        "job_function_resolution": "lons20_sex_marginal"
        if eligible
        else "not_applicable",
        "openness_score": 50.0,
        "openness_label": "average",
        "conscientiousness_score": 50.0,
        "conscientiousness_label": "average",
        "extraversion_score": 50.0,
        "extraversion_label": "average",
        "agreeableness_score": 50.0,
        "agreeableness_label": "average",
        "neuroticism_score": 50.0,
        "neuroticism_label": "average",
        "job_title": job_title,
    }
