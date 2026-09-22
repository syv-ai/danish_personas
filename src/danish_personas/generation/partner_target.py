"""Deterministic same-sex partner target policy."""

import hashlib

DEFAULT_SAME_SEX_PARTNER_PROBABILITY = 0.00701
SAME_SEX_PARTNER_POLICY_VERSION = "same-sex-partner-target-v1"
_SAME_SEX_PARTNER_DOMAIN = "danish-personas/same-sex-partner-target"


def required_partner_gender(*, sex: str, same_sex_target: bool) -> str:
    """Return the partner gender required by a persona's target and sex.

    Args:
        sex:
            Persona sex, either ``male`` or ``female``.
        same_sex_target:
            Whether the stable target requests a same-sex partner.

    Returns:
        The required partner gender.

    Raises:
        ValueError:
            If ``sex`` is not a supported binary source value.
    """
    if sex not in {"male", "female"}:
        raise ValueError(f"Unsupported sex for partner target: {sex}")
    if same_sex_target:
        return sex
    return "female" if sex == "male" else "male"


def same_sex_partner_target(*, persona_id: str, probability: float) -> bool:
    """Apply the configured Bernoulli threshold to a stable persona draw.

    Args:
        persona_id:
            Stable identifier for the generated persona.
        probability:
            Probability threshold in the closed interval ``[0, 1]``.

    Returns:
        Whether this persona has the internal same-sex partner target.

    Raises:
        ValueError:
            If ``probability`` is outside ``[0, 1]``.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Same-sex partner probability must be between 0 and 1")
    return same_sex_partner_draw(persona_id) < probability


def same_sex_partner_draw(persona_id: str) -> float:
    """Return the stable unit-interval draw for one persona ID.

    Args:
        persona_id:
            Stable identifier for the generated persona.

    Returns:
        A deterministic value in the half-open interval ``[0, 1)``.
    """
    material = "\0".join(
        (SAME_SEX_PARTNER_POLICY_VERSION, _SAME_SEX_PARTNER_DOMAIN, persona_id)
    ).encode("utf-8")
    integer = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return integer / 2**64
