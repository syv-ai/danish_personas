"""Deterministic same-sex partner target policy."""

import hashlib
from fractions import Fraction

DEFAULT_SAME_SEX_PARTNER_PROBABILITY = 0.00701
SAME_SEX_PARTNER_POLICY_VERSION = "same-sex-partner-target-v1"
_SAME_SEX_PARTNER_DOMAIN = "danish-personas/same-sex-partner-target"
_UINT64_CARDINALITY = 2**64


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
    if probability == 0.0:
        return False
    if probability == 1.0:
        return True
    threshold = _probability_threshold(probability)
    return same_sex_partner_draw(persona_id) < threshold


def _probability_threshold(probability: float) -> int:
    """Return the exact integer threshold represented by a float probability."""
    fraction = Fraction.from_float(probability)
    return (
        fraction.numerator * _UINT64_CARDINALITY + fraction.denominator - 1
    ) // fraction.denominator


def same_sex_partner_draw(persona_id: str) -> int:
    """Return the stable unsigned 64-bit draw for one persona ID.

    Args:
        persona_id:
            Stable identifier for the generated persona.

    Returns:
        A deterministic integer in the inclusive interval ``[0, 2**64 - 1]``.
    """
    material = "\0".join(
        (SAME_SEX_PARTNER_POLICY_VERSION, _SAME_SEX_PARTNER_DOMAIN, persona_id)
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
