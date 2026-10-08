"""Privacy-bounded per-row persona adjudication."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import typing as t
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydantic import Field, ValidationError, model_validator

from ..io import canonical_json
from ..models import StrictModel
from .client import OpenAIClient
from .models import GenerationConfig, LLMResponse
from .proxy_budget import (
    BASE_URL,
    SOL_ADJUDICATION_LEDGER_MAX_TOKENS,
    SOL_ADJUDICATION_MODEL,
    SOL_ADJUDICATION_PURPOSE,
    V3_ADJUDICATION_PURPOSE,
    V3_EXTENDED_ADJUDICATION_PURPOSE,
    V3_LONG_ADJUDICATION_PURPOSE,
    JSONValue,
    ProxyBudget,
)

SOL_SCHEMA_NAME = "sol_adjudication"
SOL_MAX_OUTPUT_TOKENS = 1_024
_CHECKPOINT_VERSION = 1
_MAX_PROMPT_CHARS = 10_000
_MAX_PROSE_CHARS = 20_000
_MAX_FACT_VALUE_CHARS = 600
_MAX_ROW_ATTEMPTS = 10
_MAX_LOCAL_VALIDATION_COMPLETIONS = 3
SolAdjudicationMode: t.TypeAlias = t.Literal["default", "unresolved_followup"]
_UNRESOLVED_FOLLOWUP_RULE_DA = (
    "Opfølgning på tidligere unresolved: Vælg consistent, når "
    "kandidatprosaen ikke indeholder en konkret modsigelse af en "
    "kildeunderstøttet fakta. Manglende omtale af en struktureret fakta er "
    "ikke i sig selv en modsigelse. Ved konkret modsigelse må højst to "
    "minimale, unikke og eksakt groundede rettelser foreslås. Afstå, hvis "
    "ingen understøttet og sikker rettelse findes. Opfind aldrig fakta uden "
    "kildestøtte."
)

SOL_ALLOWED_FACT_FIELDS = frozenset(
    {
        "origin_country_da",
        "municipality",
        "age",
        "sex",
        "education_level",
        "labour_market_status",
        "job_function",
        "job_title",
        "current_status",
        "marital_status",
        "legal_status_detail",
        "current_relationship_status",
        "hobbies_and_interests",
        "skills_and_expertise",
        "career_goals_and_ambitions",
        "openness_label",
        "conscientiousness_label",
        "extraversion_label",
        "agreeableness_label",
        "neuroticism_label",
    }
)
_RESTRICTED_FACT_FIELDS = frozenset(
    {
        "persona_id",
        "id",
        "origin_country",
        "origin_country_code",
        "municipality_code",
        "region_code",
        "sidecar",
        "review_sidecar",
        "sexual_orientation",
        "partner_sexual_orientation",
        "transgender",
        "transgender_status",
        "sex_characteristics_variation",
        "variation_in_sex_characteristics",
        "same_sex_partner_target",
        "same_sex_target",
        "partner_gender",
    }
)
_RESTRICTED_IDENTITY_TERMS = (
    r"sexual\s+orientation",
    r"seksu(?:el|al)\s+orientering",
    r"seksualitet",
    r"sexuality",
    r"homoseksuel",
    r"homosexual",
    r"biseksuel",
    r"bisexual",
    r"panseksuel",
    r"pansexual",
    r"aseksuel",
    r"asexual",
    r"heteroseksuel",
    r"heterosexual",
    r"lesbisk",
    r"lesbian",
    r"gay",
    r"queer",
    r"lgbtq?i?a?\+?",
    r"same[-\s]+sex",
    r"samkønnet",
    r"samkoennet",
    r"kønsidentitet",
    r"koensidentitet",
    r"gender\s+identity",
    r"transkønnet",
    r"transkoennet",
    r"transseksuel",
    r"transsexual",
    r"transgender",
    r"transperson",
    r"trans[-\s]?(?:mand|kvinde|man|woman)",
    r"non[-\s]?(?:binær|binaer|binary)",
    r"interkøn(?:net)?",
    r"interkoen(?:net)?",
    r"intersex",
    r"sex[-\s]+characteristics",
    r"kønskarakteristika",
    r"koenskarakteristika",
    r"variation\s+in\s+sex\s+characteristics",
    r"variation(?:er)?\s+i\s+kønskarakteristika",
    r"variation(?:er)?\s+i\s+koenskarakteristika",
)
_RESTRICTED_TEXT = re.compile(
    rf"(?<![\w])(?:{'|'.join(_RESTRICTED_IDENTITY_TERMS)})(?![\w])", re.IGNORECASE
)
_TOKEN_BOUNDARY = r"[0-9A-Za-zÆØÅæøå]"
_ALLOWED_FACT_VALUE_TYPES = (str, int, float, bool, type(None))


class SolEvidence(StrictModel):
    """Evidence binding a consistent or unresolved adjudication to the prose."""

    field: str = Field(min_length=1, max_length=80)
    kind: t.Literal["fact_present", "negative_evidence", "source_context"]
    quote: str = Field(min_length=1, max_length=500)


class SolPatch(StrictModel):
    """One exact prose replacement proposed by the adjudication model."""

    old_excerpt: str = Field(min_length=1, max_length=1_000)
    new_excerpt: str = Field(min_length=1, max_length=1_000)


class SolAdjudicationResponse(StrictModel):
    """Strict provider response for local Sol adjudication validation."""

    disposition: t.Literal["consistent", "patched", "unresolved"]
    reason: str = Field(min_length=1, max_length=500)
    evidence: list[SolEvidence] = Field(default_factory=list, max_length=8)
    patches: list[SolPatch] = Field(default_factory=list, max_length=2)

    @staticmethod
    def provider_json_schema() -> dict[str, object]:
        """Return the strict JSON schema sent to the local proxy.

        Returns:
            Provider-compatible JSON schema. Detailed edit and evidence rules are
            enforced locally after the response is received and usage is recorded.
        """
        return {
            "type": "object",
            "properties": {
                "disposition": {
                    "type": "string",
                    "enum": ["consistent", "patched", "unresolved"],
                },
                "reason": {"type": "string"},
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "kind": {
                                "type": "string",
                                "enum": [
                                    "fact_present",
                                    "negative_evidence",
                                    "source_context",
                                ],
                            },
                            "quote": {"type": "string"},
                        },
                        "required": ["field", "kind", "quote"],
                        "additionalProperties": False,
                    },
                },
                "patches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_excerpt": {"type": "string"},
                            "new_excerpt": {"type": "string"},
                        },
                        "required": ["old_excerpt", "new_excerpt"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["disposition", "reason", "evidence", "patches"],
            "additionalProperties": False,
        }

    @model_validator(mode="after")
    def validate_disposition_contract(self) -> "SolAdjudicationResponse":
        """Enforce cross-field disposition invariants.

        Returns:
            The validated response.

        Raises:
            ValueError:
                If the disposition-specific fields are inconsistent.
        """
        if self.disposition == "consistent":
            if self.patches or not self.evidence:
                raise ValueError("consistent responses require evidence only")
        elif self.disposition == "patched":
            if not self.patches:
                raise ValueError("patched responses require patches")
        elif self.patches:
            raise ValueError("unresolved responses cannot include patches")
        return self


def preflight_sol_adjudication_payload(
    *,
    original_persona: str,
    candidate_row: dict[str, object],
    prompt: str,
    changed_fact_hints: dict[str, dict[str, object]] | None = None,
    original_row: dict[str, object] | None = None,
    adjudication_mode: SolAdjudicationMode = "default",
) -> None:
    """Validate one outbound Sol payload without provider I/O.

    Args:
        original_persona:
            Persona prose that would be sent to the provider.
        candidate_row:
            Current structured row. Only allowlisted facts may become outbound facts.
        prompt:
            Sol adjudication prompt.
        changed_fact_hints (optional):
            Verified old/new hints for changed allowlisted facts.
        original_row (optional):
            Original row used for row-bound hint and raw-token checks.
        adjudication_mode (optional):
            Use ``unresolved_followup`` to add the fixed follow-up decision
            rule to the user payload. Defaults to ``default``.

    """
    _validated_payload(
        original_persona=original_persona,
        candidate_row=candidate_row,
        prompt=prompt,
        changed_fact_hints=changed_fact_hints,
        original_row=original_row,
        adjudication_mode=adjudication_mode,
    )


def _validated_payload(
    *,
    original_persona: str,
    candidate_row: dict[str, object],
    prompt: str,
    changed_fact_hints: dict[str, dict[str, object]] | None,
    original_row: dict[str, object] | None,
    adjudication_mode: SolAdjudicationMode,
) -> dict[str, object]:
    if not prompt.strip() or len(prompt) > _MAX_PROMPT_CHARS:
        raise SolAdjudicationError("Prompt is missing or outside the supported range")
    if (
        not isinstance(original_persona, str)
        or not original_persona.strip()
        or len(original_persona) > _MAX_PROSE_CHARS
    ):
        raise SolAdjudicationError(
            "Original persona prose is missing or outside the supported range"
        )
    mode = _validated_adjudication_mode(adjudication_mode=adjudication_mode)
    _check_restricted_text(value=prompt, label="prompt")
    _check_restricted_text(value=original_persona, label="original persona prose")
    candidate_facts = _candidate_fact_payload(candidate_row=candidate_row)
    hints = _validated_changed_fact_hints(
        changed_fact_hints=changed_fact_hints or {},
        candidate_row=candidate_row,
        original_row=original_row,
    )
    payload: dict[str, object] = {
        "persona": original_persona,
        "candidate_facts": candidate_facts,
        "changed_fact_hints": hints,
        "adjudication_scope": "single_row_no_raw_ids_no_sensitive_identity_fields",
        "evidence_rule": (
            "Brug kun fact_present, hvis citatet indeholder den eksakte værdi fra "
            "candidate_facts[field] (uden forskel på store/små bogstaver). Ved "
            "omskrivning eller fravær skal kind være source_context eller "
            "negative_evidence. Citatet skal forekomme præcis én gang ordret "
            "i personateksten."
        ),
    }
    if mode == "unresolved_followup":
        payload["unresolved_followup_rule"] = _UNRESOLVED_FOLLOWUP_RULE_DA
    for value in (prompt, canonical_json(payload)):
        _check_restricted_text(value=value, label="outbound adjudication payload")
    protected_tokens = _protected_row_tokens(
        candidate_row=candidate_row, original_row=original_row
    )
    _check_protected_tokens(
        original_persona=original_persona,
        candidate_facts=candidate_facts,
        changed_fact_hints=hints,
        candidate_row=candidate_row,
        original_row=original_row,
        protected_tokens=protected_tokens,
    )
    return payload


class SolAdjudicationError(ValueError):
    """Raised when Sol adjudication cannot be completed safely."""


def _candidate_fact_payload(
    *, candidate_row: dict[str, object]
) -> dict[str, JSONValue]:
    if not isinstance(candidate_row, dict):
        raise SolAdjudicationError("Candidate row must be a mapping")
    facts: dict[str, JSONValue] = {}
    for field in sorted(SOL_ALLOWED_FACT_FIELDS):
        if field not in candidate_row:
            continue
        facts[field] = _safe_json_value(field=field, value=candidate_row[field])
    if not facts:
        raise SolAdjudicationError("Candidate row has no allowlisted facts")
    return facts


def _safe_json_value(*, field: str, value: object) -> JSONValue:
    if field in _RESTRICTED_FACT_FIELDS or "sidecar" in field:
        raise SolAdjudicationError("Candidate row contains an unsupported fact field")
    if isinstance(value, _ALLOWED_FACT_VALUE_TYPES):
        if isinstance(value, str):
            if len(value) > _MAX_FACT_VALUE_CHARS:
                raise SolAdjudicationError("Candidate fact value is outside bounds")
            _check_restricted_text(value=value, label="candidate fact value")
        return t.cast(JSONValue, value)
    if isinstance(value, list):
        if len(value) > 12:
            raise SolAdjudicationError("Candidate fact list is outside bounds")
        return [_safe_json_value(field=field, value=item) for item in value]
    raise SolAdjudicationError("Candidate fact value has an unsupported type")


def _check_restricted_text(*, value: str, label: str) -> None:
    if _RESTRICTED_TEXT.search(value):
        raise SolAdjudicationError(f"{label} contains a restricted identity term")


def _check_protected_tokens(
    *,
    original_persona: str,
    candidate_facts: dict[str, JSONValue],
    changed_fact_hints: dict[str, dict[str, JSONValue]],
    candidate_row: dict[str, object],
    original_row: dict[str, object] | None,
    protected_tokens: frozenset[str],
) -> None:
    allowed_age_tokens = _allowed_municipality_age_tokens(
        candidate_row=candidate_row, original_row=original_row
    )
    for token in protected_tokens:
        if token in allowed_age_tokens:
            _check_municipality_age_token(
                token=token,
                original_persona=original_persona,
                candidate_facts=candidate_facts,
                changed_fact_hints=changed_fact_hints,
            )
            continue
        _reject_protected_token(
            token=token,
            values=(
                original_persona,
                canonical_json(candidate_facts),
                canonical_json(changed_fact_hints),
            ),
        )


def _allowed_municipality_age_tokens(
    *, candidate_row: dict[str, object], original_row: dict[str, object] | None
) -> frozenset[str]:
    if original_row is None:
        return frozenset()
    token_fields = _protected_row_token_fields(
        candidate_row=candidate_row, original_row=original_row
    )
    allowed_tokens: set[str] = set()
    for token, fields in token_fields.items():
        if fields != {"municipality_code"} or not token.isdecimal():
            continue
        if not _row_ages_match_token(
            token=token, candidate_row=candidate_row, original_row=original_row
        ):
            continue
        allowed_tokens.add(token)
    return frozenset(allowed_tokens)


def _protected_row_token_fields(
    *, candidate_row: dict[str, object], original_row: dict[str, object] | None
) -> dict[str, frozenset[str]]:
    rows = (candidate_row,) if original_row is None else (candidate_row, original_row)
    token_fields: dict[str, set[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for field, value in row.items():
            field_name = str(field)
            if _is_protected_token_field(field=field_name):
                for token in _string_tokens(value=value):
                    token_fields.setdefault(token, set()).add(field_name)
    return {token: frozenset(fields) for token, fields in token_fields.items()}


def _is_protected_token_field(*, field: str) -> bool:
    return field in {"persona_id", "id"} or field.endswith("_code")


def _string_tokens(*, value: object) -> set[str]:
    if isinstance(value, str):
        token = value.strip()
        return {token} if len(token) >= 3 else set()
    if isinstance(value, int) and not isinstance(value, bool):
        token = str(value)
        return {token} if len(token) >= 3 else set()
    if isinstance(value, list):
        return {token for item in value for token in _string_tokens(value=item)}
    if isinstance(value, dict):
        return {
            token for item in value.values() for token in _string_tokens(value=item)
        }
    return set()


def _row_ages_match_token(
    *, token: str, candidate_row: dict[str, object], original_row: dict[str, object]
) -> bool:
    return _age_value_matches_token(
        value=original_row.get("age"), token=token
    ) and _age_value_matches_token(value=candidate_row.get("age"), token=token)


def _age_value_matches_token(*, value: object, token: str) -> bool:
    return (
        isinstance(value, int) and not isinstance(value, bool) and str(value) == token
    )


def _check_municipality_age_token(
    *,
    token: str,
    original_persona: str,
    candidate_facts: dict[str, JSONValue],
    changed_fact_hints: dict[str, dict[str, JSONValue]],
) -> None:
    if not _all_prose_token_occurrences_are_age(token=token, text=original_persona):
        raise SolAdjudicationError(
            "Outbound adjudication payload contains a restricted row token"
        )
    for field, value in candidate_facts.items():
        if not _fact_allows_age_token(field=field, value=value, token=token):
            raise SolAdjudicationError(
                "Outbound adjudication payload contains a restricted row token"
            )
    for field, pair in changed_fact_hints.items():
        if not _hint_allows_age_token(field=field, pair=pair, token=token):
            raise SolAdjudicationError(
                "Outbound adjudication payload contains a restricted row token"
            )


def _all_prose_token_occurrences_are_age(*, token: str, text: str) -> bool:
    for match in _token_occurrences(token=token, text=text):
        suffix = text[match.end() :]
        if not re.match(r"(?:\s|-)?år(?:ig)?(?![A-Za-zÆØÅæøå])", suffix):
            return False
    return True


def _token_occurrences(*, token: str, text: str) -> list[re.Match[str]]:
    pattern = rf"(?<!{_TOKEN_BOUNDARY}){re.escape(token)}(?!{_TOKEN_BOUNDARY})"
    return list(re.finditer(pattern, text))


def _fact_allows_age_token(*, field: str, value: JSONValue, token: str) -> bool:
    if not _json_value_contains_token(value=value, token=token):
        return True
    return field == "age" and _json_value_is_numeric_token(value=value, token=token)


def _json_value_contains_token(*, value: JSONValue, token: str) -> bool:
    return _token_in_text(token=token, text=canonical_json(value))


def _token_in_text(*, token: str, text: str) -> bool:
    return bool(_token_occurrences(token=token, text=text))


def _json_value_is_numeric_token(*, value: JSONValue, token: str) -> bool:
    return (
        isinstance(value, int) and not isinstance(value, bool) and str(value) == token
    )


def _hint_allows_age_token(
    *, field: str, pair: dict[str, JSONValue], token: str
) -> bool:
    if not _json_value_contains_token(value=pair, token=token):
        return True
    if field != "age":
        return False
    return all(
        _json_value_is_numeric_token(value=value, token=token)
        for value in pair.values()
        if _json_value_contains_token(value=value, token=token)
    )


def _reject_protected_token(*, token: str, values: tuple[str, ...]) -> None:
    for value in values:
        if _token_in_text(token=token, text=value):
            raise SolAdjudicationError(
                "Outbound adjudication payload contains a restricted row token"
            )


def _protected_row_tokens(
    *, candidate_row: dict[str, object], original_row: dict[str, object] | None
) -> frozenset[str]:
    return frozenset(
        _protected_row_token_fields(
            candidate_row=candidate_row, original_row=original_row
        )
    )


def _validated_adjudication_mode(*, adjudication_mode: object) -> SolAdjudicationMode:
    if adjudication_mode not in {"default", "unresolved_followup"}:
        raise SolAdjudicationError("Unsupported Sol adjudication mode")
    return t.cast(SolAdjudicationMode, adjudication_mode)


def _validated_changed_fact_hints(
    *,
    changed_fact_hints: dict[str, dict[str, object]],
    candidate_row: dict[str, object],
    original_row: dict[str, object] | None,
) -> dict[str, dict[str, JSONValue]]:
    hints = _normalise_changed_fact_hints(
        changed_fact_hints=changed_fact_hints,
        candidate_facts=_candidate_fact_payload(candidate_row=candidate_row),
    )
    if original_row is None:
        return hints
    for field, pair in hints.items():
        if field not in original_row or field not in candidate_row:
            raise SolAdjudicationError("Changed fact hint is not row-bound")
        if pair["old"] != original_row[field] or pair["new"] != candidate_row[field]:
            raise SolAdjudicationError("Changed fact hint is not row-bound")
    return hints


def _normalise_changed_fact_hints(
    *,
    changed_fact_hints: dict[str, dict[str, object]],
    candidate_facts: dict[str, JSONValue],
) -> dict[str, dict[str, JSONValue]]:
    if not isinstance(changed_fact_hints, dict):
        raise SolAdjudicationError("Changed fact hints must be a mapping")
    normalised: dict[str, dict[str, JSONValue]] = {}
    for field, pair in changed_fact_hints.items():
        if field not in SOL_ALLOWED_FACT_FIELDS or field not in candidate_facts:
            raise SolAdjudicationError("Changed fact hint uses an unsupported field")
        if not isinstance(pair, dict) or set(pair) != {"old", "new"}:
            raise SolAdjudicationError("Changed fact hint must contain old and new")
        old = _safe_json_value(field=field, value=pair["old"])
        new = _safe_json_value(field=field, value=pair["new"])
        if old == new or new != candidate_facts[field]:
            raise SolAdjudicationError("Changed fact hint is inconsistent")
        normalised[field] = {"old": old, "new": new}
    return normalised


def _build_binding(
    *,
    original_persona: str,
    candidate_row: dict[str, object],
    payload: dict[str, object],
    prompt: str,
    schema: dict[str, object],
    budget: ProxyBudget,
) -> dict[str, str | int]:
    schema_hash = _sha(canonical_json(schema).encode("utf-8"))
    prompt_hash = _sha(prompt.encode("utf-8"))
    if (
        budget.pins.get("model") != SOL_ADJUDICATION_MODEL
        or budget.pins.get("base_url") != BASE_URL
        or budget.pins.get("max_tokens") != SOL_ADJUDICATION_LEDGER_MAX_TOKENS
        or budget.pins.get("prompt_hash") != prompt_hash
        or budget.pins.get("schema_hash") != schema_hash
        or budget.pins.get("uncapped") is not True
        or budget.pins.get("uncapped_purpose")
        not in {
            SOL_ADJUDICATION_PURPOSE,
            V3_ADJUDICATION_PURPOSE,
            V3_EXTENDED_ADJUDICATION_PURPOSE,
            V3_LONG_ADJUDICATION_PURPOSE,
        }
    ):
        raise SolAdjudicationError("Sol budget pins do not match prompt and schema")
    try:
        candidate_hash = _sha(canonical_json(candidate_row).encode("utf-8"))
        payload_hash = _sha(canonical_json(payload).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise SolAdjudicationError("Sol inputs cannot be checksum-bound") from exc
    source_pin = budget.pins.get("source_hash")
    return {
        "checkpoint_version": _CHECKPOINT_VERSION,
        "original_persona_sha256": _sha(original_persona.encode("utf-8")),
        "candidate_row_sha256": candidate_hash,
        "prompt_sha256": prompt_hash,
        "schema_sha256": schema_hash,
        "payload_sha256": payload_hash,
        "model_sha256": _sha(SOL_ADJUDICATION_MODEL.encode("utf-8")),
        "base_url_sha256": _sha(BASE_URL.encode("utf-8")),
        "source_pin_sha256": str(source_pin) if isinstance(source_pin, str) else "",
    }


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _prepare_checkpoint_parent(*, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise SolAdjudicationError("Sol checkpoint directory is unsafe")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise SolAdjudicationError("Sol checkpoint directory is not private")


def _request_adjudication(
    *,
    prompt: str,
    payload: dict[str, object],
    schema: dict[str, object],
    binding: dict[str, str | int],
    config: GenerationConfig,
    budget: ProxyBudget,
    transport: httpx.BaseTransport,
) -> LLMResponse:
    request_body = _request_body(prompt=prompt, payload=payload, schema=schema)
    budget_transport = _BudgetedSolTransport(budget=budget, transport=transport)

    request_id_prefix = _request_id_prefix(binding=binding)
    attempt_map: dict[int, int] = {}

    def reserve(attempt: int) -> None:
        global_attempt = budget.reserve_next_attempt(
            request_id_prefix,
            t.cast(dict[str, JSONValue], request_body),
            max_output_tokens=SOL_MAX_OUTPUT_TOKENS,
            max_attempts=_MAX_ROW_ATTEMPTS,
        )
        attempt_map[attempt] = global_attempt
        request_id = _request_id_from_prefix(
            prefix=request_id_prefix, attempt=global_attempt
        )
        budget_transport.activate_request(request_id=request_id)

    client = OpenAIClient(config=config, transport=budget_transport)
    try:
        response = client.complete(
            system_prompt=prompt,
            user_payload=payload,
            schema_name=SOL_SCHEMA_NAME,
            json_schema=schema,
            record_request=reserve,
        )
    finally:
        client.close()
    global_attempt = attempt_map.get(response.request_attempts)
    if global_attempt is None:
        raise SolAdjudicationError("Sol response attempt was not reserved")
    response = response.model_copy(update={"request_attempts": global_attempt})
    request_id = _request_id(binding=binding, attempt=response.request_attempts)
    budget.record_usage(
        request_id,
        input_tokens=response.prompt_tokens,
        output_tokens=response.completion_tokens,
        response_sha256=response.raw_response_sha256,
    )
    if response.model != SOL_ADJUDICATION_MODEL:
        raise SolAdjudicationError("Provider response model does not match Sol")
    if response.prompt_tokens < 0 or response.completion_tokens < 0:
        raise SolAdjudicationError("Provider token usage is invalid")
    return response


class _BudgetedSolTransport(httpx.BaseTransport):
    """Validate the exact HTTP body against a durable reservation."""

    def __init__(self, *, budget: ProxyBudget, transport: httpx.BaseTransport) -> None:
        self.budget = budget
        self.transport = transport
        self._request_id: str | None = None

    def activate_request(self, *, request_id: str) -> None:
        self._request_id = request_id

    def close(self) -> None:
        self.transport.close()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if self._request_id is None:
            raise SolAdjudicationError("HTTP request was not reserved")
        self.budget.validate_request_body(self._request_id, request.content)
        return self.transport.handle_request(request)


def _request_body(
    *, prompt: str, payload: dict[str, object], schema: dict[str, object]
) -> dict[str, object]:
    return {
        "model": SOL_ADJUDICATION_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": SOL_SCHEMA_NAME, "strict": True, "schema": schema},
        },
        "reasoning_effort": "none",
    }


def _request_id(*, binding: dict[str, str | int], attempt: int) -> str:
    return _request_id_from_prefix(
        prefix=_request_id_prefix(binding=binding), attempt=attempt
    )


def _request_id_from_prefix(*, prefix: str, attempt: int) -> str:
    return f"{prefix}-{attempt}"


def _request_id_prefix(*, binding: dict[str, str | int]) -> str:
    digest = _sha(canonical_json(binding).encode("utf-8"))
    return f"sol-adjudication-{digest}"


def _save_checkpoint(
    *, path: Path, binding: dict[str, str | int], response: str
) -> None:
    _write_private_json(
        path=path,
        value={
            "binding": binding,
            "response": response,
            "response_sha256": _sha(response.encode("utf-8")),
        },
    )


def _write_private_json(*, path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class SolAdjudicationResult:
    """Locally validated adjudication for one persona row."""

    disposition: t.Literal["consistent", "patched", "unresolved"]
    original_text: str
    proposed_text: str
    changed_fraction: float
    reason: str
    evidence: tuple[SolEvidence, ...]
    patches: tuple[SolPatch, ...]


def run_sol_adjudication(
    *,
    original_persona: str,
    candidate_row: dict[str, object],
    prompt: str,
    config: GenerationConfig,
    budget: ProxyBudget,
    checkpoint_path: Path,
    transport: httpx.BaseTransport,
    changed_fact_hints: dict[str, dict[str, object]] | None = None,
    original_row: dict[str, object] | None = None,
    adjudication_mode: SolAdjudicationMode = "default",
) -> SolAdjudicationResult:
    """Run or resume one private per-row Sol adjudication.

    Args:
        original_persona:
            Original Danish Mistral persona prose to adjudicate.
        candidate_row:
            Current structured v5 candidate row. Only allowlisted fields are sent.
        prompt:
            Danish Sol adjudication instruction pinned by the budget ledger.
        config:
            Pinned private OpenAI-compatible model configuration.
        budget:
            Durable Sol ledger, configured uncapped for ``sol_adjudication``.
        checkpoint_path:
            Private checkpoint file for this row.
        transport:
            HTTPX transport. Tests pass a mock; production passes the local proxy.
        changed_fact_hints (optional):
            Verified old/new hints for allowlisted fact fields.
        original_row (optional):
            Original structured row used only to verify changed fact hints.
        adjudication_mode (optional):
            Use ``unresolved_followup`` to add the fixed follow-up decision
            rule to the user payload. Defaults to ``default``.

    Returns:
        Locally validated adjudication result.
    """
    _validate_config(config=config)
    payload = _validated_payload(
        original_persona=original_persona,
        candidate_row=candidate_row,
        prompt=prompt,
        changed_fact_hints=changed_fact_hints,
        original_row=original_row,
        adjudication_mode=adjudication_mode,
    )
    schema = SolAdjudicationResponse.provider_json_schema()
    binding = _build_binding(
        original_persona=original_persona,
        candidate_row=candidate_row,
        payload=payload,
        prompt=prompt,
        schema=schema,
        budget=budget,
    )
    checkpoint_path = Path(checkpoint_path)
    _prepare_checkpoint_parent(path=checkpoint_path.parent)
    if checkpoint_path.exists():
        return _resume_checkpoint(
            path=checkpoint_path,
            binding=binding,
            original_persona=original_persona,
            candidate_facts=t.cast(dict[str, JSONValue], payload["candidate_facts"]),
            changed_fact_hints=changed_fact_hints or {},
        )
    return _request_validated_adjudication(
        original_persona=original_persona,
        candidate_facts=t.cast(dict[str, JSONValue], payload["candidate_facts"]),
        changed_fact_hints=changed_fact_hints or {},
        prompt=prompt,
        payload=payload,
        schema=schema,
        binding=binding,
        config=config,
        budget=budget,
        checkpoint_path=checkpoint_path,
        transport=transport,
    )


def _request_validated_adjudication(
    *,
    original_persona: str,
    candidate_facts: dict[str, JSONValue],
    changed_fact_hints: dict[str, dict[str, object]],
    prompt: str,
    payload: dict[str, object],
    schema: dict[str, object],
    binding: dict[str, str | int],
    config: GenerationConfig,
    budget: ProxyBudget,
    checkpoint_path: Path,
    transport: httpx.BaseTransport,
) -> SolAdjudicationResult:
    for completion_attempt in range(_MAX_LOCAL_VALIDATION_COMPLETIONS):
        response = _request_adjudication(
            prompt=prompt,
            payload=payload,
            schema=schema,
            binding=binding,
            config=config,
            budget=budget,
            transport=transport,
        )
        try:
            result = validate_sol_adjudication(
                original_text=original_persona,
                candidate_facts=candidate_facts,
                response=response.content,
                changed_fact_hints=changed_fact_hints,
            )
        except SolAdjudicationError:
            if completion_attempt + 1 == _MAX_LOCAL_VALIDATION_COMPLETIONS:
                message = "Sol response failed bounded local validation retries"
                raise SolAdjudicationError(message) from None
            continue
        _save_checkpoint(
            path=checkpoint_path, binding=binding, response=response.content
        )
        return result
    message = "Sol response failed bounded local validation retries"
    raise SolAdjudicationError(message)


def validate_sol_adjudication(
    *,
    original_text: str,
    candidate_facts: dict[str, JSONValue],
    response: str | dict[str, object],
    changed_fact_hints: dict[str, dict[str, object]] | None = None,
) -> SolAdjudicationResult:
    """Validate one Sol response without trusting the provider.

    Args:
        original_text:
            Original persona prose.
        candidate_facts:
            Allowlisted facts sent to the provider.
        response:
            Provider JSON text or decoded mapping.
        changed_fact_hints (optional):
            Verified old/new hints used for negative checks.

    Returns:
        A mechanically checked adjudication result.

    Raises:
        SolAdjudicationError:
            If schema, grounding, or patch checks fail.
    """
    if not isinstance(original_text, str) or not original_text:
        raise SolAdjudicationError("Original persona prose is missing")
    _check_restricted_text(value=original_text, label="original persona prose")
    facts = _validate_candidate_facts(candidate_facts=candidate_facts)
    hints = _normalise_changed_fact_hints(
        changed_fact_hints=changed_fact_hints or {}, candidate_facts=facts
    )
    review = _parse_response(response=response)
    _validate_evidence(
        review=review, original_text=original_text, candidate_facts=facts
    )
    if review.disposition == "patched":
        proposed_text, changed_fraction = _apply_validated_patches(
            original_text=original_text, review=review, changed_fact_hints=hints
        )
    else:
        proposed_text = original_text
        changed_fraction = 0.0
    return SolAdjudicationResult(
        disposition=review.disposition,
        original_text=original_text,
        proposed_text=proposed_text,
        changed_fraction=changed_fraction,
        reason=review.reason,
        evidence=tuple(review.evidence),
        patches=tuple(review.patches),
    )


def _apply_validated_patches(
    *,
    original_text: str,
    review: SolAdjudicationResponse,
    changed_fact_hints: dict[str, dict[str, JSONValue]],
) -> tuple[str, float]:
    old_excerpts = [patch.old_excerpt for patch in review.patches]
    if len(old_excerpts) != len(set(old_excerpts)):
        raise SolAdjudicationError("Sol patches must use unique excerpts")
    edit_budget = max(1, int(len(original_text) * 0.2))
    changed_extent = 0
    proposed = original_text
    for patch in review.patches:
        _check_restricted_text(value=patch.new_excerpt, label="Sol patch")
        if patch.old_excerpt == patch.new_excerpt:
            raise SolAdjudicationError("Sol patch must change text")
        if proposed.count(patch.old_excerpt) != 1:
            raise SolAdjudicationError("Sol patch excerpt is not unique")
        _check_changed_hint_regression(
            new_excerpt=patch.new_excerpt, changed_fact_hints=changed_fact_hints
        )
        changed_extent += max(len(patch.old_excerpt), len(patch.new_excerpt))
        proposed = proposed.replace(patch.old_excerpt, patch.new_excerpt, 1)
    if changed_extent > edit_budget:
        raise SolAdjudicationError("Sol patch exceeds the local edit budget")
    return proposed, changed_extent / len(original_text)


def _check_changed_hint_regression(
    *, new_excerpt: str, changed_fact_hints: dict[str, dict[str, JSONValue]]
) -> None:
    folded = new_excerpt.casefold()
    for pair in changed_fact_hints.values():
        old_values = _claim_strings(value=pair["old"])
        new_values = _claim_strings(value=pair["new"])
        if any(old.casefold() in folded for old in old_values) and not any(
            new.casefold() in folded for new in new_values
        ):
            raise SolAdjudicationError("Sol patch reintroduces an outdated fact")


def _claim_strings(*, value: JSONValue) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, bool):
        return (str(value).casefold(),)
    if isinstance(value, (int, float)):
        return (str(value),)
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, list):
        claims: list[str] = []
        for item in value:
            claims.extend(_claim_strings(value=item))
        return tuple(claims)
    return ()


def _parse_response(*, response: str | dict[str, object]) -> SolAdjudicationResponse:
    try:
        payload = json.loads(response) if isinstance(response, str) else response
        if not isinstance(payload, dict):
            raise ValueError("response is not an object")
        return SolAdjudicationResponse.model_validate(payload)
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
        raise SolAdjudicationError(
            "Sol response failed strict local validation"
        ) from exc


def _validate_candidate_facts(
    *, candidate_facts: dict[str, JSONValue]
) -> dict[str, JSONValue]:
    if not isinstance(candidate_facts, dict) or not candidate_facts:
        raise SolAdjudicationError("Candidate facts are missing")
    normalised: dict[str, JSONValue] = {}
    for field, value in candidate_facts.items():
        if field not in SOL_ALLOWED_FACT_FIELDS:
            raise SolAdjudicationError("Candidate facts include an unsupported field")
        normalised[field] = _safe_json_value(field=field, value=value)
    return normalised


def _validate_evidence(
    *,
    review: SolAdjudicationResponse,
    original_text: str,
    candidate_facts: dict[str, JSONValue],
) -> None:
    for item in review.evidence:
        if item.field not in candidate_facts:
            raise SolAdjudicationError("Sol evidence references an unsupported field")
        if item.quote not in original_text or original_text.count(item.quote) != 1:
            raise SolAdjudicationError("Sol evidence is not uniquely grounded")
        if item.kind == "fact_present" and not _quote_supports_fact(
            quote=item.quote, value=candidate_facts[item.field]
        ):
            raise SolAdjudicationError("Sol evidence does not ground the fact")


def _quote_supports_fact(*, quote: str, value: JSONValue) -> bool:
    folded = quote.casefold()
    return any(claim.casefold() in folded for claim in _claim_strings(value=value))


def _resume_checkpoint(
    *,
    path: Path,
    binding: dict[str, str | int],
    original_persona: str,
    candidate_facts: dict[str, JSONValue],
    changed_fact_hints: dict[str, dict[str, object]],
) -> SolAdjudicationResult:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SolAdjudicationError("Sol checkpoint is not readable") from exc
    if not isinstance(saved, dict) or saved.get("binding") != binding:
        raise SolAdjudicationError(
            "Sol checkpoint binding does not match current inputs"
        )
    response = saved.get("response")
    response_sha256 = saved.get("response_sha256")
    if not isinstance(response, str) or not isinstance(response_sha256, str):
        raise SolAdjudicationError("Sol checkpoint is incomplete")
    if response_sha256 != _sha(response.encode("utf-8")):
        raise SolAdjudicationError("Sol checkpoint response hash does not match")
    return validate_sol_adjudication(
        original_text=original_persona,
        candidate_facts=candidate_facts,
        response=response,
        changed_fact_hints=changed_fact_hints,
    )


def _validate_config(*, config: GenerationConfig) -> None:
    if (
        config.base_url != BASE_URL
        or config.model != SOL_ADJUDICATION_MODEL
        or config.max_tokens is not None
        or config.reasoning_effort != "none"
        or config.enable_thinking is not None
        or not 1 <= config.maximum_http_attempts <= 5
    ):
        raise SolAdjudicationError("Generation configuration is not pinned for Sol")
