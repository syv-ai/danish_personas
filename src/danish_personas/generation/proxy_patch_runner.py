"""Offline-testable, privacy-bounded local proxy prose patch runner."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .client import OpenAIClient
from .models import GenerationConfig, LLMResponse
from .prose_patch import ProsePatchError, ProsePatchResponse, apply_patches
from .proxy_budget import BASE_URL, MODEL, ProxyBudget

_ALLOWED_FACTS = frozenset(
    {
        "marital_status",
        "legal_status_detail",
        "education_level",
        "labour_market_status",
        "detailed_status",
        "job_function",
        "job_title",
        "current_relationship_status",
        "hobbies_and_interests",
        "skills_and_expertise",
        "age",
    }
)
_SENSITIVE_TEXT = re.compile(
    r"\b(?:sexual\s+orientation|seksuel\s+orientering|seksual\s+orientation|"
    r"homoseksuel|biseksuel|transkønnet|transseksuel|interkønnet|"
    r"transgender|partner_transgender|variation\s+in\s+sex\s+characteristics|"
    r"intersex|partner_sexual_orientation|sexuality)\b",
    re.IGNORECASE,
)


class ProxyPatchError(ValueError):
    """Raised when a proxy patch cannot safely be proposed."""


@dataclass(frozen=True)
class ProxyPatchProposal:
    """A provisional prose change for independent review."""

    persona_text: str
    changed_fraction: float
    evidence: tuple[dict[str, str], ...]
    checkpoint_path: Path


class _BoundedTransport(httpx.BaseTransport):
    """Reject a request whose actual body exceeds its durable reservation."""

    def __init__(self, transport: httpx.BaseTransport, limit: int) -> None:
        self.transport = transport
        self.limit = limit

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if len(request.content) > self.limit:
            raise ProxyPatchError("Constructed HTTP body exceeds reserved input bound")
        return self.transport.handle_request(request)

    def close(self) -> None:
        self.transport.close()


def run_proxy_patch(
    *,
    row: dict[str, Any],
    changed_facts: dict[str, dict[str, object]],
    gender: str | None,
    partner_gender: str | None,
    prompt: str,
    config: GenerationConfig,
    budget: ProxyBudget,
    checkpoint_path: Path,
    transport: httpx.BaseTransport,
) -> ProxyPatchProposal:
    """Return a provisional patch, never modifying the input row or dataset."""
    _validate_config(config)
    old_text, payload = _validated_input(
        row, changed_facts, gender, partner_gender, prompt
    )
    schema = ProsePatchResponse.provider_json_schema()
    binding = _build_binding(row, old_text, payload, prompt, schema, budget)
    checkpoint_path = Path(checkpoint_path)
    _prepare_checkpoint_parent(checkpoint_path.parent)
    if checkpoint_path.exists():
        return _resume_checkpoint(checkpoint_path, binding, old_text)
    response = _request_patch(
        prompt, payload, schema, binding, config, budget, transport
    )
    return _save_proposal(checkpoint_path, response, old_text, binding)


def _validated_input(
    row: dict[str, Any],
    changed_facts: dict[str, dict[str, object]],
    gender: str | None,
    partner_gender: str | None,
    prompt: str,
) -> tuple[str, dict[str, object]]:
    if not prompt.strip():
        raise ProxyPatchError("Prompt must not be empty")
    old_text = row.get("persona")
    if not isinstance(old_text, str) or not 300 <= len(old_text) <= 900:
        raise ProxyPatchError("Persona text is missing or outside the supported range")
    if _SENSITIVE_TEXT.search(old_text):
        raise ProxyPatchError("Persona text contains a sensitive-identity term")
    if not changed_facts or set(changed_facts) - _ALLOWED_FACTS:
        raise ProxyPatchError("Changed facts contain fields outside the allowlist")
    for field, pair in changed_facts.items():
        if not isinstance(pair, dict) or set(pair) != {"old", "new"}:
            raise ProxyPatchError(f"Changed fact {field!r} must contain old and new")
        if any(not _safe_fact_value(value) for value in pair.values()):
            raise ProxyPatchError(f"Changed fact {field!r} contains an unsafe value")
    payload: dict[str, object] = {
        "persona": old_text,
        "changed_facts": changed_facts,
    }
    if gender is not None:
        payload["gender"] = gender
    if partner_gender is not None:
        payload["partner_gender"] = partner_gender
    return old_text, payload


def _safe_fact_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return abs(value) <= 150
    if isinstance(value, str):
        return len(value) <= 120 and _SENSITIVE_TEXT.search(value) is None
    return False


def _build_binding(
    row: dict[str, Any],
    old_text: str,
    payload: dict[str, object],
    prompt: str,
    schema: dict[str, object],
    budget: ProxyBudget,
) -> dict[str, str]:
    schema_hash = _sha(_canonical(schema))
    prompt_hash = _sha(prompt.encode("utf-8"))
    if (
        budget.pins.get("model") != MODEL
        or budget.pins.get("base_url") != BASE_URL
        or budget.pins.get("max_tokens") != 128_000
        or budget.pins.get("prompt_hash") != prompt_hash
        or budget.pins.get("schema_hash") != schema_hash
    ):
        raise ProxyPatchError("Proxy budget pins do not match prompt and schema")
    try:
        row_hash = _sha(_canonical(row))
    except (TypeError, ValueError) as exc:
        raise ProxyPatchError("Source row cannot be checksum-bound") from exc
    return {
        "source_sha256": _sha(old_text.encode("utf-8")),
        "row_sha256": row_hash,
        "facts_sha256": _sha(_canonical(payload)),
        "prompt_sha256": prompt_hash,
        "schema_sha256": schema_hash,
        "model": MODEL,
    }


def _prepare_checkpoint_parent(parent: Path) -> None:
    missing: list[Path] = []
    current = parent
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
    if parent.stat().st_mode & 0o077:
        raise ProxyPatchError("Checkpoint directory must be private (mode 0700)")


def _resume_checkpoint(
    path: Path, binding: dict[str, str], old_text: str
) -> ProxyPatchProposal:
    checkpoint = _read_checkpoint(path)
    if any(checkpoint.get(key) != value for key, value in binding.items()):
        raise ProxyPatchError("Checkpoint inputs, prompt, schema, or model changed")
    proposed = checkpoint.get("proposed_persona_text")
    evidence = checkpoint.get("evidence")
    fraction = checkpoint.get("changed_fraction")
    if not isinstance(proposed, str) or not isinstance(evidence, list):
        raise ProxyPatchError("Provisional checkpoint is malformed")
    parsed_evidence = _validate_evidence(evidence)
    try:
        recomputed = apply_patches(old_text, parsed_evidence)
    except ProsePatchError as exc:
        raise ProxyPatchError("Provisional checkpoint evidence is invalid") from exc
    actual_fraction = sum(
        len(item["old_excerpt"]) for item in parsed_evidence
    ) / len(old_text)
    if (
        proposed != recomputed
        or not isinstance(fraction, (int, float))
        or fraction != actual_fraction
    ):
        raise ProxyPatchError(
            "Provisional checkpoint does not match its recorded evidence"
        )
    digest = checkpoint.get("checkpoint_sha256")
    unsigned = {
        key: value
        for key, value in checkpoint.items()
        if key != "checkpoint_sha256"
    }
    if digest != _sha(_canonical(unsigned)):
        raise ProxyPatchError("Provisional checkpoint checksum is invalid")
    return ProxyPatchProposal(proposed, actual_fraction, tuple(parsed_evidence), path)


def _validate_evidence(evidence: list[object]) -> list[dict[str, str]]:
    validated: list[dict[str, str]] = []
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"old_excerpt", "new_excerpt"}:
            raise ProxyPatchError("Provisional checkpoint evidence is malformed")
        old, new = item["old_excerpt"], item["new_excerpt"]
        if not isinstance(old, str) or not isinstance(new, str):
            raise ProxyPatchError("Provisional checkpoint evidence is malformed")
        validated.append({"old_excerpt": old, "new_excerpt": new})
    return validated


def _request_patch(
    prompt: str,
    payload: dict[str, object],
    schema: dict[str, object],
    binding: dict[str, str],
    config: GenerationConfig,
    budget: ProxyBudget,
    transport: httpx.BaseTransport,
) -> LLMResponse:
    request_body = _request_body(prompt, payload, schema)
    reservation_bytes = len(_canonical(request_body)) + budget.overhead
    request_id = _sha(_canonical(binding))
    reserved = False

    def reserve(attempt: int) -> None:
        nonlocal reserved
        if attempt != 1 or reserved:
            raise ProxyPatchError("Only one HTTP attempt is permitted")
        budget.reserve_attempt(request_id, request_body)
        reserved = True

    client = OpenAIClient(
        config=config, transport=_BoundedTransport(transport, reservation_bytes)
    )
    try:
        response = client.complete(
            system_prompt=prompt, user_payload=payload, schema_name="prose_patch",
            json_schema=schema, record_request=reserve,
        )
    finally:
        client.close()
    if not reserved:
        raise ProxyPatchError("Request was not durably reserved")
    if response.model != MODEL:
        raise ProxyPatchError("Provider response model does not match the pinned model")
    if response.completion_tokens > 128_000 or response.prompt_tokens < 0:
        raise ProxyPatchError("Provider token usage exceeds the reservation policy")
    budget.record_usage(
        request_id, input_tokens=response.prompt_tokens,
        output_tokens=response.completion_tokens, response=response.raw_response_sha256,
    )
    return response


def _save_proposal(
    path: Path, response: LLMResponse, old_text: str, binding: dict[str, str]
) -> ProxyPatchProposal:
    try:
        patched = apply_patches(old_text, response.content)
    except ProsePatchError as exc:
        raise ProxyPatchError("Provider patch failed local validation") from exc
    parsed = ProsePatchResponse.model_validate_json(response.content)
    fraction = sum(len(patch.old_excerpt) for patch in parsed.patches) / len(old_text)
    evidence = tuple(
        {"old_excerpt": patch.old_excerpt, "new_excerpt": patch.new_excerpt}
        for patch in parsed.patches
    )
    document: dict[str, object] = {
        **binding, "proposed_persona_text": patched,
        "changed_fraction": fraction, "evidence": list(evidence),
    }
    document["checkpoint_sha256"] = _sha(_canonical(document))
    _write_checkpoint(path, document)
    return ProxyPatchProposal(patched, fraction, evidence, path)


def _validate_config(config: GenerationConfig) -> None:
    if (
        config.base_url != BASE_URL
        or config.model != MODEL
        or config.max_tokens is not None
        or config.reasoning_effort != "none"
        or config.maximum_http_attempts != 1
        or config.enable_thinking is not None
    ):
        raise ProxyPatchError("Generation configuration is not the pinned local proxy")


def _request_body(
    prompt: str, payload: dict[str, object], schema: dict[str, object]
) -> dict[str, object]:
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "prose_patch", "strict": True, "schema": schema},
        },
        "reasoning_effort": "none",
    }


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_checkpoint(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ProxyPatchError("Provisional checkpoint is unreadable") from exc


def _write_checkpoint(path: Path, value: dict[str, object]) -> None:
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
