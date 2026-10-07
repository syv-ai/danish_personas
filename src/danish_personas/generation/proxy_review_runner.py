"""Offline-testable, privacy-bounded local proxy prose review runner."""

from __future__ import annotations

import hashlib
import json
import os
import typing as t
import uuid
from pathlib import Path

import httpx

from .client import OpenAIClient
from .models import GenerationConfig, LLMResponse
from .prose_review import ProseReviewError as LocalProseReviewError
from .prose_review import ProseReviewResponse, ProseReviewResult, validate_prose_review
from .proxy_budget import BASE_URL, MODEL, JSONValue, ProxyBudget
from .proxy_patch_runner import ProxyPatchError, _validate_config, _validated_input

_CHECKPOINT_VERSION = 1
_SCHEMA_NAME = "prose_review"


def run_proxy_review(
    *,
    row: dict[str, t.Any],
    candidate_row: dict[str, t.Any],
    changed_facts: dict[str, dict[str, object]],
    prompt: str,
    config: GenerationConfig,
    budget: ProxyBudget,
    checkpoint_path: Path,
    transport: httpx.BaseTransport,
) -> ProseReviewResult:
    """Run or resume one privacy-bounded local proxy prose review.

    Args:
        row:
            Original generated row with the unchanged Mistral persona prose.
        candidate_row:
            Candidate row whose allowlisted changed facts have been verified.
        changed_facts:
            Allowlisted old/new fact deltas to review against the prose.
        prompt:
            Review prompt pinned by the proxy budget.
        config:
            Pinned local proxy generation configuration.
        budget:
            Durable proxy budget gate.
        checkpoint_path:
            Private checkpoint path for this review decision.
        transport:
            HTTPX transport, usually a mock in tests or the local proxy.

    Returns:
        Locally validated prose-review result.

    Raises:
        ProxyReviewError:
            If inputs, checkpoint state, or proxy accounting are unsafe.
    """
    try:
        _validate_config(config)
        original_text, payload = _validated_input(
            row, candidate_row, changed_facts, None, None, prompt
        )
    except ProxyPatchError as exc:
        raise ProxyReviewError(str(exc)) from exc

    schema = ProseReviewResponse.provider_json_schema()
    binding = _build_binding(
        row=row,
        candidate_row=candidate_row,
        original_text=original_text,
        changed_facts=changed_facts,
        prompt=prompt,
        schema=schema,
        budget=budget,
    )
    checkpoint_path = Path(checkpoint_path)
    _prepare_checkpoint_parent(checkpoint_path.parent)
    if checkpoint_path.exists():
        return _resume_checkpoint(
            path=checkpoint_path,
            binding=binding,
            original_text=original_text,
            changed_facts=changed_facts,
        )

    response = _request_review(
        prompt=prompt,
        payload=payload,
        schema=schema,
        binding=binding,
        config=config,
        budget=budget,
        transport=transport,
    )
    try:
        result = validate_prose_review(
            original_text=original_text,
            changed_facts=changed_facts,
            response=response.content,
        )
    except LocalProseReviewError:
        result = _insufficient_evidence_result(
            original_text=original_text, changed_facts=changed_facts
        )
    _save_checkpoint(path=checkpoint_path, binding=binding, result=result)
    return result


class ProxyReviewError(ValueError):
    """Raised when a proxy review cannot safely be completed."""


def _build_binding(
    *,
    row: dict[str, t.Any],
    candidate_row: dict[str, t.Any],
    original_text: str,
    changed_facts: dict[str, dict[str, object]],
    prompt: str,
    schema: dict[str, object],
    budget: ProxyBudget,
) -> dict[str, str | int]:
    schema_hash = _sha(_canonical(schema))
    prompt_hash = _sha(prompt.encode("utf-8"))
    campaign = budget.pins.get("campaign")
    if (
        budget.pins.get("model") != MODEL
        or budget.pins.get("base_url") != BASE_URL
        or budget.pins.get("max_tokens") != 128_000
        or budget.pins.get("prompt_hash") != prompt_hash
        or budget.pins.get("schema_hash") != schema_hash
        or not isinstance(campaign, str)
    ):
        raise ProxyReviewError("Proxy budget pins do not match prompt and schema")
    try:
        row_hash = _sha(_canonical(row))
        candidate_hash = _sha(_canonical(candidate_row))
        facts_hash = _sha(_canonical(changed_facts))
    except (TypeError, ValueError) as exc:
        raise ProxyReviewError("Review inputs cannot be checksum-bound") from exc
    source_pin = budget.pins.get("source_hash")
    return {
        "checkpoint_version": _CHECKPOINT_VERSION,
        "original_text_sha256": _sha(original_text.encode("utf-8")),
        "row_sha256": row_hash,
        "candidate_row_sha256": candidate_hash,
        "changed_facts_sha256": facts_hash,
        "prompt_sha256": prompt_hash,
        "schema_sha256": schema_hash,
        "campaign_sha256": _sha(campaign.encode("utf-8")),
        "model_sha256": _sha(MODEL.encode("utf-8")),
        "base_url_sha256": _sha(BASE_URL.encode("utf-8")),
        "source_pin_sha256": str(source_pin) if isinstance(source_pin, str) else "",
    }


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _insufficient_evidence_result(
    *, original_text: str, changed_facts: dict[str, dict[str, object]]
) -> ProseReviewResult:
    return validate_prose_review(
        original_text=original_text,
        changed_facts=changed_facts,
        response={
            "disposition": "needs_manual_review",
            "patches": [],
            "unchanged_evidence": [],
            "manual_review_reason": "insufficient_evidence",
        },
    )


def _prepare_checkpoint_parent(parent: Path) -> None:
    missing: list[Path] = []
    current = parent
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
    if parent.stat().st_mode & 0o077:
        raise ProxyReviewError("Checkpoint directory must be private (mode 0700)")


def _request_review(
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
    reservation_bytes = len(_canonical(request_body)) + budget.overhead
    request_id = _request_id(binding=binding)
    reserved = False

    def reserve(attempt: int) -> None:
        nonlocal reserved
        if attempt != 1 or reserved:
            raise ProxyReviewError("Only one HTTP attempt is permitted")
        budget.reserve_attempt(request_id, t.cast(dict[str, JSONValue], request_body))
        reserved = True

    client = OpenAIClient(
        config=config, transport=_BoundedTransport(transport, reservation_bytes)
    )
    try:
        response = client.complete(
            system_prompt=prompt,
            user_payload=payload,
            schema_name=_SCHEMA_NAME,
            json_schema=schema,
            record_request=reserve,
        )
    finally:
        client.close()
    if not reserved:
        raise ProxyReviewError("Request was not durably reserved")
    budget.record_usage(
        request_id,
        input_tokens=response.prompt_tokens,
        output_tokens=response.completion_tokens,
        response_sha256=response.raw_response_sha256,
    )
    if response.model != MODEL:
        raise ProxyReviewError(
            "Provider response model does not match the pinned model"
        )
    if response.completion_tokens > 128_000 or response.prompt_tokens < 0:
        raise ProxyReviewError("Provider token usage exceeds the reservation policy")
    return response


class _BoundedTransport(httpx.BaseTransport):
    """Reject a request whose actual body exceeds its durable reservation."""

    def __init__(self, transport: httpx.BaseTransport, limit: int) -> None:
        self.transport = transport
        self.limit = limit

    def close(self) -> None:
        self.transport.close()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if len(request.content) > self.limit:
            raise ProxyReviewError("Constructed HTTP body exceeds reserved input bound")
        return self.transport.handle_request(request)


def _request_body(
    *, prompt: str, payload: dict[str, object], schema: dict[str, object]
) -> dict[str, object]:
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": _SCHEMA_NAME, "strict": True, "schema": schema},
        },
        "reasoning_effort": "none",
    }


def _request_id(*, binding: dict[str, str | int]) -> str:
    binding_hash = _sha(_canonical(binding))[:16]
    return f"review-{binding_hash}-{uuid.uuid4().hex}"


def _resume_checkpoint(
    *,
    path: Path,
    binding: dict[str, str | int],
    original_text: str,
    changed_facts: dict[str, dict[str, object]],
) -> ProseReviewResult:
    _require_private_checkpoint(path)
    checkpoint = _read_checkpoint(path)
    expected_keys = _checkpoint_keys()
    if set(checkpoint) != expected_keys:
        raise ProxyReviewError("Review checkpoint is malformed")
    if any(checkpoint.get(key) != value for key, value in binding.items()):
        raise ProxyReviewError("Checkpoint inputs, prompt, schema, or model changed")
    digest = checkpoint.get("checkpoint_sha256")
    unsigned = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    if digest != _sha(_canonical(unsigned)):
        raise ProxyReviewError("Review checkpoint checksum is invalid")
    response = _checkpoint_response(checkpoint)
    try:
        result = validate_prose_review(
            original_text=original_text, changed_facts=changed_facts, response=response
        )
    except LocalProseReviewError as exc:
        raise ProxyReviewError("Review checkpoint decision is invalid") from exc
    _verify_checkpoint_result(checkpoint=checkpoint, result=result)
    return result


def _checkpoint_keys() -> set[str]:
    return {
        "checkpoint_version",
        "original_text_sha256",
        "row_sha256",
        "candidate_row_sha256",
        "changed_facts_sha256",
        "prompt_sha256",
        "schema_sha256",
        "campaign_sha256",
        "model_sha256",
        "base_url_sha256",
        "source_pin_sha256",
        "disposition",
        "changed_fraction",
        "proposed_text_sha256",
        "patches",
        "unchanged_evidence",
        "manual_review_reason",
        "unchanged_consistent_note",
        "checkpoint_sha256",
    }


def _checkpoint_response(checkpoint: dict[str, object]) -> dict[str, object]:
    patches = checkpoint.get("patches")
    evidence = checkpoint.get("unchanged_evidence")
    if not isinstance(patches, list) or not isinstance(evidence, list):
        raise ProxyReviewError("Review checkpoint is malformed")
    return {
        "disposition": checkpoint.get("disposition"),
        "patches": patches,
        "unchanged_evidence": evidence,
        "manual_review_reason": checkpoint.get("manual_review_reason"),
    }


def _read_checkpoint(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError
        return t.cast(dict[str, object], value)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ProxyReviewError("Review checkpoint is unreadable") from exc


def _require_private_checkpoint(path: Path) -> None:
    if path.stat().st_mode & 0o777 != 0o600:
        raise ProxyReviewError("Review checkpoint must be private (mode 0600)")


def _verify_checkpoint_result(
    *, checkpoint: dict[str, object], result: ProseReviewResult
) -> None:
    if (
        checkpoint.get("disposition") != result.disposition
        or checkpoint.get("changed_fraction") != result.changed_fraction
        or checkpoint.get("proposed_text_sha256")
        != _sha(result.proposed_text.encode("utf-8"))
        or checkpoint.get("manual_review_reason") != result.manual_review_reason
        or checkpoint.get("unchanged_consistent_note")
        != result.unchanged_consistent_note
    ):
        raise ProxyReviewError("Review checkpoint does not match its decision")


def _save_checkpoint(
    *, path: Path, binding: dict[str, str | int], result: ProseReviewResult
) -> None:
    document: dict[str, object] = {
        **binding,
        "disposition": result.disposition,
        "changed_fraction": result.changed_fraction,
        "proposed_text_sha256": _sha(result.proposed_text.encode("utf-8")),
        "patches": [
            {"old_excerpt": patch.old_excerpt, "new_excerpt": patch.new_excerpt}
            for patch in result.patches
        ],
        "unchanged_evidence": [
            {"field": item.field, "kind": item.kind, "quote": item.quote}
            for item in result.unchanged_evidence
        ],
        "manual_review_reason": result.manual_review_reason,
        "unchanged_consistent_note": result.unchanged_consistent_note,
    }
    document["checkpoint_sha256"] = _sha(_canonical(document))
    _write_checkpoint(path, document)


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


__all__ = ["ProxyReviewError", "run_proxy_review"]
