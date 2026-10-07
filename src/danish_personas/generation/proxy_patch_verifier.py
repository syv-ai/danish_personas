"""Local-proxy verifier for first-pass persona prose patches."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import typing as t
import uuid
from pathlib import Path

import httpx

from ..release.prose_patch_verification import (
    ProsePatchSecondReview,
    ProsePatchVerificationResult,
    verify_prose_patch_proposal,
)
from ..release.prose_patch_verification import (
    ProsePatchVerificationError as LocalVerificationError,
)
from .client import OpenAIClient
from .models import GenerationConfig, LLMResponse
from .proxy_budget import (
    BASE_URL,
    EDUCATION_VERIFICATION_PURPOSE,
    MODEL,
    PATCH_VERIFICATION_PURPOSE,
    JSONValue,
    ProxyBudget,
)
from .proxy_patch_runner import (
    ProxyPatchError,
    _contains_sensitive_text,
    _fact_text,
    _validated_input,
)

_CHECKPOINT_VERSION = 1
_SCHEMA_NAME = "prose_patch_verification"
_VERIFICATION_PURPOSES = frozenset(
    {PATCH_VERIFICATION_PURPOSE, EDUCATION_VERIFICATION_PURPOSE}
)
_LEGAL_STATUS_DETAIL_NULL_CONTEXT = "legal_status_detail_null_context"
_NULL_DETAIL_MARITAL_STATUS_DA = {
    "divorced": "skilt",
    "widowed": "enkestand",
    "never_married": "aldrig gift",
}
_NULL_DETAIL_EXPLANATION_DA = (
    "legal_status_detail = null betyder, at en ikke-understoettet fin detalje er "
    "fjernet. Den aktuelle kildeunderstoettede civilstandskategori her er maalet; "
    "udled ingen nye personlige traek."
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


def run_proxy_patch_verification(
    *,
    row: dict[str, object],
    candidate_row: dict[str, object],
    changed_facts: dict[str, dict[str, object]],
    proposed_text: str,
    patches: list[dict[str, str]],
    first_checkpoint_sha256: str,
    prompt: str,
    config: GenerationConfig,
    budget: ProxyBudget,
    checkpoint_path: Path,
    transport: httpx.BaseTransport,
) -> ProsePatchVerificationResult:
    """Run or resume a private second-pass local-proxy patch verification.

    Args:
        row:
            Original generated row whose persona prose was patched.
        candidate_row:
            Candidate row containing only source-verified structured changes.
        changed_facts:
            Source-verified old/new changed facts.
        proposed_text:
            First-pass proposed persona prose after exact patch application.
        patches:
            Exact old/new patch snippets from the first pass.
        first_checkpoint_sha256:
            SHA-256 of the first-pass patch checkpoint being verified.
        prompt:
            Danish verifier prompt pinned by the proxy budget.
        config:
            Pinned local proxy generation configuration.
        budget:
            Durable uncapped patch-verification budget.
        checkpoint_path:
            Private checkpoint path for the verifier verdict.
        transport:
            HTTPX transport, normally a mock in tests or the local proxy.

    Returns:
        Locally revalidated prose-patch verification result.

    Raises:
        ProxyPatchVerificationError:
            If inputs, privacy checks, checkpoint state, or accounting are unsafe.
    """
    try:
        _validate_config(config=config)
        original_text, base_payload = _validated_input(
            row, candidate_row, changed_facts, None, None, prompt
        )
    except ProxyPatchError as exc:
        raise ProxyPatchVerificationError(str(exc)) from exc
    _validate_checkpoint_sha(first_checkpoint_sha256=first_checkpoint_sha256)
    _scan_verifier_text(
        proposed_text=proposed_text,
        patches=patches,
        changed_facts=changed_facts,
        prompt=prompt,
    )
    _prevalidate_patch_inputs(
        original_text=original_text,
        changed_facts=changed_facts,
        proposed_text=proposed_text,
        patches=patches,
        first_checkpoint_sha256=first_checkpoint_sha256,
    )

    schema = ProsePatchSecondReview.provider_json_schema()
    payload = _payload(
        original_text=original_text,
        proposed_text=proposed_text,
        patches=patches,
        changed_facts=changed_facts,
        base_payload=base_payload,
        row=row,
        candidate_row=candidate_row,
    )
    binding = _build_binding(
        row=row,
        candidate_row=candidate_row,
        original_text=original_text,
        proposed_text=proposed_text,
        patches=patches,
        changed_facts=changed_facts,
        first_checkpoint_sha256=first_checkpoint_sha256,
        prompt=prompt,
        schema=schema,
        payload=payload,
        budget=budget,
    )
    checkpoint_path = Path(checkpoint_path)
    _prepare_checkpoint_parent(parent=checkpoint_path.parent)
    if checkpoint_path.exists():
        return _resume_checkpoint(
            path=checkpoint_path,
            binding=binding,
            original_text=original_text,
            changed_facts=changed_facts,
            proposed_text=proposed_text,
            patches=patches,
            first_checkpoint_sha256=first_checkpoint_sha256,
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
        result = verify_prose_patch_proposal(
            original_text=original_text,
            changed_facts=changed_facts,
            proposed_text=proposed_text,
            patches=patches,
            second_review=response.content,
            original_checkpoint_sha256=first_checkpoint_sha256,
        )
    except LocalVerificationError:
        result = _conservative_result(
            original_text=original_text,
            changed_facts=changed_facts,
            proposed_text=proposed_text,
            patches=patches,
            first_checkpoint_sha256=first_checkpoint_sha256,
        )
    _save_checkpoint(path=checkpoint_path, binding=binding, result=result)
    return result


class ProxyPatchVerificationError(ValueError):
    """Raised when a proxy patch verification cannot be completed safely."""


def _build_binding(
    *,
    row: dict[str, object],
    candidate_row: dict[str, object],
    original_text: str,
    proposed_text: str,
    patches: list[dict[str, str]],
    changed_facts: dict[str, dict[str, object]],
    first_checkpoint_sha256: str,
    prompt: str,
    schema: dict[str, object],
    payload: dict[str, object],
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
        or not budget.uncapped
        or budget.uncapped_purpose not in _VERIFICATION_PURPOSES
        or budget.pins.get("uncapped_purpose") != budget.uncapped_purpose
    ):
        raise ProxyPatchVerificationError(
            "Proxy budget pins do not match patch-verification prompt and schema"
        )
    try:
        row_hash = _sha(_canonical(row))
        candidate_hash = _sha(_canonical(candidate_row))
        facts_hash = _sha(_canonical(changed_facts))
        patch_hash = _sha(_canonical(patches))
        payload_hash = _sha(_canonical(payload))
    except (TypeError, ValueError) as exc:
        raise ProxyPatchVerificationError(
            "Verification inputs cannot be checksum-bound"
        ) from exc
    source_pin = budget.pins.get("source_hash")
    return {
        "checkpoint_version": _CHECKPOINT_VERSION,
        "first_checkpoint_sha256": first_checkpoint_sha256.lower(),
        "original_text_sha256": _sha(original_text.encode("utf-8")),
        "proposed_text_sha256": _sha(proposed_text.encode("utf-8")),
        "row_sha256": row_hash,
        "candidate_row_sha256": candidate_hash,
        "changed_facts_sha256": facts_hash,
        "patches_sha256": patch_hash,
        "payload_sha256": payload_hash,
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


def _conservative_result(
    *,
    original_text: str,
    changed_facts: dict[str, dict[str, object]],
    proposed_text: str,
    patches: list[dict[str, str]],
    first_checkpoint_sha256: str,
) -> ProsePatchVerificationResult:
    return verify_prose_patch_proposal(
        original_text=original_text,
        changed_facts=changed_facts,
        proposed_text=proposed_text,
        patches=patches,
        second_review={
            "verdict": "needs_manual_review",
            "reasons": ["invalid_quote_evidence"],
            "fact_evidence": [],
        },
        original_checkpoint_sha256=first_checkpoint_sha256,
    )


def _payload(
    *,
    original_text: str,
    proposed_text: str,
    patches: list[dict[str, str]],
    changed_facts: dict[str, dict[str, object]],
    base_payload: dict[str, object],
    row: dict[str, object],
    candidate_row: dict[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "original_persona": original_text,
        "proposed_persona": proposed_text,
        "patches": patches,
        "changed_facts": changed_facts,
    }
    if "gender" in base_payload or "partner_gender" in base_payload:
        raise ProxyPatchVerificationError(
            "Verifier payload must not contain gender fields"
        )
    return _payload_with_null_detail_context(
        payload=payload,
        row=row,
        candidate_row=candidate_row,
        changed_facts=changed_facts,
    )


def _payload_with_null_detail_context(
    *,
    payload: dict[str, object],
    row: dict[str, object],
    candidate_row: dict[str, object],
    changed_facts: dict[str, dict[str, object]],
) -> dict[str, object]:
    detail_change = changed_facts.get("legal_status_detail")
    if detail_change is None or detail_change.get("new") is not None:
        return payload
    marital_status = candidate_row.get("marital_status")
    if marital_status not in _NULL_DETAIL_MARITAL_STATUS_DA:
        raise ProxyPatchVerificationError(
            "legal_status_detail null verification requires a concrete marital_status"
        )
    if not _marital_status_is_verified(
        marital_status=marital_status, row=row, changed_facts=changed_facts
    ):
        raise ProxyPatchVerificationError(
            "legal_status_detail null context must be verified by the rows"
        )
    with_context = dict(payload)
    with_context[_LEGAL_STATUS_DETAIL_NULL_CONTEXT] = {
        "target_marital_category_da": _NULL_DETAIL_MARITAL_STATUS_DA[
            t.cast(str, marital_status)
        ],
        "explanation": _NULL_DETAIL_EXPLANATION_DA,
    }
    return with_context


def _marital_status_is_verified(
    *,
    marital_status: object,
    row: dict[str, object],
    changed_facts: dict[str, dict[str, object]],
) -> bool:
    change = changed_facts.get("marital_status")
    if change is not None:
        return change.get("new") == marital_status and change.get("old") == row.get(
            "marital_status"
        )
    return row.get("marital_status") == marital_status


def _prepare_checkpoint_parent(*, parent: Path) -> None:
    missing: list[Path] = []
    current = parent
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        try:
            directory.mkdir(mode=0o700, exist_ok=True)
        except OSError as exc:
            raise ProxyPatchVerificationError(
                "Checkpoint directory path is unsafe"
            ) from exc
        _validate_checkpoint_directory(directory=directory)
    _validate_checkpoint_directory(directory=parent)


def _validate_checkpoint_directory(*, directory: Path) -> None:
    try:
        mode = directory.lstat().st_mode
    except OSError as exc:
        raise ProxyPatchVerificationError(
            "Checkpoint directory path is unsafe"
        ) from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise ProxyPatchVerificationError(
            "Checkpoint directory must be a real directory"
        )
    if stat.S_IMODE(mode) != 0o700:
        raise ProxyPatchVerificationError(
            "Checkpoint directory must be private (mode 0700)"
        )


def _prevalidate_patch_inputs(
    *,
    original_text: str,
    changed_facts: dict[str, dict[str, object]],
    proposed_text: str,
    patches: list[dict[str, str]],
    first_checkpoint_sha256: str,
) -> None:
    try:
        _conservative_result(
            original_text=original_text,
            changed_facts=changed_facts,
            proposed_text=proposed_text,
            patches=patches,
            first_checkpoint_sha256=first_checkpoint_sha256,
        )
    except LocalVerificationError as exc:
        raise ProxyPatchVerificationError(
            "Patch inputs failed local verification"
        ) from exc


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
    binding_prefix = _sha(_canonical(binding))[:16]
    request_ids: dict[int, str] = {}

    def reserve(attempt: int) -> None:
        request_id = f"patch-verification-{binding_prefix}-{attempt}-{uuid.uuid4().hex}"
        if attempt in request_ids:
            raise ProxyPatchVerificationError("HTTP attempt was reserved twice")
        budget.reserve_attempt(request_id, t.cast(dict[str, JSONValue], request_body))
        request_ids[attempt] = request_id

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
    request_id = request_ids.get(response.request_attempts)
    if request_id is None:
        raise ProxyPatchVerificationError("Request was not durably reserved")
    budget.record_usage(
        request_id,
        input_tokens=response.prompt_tokens,
        output_tokens=response.completion_tokens,
        response_sha256=response.raw_response_sha256,
    )
    if response.model != MODEL:
        raise ProxyPatchVerificationError(
            "Provider response model does not match the pinned model"
        )
    if response.completion_tokens > 128_000 or response.prompt_tokens < 0:
        raise ProxyPatchVerificationError(
            "Provider token usage exceeds the reservation policy"
        )
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
            raise ProxyPatchVerificationError(
                "Constructed HTTP body exceeds reserved input bound"
            )
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


def _resume_checkpoint(
    *,
    path: Path,
    binding: dict[str, str | int],
    original_text: str,
    changed_facts: dict[str, dict[str, object]],
    proposed_text: str,
    patches: list[dict[str, str]],
    first_checkpoint_sha256: str,
) -> ProsePatchVerificationResult:
    _require_private_checkpoint(path=path)
    checkpoint = _read_checkpoint(path=path)
    if set(checkpoint) != _checkpoint_keys(binding=binding):
        raise ProxyPatchVerificationError("Patch-verification checkpoint is malformed")
    if any(checkpoint.get(key) != value for key, value in binding.items()):
        raise ProxyPatchVerificationError(
            "Checkpoint inputs, prompt, schema, or model changed"
        )
    digest = checkpoint.get("checkpoint_sha256")
    unsigned = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    if digest != _sha(_canonical(unsigned)):
        raise ProxyPatchVerificationError(
            "Patch-verification checkpoint checksum is invalid"
        )
    result = verify_prose_patch_proposal(
        original_text=original_text,
        changed_facts=changed_facts,
        proposed_text=proposed_text,
        patches=patches,
        second_review=_checkpoint_review(checkpoint=checkpoint),
        original_checkpoint_sha256=first_checkpoint_sha256,
    )
    _verify_checkpoint_result(checkpoint=checkpoint, result=result)
    return result


def _checkpoint_keys(*, binding: dict[str, str | int]) -> set[str]:
    return set(binding) | {
        "accepted",
        "review_verdict",
        "reasons",
        "changed_fraction",
        "changed_characters",
        "patch_count",
        "original_checkpoint_sha256",
        "provisional",
        "requires_later_release_gate",
        "fact_evidence",
        "checkpoint_sha256",
    }


def _checkpoint_review(*, checkpoint: dict[str, object]) -> dict[str, object]:
    evidence = checkpoint.get("fact_evidence")
    reasons = checkpoint.get("reasons")
    if not isinstance(evidence, list) or not isinstance(reasons, list):
        raise ProxyPatchVerificationError("Patch-verification checkpoint is malformed")
    return {
        "verdict": checkpoint.get("review_verdict"),
        "reasons": reasons,
        "fact_evidence": evidence,
    }


def _read_checkpoint(*, path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError
        return t.cast(dict[str, object], value)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ProxyPatchVerificationError(
            "Patch-verification checkpoint is unreadable"
        ) from exc


def _require_private_checkpoint(*, path: Path) -> None:
    if path.stat().st_mode & 0o777 != 0o600:
        raise ProxyPatchVerificationError(
            "Patch-verification checkpoint must be private (mode 0600)"
        )


def _verify_checkpoint_result(
    *, checkpoint: dict[str, object], result: ProsePatchVerificationResult
) -> None:
    if (
        checkpoint.get("accepted") != result.accepted
        or checkpoint.get("review_verdict") != result.review_verdict
        or checkpoint.get("reasons") != result.reasons
        or checkpoint.get("changed_fraction") != result.changed_fraction
        or checkpoint.get("changed_characters") != result.changed_characters
        or checkpoint.get("patch_count") != result.patch_count
        or checkpoint.get("original_checkpoint_sha256")
        != result.original_checkpoint_sha256
        or checkpoint.get("provisional") != result.provisional
        or checkpoint.get("requires_later_release_gate")
        != result.requires_later_release_gate
        or checkpoint.get("fact_evidence") != _serialise_evidence(result=result)
    ):
        raise ProxyPatchVerificationError(
            "Patch-verification checkpoint does not match its decision"
        )


def _serialise_evidence(
    result: ProsePatchVerificationResult,
) -> list[dict[str, object]]:
    return [
        {
            "field": item.field,
            "status": item.status,
            "original_quote": item.original_quote,
            "proposed_quote": item.proposed_quote,
        }
        for item in result.fact_evidence
    ]


def _save_checkpoint(
    *, path: Path, binding: dict[str, str | int], result: ProsePatchVerificationResult
) -> None:
    document: dict[str, object] = {
        **binding,
        "accepted": result.accepted,
        "review_verdict": result.review_verdict,
        "reasons": result.reasons,
        "changed_fraction": result.changed_fraction,
        "changed_characters": result.changed_characters,
        "patch_count": result.patch_count,
        "original_checkpoint_sha256": result.original_checkpoint_sha256,
        "provisional": result.provisional,
        "requires_later_release_gate": result.requires_later_release_gate,
        "fact_evidence": _serialise_evidence(result=result),
    }
    document["checkpoint_sha256"] = _sha(_canonical(document))
    _write_checkpoint(path=path, value=document)


def _write_checkpoint(*, path: Path, value: dict[str, object]) -> None:
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


def _scan_verifier_text(
    *,
    proposed_text: str,
    patches: list[dict[str, str]],
    changed_facts: dict[str, dict[str, object]],
    prompt: str,
) -> None:
    if _contains_sensitive_text(proposed_text) or _contains_sensitive_text(prompt):
        raise ProxyPatchVerificationError(
            "Outbound verifier text contains a sensitive-identity term"
        )
    excerpt_values: list[str] = []
    for patch in patches:
        if not isinstance(patch, dict):
            raise ProxyPatchVerificationError("Patch excerpts must be dictionaries")
        excerpt_values.extend(str(value) for value in patch.values())
    outbound = [*excerpt_values, *_fact_text(changed_facts)]
    if any(_contains_sensitive_text(value) for value in outbound):
        raise ProxyPatchVerificationError(
            "Outbound verifier text contains a sensitive-identity term"
        )


def _validate_checkpoint_sha(*, first_checkpoint_sha256: str) -> None:
    if _SHA256_RE.fullmatch(first_checkpoint_sha256) is None:
        raise ProxyPatchVerificationError(
            "First checkpoint SHA must be a SHA-256 digest"
        )


def _validate_config(*, config: GenerationConfig) -> None:
    if (
        config.base_url != BASE_URL
        or config.model != MODEL
        or config.max_tokens is not None
        or config.reasoning_effort != "none"
        or config.enable_thinking is not None
    ):
        raise ProxyPatchVerificationError(
            "Generation configuration is not the pinned local proxy"
        )
