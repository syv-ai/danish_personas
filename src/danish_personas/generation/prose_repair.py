"""Offline-testable, checksum-bound prose-only repair runner.

The runner never constructs a network client; callers must explicitly inject one.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import typing as t
from hashlib import sha256
from pathlib import Path

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..io import canonical_json
from .models import GenerationConfig, LLMResponse


class _Prose(BaseModel):
    """Strict prose-only response contract."""

    model_config = ConfigDict(extra="forbid", strict=True)
    persona: str = Field(min_length=300, max_length=900)


# Only these existing facts may cross the provider boundary. Keep sensitive
# relationship/orientation and internal sampling-target fields out by construction.
PAYLOAD_ALLOWLIST = frozenset(
    {
        "age",
        "gender",
        "municipality",
        "municipality_name",
        "region",
        "region_name",
        "landsdel",
        "origin_country_da",
        "education",
        "education_level",
        "labour_status",
        "status",
        "job_function",
        "job_function_da",
        "job_title",
        "cultural_context",
        "skills_and_expertise",
        "hobbies_and_interests",
        "career_goals_and_ambitions",
        "current_relationship_status",
        "legal_status_detail",
        "ocean",
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
        "personality_tendencies",
        "partner_gender",
    }
)
FORBIDDEN = frozenset(
    {
        "sexual_orientation",
        "partner_sexual_orientation",
        "transgender",
        "partner_transgender",
        "variation_in_sex_characteristics",
        "same_sex_partner_probability",
        "same_sex_partner_target",
    }
)
USD_PER_EUR = 1.30
INPUT_EUR_PER_MILLION = 0.10
OUTPUT_EUR_PER_MILLION = 0.25


class RepairError(RuntimeError):
    """Raised when a repair cannot proceed safely."""


class _RepairClient(t.Protocol):
    def complete(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, object],
        schema_name: str,
        json_schema: dict[str, object],
        record_request: t.Callable[[int], None],
    ) -> LLMResponse: ...


def _digest(value: object) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_prose_repair(
    *,
    rows: list[dict[str, t.Any]],
    changed_fields: dict[str, set[str] | list[str]],
    id_field: str,
    prompt: str,
    model: str,
    config: GenerationConfig,
    input_manifest_sha256: str,
    sidecar_sha256: str,
    output_dir: Path,
    cost_cap_usd: float | None,
    client: _RepairClient,
) -> list[dict[str, t.Any]]:
    """Repair prose for affected IDs while preserving structured fields.

    The injected client must expose ``complete`` and must not be used to send any
    request unless the caller has authorised that provider interaction.

    Args:
        rows: Source records to repair.
        changed_fields: Changed-field reason metadata indexed by record ID.
        id_field: Field containing each record's unique string ID.
        prompt: System prompt used for the prose response.
        model: Model name, which must match the generation configuration.
        config: Bounded generation configuration.
        input_manifest_sha256: Hash identifying the source manifest.
        sidecar_sha256: Hash identifying the response schema sidecar.
        output_dir: Private directory for the lock, ledger, and checkpoints.
        cost_cap_usd: Positive finite cumulative USD cap no greater than 100.
        client: Explicitly injected client implementing ``complete``.

    Returns:
        The input rows with repaired persona prose where requested.

    """
    identifiers = _validate_inputs(
        rows=rows,
        changed_fields=changed_fields,
        id_field=id_field,
        prompt=prompt,
        model=model,
        config=config,
        input_manifest_sha256=input_manifest_sha256,
        sidecar_sha256=sidecar_sha256,
        cost_cap_usd=cost_cap_usd,
    )
    if not rows:
        return []
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output_dir, 0o700)
    lock_path = output_dir / ".repair.lock"
    lock_stream = lock_path.open("a+")
    os.chmod(lock_path, 0o600)
    try:
        _acquire_lock(lock_stream=lock_stream)
        return _run_locked(
            rows=rows,
            identifiers=identifiers,
            changed_fields=changed_fields,
            id_field=id_field,
            prompt=prompt,
            model=model,
            config=config,
            input_manifest_sha256=input_manifest_sha256,
            sidecar_sha256=sidecar_sha256,
            output_dir=output_dir,
            cost_cap_usd=cost_cap_usd,
            client=client,
        )
    finally:
        lock_stream.close()


def _validate_inputs(
    *,
    rows: list[dict[str, t.Any]],
    changed_fields: dict[str, set[str] | list[str]],
    id_field: str,
    prompt: str,
    model: str,
    config: GenerationConfig,
    input_manifest_sha256: str,
    sidecar_sha256: str,
    cost_cap_usd: float | None,
) -> list[str]:
    """Validate request bounds and its row/change-set binding.

    Returns:
        Validated identifiers in input order.

    Raises:
        RepairError: If any bound or input metadata is invalid.
    """
    if (
        cost_cap_usd is None
        or not math.isfinite(cost_cap_usd)
        or cost_cap_usd <= 0
        or cost_cap_usd > 100
    ):
        raise RepairError("Cost cap must be finite, positive, and no greater than $100")
    if not model or config.model != model:
        raise RepairError("Model must match configuration")
    if config.max_tokens is None or config.max_tokens <= 0:
        raise RepairError("max_tokens must be positive and bounded")
    if config.max_tokens > 4096:
        raise RepairError("max_tokens exceeds the repair safety limit")
    if not input_manifest_sha256 or not sidecar_sha256:
        raise RepairError("Input manifest and schema sidecar hashes are required")
    return _validate_row_metadata(
        rows=rows, changed_fields=changed_fields, id_field=id_field
    )


def _validate_row_metadata(
    *,
    rows: list[dict[str, t.Any]],
    changed_fields: dict[str, set[str] | list[str]],
    id_field: str,
) -> list[str]:
    """Validate unique row IDs and changed-field reason metadata.

    Returns:
        Validated row identifiers in input order.

    Raises:
        RepairError: If row identifiers or changed-field metadata are invalid.
    """
    identifiers = [row.get(id_field) for row in rows]
    if any(
        not isinstance(identifier, str) or not identifier for identifier in identifiers
    ):
        raise RepairError("Every input row requires a non-empty string ID")
    if len(set(identifiers)) != len(identifiers):
        raise RepairError("Duplicate input IDs are not allowed")
    if set(changed_fields) - set(identifiers):
        raise RepairError("Changed-field set references an unknown ID")
    for values in changed_fields.values():
        if not values or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise RepairError("Changed-field metadata must contain field names")
    return identifiers


def _acquire_lock(*, lock_stream: t.TextIO) -> None:
    """Acquire the exclusive, non-blocking repair lock.

    Raises:
        RepairError: If another repair execution holds the lock.
    """
    try:
        if os.name == "nt":
            lock_stream.seek(0)
            msvcrt.locking(lock_stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as error:
        raise RepairError("Another repair execution holds the output lock") from error


def _run_locked(
    *,
    rows: list[dict[str, t.Any]],
    identifiers: list[str],
    changed_fields: dict[str, set[str] | list[str]],
    id_field: str,
    prompt: str,
    model: str,
    config: GenerationConfig,
    input_manifest_sha256: str,
    sidecar_sha256: str,
    output_dir: Path,
    cost_cap_usd: float | None,
    client: _RepairClient,
) -> list[dict[str, t.Any]]:
    schema = _Prose.model_json_schema()
    binding = _make_binding(
        rows=rows,
        changed_fields=changed_fields,
        prompt=prompt,
        model=model,
        config=config,
        input_manifest_sha256=input_manifest_sha256,
        sidecar_sha256=sidecar_sha256,
        schema=schema,
        cost_cap_usd=cost_cap_usd,
    )
    binding_hash = _digest(binding)
    ledger_path = output_dir / "ledger.json"
    ledger, reserved = _load_ledger(
        path=ledger_path,
        binding=binding,
        identifiers=identifiers,
        cost_cap_usd=cost_cap_usd,
    )
    completed: list[dict[str, t.Any]] = []
    for row in rows:
        completed.append(
            _repair_row(
                row=row,
                identifier=row[id_field],
                changed=row[id_field] in changed_fields,
                prompt=prompt,
                model=model,
                config=config,
                schema=schema,
                binding_hash=binding_hash,
                output_dir=output_dir,
                ledger=ledger,
                ledger_path=ledger_path,
                reserved=reserved,
                cost_cap_usd=cost_cap_usd,
                client=client,
            )
        )
        reserved[0] = sum(item["usd"] for item in ledger["reservations"])
    return completed


def _make_binding(
    *,
    rows: list[dict[str, t.Any]],
    changed_fields: dict[str, set[str] | list[str]],
    prompt: str,
    model: str,
    config: GenerationConfig,
    input_manifest_sha256: str,
    sidecar_sha256: str,
    schema: dict[str, object],
    cost_cap_usd: float | None,
) -> dict[str, object]:
    """Bind durable repair state to all inputs that can affect output.

    Returns:
        The canonical binding data.
    """
    return {
        "input_manifest_sha256": input_manifest_sha256,
        "response_schema_sha256": _digest(schema),
        "rows_sha256": _digest(rows),
        "changed_fields_sha256": _digest(
            {key: sorted(value) for key, value in changed_fields.items()}
        ),
        "prompt_sha256": sha256(prompt.encode()).hexdigest(),
        "model": model,
        "sidecar_sha256": sidecar_sha256,
        "max_tokens": config.max_tokens,
        "cost_cap_usd": cost_cap_usd,
    }


def _load_ledger(
    *,
    path: Path,
    binding: dict[str, object],
    identifiers: list[str],
    cost_cap_usd: float | None,
) -> tuple[dict[str, t.Any], list[float]]:
    """Load and fail closed on inconsistent durable request reservations.

    Returns:
        The validated ledger and its existing reserved cost.

    Raises:
        RepairError: If the ledger is malformed, stale, or over budget.
    """
    if path.exists():
        try:
            ledger = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise RepairError("Stale or malformed repair ledger") from error
        if not isinstance(ledger, dict) or ledger.get("binding") != binding:
            raise RepairError("Stale or malformed repair ledger")
        reservations = ledger.get("reservations")
        if not isinstance(reservations, list):
            raise RepairError("Stale or malformed repair ledger")
    else:
        ledger = {"binding": binding, "reservations": []}
        _atomic_json(path, ledger)
        reservations = ledger["reservations"]
    try:
        costs = [
            float(item["usd"])
            for item in reservations
            if isinstance(item, dict)
            and item.get("id") in identifiers
            and isinstance(item.get("usd"), (int, float))
            and math.isfinite(float(item["usd"]))
            and float(item["usd"]) > 0
        ]
    except (TypeError, ValueError, OverflowError) as error:
        raise RepairError("Malformed repair ledger reservation") from error
    if len(costs) != len(reservations):
        raise RepairError("Malformed repair ledger reservation")
    reserved = sum(costs)
    if not math.isfinite(reserved) or reserved > cost_cap_usd:
        raise RepairError("Existing reservations exceed the configured cap")
    return ledger, [reserved]


def _repair_row(
    *,
    row: dict[str, t.Any],
    identifier: str,
    changed: bool,
    prompt: str,
    model: str,
    config: GenerationConfig,
    schema: dict[str, object],
    binding_hash: str,
    output_dir: Path,
    ledger: dict[str, t.Any],
    ledger_path: Path,
    reserved: list[float],
    cost_cap_usd: float | None,
    client: _RepairClient,
) -> dict[str, t.Any]:
    checkpoint_path = output_dir / f"{sha256(identifier.encode()).hexdigest()}.json"
    row_hash = _digest(row)
    checkpoint = _load_checkpoint(
        path=checkpoint_path, binding_hash=binding_hash, row_hash=row_hash
    )
    if checkpoint is not None:
        return {**row, **checkpoint}
    if not changed:
        return dict(row)
    payload = _make_payload(row=row)
    input_token_bound = _input_token_bound(
        prompt=prompt, payload=payload, schema=schema
    )
    per_attempt_usd = _attempt_cost(
        input_token_bound=input_token_bound, max_tokens=config.max_tokens
    )

    def reserve(_attempt_number: int) -> None:
        if reserved[0] + per_attempt_usd > cost_cap_usd:
            raise RepairError("Cost cap would be exceeded before network request")
        ledger["reservations"].append({"id": identifier, "usd": per_attempt_usd})
        _atomic_json(ledger_path, ledger)
        reserved[0] += per_attempt_usd

    response = client.complete(
        system_prompt=prompt,
        user_payload=payload,
        schema_name="persona_prose",
        json_schema=schema,
        record_request=reserve,
    )
    _verify_response(
        response=response,
        model=model,
        input_token_bound=input_token_bound,
        max_tokens=config.max_tokens,
    )
    try:
        prose = _Prose.model_validate_json(response.content)
    except (ValidationError, ValueError) as error:
        raise RepairError("Model response failed strict prose schema") from error
    output = prose.model_dump(mode="json")
    _atomic_json(
        checkpoint_path,
        {
            "binding_sha256": binding_hash,
            "row_sha256": row_hash,
            "response_id": response.response_id,
            "response_sha256": response.raw_response_sha256,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "output": output,
        },
    )
    return {**row, **output}


def _load_checkpoint(
    *, path: Path, binding_hash: str, row_hash: str
) -> dict[str, str] | None:
    """Validate an existing checkpoint before using its prose output.

    Returns:
        Validated prose fields, or ``None`` when no checkpoint exists.

    Raises:
        RepairError: If the checkpoint is malformed or stale.
    """
    if not path.exists():
        return None
    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise RepairError("Malformed prose checkpoint") from error
    if not isinstance(checkpoint, dict) or (
        checkpoint.get("binding_sha256") != binding_hash
        or checkpoint.get("row_sha256") != row_hash
    ):
        raise RepairError("Checkpoint does not match current row or inputs")
    try:
        prose = _Prose.model_validate(checkpoint["output"])
    except (ValidationError, KeyError, TypeError) as error:
        raise RepairError("Malformed prose checkpoint output") from error
    return prose.model_dump(mode="json")


def _make_payload(*, row: dict[str, t.Any]) -> dict[str, object]:
    """Apply the explicit provider boundary allowlist.

    Returns:
        Only fields explicitly permitted in the provider request.

    Raises:
        RepairError: If any forbidden field crosses the provider boundary.
    """
    payload = {key: value for key, value in row.items() if key in PAYLOAD_ALLOWLIST}
    if FORBIDDEN.intersection(payload):
        raise RepairError("Forbidden provider payload field")
    return payload


def _input_token_bound(
    *, prompt: str, payload: dict[str, object], schema: dict[str, object]
) -> int:
    """Conservatively bound request input from its encoded body size.

    Returns:
        The maximum estimated input token count.
    """
    body_bytes = len(
        (prompt + canonical_json(payload) + canonical_json(schema)).encode()
    )
    return body_bytes + 512


def _attempt_cost(*, input_token_bound: int, max_tokens: int | None) -> float:
    """Estimate a per-attempt USD reservation using conservative token bounds.

    Returns:
        The estimated cost in USD.
    """
    assert max_tokens is not None
    return (
        USD_PER_EUR
        * (
            input_token_bound * INPUT_EUR_PER_MILLION
            + max_tokens * OUTPUT_EUR_PER_MILLION
        )
        / 1_000_000
    )


def _verify_response(
    *, response: LLMResponse, model: str, input_token_bound: int, max_tokens: int | None
) -> None:
    """Reject metadata that violates the requested model or token reservation.

    Raises:
        RepairError: If response metadata exceeds the recorded bounds.
    """
    if response.model != model:
        raise RepairError("Provider response model does not match requested model")
    if (
        max_tokens is None
        or response.prompt_tokens > input_token_bound
        or response.completion_tokens > max_tokens
        or response.total_tokens != response.prompt_tokens + response.completion_tokens
    ):
        raise RepairError("Provider response usage exceeds the reserved token bounds")
