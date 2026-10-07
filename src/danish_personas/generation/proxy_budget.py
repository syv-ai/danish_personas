"""Durable conservative budget gate for the local Pi OpenAI proxy."""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import TypeAlias, TypeVar

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

MODEL = "gpt-6-luna"
BASE_URL = "http://127.0.0.1:18080/v1"
HARD_CAP_USD = Decimal("100")
INTERNAL_CAP_USD = Decimal("90")
USER_BUDGET_PATH = Path.home() / ".danish-personas" / "proxy-budget.jsonl"
JSONValue: TypeAlias = (
    None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)
Result = TypeVar("Result")

PRIOR_RESERVATIONS = {
    "prior-failed-melious": Decimal("0.00064415"),
    "prior-failed-mistral": Decimal("0.000935"),
    "synthetic-proxy-smoke": Decimal("0.065"),
}


class ProxyBudget:
    """Append-only request budget bound to pinned model and campaign inputs.

    The registry is expected to contain the ``openai-codex`` provider with a
    model entry having ``id`` (or ``name``), ``maxTokens``, and ``cost`` with
    ``input`` and ``output`` prices in USD per million tokens. ``models`` may be
    a mapping or list. A flat ``models`` registry is also supported.
    """

    def __init__(
        self,
        *,
        ledger_path: Path | None = None,
        registry_path: Path,
        campaign: str,
        source_hash: str,
        prompt_hash: str,
        schema_hash: str,
        model: str = MODEL,
        base_url: str = BASE_URL,
        max_tokens: int = 128_000,
        input_usd_per_million: str = "0.1",
        output_usd_per_million: str = "0.5",
        cap_usd: Decimal = INTERNAL_CAP_USD,
        request_overhead_bytes: int = 4096,
    ) -> None:
        """Create or reopen a ledger after checking pinned registry and policy.

        Raises:
            ProxyBudgetError: If configuration or pinned model metadata is invalid.
        """
        # ``ledger_path`` is accepted for source compatibility, but never selects
        # the ledger: all proxy campaigns share the same user-level budget.
        del ledger_path
        self.path = USER_BUDGET_PATH
        self.registry_path = Path(registry_path)
        self.pins: dict[str, JSONValue] = {
            "type": "header",
            "model": model,
            "base_url": base_url,
            "max_tokens": max_tokens,
            "input_usd_per_million": str(Decimal(input_usd_per_million)),
            "output_usd_per_million": str(Decimal(output_usd_per_million)),
            "campaign": campaign,
            "source_hash": source_hash,
            "prompt_hash": prompt_hash,
            "schema_hash": schema_hash,
        }
        self.cap = Decimal(cap_usd)
        self.overhead = request_overhead_bytes
        if (
            model != MODEL
            or base_url != BASE_URL
            or max_tokens != 128_000
            or Decimal(input_usd_per_million) != Decimal("0.1")
            or Decimal(output_usd_per_million) != Decimal("0.5")
            or not Decimal("0") < self.cap <= INTERNAL_CAP_USD
            or request_overhead_bytes < 0
            or not all((campaign, source_hash, prompt_hash, schema_hash))
        ):
            raise ProxyBudgetError(
                "Proxy budget configuration is not within pinned policy"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._locked(self._initialise)

    def _locked(self, function: Callable[[], Result]) -> Result:
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        locked = False
        try:
            if os.name == "nt":
                os.chmod(lock_path, 0o600)
                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"\0")
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            else:
                os.fchmod(fd, 0o600)
                fcntl.flock(fd, fcntl.LOCK_EX)
            locked = True
            return function()
        finally:
            if locked:
                if os.name == "nt":
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _initialise(self) -> None:
        self._check_registry()
        if self.path.exists():
            header, _ = self._load()
            self._check_header(header)
            os.chmod(self.path, 0o600)
            return
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as ledger:
            ledger.write(_canonical_json(self.pins).decode() + "\n")
            for request_id, amount in PRIOR_RESERVATIONS.items():
                ledger.write(
                    _canonical_json(
                        {
                            "type": "reservation",
                            "request_id": request_id,
                            "usd": str(amount),
                            "historical": True,
                        }
                    ).decode()
                    + "\n"
                )
            ledger.flush()
            os.fsync(ledger.fileno())
        os.chmod(self.path, 0o600)

    def _check_header(self, header: dict[str, JSONValue]) -> None:
        self._check_registry()
        if header != self.pins:
            raise ProxyBudgetError(
                "Budget ledger pins do not match current configuration"
            )

    def _check_registry(self) -> None:
        try:
            document = json.loads(self.registry_path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError("invalid models registry")
            if "models" in document:
                models = document["models"]
            else:
                provider = document.get("openai-codex")
                if not isinstance(provider, dict):
                    raise ValueError("missing openai-codex provider")
                models = provider.get("models")
            if isinstance(models, dict):
                candidates = [
                    dict(value, id=key) if isinstance(value, dict) else value
                    for key, value in models.items()
                ]
            elif isinstance(models, list):
                candidates = models
            else:
                raise ValueError("invalid models registry")
            matching_models = [
                item
                for item in candidates
                if isinstance(item, dict)
                and item.get("id", item.get("name")) == self.pins["model"]
            ]
            if len(matching_models) != 1:
                raise ValueError("model is missing or duplicated")
            model = matching_models[0]
            cost = model["cost"]
            inputs, outputs = cost["input"], cost["output"]
            if (
                model["maxTokens"] != self.pins["max_tokens"]
                or Decimal(str(inputs))
                != Decimal(str(self.pins["input_usd_per_million"]))
                or Decimal(str(outputs))
                != Decimal(str(self.pins["output_usd_per_million"]))
            ):
                raise ValueError("model metadata changed")
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            StopIteration,
            json.JSONDecodeError,
        ) as exc:
            raise ProxyBudgetError(
                "Model registry is missing, changed, or unbounded"
            ) from exc

    def _load(self) -> tuple[dict[str, JSONValue], list[dict[str, JSONValue]]]:
        try:
            contents = self.path.read_text(encoding="utf-8")
            if not contents.endswith("\n"):
                raise ValueError("ledger does not end at a complete record")
            lines = contents.splitlines()
            if not lines or any(not line.strip() for line in lines):
                raise ValueError("empty ledger line")
            records = [json.loads(line) for line in lines]
            header = records[0]
            if not isinstance(header, dict) or header.get("type") != "header":
                raise ValueError("missing header")
            return header, _validate_records(records[1:])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ProxyBudgetError("Budget ledger is malformed or truncated") from exc

    def record_usage(
        self,
        request_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
        response_sha256: str,
    ) -> None:
        """Record observed usage and response hash without refunding a reservation.

        Raises:
            ProxyBudgetError: If usage is invalid or has no matching reservation.
        """
        if input_tokens < 0 or output_tokens < 0:
            raise ProxyBudgetError("Observed token counts must be non-negative")
        if (
            not isinstance(response_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", response_sha256) is None
        ):
            raise ProxyBudgetError("Response SHA-256 must be a lowercase hex digest")

        def operation() -> None:
            header, records = self._load()
            self._check_header(header)
            reservations = {
                r["request_id"] for r in records if r["type"] == "reservation"
            }
            if request_id not in reservations:
                raise ProxyBudgetError("Usage references an unknown request ID")
            if any(
                r.get("request_id") == request_id and r["type"] == "usage"
                for r in records
            ):
                raise ProxyBudgetError("Usage already recorded for request ID")
            self._append(
                {
                    "type": "usage",
                    "request_id": request_id,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "response_sha256": response_sha256,
                }
            )

        self._locked(operation)

    def _append(self, record: dict[str, JSONValue]) -> None:
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND)
        try:
            if os.name == "nt":
                os.chmod(self.path, 0o600)
            else:
                os.fchmod(fd, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8", closefd=False) as ledger:
                ledger.write(_canonical_json(record).decode() + "\n")
                ledger.flush()
                os.fsync(fd)
        finally:
            os.close(fd)

    def reserve_attempt(
        self, request_id: str, request: dict[str, JSONValue]
    ) -> Decimal:
        """Durably reserve worst-case cost before an HTTP attempt.

        Args:
            request_id: Unique identifier for this HTTP attempt.
            request: JSON-compatible provider request payload.

        Returns:
            The USD amount reserved for this attempt.

        Raises:
            ProxyBudgetError: If the request is invalid or the budget is exhausted.
        """
        if not request_id or not isinstance(request_id, str):
            raise ProxyBudgetError("Request ID must be a non-empty string")
        request_bytes = len(_canonical_json(request)) + self.overhead
        # One token per UTF-8 byte is deliberately conservative.
        input_tokens = request_bytes
        per_request = (
            Decimal(input_tokens) * Decimal(str(self.pins["input_usd_per_million"]))
            + Decimal(str(self.pins["max_tokens"]))
            * Decimal(str(self.pins["output_usd_per_million"]))
        ) / Decimal(1_000_000)

        def operation() -> Decimal:
            header, records = self._load()
            self._check_header(header)
            reservations = {
                str(r["request_id"]): Decimal(str(r["usd"]))
                for r in records
                if r["type"] == "reservation"
            }
            if request_id in reservations:
                raise ProxyBudgetError("Request ID is already reserved")
            total = sum(reservations.values(), Decimal(0))
            if total + per_request > self.cap or total + per_request > HARD_CAP_USD:
                raise ProxyBudgetError("Proxy campaign budget cap exhausted")
            self._append(
                {
                    "type": "reservation",
                    "request_id": request_id,
                    "usd": str(per_request),
                    "input_byte_bound": request_bytes,
                    "max_output_tokens": self.pins["max_tokens"],
                }
            )
            return per_request

        return self._locked(operation)


class ProxyBudgetError(RuntimeError):
    """Raised when the durable proxy budget cannot safely authorise a request."""


def _canonical_json(value: JSONValue) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _validate_records(records: list[object]) -> list[dict[str, JSONValue]]:
    """Validate ledger events and enforce the immutable hard cap.

    Args:
        records: Decoded JSON lines after the header.

    Returns:
        Validated reservation and usage records.

    Raises:
        ValueError: If a record is invalid or the hard cap is exceeded.
    """
    validated: list[dict[str, JSONValue]] = []
    reservations: set[str] = set()
    total = Decimal(0)
    for value in records:
        if not isinstance(value, dict) or value.get("type") not in {
            "reservation",
            "usage",
        }:
            raise ValueError("unknown ledger record")
        record: dict[str, JSONValue] = value
        identifier = record.get("request_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("invalid request ID")
        if record["type"] == "reservation":
            _validate_reservation(record=record)
            amount = Decimal(str(record["usd"]))
            if identifier in reservations:
                raise ValueError("invalid/duplicate reservation")
            reservations.add(identifier)
            total += amount
        else:
            _validate_usage(
                record=record, identifier=identifier, reservations=reservations
            )
        validated.append(record)
    if total > HARD_CAP_USD:
        raise ValueError("historical reservations exceed hard cap")
    return validated


def _validate_reservation(*, record: dict[str, JSONValue]) -> None:
    allowed = {
        "type",
        "request_id",
        "usd",
        "input_byte_bound",
        "max_output_tokens",
        "historical",
    }
    amount = record.get("usd")
    if set(record) - allowed or not isinstance(amount, str):
        raise ValueError("invalid reservation fields")
    value = Decimal(amount)
    if not value.is_finite() or value <= 0:
        raise ValueError("invalid reservation")


def _validate_usage(
    *, record: dict[str, JSONValue], identifier: str, reservations: set[str]
) -> None:
    if set(record) != {
        "type",
        "request_id",
        "input_tokens",
        "output_tokens",
        "response_sha256",
    }:
        raise ValueError("invalid usage record")
    if identifier not in reservations:
        raise ValueError("usage without reservation")
    input_tokens = record["input_tokens"]
    output_tokens = record["output_tokens"]
    response_hash = record["response_sha256"]
    if (
        not isinstance(input_tokens, int)
        or not isinstance(output_tokens, int)
        or input_tokens < 0
        or output_tokens < 0
        or not isinstance(response_hash, str)
        or len(response_hash) != 64
    ):
        raise ValueError("invalid usage fields")
