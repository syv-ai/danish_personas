"""Durable conservative budget gate for the local Pi OpenAI proxy."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

MODEL = "gpt-6-luna"
BASE_URL = "http://127.0.0.1:18080/v1"
HARD_CAP_USD = Decimal("100")
INTERNAL_CAP_USD = Decimal("90")
PRIOR_RESERVATIONS = {
    "prior-failed-melious": Decimal("0.00064415"),
    "prior-failed-mistral": Decimal("0.000935"),
    "synthetic-proxy-smoke": Decimal("0.065"),
}


class ProxyBudgetError(RuntimeError):
    """Raised when the durable proxy budget cannot safely authorise a request."""


class ProxyBudget:
    """Append-only request budget bound to pinned model and campaign inputs.

    The registry is expected to contain a model entry with ``id`` (or ``name``),
    ``maxTokens``, and ``cost`` containing ``input`` and ``output`` prices in
    USD per million tokens. ``models`` may be a mapping or list.
    """

    def __init__(
        self,
        *,
        ledger_path: Path,
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
        self.path = Path(ledger_path)
        self.registry_path = Path(registry_path)
        self.pins: dict[str, Any] = {
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

    def reserve_attempt(self, request_id: str, request: Any) -> Decimal:
        """Durably reserve worst-case cost before an HTTP attempt; return USD."""
        if not request_id or not isinstance(request_id, str):
            raise ProxyBudgetError("Request ID must be a non-empty string")
        request_bytes = len(_canonical_json(request)) + self.overhead
        # One token per UTF-8 byte is deliberately conservative.
        input_tokens = request_bytes
        per_request = (
            Decimal(input_tokens) * Decimal(self.pins["input_usd_per_million"])
            + Decimal(self.pins["max_tokens"])
            * Decimal(self.pins["output_usd_per_million"])
        ) / Decimal(1_000_000)

        def operation() -> Decimal:
            header, records = self._load()
            self._check_header(header)
            reservations = {
                r["request_id"]: Decimal(r["usd"])
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

    def record_usage(
        self,
        request_id: str,
        *,
        input_tokens: int,
        output_tokens: int,
        response: bytes | str,
    ) -> None:
        """Record observed usage and response hash; never refunds a reservation."""
        if input_tokens < 0 or output_tokens < 0:
            raise ProxyBudgetError("Observed token counts must be non-negative")
        response_bytes = (
            response.encode("utf-8") if isinstance(response, str) else response
        )
        response_hash = hashlib.sha256(response_bytes).hexdigest()

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
                    "response_sha256": response_hash,
                }
            )

        self._locked(operation)

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

    def _check_registry(self) -> None:
        try:
            document = json.loads(self.registry_path.read_text(encoding="utf-8"))
            models = document.get("models", document)
            if isinstance(models, dict):
                candidates = [
                    dict(value, id=key) if isinstance(value, dict) else value
                    for key, value in models.items()
                ]
            elif isinstance(models, list):
                candidates = models
            else:
                raise ValueError("invalid models registry")
            model = next(
                item
                for item in candidates
                if isinstance(item, dict)
                and item.get("id", item.get("name")) == self.pins["model"]
            )
            cost = model["cost"]
            inputs, outputs = cost["input"], cost["output"]
            if (
                model["maxTokens"] != self.pins["max_tokens"]
                or Decimal(str(inputs)) != Decimal(self.pins["input_usd_per_million"])
                or Decimal(str(outputs)) != Decimal(self.pins["output_usd_per_million"])
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

    def _check_header(self, header: dict[str, Any]) -> None:
        self._check_registry()
        if header != self.pins:
            raise ProxyBudgetError(
                "Budget ledger pins do not match current configuration"
            )

    def _load(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        try:
            contents = self.path.read_text(encoding="utf-8")
            if not contents.endswith("\\n"):
                raise ValueError("ledger does not end at a complete record")
            lines = contents.splitlines()
            if not lines or any(not line.strip() for line in lines):
                raise ValueError("empty ledger line")
            records = [json.loads(line) for line in lines]
            header = records[0]
            if header.get("type") != "header" or not isinstance(header, dict):
                raise ValueError("missing header")
            reservations: set[str] = set()
            for record in records[1:]:
                if not isinstance(record, dict) or record.get("type") not in {
                    "reservation",
                    "usage",
                }:
                    raise ValueError("unknown ledger record")
                identifier = record["request_id"]
                if not isinstance(identifier, str) or not identifier:
                    raise ValueError("invalid request ID")
                if record["type"] == "reservation":
                    allowed = {
                        "type",
                        "request_id",
                        "usd",
                        "input_byte_bound",
                        "max_output_tokens",
                        "historical",
                    }
                    if set(record) - allowed:
                        raise ValueError("unknown reservation fields")
                    value = Decimal(record["usd"])
                    if (
                        not value.is_finite()
                        or value <= 0
                        or identifier in reservations
                    ):
                        raise ValueError("invalid/duplicate reservation")
                    reservations.add(identifier)
                else:
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
                    if (
                        not isinstance(record["input_tokens"], int)
                        or not isinstance(record["output_tokens"], int)
                        or record["input_tokens"] < 0
                        or record["output_tokens"] < 0
                        or not isinstance(record["response_sha256"], str)
                        or len(record["response_sha256"]) != 64
                    ):
                        raise ValueError("invalid usage fields")
            if (
                sum(
                    (
                        Decimal(r["usd"])
                        for r in records[1:]
                        if r["type"] == "reservation"
                    ),
                    Decimal(0),
                )
                > HARD_CAP_USD
            ):
                raise ValueError("historical reservations exceed hard cap")
            return header, records[1:]
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ProxyBudgetError("Budget ledger is malformed or truncated") from exc

    def _append(self, record: dict[str, Any]) -> None:
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8", closefd=False) as ledger:
                ledger.write(_canonical_json(record).decode() + "\n")
                ledger.flush()
                os.fsync(fd)
        finally:
            os.close(fd)

    def _locked(self, function: Any) -> Any:
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.fchmod(fd, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            return function()
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
