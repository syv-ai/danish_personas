"""Durable conservative budget gate for the local Pi OpenAI proxy."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Iterable
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TypeAlias, TypeVar

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

REVIEW_MODEL_ENV = "DANISH_PERSONAS_REVIEW_MODEL"
ADJUDICATION_MODEL_ENV = "DANISH_PERSONAS_ADJUDICATION_MODEL"
MODEL = os.environ.get(REVIEW_MODEL_ENV, "review-model").strip()
SOL_ADJUDICATION_MODEL = os.environ.get(
    ADJUDICATION_MODEL_ENV, "adjudication-model"
).strip()


def require_runtime_model(environment_variable: str) -> str:
    """Require a private runtime model choice before a live campaign runs.

    Neutral fallback identifiers keep offline fixtures usable; they are never
    accepted for a live campaign unless explicitly configured in the environment.

    Returns:
        The selected private model identifier.

    Raises:
        ProxyBudgetError: If the environment variable is missing or blank.
    """
    model = os.environ.get(environment_variable, "").strip()
    if not model:
        raise ProxyBudgetError(
            f"Set {environment_variable} to a private runtime model name before "
            "running this campaign"
        )
    configured_model = {
        REVIEW_MODEL_ENV: MODEL,
        ADJUDICATION_MODEL_ENV: SOL_ADJUDICATION_MODEL,
    }.get(environment_variable)
    if configured_model != model:
        raise ProxyBudgetError(
            f"Set {environment_variable} before starting the process so the "
            "campaign budget and request use the same model"
        )
    return model


BASE_URL = "http://127.0.0.1:18080/v1"
DEFAULT_MAX_TOKENS = 128_000
SOL_ADJUDICATION_LEDGER_MAX_TOKENS = DEFAULT_MAX_TOKENS
HARD_CAP_USD = Decimal("100")
INTERNAL_CAP_USD = Decimal("90")
USER_BUDGET_PATH = Path.home() / ".danish-personas" / "proxy-budget.jsonl"
USER_UNCAPPED_BUDGET_PATH = Path.home() / ".danish-personas" / "proxy-uncapped.jsonl"
USER_SOL_ADJUDICATION_BUDGET_PATH = (
    Path.home() / ".danish-personas" / "proxy-sol-adjudication.jsonl"
)
USER_V3_ADJUDICATION_BUDGET_PATH = (
    Path.home() / ".danish-personas" / "proxy-v3-adjudication.jsonl"
)
USER_V3_EXTENDED_ADJUDICATION_BUDGET_PATH = (
    Path.home() / ".danish-personas" / "proxy-v3-extended-adjudication.jsonl"
)
USER_V3_LONG_ADJUDICATION_BUDGET_PATH = (
    Path.home() / ".danish-personas" / "proxy-v3-long-adjudication.jsonl"
)
USER_PATCH_VERIFICATION_BUDGET_PATH = (
    Path.home() / ".danish-personas" / "proxy-patch-verification.jsonl"
)
USER_EDUCATION_REVIEW_BUDGET_PATH = (
    Path.home() / ".danish-personas" / "proxy-education-review.jsonl"
)
USER_EDUCATION_VERIFICATION_BUDGET_PATH = (
    Path.home() / ".danish-personas" / "proxy-education-verification.jsonl"
)
PATCH_VERIFICATION_PURPOSE = "patch_verification"
EDUCATION_REVIEW_PURPOSE = "h90_v5"
EDUCATION_VERIFICATION_PURPOSE = "h90_v5_verification"
SOL_ADJUDICATION_PURPOSE = "sol_adjudication"
V3_ADJUDICATION_PURPOSE = "v3_adjudication"
V3_EXTENDED_ADJUDICATION_PURPOSE = "v3_extended_adjudication"
V3_LONG_ADJUDICATION_PURPOSE = "v3_long_adjudication"
V3_TARGETED_FOLLOWUP_PURPOSE = "v3_targeted_followup"
UNLIMITED_PURPOSES = frozenset(
    {
        PATCH_VERIFICATION_PURPOSE,
        EDUCATION_REVIEW_PURPOSE,
        EDUCATION_VERIFICATION_PURPOSE,
        SOL_ADJUDICATION_PURPOSE,
        V3_ADJUDICATION_PURPOSE,
        V3_EXTENDED_ADJUDICATION_PURPOSE,
        V3_LONG_ADJUDICATION_PURPOSE,
        V3_TARGETED_FOLLOWUP_PURPOSE,
    }
)
JSONValue: TypeAlias = (
    None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)
Result = TypeVar("Result")
IS_WINDOWS = sys.platform == "win32"

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
        max_tokens: int = DEFAULT_MAX_TOKENS,
        input_usd_per_million: str = "0.1",
        output_usd_per_million: str = "0.5",
        cap_usd: Decimal = INTERNAL_CAP_USD,
        request_overhead_bytes: int = 4096,
        uncapped: bool = False,
        uncapped_purpose: str | None = None,
    ) -> None:
        """Create or reopen a ledger after checking pinned registry and policy.

        Raises:
            ProxyBudgetError: If configuration or pinned model metadata is invalid.
        """
        # Purpose-specific ledgers are private and durable; callers cannot redirect
        # uncapped campaigns to an arbitrary ledger path.
        del ledger_path
        self.uncapped = uncapped
        self.uncapped_purpose = uncapped_purpose
        self._requires_capped_ledger = uncapped and uncapped_purpose not in {
            SOL_ADJUDICATION_PURPOSE,
            V3_ADJUDICATION_PURPOSE,
            V3_EXTENDED_ADJUDICATION_PURPOSE,
            V3_LONG_ADJUDICATION_PURPOSE,
            V3_TARGETED_FOLLOWUP_PURPOSE,
        }
        self.path = self._ledger_path(uncapped=uncapped, purpose=uncapped_purpose)
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
        if uncapped:
            self.pins["uncapped"] = True
            if uncapped_purpose is not None:
                self.pins["uncapped_purpose"] = uncapped_purpose
        self.cap = Decimal(cap_usd)
        self.overhead = request_overhead_bytes
        expected_model = MODEL
        expected_max_tokens = DEFAULT_MAX_TOKENS
        expected_input_price = Decimal("0.1")
        expected_output_price = Decimal("0.5")
        if uncapped_purpose in {
            SOL_ADJUDICATION_PURPOSE,
            V3_ADJUDICATION_PURPOSE,
            V3_EXTENDED_ADJUDICATION_PURPOSE,
            V3_LONG_ADJUDICATION_PURPOSE,
            V3_TARGETED_FOLLOWUP_PURPOSE,
        }:
            expected_model = SOL_ADJUDICATION_MODEL
            expected_max_tokens = SOL_ADJUDICATION_LEDGER_MAX_TOKENS
            if uncapped_purpose == SOL_ADJUDICATION_PURPOSE:
                expected_input_price = Decimal("2")
                expected_output_price = Decimal("10")
        try:
            input_price = Decimal(input_usd_per_million)
            output_price = Decimal(output_usd_per_million)
        except InvalidOperation as exc:
            raise ProxyBudgetError(
                "Proxy budget configuration is not within pinned policy"
            ) from exc
        if (
            model != expected_model
            or base_url != BASE_URL
            or max_tokens != expected_max_tokens
            or input_price != expected_input_price
            or output_price != expected_output_price
            or (not uncapped and not Decimal("0") < self.cap <= INTERNAL_CAP_USD)
            or (
                uncapped_purpose is not None
                and (not uncapped or uncapped_purpose not in UNLIMITED_PURPOSES)
            )
            or request_overhead_bytes < 0
            or not all((campaign, source_hash, prompt_hash, schema_hash))
        ):
            raise ProxyBudgetError(
                "Proxy budget configuration is not within pinned policy"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._locked(self._initialise)

    @staticmethod
    def _ledger_path(*, uncapped: bool, purpose: str | None) -> Path:
        if not uncapped:
            return USER_BUDGET_PATH
        purpose_paths = {
            SOL_ADJUDICATION_PURPOSE: USER_SOL_ADJUDICATION_BUDGET_PATH,
            V3_ADJUDICATION_PURPOSE: USER_V3_ADJUDICATION_BUDGET_PATH,
            V3_EXTENDED_ADJUDICATION_PURPOSE: (
                USER_V3_EXTENDED_ADJUDICATION_BUDGET_PATH
            ),
            V3_LONG_ADJUDICATION_PURPOSE: USER_V3_LONG_ADJUDICATION_BUDGET_PATH,
            V3_TARGETED_FOLLOWUP_PURPOSE: Path.home()
            / ".danish-personas/proxy-v3-targeted-followup.jsonl",
            PATCH_VERIFICATION_PURPOSE: USER_PATCH_VERIFICATION_BUDGET_PATH,
            EDUCATION_REVIEW_PURPOSE: USER_EDUCATION_REVIEW_BUDGET_PATH,
            EDUCATION_VERIFICATION_PURPOSE: USER_EDUCATION_VERIFICATION_BUDGET_PATH,
        }
        return purpose_paths.get(purpose, USER_UNCAPPED_BUDGET_PATH)

    def _locked(self, function: Callable[[], Result]) -> Result:
        if self._requires_capped_ledger:
            return _locked_path(
                USER_BUDGET_PATH, lambda: _locked_path(self.path, function)
            )
        return _locked_path(self.path, function)

    def _initialise(self) -> None:
        self._check_registry()
        if self._requires_capped_ledger:
            self._refresh_old_ledger_pin()
        if self.path.exists():
            header, _ = self._load()
            self._check_header(header)
            os.chmod(self.path, 0o600)
            return
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as ledger:
            ledger.write(_canonical_json(self.pins).decode() + "\n")
            if not self.uncapped:
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
        _fsync_directory(self.path.parent)

    def _check_header(self, header: dict[str, JSONValue]) -> None:
        self._check_registry()
        if self._requires_capped_ledger:
            self._refresh_old_ledger_pin()
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
            registry_token_limit = (
                DEFAULT_MAX_TOKENS
                if self.pins["model"] == SOL_ADJUDICATION_MODEL
                else self.pins["max_tokens"]
            )
            if (
                model["maxTokens"] != registry_token_limit
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
            InvalidOperation,
            json.JSONDecodeError,
        ) as exc:
            raise ProxyBudgetError(
                "Model registry is missing, changed, or unbounded"
            ) from exc

    def _refresh_old_ledger_pin(self) -> None:
        try:
            _, _, contents = _read_ledger(USER_BUDGET_PATH)
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            InvalidOperation,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ProxyBudgetError(
                "Original capped proxy budget ledger is missing or incomplete"
            ) from exc
        self.pins["old_ledger_sha256"] = hashlib.sha256(contents).hexdigest()

    def _load(self) -> tuple[dict[str, JSONValue], list[dict[str, JSONValue]]]:
        try:
            header, records, _ = _read_ledger(
                self.path, enforce_hard_cap=not self.uncapped
            )
            return header, records
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            InvalidOperation,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
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
                str(r["request_id"]): r for r in records if r["type"] == "reservation"
            }
            reservation = reservations.get(request_id)
            if reservation is None:
                raise ProxyBudgetError("Usage references an unknown request ID")
            if any(
                r.get("request_id") == request_id and r["type"] == "usage"
                for r in records
            ):
                raise ProxyBudgetError("Usage already recorded for request ID")
            input_bound = reservation.get("input_byte_bound")
            output_bound = reservation.get("max_output_tokens")
            if not isinstance(input_bound, int) or not isinstance(output_bound, int):
                raise ProxyBudgetError(
                    "Usage references a reservation without token bounds"
                )
            if input_tokens > input_bound:
                raise ProxyBudgetError("Observed token usage exceeds reserved bounds")
            output_overage = output_tokens - output_bound
            if output_overage > 0 and not self._records_unbounded_sol_output():
                raise ProxyBudgetError("Observed token usage exceeds reserved bounds")
            usage: dict[str, JSONValue] = {
                "type": "usage",
                "request_id": request_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "response_sha256": response_sha256,
            }
            if output_overage > 0:
                usage["unbounded_output"] = True
                usage["reserved_output_tokens"] = output_bound
                usage["output_tokens_over_reserved"] = output_overage
            self._append(usage)

        self._locked(operation)

    def _append(self, record: dict[str, JSONValue]) -> None:
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND)
        try:
            if IS_WINDOWS:
                os.chmod(self.path, 0o600)
            else:
                os.fchmod(fd, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8", closefd=False) as ledger:
                ledger.write(_canonical_json(record).decode() + "\n")
                ledger.flush()
                os.fsync(fd)
        finally:
            os.close(fd)

    def _records_unbounded_sol_output(self) -> bool:
        return self.uncapped and self.uncapped_purpose in {
            SOL_ADJUDICATION_PURPOSE,
            V3_ADJUDICATION_PURPOSE,
            V3_EXTENDED_ADJUDICATION_PURPOSE,
            V3_LONG_ADJUDICATION_PURPOSE,
        }

    def reserve_attempt(
        self,
        request_id: str,
        request: dict[str, JSONValue],
        *,
        max_output_tokens: int | None = None,
    ) -> Decimal:
        """Durably reserve worst-case cost before an HTTP attempt.

        Args:
            request_id: Unique identifier for this HTTP attempt.
            request: JSON-compatible provider request payload.
            max_output_tokens (optional): Smaller response-token bound to reserve.
                Defaults to the pinned model maximum.

        Returns:
            The USD amount reserved for this attempt.

        Raises:
            ProxyBudgetError: If the request is invalid or the budget is exhausted.
        """
        if not request_id or not isinstance(request_id, str):
            raise ProxyBudgetError("Request ID must be a non-empty string")
        output_bound = self._normalise_output_bound(max_output_tokens)
        request_bytes = len(_canonical_json(request)) + self.overhead
        # One token per UTF-8 byte is deliberately conservative.
        input_tokens = request_bytes
        per_request = (
            Decimal(input_tokens) * Decimal(str(self.pins["input_usd_per_million"]))
            + Decimal(output_bound) * Decimal(str(self.pins["output_usd_per_million"]))
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
            if not self.uncapped and (
                total + per_request > self.cap or total + per_request > HARD_CAP_USD
            ):
                raise ProxyBudgetError("Proxy campaign budget cap exhausted")
            self._append(
                {
                    "type": "reservation",
                    "request_id": request_id,
                    "usd": str(per_request),
                    "input_byte_bound": request_bytes,
                    "max_output_tokens": output_bound,
                }
            )
            return per_request

        return self._locked(operation)

    def _normalise_output_bound(self, max_output_tokens: int | None) -> int:
        model_limit = self.pins["max_tokens"]
        if not isinstance(model_limit, int):
            raise ProxyBudgetError("Pinned model token limit is invalid")
        if max_output_tokens is None:
            return model_limit
        if (
            not isinstance(max_output_tokens, int)
            or max_output_tokens <= 0
            or max_output_tokens > model_limit
        ):
            raise ProxyBudgetError("Response token bound is outside pinned policy")
        return max_output_tokens

    def reserve_next_attempt(
        self,
        request_id_prefix: str,
        request: dict[str, JSONValue],
        *,
        max_output_tokens: int | None = None,
        max_attempts: int,
    ) -> int:
        """Reserve the next durable per-row attempt suffix.

        Args:
            request_id_prefix:
                Stable request ID prefix without the trailing attempt suffix.
            request:
                JSON-compatible provider request payload.
            max_output_tokens (optional):
                Smaller response-token bound to reserve.
            max_attempts:
                Maximum lifetime attempts for the row.

        Returns:
            Reserved one-based attempt suffix.

        Raises:
            ProxyBudgetError:
                If all lifetime attempts are already reserved or the reservation is
                invalid.
        """
        if not request_id_prefix or not isinstance(request_id_prefix, str):
            raise ProxyBudgetError("Request ID prefix must be a non-empty string")
        if not isinstance(max_attempts, int) or max_attempts <= 0:
            raise ProxyBudgetError("Maximum attempt count must be positive")
        output_bound = self._normalise_output_bound(max_output_tokens)
        request_bytes = len(_canonical_json(request)) + self.overhead
        input_tokens = request_bytes
        per_request = (
            Decimal(input_tokens) * Decimal(str(self.pins["input_usd_per_million"]))
            + Decimal(output_bound) * Decimal(str(self.pins["output_usd_per_million"]))
        ) / Decimal(1_000_000)

        def operation() -> int:
            header, records = self._load()
            self._check_header(header)
            reservations = {
                str(r["request_id"]): Decimal(str(r["usd"]))
                for r in records
                if r["type"] == "reservation"
            }
            attempt = _next_attempt_suffix(
                request_ids=reservations.keys(),
                prefix=request_id_prefix,
                max_attempts=max_attempts,
            )
            total = sum(reservations.values(), Decimal(0))
            if not self.uncapped and (
                total + per_request > self.cap or total + per_request > HARD_CAP_USD
            ):
                raise ProxyBudgetError("Proxy campaign budget cap exhausted")
            self._append(
                {
                    "type": "reservation",
                    "request_id": f"{request_id_prefix}-{attempt}",
                    "usd": str(per_request),
                    "input_byte_bound": request_bytes,
                    "max_output_tokens": output_bound,
                }
            )
            return attempt

        return self._locked(operation)

    def usage_summary(self) -> dict[str, JSONValue]:
        """Return aggregate usage and an unverified billing estimate.

        The estimate is derived only from durable ledger records and pinned list
        prices. It is not a provider invoice or a confirmation of billed spend.
        """

        def operation() -> dict[str, JSONValue]:
            header, records = self._load()
            self._check_header(header)
            reservations = [r for r in records if r["type"] == "reservation"]
            usages = [r for r in records if r["type"] == "usage"]
            reserved_usd = sum(
                (Decimal(str(r["usd"])) for r in reservations), Decimal(0)
            )
            input_tokens = sum(
                _usage_token_count(record=r, key="input_tokens") for r in usages
            )
            output_tokens = sum(
                _usage_token_count(record=r, key="output_tokens") for r in usages
            )
            input_price = Decimal(str(self.pins["input_usd_per_million"]))
            output_price = Decimal(str(self.pins["output_usd_per_million"]))
            usage_estimate = (
                Decimal(input_tokens) * input_price
                + Decimal(output_tokens) * output_price
            ) / Decimal(1_000_000)
            return {
                "ledger_path": str(self.path),
                "model": str(self.pins["model"]),
                "uncapped": self.uncapped,
                "reservation_count": len(reservations),
                "usage_count": len(usages),
                "recorded_input_tokens": input_tokens,
                "recorded_output_tokens": output_tokens,
                "reserved_usd": str(reserved_usd),
                "estimated_recorded_usage_usd": str(usage_estimate),
                "invoice_verified": False,
            }

        return self._locked(operation)

    def validate_request_body(self, request_id: str, body: bytes) -> None:
        """Ensure an actual HTTP body fits a durable reservation.

        Callers should reserve an attempt and then validate the exact request
        body immediately before I/O. This check does not reserve, refund, or
        mutate ledger state.

        Raises:
            ProxyBudgetError:
                If the body or reservation is invalid.
        """
        if not isinstance(body, bytes):
            raise ProxyBudgetError("HTTP request body must be bytes")

        def operation() -> None:
            header, records = self._load()
            self._check_header(header)
            reservation = next(
                (
                    r
                    for r in records
                    if r["type"] == "reservation" and r.get("request_id") == request_id
                ),
                None,
            )
            if reservation is None:
                raise ProxyBudgetError("Body check references an unknown request ID")
            input_bound = reservation.get("input_byte_bound")
            if not isinstance(input_bound, int):
                raise ProxyBudgetError(
                    "Body check references a reservation without input bounds"
                )
            if len(body) > input_bound:
                raise ProxyBudgetError("HTTP request body exceeds reserved bounds")

        self._locked(operation)


class ProxyBudgetError(RuntimeError):
    """Raised when the durable proxy budget cannot safely authorise a request."""


def _canonical_json(value: JSONValue) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    if IS_WINDOWS:
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _locked_path(path: Path, function: Callable[[], Result]) -> Result:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    locked = False
    try:
        if IS_WINDOWS:
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
            if IS_WINDOWS:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _next_attempt_suffix(
    *, request_ids: Iterable[str], prefix: str, max_attempts: int
) -> int:
    used: set[int] = set()
    marker = f"{prefix}-"
    for request_id in request_ids:
        if not isinstance(request_id, str) or not request_id.startswith(marker):
            continue
        suffix = request_id.removeprefix(marker)
        if suffix.isdecimal():
            used.add(int(suffix))
    for attempt in range(1, max_attempts + 1):
        if attempt not in used:
            return attempt
    raise ProxyBudgetError("Per-row proxy attempt lifetime exhausted")


def _read_ledger(
    path: Path, *, enforce_hard_cap: bool = True
) -> tuple[dict[str, JSONValue], list[dict[str, JSONValue]], bytes]:
    contents = path.read_bytes()
    if not contents.endswith(b"\n"):
        raise ValueError("ledger does not end at a complete record")
    text = contents.decode("utf-8")
    lines = text.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("empty ledger line")
    records = [json.loads(line) for line in lines]
    header = records[0]
    if not isinstance(header, dict) or header.get("type") != "header":
        raise ValueError("missing header")
    return (
        header,
        _validate_records(
            records[1:],
            enforce_hard_cap=enforce_hard_cap,
            allow_unbounded_output=(
                header.get("uncapped") is True
                and header.get("uncapped_purpose")
                in {
                    SOL_ADJUDICATION_PURPOSE,
                    V3_ADJUDICATION_PURPOSE,
                    V3_EXTENDED_ADJUDICATION_PURPOSE,
                    V3_LONG_ADJUDICATION_PURPOSE,
                }
            ),
        ),
        contents,
    )


def _validate_records(
    records: list[object],
    *,
    enforce_hard_cap: bool = True,
    allow_unbounded_output: bool = False,
) -> list[dict[str, JSONValue]]:
    """Validate ledger events and optionally enforce the immutable hard cap.

    Args:
        records: Decoded JSON lines after the header.
        enforce_hard_cap: Whether reservations must remain below the capped
            ledger's immutable hard cap.
        allow_unbounded_output: Whether usage may record output above the
            reserved response bound with explicit overage fields.

    Returns:
        Validated reservation and usage records.

    Raises:
        ValueError: If a record is invalid or the requested cap is exceeded.
    """
    validated: list[dict[str, JSONValue]] = []
    reservations: dict[str, dict[str, JSONValue]] = {}
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
            reservations[identifier] = record
            total += amount
        else:
            _validate_usage(
                record=record,
                identifier=identifier,
                reservations=reservations,
                allow_unbounded_output=allow_unbounded_output,
            )
        validated.append(record)
    if enforce_hard_cap and total > HARD_CAP_USD:
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
    input_bound = record.get("input_byte_bound")
    output_bound = record.get("max_output_tokens")
    if set(record) - allowed or not isinstance(amount, str):
        raise ValueError("invalid reservation fields")
    if "input_byte_bound" in record and (
        not isinstance(input_bound, int) or input_bound < 0
    ):
        raise ValueError("invalid reservation input bound")
    if "max_output_tokens" in record and (
        not isinstance(output_bound, int) or output_bound <= 0
    ):
        raise ValueError("invalid reservation output bound")
    value = Decimal(amount)
    if not value.is_finite() or value <= 0:
        raise ValueError("invalid reservation")


def _validate_usage(
    *,
    record: dict[str, JSONValue],
    identifier: str,
    reservations: dict[str, dict[str, JSONValue]],
    allow_unbounded_output: bool,
) -> None:
    required_fields = {
        "type",
        "request_id",
        "input_tokens",
        "output_tokens",
        "response_sha256",
    }
    allowed_fields = set(required_fields)
    if allow_unbounded_output:
        allowed_fields |= {
            "unbounded_output",
            "reserved_output_tokens",
            "output_tokens_over_reserved",
        }
    if not required_fields <= set(record) or set(record) - allowed_fields:
        raise ValueError("invalid usage record")
    reservation = reservations.get(identifier)
    if reservation is None:
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
        or re.fullmatch(r"[0-9a-f]{64}", response_hash) is None
    ):
        raise ValueError("invalid usage fields")
    input_bound = reservation.get("input_byte_bound")
    output_bound = reservation.get("max_output_tokens")
    if not isinstance(input_bound, int) or not isinstance(output_bound, int):
        raise ValueError("usage references unbounded reservation")
    if input_tokens > input_bound:
        raise ValueError("usage exceeds reserved bounds")
    output_overage = output_tokens - output_bound
    overage_fields = {
        "unbounded_output",
        "reserved_output_tokens",
        "output_tokens_over_reserved",
    }
    if output_overage <= 0:
        if set(record) & overage_fields:
            raise ValueError("usage overage accounting is invalid")
        return
    if (
        record.get("unbounded_output") is not True
        or record.get("reserved_output_tokens") != output_bound
        or record.get("output_tokens_over_reserved") != output_overage
    ):
        raise ValueError("usage overage accounting is invalid")


def _usage_token_count(*, record: dict[str, JSONValue], key: str) -> int:
    """Reject malformed usage tokens rather than coercing ledger values.

    Returns:
        The recorded non-negative token count.

    Raises:
        ProxyBudgetError:
            If the ledger token count is malformed.
    """
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProxyBudgetError("Usage ledger token count is malformed")
    return value
