"""Minimal retrying client for OpenAI-compatible chat completions."""

import json
import time
from hashlib import sha256
from time import monotonic

import httpx
from pydantic import BaseModel

from .models import GenerationConfig, LLMResponse


class OpenAIClient:
    """Call one OpenAI-compatible model with bounded retries."""

    def __init__(
        self,
        config: GenerationConfig,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Initialise the client.

        Args:
            config:
                Validated generation configuration.
            api_key (optional):
                Optional bearer token. Defaults to no authentication.
            transport (optional):
                Optional HTTPX transport for tests. Defaults to network transport.

        Raises:
            ValueError:
                If the enabled configuration omits its base URL or model.
        """
        if not config.base_url or not config.model:
            message = "Enabled generation requires base_url and model"
            raise ValueError(message)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._config = config
        self._requests_made = 0
        self._client = httpx.Client(
            base_url=config.base_url.rstrip("/") + "/",
            headers=headers,
            timeout=config.timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def complete(
        self,
        system_prompt: str,
        user_payload: dict[str, object],
        schema_name: str,
        json_schema: dict[str, object],
    ) -> LLMResponse:
        """Return one schema-constrained chat completion.

        Args:
            system_prompt:
                Stage-specific Danish instruction.
            user_payload:
                Immutable structured input for the model.
            schema_name:
                Response schema name.
            json_schema:
                JSON Schema accepted by the endpoint.

        Returns:
            Completion text and auditable metadata.

        Raises:
            httpx.HTTPStatusError:
                If a permanent HTTP error or final retry fails.
            ValueError:
                If the response omits a completion choice.
            RequestBudgetExceeded:
                If another HTTP request would exceed the smoke budget.
            RuntimeError:
                If retry handling ends without a captured HTTP error.
        """
        body = self._request_body(
            system_prompt=system_prompt,
            user_payload=user_payload,
            schema_name=schema_name,
            json_schema=json_schema,
        )
        last_error: httpx.HTTPStatusError | httpx.TransportError | None = None
        for attempt in range(self._config.maximum_http_attempts):
            if self._requests_made >= self._config.maximum_total_requests:
                message = "Generation HTTP request budget is exhausted"
                raise RequestBudgetExceeded(message)
            started = monotonic()
            try:
                self._requests_made += 1
                response = self._client.post("chat/completions", json=body)
                response.raise_for_status()
                raw = response.content
                parsed = _Completion.model_validate(response.json())
                if not parsed.choices:
                    message = "Completion response contains no choices"
                    raise ValueError(message)
                usage = parsed.usage
                return LLMResponse(
                    response_id=parsed.id,
                    model=parsed.model,
                    content=parsed.choices[0].message.content,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    request_attempts=attempt + 1,
                    latency_seconds=monotonic() - started,
                    estimated_cost_usd=usage.estimated_cost,
                    inference_provider=response.headers.get("x-inference-provider"),
                    raw_response_sha256=sha256(raw).hexdigest(),
                )
            except httpx.HTTPStatusError as error:
                last_error = error
                if error.response.status_code not in {429, 500, 502, 503, 504}:
                    raise
            except httpx.TransportError as error:
                last_error = error
            if attempt + 1 < self._config.maximum_http_attempts:
                time.sleep(self._config.retry_backoff_seconds * (2**attempt))
        if last_error is None:
            message = "Completion failed without a captured HTTP error"
            raise RuntimeError(message)
        raise last_error

    def _request_body(
        self,
        system_prompt: str,
        user_payload: dict[str, object],
        schema_name: str,
        json_schema: dict[str, object],
    ) -> dict[str, object]:
        body: dict[str, object] = {
            "model": self._config.model or "",
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            "response_format": self._response_format(
                schema_name=schema_name, json_schema=json_schema
            ),
        }
        if self._config.max_tokens is not None:
            body["max_tokens"] = self._config.max_tokens
        if self._config.enable_thinking is not None:
            body["chat_template_kwargs"] = {
                "enable_thinking": self._config.enable_thinking
            }
        return body

    def _response_format(
        self, schema_name: str, json_schema: dict[str, object]
    ) -> dict[str, object]:
        if self._config.response_format == "json_object":
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": json_schema},
        }

    @property
    def requests_made(self) -> int:
        """HTTP attempts made by this client instance."""
        return self._requests_made


class RequestBudgetExceeded(RuntimeError):
    """Raised before a request would exceed the configured smoke budget."""


class _Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float | None = None


class _Message(BaseModel):
    content: str


class _Choice(BaseModel):
    message: _Message


class _Completion(BaseModel):
    id: str
    model: str
    choices: list[_Choice]
    usage: _Usage = _Usage()
