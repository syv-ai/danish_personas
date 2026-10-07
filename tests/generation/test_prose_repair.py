"""Offline contract tests for prose-only repair."""

from __future__ import annotations

import collections.abc as c
import json
import os
from hashlib import sha256
from pathlib import Path

import httpx
import pytest

from danish_personas.generation.client import OpenAIClient
from danish_personas.generation.models import GenerationConfig, LLMResponse
from danish_personas.generation.prose_repair import RepairError, run_prose_repair


@pytest.fixture
def config() -> GenerationConfig:
    """Return a bounded offline generation configuration."""
    return GenerationConfig.model_construct(
        base_url="https://api.mistral.ai/v1",
        model="mistral-small-2603",
        api_key_env="MISTRAL_API_KEY",
        timeout_seconds=1,
        maximum_http_attempts=2,
        maximum_total_requests=None,
        retry_backoff_seconds=0,
        maximum_rows_per_shard=1,
        same_sex_partner_probability=0.1,
        max_tokens=800,
        enable_thinking=None,
        reasoning_effort=None,
        prompt=Path("repair.md"),
        job_title_mapping=None,
        origin_label_contract=Path("contract.yaml"),
    )


def test_attempt_reservations_are_durable_and_cap_checked(
    tmp_path: Path, config: GenerationConfig
) -> None:
    """Reject an unaffordable attempt before recording a reservation."""
    client = FakeClient(json.dumps({"persona": "d" * 300}), attempts=2)
    with pytest.raises(RepairError, match="Cost cap"):
        _run(tmp_path, config, client, cost_cap_usd=0.0000001)
    ledger = [
        json.loads(line)
        for line in (tmp_path / "ledger.jsonl").read_text().splitlines()
    ]
    assert len(ledger) == 1
    assert ledger[0]["type"] == "header"

    retry_dir = tmp_path / "retry"
    _run(retry_dir, config, client, cost_cap_usd=100.0)
    retry_ledger = [
        json.loads(line)
        for line in (retry_dir / "ledger.jsonl").read_text().splitlines()
    ]
    assert len(retry_ledger[1:]) == 2
    assert client.reservations == 2


class FakeClient:
    """Capture the request payload and emulate durable attempt callbacks."""

    def __init__(
        self,
        content: str,
        attempts: int = 1,
        model: str = "mistral-small-2603",
        prompt_tokens: int = 10,
        completion_tokens: int = 20,
    ) -> None:
        """Initialise canned response content and request count."""
        self.content = content
        self.attempts = attempts
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.payload: dict[str, object] | None = None
        self.reservations = 0

    def complete(
        self,
        *,
        user_payload: dict[str, object],
        record_request: c.Callable[[int], None],
        **kwargs: object,
    ) -> LLMResponse:
        """Capture a request and invoke its reservation callback per attempt.

        Returns:
            A canned completion response.
        """
        del kwargs
        self.payload = user_payload
        for number in range(1, self.attempts + 1):
            record_request(number)
            self.reservations += 1
        return LLMResponse(
            response_id="response-1",
            model=self.model,
            content=self.content,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.prompt_tokens + self.completion_tokens,
            request_attempts=self.attempts,
            latency_seconds=0,
            raw_response_sha256=sha256(b"response").hexdigest(),
        )


def _run(
    tmp_path: Path, config: GenerationConfig, client: FakeClient, **kwargs: object
) -> list[dict[str, object]]:
    """Run the repair fixture with overridable input values.

    Returns:
        The repaired fixture rows.
    """
    defaults: dict[str, object] = {
        "rows": [
            {
                "id": "a",
                "age": 40,
                "gender": "female",
                "partner_gender": "male",
                "sexual_orientation": "private",
                "same_sex_partner_target": True,
                "job_title": "lærer",
            }
        ],
        "changed_fields": {"a": {"job_title"}},
        "id_field": "id",
        "prompt": "Skriv persona.",
        "model": "mistral-small-2603",
        "config": config,
        "input_manifest_sha256": "manifest",
        "sidecar_sha256": "schema",
        "output_dir": tmp_path,
        "cost_cap_usd": 1.0,
        "client": client,
    }
    defaults.update(kwargs)
    return run_prose_repair(**defaults)  # type: ignore[arg-type]


def test_changed_fields_are_reason_metadata_not_payload_allowlist(
    tmp_path: Path, config: GenerationConfig
) -> None:
    """Accept changes to non-provider fields without exposing those fields."""
    client = FakeClient(json.dumps({"persona": "d" * 300}))
    row = {"id": "a", "age": 40, "age_band": "35-44", "marital_status": "single"}
    _run(
        tmp_path,
        config,
        client,
        rows=[row],
        changed_fields={"a": {"age_band", "marital_status"}},
    )
    assert client.payload == {"age": 40}


def test_changed_row_and_unbounded_tokens_fail_closed(
    tmp_path: Path, config: GenerationConfig
) -> None:
    """Reject stale checkpoints and non-positive token limits."""
    _run(tmp_path, config, FakeClient(json.dumps({"persona": "d" * 300})))
    changed = [{"id": "a", "age": 41, "gender": "female", "job_title": "lærer"}]
    with pytest.raises(RepairError, match="Stale or malformed repair ledger"):
        _run(tmp_path, config, FakeClient("unused"), rows=changed)
    unbounded = GenerationConfig.model_construct(**{**config.__dict__, "max_tokens": 0})
    with pytest.raises(RepairError, match="max_tokens"):
        _run(tmp_path / "other", unbounded, FakeClient("unused"))


def test_cost_cap_of_one_hundred_usd_is_allowed(
    tmp_path: Path, config: GenerationConfig
) -> None:
    """Accept the inclusive maximum cap without weakening per-attempt checks."""
    client = FakeClient(json.dumps({"persona": "d" * 300}))
    _run(tmp_path, config, client, cost_cap_usd=100.0)
    assert client.reservations == 1


def test_interrupted_request_resumes_with_cumulative_reservation(
    tmp_path: Path, config: GenerationConfig
) -> None:
    """Keep a durable reservation after interruption and resume without rewriting it."""

    class InterruptedClient(FakeClient):
        def complete(
            self, *, record_request: c.Callable[[int], None], **kwargs: object
        ) -> LLMResponse:
            del kwargs
            record_request(1)
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        _run(tmp_path, config, InterruptedClient("unused"))

    ledger_path = tmp_path / "ledger.jsonl"
    first_run = ledger_path.read_text().splitlines()
    assert len(first_run) == 2
    _run(tmp_path, config, FakeClient(json.dumps({"persona": "d" * 300})))
    resumed = ledger_path.read_text().splitlines()
    assert len(resumed) == 3
    assert resumed[:2] == first_run


@pytest.mark.parametrize("cap", [float("nan"), float("inf"), 100.01])
def test_invalid_cost_cap_fails_closed(
    tmp_path: Path, config: GenerationConfig, cap: float
) -> None:
    """Reject non-finite and over-limit cost caps before starting a repair."""
    with pytest.raises(RepairError, match="Cost cap"):
        _run(tmp_path, config, FakeClient("unused"), cost_cap_usd=cap)


def test_malformed_ledger_reservation_fails_closed(
    tmp_path: Path, config: GenerationConfig
) -> None:
    """Reject malformed durable request accounting before any provider call."""
    _run(tmp_path, config, FakeClient(json.dumps({"persona": "d" * 300})))
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text(
        "\n".join(
            [
                *ledger_path.read_text().splitlines()[:1],
                json.dumps({"type": "reservation", "id": "a", "usd": "not-a-number"}),
            ]
        )
        + "\n"
    )
    client = FakeClient("unused")
    with pytest.raises(RepairError, match="Malformed repair ledger reservation"):
        _run(tmp_path, config, client)
    assert client.reservations == 0


def test_model_argument_must_match_mistral_model(
    tmp_path: Path, config: GenerationConfig
) -> None:
    """Reject an alternate requested model before invoking the client."""
    client = FakeClient("unused")
    with pytest.raises(RepairError, match="mistral-small-2603"):
        _run(tmp_path, config, client, model="another-model")
    assert client.payload is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base_url", "https://wrong.example/v1"),
        ("model", "another-model"),
        ("api_key_env", "OTHER_API_KEY"),
    ],
)
def test_provider_configuration_must_match_mistral(
    tmp_path: Path, config: GenerationConfig, field: str, value: str
) -> None:
    """Reject non-Mistral provider settings before invoking the client."""
    wrong_config = GenerationConfig.model_construct(**{**config.__dict__, field: value})
    client = FakeClient("unused")
    with pytest.raises(RepairError, match="Mistral|mistral-small-2603|MISTRAL_API_KEY"):
        _run(tmp_path, wrong_config, client)
    assert client.payload is None
    assert not (tmp_path / "ledger.jsonl").exists()


def test_real_client_mock_reserves_before_network(
    tmp_path: Path, config: GenerationConfig
) -> None:
    """Exercise the actual HTTP client with an offline transport and hard cap."""
    network_requests: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        network_requests.append(request.content)
        return httpx.Response(
            status_code=200,
            json={
                "id": "offline-response",
                "model": "mistral-small-2603",
                "choices": [
                    {"message": {"content": json.dumps({"persona": "d" * 300})}}
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 100,
                    "total_tokens": 112,
                },
            },
        )

    row = {
        "id": "private-id",
        "age": 30,
        "gender": "nonbinary",
        "partner_gender": "woman",
        "sexual_orientation": "secret-orientation-sentinel",
        "transgender": "secret-trans-sentinel",
    }
    client = OpenAIClient(config=config, transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(RepairError, match="Cost cap"):
            _run(
                tmp_path / "rejected",
                config,
                client,
                rows=[row],
                changed_fields={"private-id": ["gender"]},
                cost_cap_usd=0.000001,
            )
        assert network_requests == []
        repaired = _run(
            tmp_path / "accepted",
            config,
            client,
            rows=[row],
            changed_fields={"private-id": ["gender"]},
            cost_cap_usd=1.0,
        )
    finally:
        client.close()
    assert repaired[0]["persona"] == "d" * 300
    assert len(network_requests) == 1
    assert b"secret-orientation-sentinel" not in network_requests[0]
    assert b"secret-trans-sentinel" not in network_requests[0]
    assert b"nonbinary" in network_requests[0]
    assert b"woman" in network_requests[0]


@pytest.mark.parametrize(
    ("response_model", "prompt_tokens", "completion_tokens"),
    [
        ("other-model", 10, 20),
        ("mistral-small-2603", 100_000, 20),
        ("mistral-small-2603", 10, 801),
    ],
)
def test_response_usage_and_model_must_match_reservation(
    tmp_path: Path,
    config: GenerationConfig,
    response_model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    """Fail closed when provider metadata exceeds the requested bounds."""
    client = FakeClient(
        json.dumps({"persona": "d" * 300}),
        model=response_model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    with pytest.raises(RepairError, match="response model|usage exceeds"):
        _run(tmp_path, config, client)


def test_schema_prose_repair_payload_allowlist_and_resume(
    tmp_path: Path, config: GenerationConfig
) -> None:
    """Repair prose while excluding private fields and supporting resume."""
    prose = "d" * 300
    client = FakeClient(json.dumps({"persona": prose}))
    row = {
        "id": "a",
        "age": 40,
        "gender": "female",
        "partner_gender": "male",
        "sexual_orientation": "private",
        "partner_sexual_orientation": "private",
        "transgender": True,
        "partner_transgender": True,
        "variation_in_sex_characteristics": True,
        "partner_variation_in_sex_characteristics": True,
        "same_sex_partner_target": True,
        "job_title": "lærer",
    }
    result = _run(tmp_path, config, client, rows=[row])
    assert result[0]["persona"] == prose
    assert client.payload == {
        "age": 40,
        "gender": "female",
        "partner_gender": "male",
        "job_title": "lærer",
    }
    assert {"gender", "partner_gender"}.issubset(client.payload)
    assert not set(client.payload).intersection(
        {
            "sexual_orientation",
            "partner_sexual_orientation",
            "transgender",
            "partner_transgender",
            "variation_in_sex_characteristics",
            "partner_variation_in_sex_characteristics",
        }
    )
    assert set(row.items()).issubset(set(result[0].items()))
    resumed = _run(tmp_path, config, FakeClient("unused"), rows=[row])
    assert resumed == result
    ledger_path = tmp_path / "ledger.jsonl"
    lines = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert lines[0]["type"] == "header"
    binding = lines[0]["binding"]
    assert binding["pricing_currency"] == "USD"
    assert binding["input_usd_per_million"] == 0.15
    assert binding["output_usd_per_million"] == 0.60
    assert binding["pricing_source"] == ("https://docs.mistral.ai/inference/pricing")
    assert "usd_per_eur" not in binding
    assert len(lines[1:]) == 1
    assert (ledger_path.stat().st_mode & 0o777) == 0o600
    assert os.stat(tmp_path).st_mode & 0o777 == 0o700
