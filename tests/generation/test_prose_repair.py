"""Offline contract tests for prose-only repair."""

from __future__ import annotations

import collections.abc as c
import json
import os
from hashlib import sha256
from pathlib import Path

import pytest

from danish_personas.generation.models import GenerationConfig, LLMResponse
from danish_personas.generation.prose_repair import RepairError, run_prose_repair


@pytest.fixture
def config() -> GenerationConfig:
    """Return a bounded offline generation configuration."""
    return GenerationConfig.model_construct(
        base_url="https://offline.invalid/v1",
        model="local-model",
        api_key_env=None,
        timeout_seconds=1,
        maximum_http_attempts=2,
        maximum_total_requests=None,
        retry_backoff_seconds=0,
        maximum_rows_per_shard=1,
        same_sex_partner_probability=0.1,
        max_tokens=256,
        enable_thinking=None,
        reasoning_effort=None,
        prompt=Path("repair.md"),
        job_title_mapping=None,
        origin_label_contract=Path("contract.yaml"),
    )


class FakeClient:
    """Capture the request payload and emulate durable attempt callbacks."""

    def __init__(self, content: str, attempts: int = 1) -> None:
        """Initialise canned response content and request count."""
        self.content = content
        self.attempts = attempts
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
            model="local-model",
            content=self.content,
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
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
        "model": "local-model",
        "config": config,
        "input_manifest_sha256": "manifest",
        "sidecar_sha256": "schema",
        "output_dir": tmp_path,
        "cost_cap_usd": 1.0,
        "client": client,
    }
    defaults.update(kwargs)
    return run_prose_repair(**defaults)  # type: ignore[arg-type]


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
    assert set(row.items()).issubset(set(result[0].items()))
    resumed = _run(tmp_path, config, FakeClient("unused"), rows=[row])
    assert resumed == result
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert len(ledger["reservations"]) == 1
    assert (tmp_path / "ledger.json").stat().st_mode & 0o777 == 0o600
    assert os.stat(tmp_path).st_mode & 0o777 == 0o700


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


def test_attempt_reservations_are_durable_and_cap_checked(
    tmp_path: Path, config: GenerationConfig
) -> None:
    """Reject an unaffordable attempt before recording a reservation."""
    client = FakeClient(json.dumps({"persona": "d" * 300}), attempts=2)
    with pytest.raises(RepairError, match="Cost cap"):
        _run(tmp_path, config, client, cost_cap_usd=0.0000001)
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert ledger["reservations"] == []


@pytest.mark.parametrize("cap", [float("nan"), float("inf"), 100.01])
def test_invalid_cost_cap_fails_closed(
    tmp_path: Path, config: GenerationConfig, cap: float
) -> None:
    """Reject non-finite and over-limit cost caps before starting a repair."""
    with pytest.raises(RepairError, match="Cost cap"):
        _run(tmp_path, config, FakeClient("unused"), cost_cap_usd=cap)


def test_changed_row_and_unbounded_tokens_fail_closed(
    tmp_path: Path, config: GenerationConfig
) -> None:
    """Reject stale checkpoints and non-positive token limits."""
    _run(tmp_path, config, FakeClient(json.dumps({"persona": "d" * 300})))
    changed = [{"id": "a", "age": 41, "gender": "female", "job_title": "lærer"}]
    with pytest.raises(RepairError, match="Checkpoint"):
        _run(tmp_path, config, FakeClient("unused"), rows=changed)
    unbounded = GenerationConfig.model_construct(**{**config.__dict__, "max_tokens": 0})
    with pytest.raises(RepairError, match="max_tokens"):
        _run(tmp_path / "other", unbounded, FakeClient("unused"))
