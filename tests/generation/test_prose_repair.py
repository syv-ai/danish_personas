"""Offline contract tests for prose-only repair."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from danish_personas.generation.models import GenerationConfig, LLMResponse
from danish_personas.generation.prose_repair import RepairError, run_prose_repair


@pytest.fixture
def config() -> GenerationConfig:
    return GenerationConfig.model_construct(
        base_url="https://offline.invalid/v1", model="local-model", api_key_env=None,
        timeout_seconds=1, maximum_http_attempts=2, maximum_total_requests=None,
        retry_backoff_seconds=0, maximum_rows_per_shard=1,
        same_sex_partner_probability=0.1, max_tokens=256, enable_thinking=None,
        reasoning_effort=None, prompt=Path("repair.md"), job_title_mapping=None,
        origin_label_contract=Path("contract.yaml"),
    )


class FakeClient:
    def __init__(self, content: str, attempts: int = 1) -> None:
        self.content = content
        self.attempts = attempts
        self.payload: dict[str, Any] | None = None
        self.reservations = 0

    def complete(self, *, user_payload: dict[str, Any], record_request: Any, **_: Any) -> LLMResponse:
        self.payload = user_payload
        for number in range(1, self.attempts + 1):
            record_request(number)
            self.reservations += 1
        return LLMResponse(
            response_id="response-1", model="local-model", content=self.content,
            prompt_tokens=10, completion_tokens=20, total_tokens=30,
            request_attempts=self.attempts, latency_seconds=0,
            raw_response_sha256=sha256(b"response").hexdigest(),
        )


def _run(tmp_path: Path, config: GenerationConfig, client: FakeClient, **kwargs: Any) -> list[dict[str, Any]]:
    defaults = {
        "rows": [{"id": "a", "age": 40, "gender": "female", "partner_gender": "male",
                  "sexual_orientation": "private", "same_sex_partner_target": True,
                  "job_title": "lærer"}],
        "changed_fields": {"a": {"job_title"}}, "id_field": "id",
        "prompt": "Skriv persona.", "model": "local-model", "config": config,
        "input_manifest_sha256": "manifest", "sidecar_sha256": "schema",
        "output_dir": tmp_path, "cost_cap_usd": 1.0, "client": client,
    }
    defaults.update(kwargs)
    return run_prose_repair(**defaults)


def test_schema_prose_repair_payload_allowlist_and_resume(tmp_path: Path, config: GenerationConfig) -> None:
    prose = "d" * 300
    client = FakeClient(json.dumps({"persona": prose}))
    row = {"id": "a", "age": 40, "gender": "female", "partner_gender": "male",
           "sexual_orientation": "private", "same_sex_partner_target": True,
           "job_title": "lærer"}
    result = _run(tmp_path, config, client, rows=[row])
    assert result[0]["persona"] == prose
    assert client.payload == {"age": 40, "gender": "female", "partner_gender": "male", "job_title": "lærer"}
    assert set(row.items()).issubset(set(result[0].items()))
    resumed = _run(tmp_path, config, FakeClient("unused"), rows=[row])
    assert resumed == result
    assert len(json.loads((tmp_path / "ledger.json").read_text())["reservations"]) == 1


def test_attempt_reservations_are_durable_and_cap_checked(tmp_path: Path, config: GenerationConfig) -> None:
    client = FakeClient(json.dumps({"persona": "d" * 300}), attempts=2)
    with pytest.raises(RepairError, match="Cost cap"):
        _run(tmp_path, config, client, cost_cap_usd=0.0000001)
    # Refusal takes place in the pre-request callback; reservation is not refunded.
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert ledger["reservations"] == []


def test_changed_row_and_unbounded_tokens_fail_closed(tmp_path: Path, config: GenerationConfig) -> None:
    _run(tmp_path, config, FakeClient(json.dumps({"persona": "d" * 300})))
    changed = [{"id": "a", "age": 41, "gender": "female", "partner_gender": "male",
                "job_title": "lærer"}]
    with pytest.raises(RepairError, match="Checkpoint"):
        _run(tmp_path, config, FakeClient("unused"), rows=changed)
    unbounded = GenerationConfig.model_construct(**{**config.__dict__, "max_tokens": None})
    with pytest.raises(RepairError, match="max_tokens"):
        _run(tmp_path / "other", unbounded, FakeClient("unused"))
