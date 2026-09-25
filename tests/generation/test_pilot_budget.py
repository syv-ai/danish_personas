"""Offline campaign-cost reservation tests."""

import concurrent.futures as futures
import json
import typing as t
from pathlib import Path

import pytest
from generation_test_helpers import (
    InterruptingGenerationClient,
    MockGenerationClient,
    RejectingGenerationClient,
    write_generation_inputs,
)

from danish_personas.generation.pilot import _PilotCostBudget, run_pilot


class _PilotCostKwargs(t.TypedDict):
    """Arguments shared by conservative settlement tests."""

    input_path: Path
    sample_manifest_path: Path
    config_path: Path
    output_dir: Path
    rows: int
    batch_size: int
    concurrency: int
    delay_between_batches: float
    maximum_total_requests: int
    input_price_per_million: float
    output_price_per_million: float
    maximum_campaign_cost_usd: float
    maximum_shard_cost_usd: float


def test_concurrent_reservations_stop_at_campaign_boundary(tmp_path: Path) -> None:
    """Concurrent shard scheduling cannot reserve beyond the cap."""
    budget = _budget(tmp_path / "ledger.json")

    def reserve(offset: int) -> type[None] | type[ValueError]:
        try:
            budget.reserve(offset=offset)
        except ValueError:
            return ValueError
        return type(None)

    with futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(reserve, range(4)))

    assert results.count(type(None)) == 2
    assert results.count(ValueError) == 2


def _budget(path: Path) -> _PilotCostBudget:
    return _PilotCostBudget(
        path=path,
        input_sha256="input",
        generation_config_sha256="config",
        model="mistral-small-4",
        input_price_per_million=0.3,
        output_price_per_million=1.2,
        maximum_campaign_cost_usd=2.0,
        maximum_shard_cost_usd=1.0,
        maximum_shard_requests=2,
    )


def test_failed_reservation_survives_resume_and_tampering_fails_closed(
    tmp_path: Path,
) -> None:
    """Unknown attempts remain reserved, and an altered identity is rejected."""
    path = tmp_path / "ledger.json"
    budget = _budget(path)
    budget.reserve(offset=0)
    resumed = _budget(path)
    resumed.reserve(offset=0)
    resumed.reserve(offset=1)
    with pytest.raises(ValueError, match="cap"):
        resumed.reserve(offset=2)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model"] = "different-model"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        _budget(path)

    payload["model"] = "mistral-small-4"
    payload["maximum_shard_requests"] = 3
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        _budget(path)


@pytest.mark.parametrize(
    ("client", "failure_attribute"),
    [
        (RejectingGenerationClient, "reject_once"),
        (InterruptingGenerationClient, "interrupt_once"),
    ],
)
def test_resume_retains_cost_for_unaccounted_http_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    client: type[MockGenerationClient],
    failure_attribute: str,
) -> None:
    """Malformed and pre-checkpoint responses retain conservative cost."""
    paths = write_generation_inputs(root=tmp_path)
    monkeypatch.setattr("danish_personas.generation.pipeline.OpenAIClient", client)
    client.requests = 0
    setattr(client, failure_attribute, True)
    kwargs: _PilotCostKwargs = {
        "input_path": paths["sample"],
        "sample_manifest_path": paths["sample_manifest"],
        "config_path": paths["config"],
        "output_dir": tmp_path / "pilot",
        "rows": 2,
        "batch_size": 1,
        "concurrency": 1,
        "delay_between_batches": 0.0,
        "maximum_total_requests": 10,
        "input_price_per_million": 0.3,
        "output_price_per_million": 1.2,
        "maximum_campaign_cost_usd": 1.1,
        "maximum_shard_cost_usd": 1.0,
    }
    with pytest.raises((ValueError, RuntimeError)):
        run_pilot(**kwargs)

    setattr(client, failure_attribute, False)
    with pytest.raises(ValueError, match="cap"):
        run_pilot(**kwargs)

    ledger_paths = list((tmp_path / "pilot").glob("*/pilot-cost-ledger.json"))
    payload = json.loads(ledger_paths[0].read_text(encoding="utf-8"))
    reservation = payload["reservations"][0]
    assert reservation["status"] == "settled"
    assert reservation["settled_cost_usd"] == pytest.approx(0.2 + 0.000027)
