"""Offline campaign-cost reservation tests."""

import concurrent.futures as futures
import json
from pathlib import Path

import pytest

from danish_personas.generation.pilot import _PilotCostBudget


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
