"""Tests for durable local proxy request reservations."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from danish_personas.generation.proxy_budget import ProxyBudget, ProxyBudgetError


def _registry(path: Path, *, price: str = "0.1", model: str = "gpt-6-luna") -> Path:
    path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": model,
                        "maxTokens": 128_000,
                        "cost": {"input": price, "output": "0.5"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def _budget(tmp_path: Path, *, cap: str = "1") -> ProxyBudget:
    return ProxyBudget(
        ledger_path=tmp_path / "budget.jsonl",
        registry_path=_registry(tmp_path / "models-store.json"),
        campaign="campaign-1",
        source_hash="a" * 64,
        prompt_hash="b" * 64,
        schema_hash="c" * 64,
        cap_usd=Decimal(cap),
    )


def test_reservation_is_durable_conservative_and_usage_does_not_refund(
    tmp_path: Path,
) -> None:
    budget = _budget(tmp_path)
    reserved = budget.reserve_attempt("attempt-1", {"content": "fødselsdag"})
    lines = (tmp_path / "budget.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5  # pinned header, three historical, new attempt
    assert reserved >= Decimal("0.064")
    assert (tmp_path / "budget.jsonl").stat().st_mode & 0o777 == 0o600

    budget.record_usage(
        "attempt-1", input_tokens=3, output_tokens=5, response="private response"
    )
    assert budget.reserve_attempt("attempt-2", {"content": "fødselsdag"}) == reserved
    ledger_text = (tmp_path / "budget.jsonl").read_text(encoding="utf-8")
    assert "private response" not in ledger_text
    assert "fødselsdag" not in ledger_text
    assert '"response_sha256"' in ledger_text


def test_restart_keeps_prior_reservations_and_rejects_unknown_usage(
    tmp_path: Path,
) -> None:
    budget = _budget(tmp_path)
    budget.reserve_attempt("attempt-1", {"x": 1})
    restarted = _budget(tmp_path)
    with pytest.raises(ProxyBudgetError, match="unknown request ID"):
        restarted.record_usage(
            "unknown", input_tokens=1, output_tokens=1, response=b"response"
        )
    with pytest.raises(ProxyBudgetError, match="already reserved"):
        restarted.reserve_attempt("attempt-1", {"x": 1})


def test_truncated_ledger_fails_closed(tmp_path: Path) -> None:
    budget = _budget(tmp_path)
    budget.reserve_attempt("attempt-1", {"x": 1})
    with (tmp_path / "budget.jsonl").open("a", encoding="utf-8") as ledger:
        ledger.write('{"type":"reservation"')
    with pytest.raises(ProxyBudgetError, match="malformed or truncated"):
        budget.reserve_attempt("attempt-2", {"x": 1})


def test_internal_cap_includes_historical_reservations(tmp_path: Path) -> None:
    budget = _budget(tmp_path, cap="0.065")
    with pytest.raises(ProxyBudgetError, match="cap exhausted"):
        budget.reserve_attempt("attempt-1", {"x": 1})


@pytest.mark.parametrize(
    ("price", "model"), [("0.2", "gpt-6-luna"), ("0.1", "different-model")]
)
def test_registry_price_or_model_change_fails_closed(
    tmp_path: Path, price: str, model: str
) -> None:
    registry = _registry(tmp_path / "models-store.json")
    budget = ProxyBudget(
        ledger_path=tmp_path / "budget.jsonl",
        registry_path=registry,
        campaign="campaign-1",
        source_hash="a" * 64,
        prompt_hash="b" * 64,
        schema_hash="c" * 64,
    )
    _registry(registry, price=price, model=model)
    with pytest.raises(ProxyBudgetError, match="registry"):
        budget.reserve_attempt("attempt-1", {"x": 1})


def test_reservation_rejects_ids_reused_for_historical_entries(tmp_path: Path) -> None:
    budget = _budget(tmp_path)
    with pytest.raises(ProxyBudgetError, match="already reserved"):
        budget.reserve_attempt("prior-failed-melious", {"x": 1})
