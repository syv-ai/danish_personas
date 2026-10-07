"""Tests for durable local proxy request reservations."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path

import pytest

from danish_personas.generation import proxy_budget
from danish_personas.generation.proxy_budget import ProxyBudget, ProxyBudgetError


@pytest.fixture(autouse=True)
def _private_budget_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep all test campaigns on isolated user-level ledgers."""
    monkeypatch.setattr(proxy_budget, "USER_BUDGET_PATH", tmp_path / "budget.jsonl")
    monkeypatch.setattr(
        proxy_budget, "USER_UNCAPPED_BUDGET_PATH", tmp_path / "uncapped.jsonl"
    )


def test_alternate_ledger_path_cannot_reset_shared_budget(tmp_path: Path) -> None:
    """Caller-provided paths cannot create a fresh campaign budget."""
    budget = _budget(tmp_path, cap="0.15")
    budget.reserve_attempt("attempt-1", {"x": 1})
    second = ProxyBudget(
        ledger_path=tmp_path / "different-ledger.jsonl",
        registry_path=_registry(tmp_path / "models-store.json"),
        campaign="campaign-1",
        source_hash="a" * 64,
        prompt_hash="b" * 64,
        schema_hash="c" * 64,
        cap_usd=Decimal("0.15"),
    )
    assert second.path == budget.path == proxy_budget.USER_BUDGET_PATH
    with pytest.raises(ProxyBudgetError, match="cap exhausted"):
        second.reserve_attempt("attempt-2", {"x": 1})
    assert not (tmp_path / "different-ledger.jsonl").exists()


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


def _registry(path: Path, *, price: str = "0.1", model: str = "gpt-6-luna") -> Path:
    path.write_text(
        json.dumps(
            {
                "openai-codex": {
                    "models": [
                        {
                            "id": model,
                            "maxTokens": 128_000,
                            "cost": {"input": price, "output": "0.5"},
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_changed_campaign_pins_fail_closed(tmp_path: Path) -> None:
    """A shared ledger cannot be reopened under changed source or prompt pins."""
    _budget(tmp_path)
    changed_pins = (
        ("source_hash", "e" * 64),
        ("prompt_hash", "f" * 64),
        ("schema_hash", "0" * 64),
    )
    for field, value in changed_pins:
        pins = {
            "campaign": "campaign-1",
            "source_hash": "a" * 64,
            "prompt_hash": "b" * 64,
            "schema_hash": "c" * 64,
        }
        pins[field] = value
        with pytest.raises(ProxyBudgetError, match="pins do not match"):
            ProxyBudget(
                registry_path=_registry(tmp_path / "models-store.json"),
                campaign=pins["campaign"],
                source_hash=pins["source_hash"],
                prompt_hash=pins["prompt_hash"],
                schema_hash=pins["schema_hash"],
            )


def test_internal_cap_includes_historical_reservations(tmp_path: Path) -> None:
    """Count pinned historical charges against the campaign's internal cap."""
    budget = _budget(tmp_path, cap="0.065")
    with pytest.raises(ProxyBudgetError, match="cap exhausted"):
        budget.reserve_attempt("attempt-1", {"x": 1})


@pytest.mark.parametrize(
    ("price", "model"), [("0.2", "gpt-6-luna"), ("0.1", "different-model")]
)
def test_registry_price_or_model_change_fails_closed(
    tmp_path: Path, price: str, model: str
) -> None:
    """Fail closed when pinned model identity or pricing changes."""
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


def test_registry_rejects_duplicate_pinned_models(tmp_path: Path) -> None:
    """Reject ambiguous duplicate entries in the configured provider registry."""
    registry = _registry(tmp_path / "models-store.json")
    document = json.loads(registry.read_text(encoding="utf-8"))
    document["openai-codex"]["models"].append(
        document["openai-codex"]["models"][0].copy()
    )
    registry.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ProxyBudgetError, match="registry"):
        ProxyBudget(
            ledger_path=tmp_path / "budget.jsonl",
            registry_path=registry,
            campaign="campaign-1",
            source_hash="a" * 64,
            prompt_hash="b" * 64,
            schema_hash="c" * 64,
        )


def test_registry_rejects_model_under_wrong_provider(tmp_path: Path) -> None:
    """Do not accept a matching model entry from a different provider."""
    registry = _registry(tmp_path / "models-store.json")
    document = json.loads(registry.read_text(encoding="utf-8"))
    document["wrong-provider"] = document.pop("openai-codex")
    registry.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ProxyBudgetError, match="registry"):
        ProxyBudget(
            ledger_path=tmp_path / "budget.jsonl",
            registry_path=registry,
            campaign="campaign-1",
            source_hash="a" * 64,
            prompt_hash="b" * 64,
            schema_hash="c" * 64,
        )


def test_reservation_is_durable_conservative_and_usage_does_not_refund(
    tmp_path: Path,
) -> None:
    """Persist reservations and keep their budget charge after usage is recorded."""
    budget = _budget(tmp_path)
    reserved = budget.reserve_attempt("attempt-1", {"content": "fødselsdag"})
    lines = (tmp_path / "budget.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5  # pinned header, three historical, new attempt
    assert reserved >= Decimal("0.064")
    assert (tmp_path / "budget.jsonl").stat().st_mode & 0o777 == 0o600

    budget.record_usage(
        "attempt-1", input_tokens=3, output_tokens=5, response_sha256="d" * 64
    )
    assert budget.reserve_attempt("attempt-2", {"content": "fødselsdag"}) == reserved
    ledger_text = (tmp_path / "budget.jsonl").read_text(encoding="utf-8")
    assert "d" * 64 in ledger_text
    assert "private response" not in ledger_text
    assert "fødselsdag" not in ledger_text
    assert '"response_sha256"' in ledger_text


def test_reservation_rejects_ids_reused_for_historical_entries(tmp_path: Path) -> None:
    """Do not allow new attempts to alias a prior reservation."""
    budget = _budget(tmp_path)
    with pytest.raises(ProxyBudgetError, match="already reserved"):
        budget.reserve_attempt("prior-failed-melious", {"x": 1})


def test_restart_keeps_prior_reservations_and_rejects_unknown_usage(
    tmp_path: Path,
) -> None:
    """Reload complete newline-terminated records and prevent duplicate attempts."""
    budget = _budget(tmp_path)
    budget.reserve_attempt("attempt-1", {"x": 1})
    restarted = _budget(tmp_path)
    with pytest.raises(ProxyBudgetError, match="unknown request ID"):
        restarted.record_usage(
            "unknown", input_tokens=1, output_tokens=1, response_sha256="d" * 64
        )
    with pytest.raises(ProxyBudgetError, match="already reserved"):
        restarted.reserve_attempt("attempt-1", {"x": 1})


def test_truncated_ledger_fails_closed(tmp_path: Path) -> None:
    """Reject an incomplete trailing record rather than authorising a request."""
    budget = _budget(tmp_path)
    budget.reserve_attempt("attempt-1", {"x": 1})
    with (tmp_path / "budget.jsonl").open("a", encoding="utf-8") as ledger:
        ledger.write('{"type":"reservation"')
    with pytest.raises(ProxyBudgetError, match="malformed or truncated"):
        budget.reserve_attempt("attempt-2", {"x": 1})


def test_uncapped_rejects_model_and_manifest_changes(tmp_path: Path) -> None:
    """Fail closed when uncapped model or campaign pins change."""
    _budget(tmp_path)
    _uncapped_budget(tmp_path)

    with pytest.raises(ProxyBudgetError, match="pinned policy"):
        _uncapped_budget(tmp_path, model="different-model")
    with pytest.raises(ProxyBudgetError, match="pins do not match"):
        _uncapped_budget(tmp_path, prompt_hash="f" * 64)

    with proxy_budget.USER_BUDGET_PATH.open("a", encoding="utf-8") as old_ledger:
        old_ledger.write("\n")
    with pytest.raises(ProxyBudgetError, match="missing or incomplete"):
        _uncapped_budget(tmp_path)


def _uncapped_budget(
    tmp_path: Path,
    *,
    model: str = "gpt-6-luna",
    campaign: str = "campaign-1",
    prompt_hash: str = "b" * 64,
    request_overhead_bytes: int = 4096,
) -> ProxyBudget:
    return ProxyBudget(
        ledger_path=tmp_path / "ignored-uncapped.jsonl",
        registry_path=_registry(tmp_path / "models-store.json", model=model),
        campaign=campaign,
        source_hash="a" * 64,
        prompt_hash=prompt_hash,
        schema_hash="c" * 64,
        model=model,
        request_overhead_bytes=request_overhead_bytes,
        uncapped=True,
    )


def test_uncapped_requires_existing_complete_capped_ledger(tmp_path: Path) -> None:
    """Uncapped campaigns must bind a complete capped pilot ledger."""
    with pytest.raises(ProxyBudgetError, match="missing or incomplete"):
        _uncapped_budget(tmp_path)

    proxy_budget.USER_BUDGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    proxy_budget.USER_BUDGET_PATH.write_text(
        '{"type":"header","model":"gpt-6-luna"}', encoding="utf-8"
    )
    with pytest.raises(ProxyBudgetError, match="missing or incomplete"):
        _uncapped_budget(tmp_path)


def test_uncapped_reservations_bind_old_sha_and_do_not_touch_old_ledger(
    tmp_path: Path,
) -> None:
    """Permit large uncapped reservations while preserving capped history bytes."""
    _budget(tmp_path)
    old_bytes = proxy_budget.USER_BUDGET_PATH.read_bytes()
    huge_overhead = 1_001_000_000

    capped = ProxyBudget(
        registry_path=_registry(tmp_path / "models-store.json"),
        campaign="campaign-1",
        source_hash="a" * 64,
        prompt_hash="b" * 64,
        schema_hash="c" * 64,
        request_overhead_bytes=huge_overhead,
    )
    with pytest.raises(ProxyBudgetError, match="cap exhausted"):
        capped.reserve_attempt("giant-capped", {"x": 1})

    uncapped = _uncapped_budget(tmp_path, request_overhead_bytes=huge_overhead)
    reserved = uncapped.reserve_attempt("giant-uncapped", {"x": 1})

    assert reserved > Decimal("100")
    assert proxy_budget.USER_BUDGET_PATH.read_bytes() == old_bytes
    uncapped_lines = proxy_budget.USER_UNCAPPED_BUDGET_PATH.read_text(
        encoding="utf-8"
    ).splitlines()
    header = json.loads(uncapped_lines[0])
    assert header["uncapped"] is True
    assert header["old_ledger_sha256"] == hashlib.sha256(old_bytes).hexdigest()
    assert header["model"] == "gpt-6-luna"
    assert header["base_url"] == "http://127.0.0.1:18080/v1"
    assert proxy_budget.USER_UNCAPPED_BUDGET_PATH.stat().st_mode & 0o777 == 0o600


def test_uncapped_restart_keeps_reservations_and_usage_idempotent(
    tmp_path: Path,
) -> None:
    """Reload uncapped reservations without double-counting prior capped charges."""
    _budget(tmp_path)
    uncapped = _uncapped_budget(tmp_path)
    uncapped.reserve_attempt("uncapped-attempt-1", {"x": 1})
    uncapped.record_usage(
        "uncapped-attempt-1", input_tokens=2, output_tokens=3, response_sha256="e" * 64
    )

    restarted = _uncapped_budget(tmp_path)
    with pytest.raises(ProxyBudgetError, match="already reserved"):
        restarted.reserve_attempt("uncapped-attempt-1", {"x": 1})
    with pytest.raises(ProxyBudgetError, match="already recorded"):
        restarted.record_usage(
            "uncapped-attempt-1",
            input_tokens=2,
            output_tokens=3,
            response_sha256="e" * 64,
        )
    uncapped_records = proxy_budget.USER_UNCAPPED_BUDGET_PATH.read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(uncapped_records) == 3


def test_usage_rejects_invalid_response_digest(tmp_path: Path) -> None:
    """Only an existing lowercase SHA-256 digest may be recorded."""
    budget = _budget(tmp_path)
    budget.reserve_attempt("attempt-1", {"x": 1})
    for digest in ("A" * 64, "d" * 63, "not-a-digest"):
        with pytest.raises(ProxyBudgetError, match="lowercase hex digest"):
            budget.record_usage(
                "attempt-1", input_tokens=1, output_tokens=1, response_sha256=digest
            )
