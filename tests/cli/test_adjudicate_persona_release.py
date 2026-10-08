"""Offline contracts for the private full-release Sol CLI."""

from __future__ import annotations

import json
import os
import stat
import threading
from pathlib import Path

import httpx
import polars as pl
import pytest

import danish_personas.generation.proxy_budget as proxy_budget
from danish_personas.generation.models import GenerationConfig
from danish_personas.generation.proxy_budget import ProxyBudget
from danish_personas.generation.sol_adjudication import (
    SolAdjudicationResult,
    run_sol_adjudication,
)
from danish_personas.io import sha256_file
from scripts import adjudicate_persona_release as adjudicate


def test_completed_status_skips_provider_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows already completed in status are not requested again."""
    paths = _fixture_paths(tmp_path, rows=5)
    _patch_budget_path(tmp_path=tmp_path, monkeypatch=monkeypatch)
    runner = _RecordingRunner()

    first = adjudicate.run_release_adjudication(
        paths=paths,
        execute=True,
        max_rows=2,
        workers=1,
        expected_original_sha256=sha256_file(paths.original),
        expected_row_count=5,
        sol_runner=runner,
    )
    second = adjudicate.run_release_adjudication(
        paths=paths,
        execute=True,
        max_rows=2,
        workers=1,
        expected_original_sha256=sha256_file(paths.original),
        expected_row_count=5,
        sol_runner=runner,
    )

    assert first["processed"] == 2
    assert second["processed"] == 2
    assert runner.calls == 2


class _RecordingRunner:
    def __init__(self) -> None:
        self.calls = 0
        self.checkpoints: set[Path] = set()
        self._lock = threading.Lock()

    def __call__(
        self,
        original_persona: str,
        candidate_row: dict[str, object],
        prompt: str,
        config: GenerationConfig,
        budget: ProxyBudget,
        checkpoint_path: Path,
        transport: httpx.BaseTransport,
        changed_fact_hints: dict[str, dict[str, object]],
        original_row: dict[str, object],
    ) -> SolAdjudicationResult:
        del transport
        with self._lock:
            self.calls += 1
            self.checkpoints.add(checkpoint_path)
        result = run_sol_adjudication(
            original_persona=original_persona,
            candidate_row=candidate_row,
            prompt=prompt,
            config=config,
            budget=budget,
            checkpoint_path=checkpoint_path,
            transport=_sol_transport(original_persona=original_persona),
            changed_fact_hints=changed_fact_hints,
            original_row=original_row,
        )
        assert stat.S_IMODE(checkpoint_path.stat().st_mode) == 0o600
        return result


def _sol_transport(*, original_persona: str) -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "gpt-6-sol"
        assert "max_tokens" not in body
        assert "max_completion_tokens" not in body
        return httpx.Response(
            200,
            json={
                "id": "response-1",
                "model": "gpt-6-sol",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "disposition": "consistent",
                                    "reason": "ok",
                                    "evidence": [
                                        {
                                            "field": "age",
                                            "kind": "negative_evidence",
                                            "quote": original_persona,
                                        }
                                    ],
                                    "patches": [],
                                }
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )

    return httpx.MockTransport(respond)


def _fixture_paths(
    tmp_path: Path, *, rows: int, restricted: bool = False, secret: bool = False
) -> adjudicate.ReleasePaths:
    root = tmp_path / "audit"
    root.mkdir()
    original = root / "train.parquet"
    candidate = root / "candidate.parquet"
    report = root / "candidate.json"
    prompt = root / "prompt.md"
    registry = root / "models.json"
    output = root / "sol-output"
    original_frame = _frame(
        rows=rows, candidate=False, restricted=restricted, secret=secret
    )
    candidate_frame = _frame(
        rows=rows, candidate=True, restricted=restricted, secret=secret
    )
    original_frame.write_parquet(original)
    candidate_frame.write_parquet(candidate)
    report.write_text(
        json.dumps(
            {
                "row_count": rows,
                "preview_sha256": adjudicate._frame_hash(frame=candidate_frame),
                "source_hashes": {"original_v1_sha256": sha256_file(original)},
            }
        ),
        encoding="utf-8",
    )
    prompt.write_text(
        "Vurder personaen konservativt og returnér JSON.\n", encoding="utf-8"
    )
    registry.write_text(
        json.dumps(
            {
                "openai-codex": {
                    "models": [
                        {
                            "id": "gpt-6-sol",
                            "maxTokens": proxy_budget.DEFAULT_MAX_TOKENS,
                            "cost": {"input": "2", "output": "10"},
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    return adjudicate.ReleasePaths(
        original=original,
        candidate=candidate,
        report=report,
        prompt=prompt,
        output_dir=output,
        registry=registry,
    )


def _frame(
    *, rows: int, candidate: bool, restricted: bool, secret: bool
) -> pl.DataFrame:
    prefix = "raw-secret-id" if secret else "raw-id"
    prose = "unik hemmelig prosatekst" if secret else "almindelig prosatekst"
    if restricted:
        prose = "personen omtales som homoseksuel i teksten"
    return pl.DataFrame(
        {
            "persona_id": [f"{prefix}-{index:03d}" for index in range(rows)],
            "persona": [f"{prose} {index:03d}" for index in range(rows)],
            "age": [30 + index for index in range(rows)],
            "municipality": ["Aarhus" for _ in range(rows)],
            "job_title": ["lærer" if candidate else "pædagog" for _ in range(rows)],
            "sidecar": ["must not be sent" for _ in range(rows)],
        }
    )


def _patch_budget_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        proxy_budget, "USER_SOL_ADJUDICATION_BUDGET_PATH", tmp_path / "sol-budget.jsonl"
    )


def test_dry_run_selects_all_fixture_rows_without_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dry-run covers the full ordered fixture and performs no provider I/O."""
    paths = _fixture_paths(tmp_path, rows=7)
    _patch_budget_path(tmp_path=tmp_path, monkeypatch=monkeypatch)

    def fail_runner(
        original_persona: str,
        candidate_row: dict[str, object],
        prompt: str,
        config: GenerationConfig,
        budget: ProxyBudget,
        checkpoint_path: Path,
        transport: httpx.BaseTransport,
        changed_fact_hints: dict[str, dict[str, object]],
        original_row: dict[str, object],
    ) -> SolAdjudicationResult:
        raise AssertionError("dry-run must not call the row runner")

    summary = adjudicate.run_release_adjudication(
        paths=paths,
        execute=False,
        max_rows=None,
        workers=1,
        expected_original_sha256=sha256_file(paths.original),
        expected_row_count=7,
        sol_runner=fail_runner,
    )

    assert summary["selected"] == 7
    assert summary["pending"] == 7
    assert not paths.output_dir.exists()


def test_existing_public_output_directory_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The campaign refuses to write into a non-private directory."""
    paths = _fixture_paths(tmp_path, rows=2)
    _patch_budget_path(tmp_path=tmp_path, monkeypatch=monkeypatch)
    paths.output_dir.mkdir()
    os.chmod(paths.output_dir, 0o755)

    with pytest.raises(adjudicate.PersonaReleaseAdjudicationError):
        adjudicate.run_release_adjudication(
            paths=paths,
            execute=True,
            max_rows=None,
            workers=1,
            expected_original_sha256=sha256_file(paths.original),
            expected_row_count=2,
            sol_runner=_RecordingRunner(),
        )


def test_fatal_row_stops_without_queueing_all_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hard row failure stops after at most the bounded in-flight window."""
    paths = _fixture_paths(tmp_path, rows=12)
    _patch_budget_path(tmp_path=tmp_path, monkeypatch=monkeypatch)
    runner = _FailingFirstRunner()

    with pytest.raises(adjudicate.PersonaReleaseAdjudicationError):
        adjudicate.run_release_adjudication(
            paths=paths,
            execute=True,
            max_rows=None,
            workers=4,
            expected_original_sha256=sha256_file(paths.original),
            expected_row_count=12,
            sol_runner=runner,
        )

    assert 1 <= runner.calls <= 4


class _FailingFirstRunner(_RecordingRunner):
    def __call__(
        self,
        original_persona: str,
        candidate_row: dict[str, object],
        prompt: str,
        config: GenerationConfig,
        budget: ProxyBudget,
        checkpoint_path: Path,
        transport: httpx.BaseTransport,
        changed_fact_hints: dict[str, dict[str, object]],
        original_row: dict[str, object],
    ) -> SolAdjudicationResult:
        del transport
        with self._lock:
            self.calls += 1
            call_number = self.calls
            self.checkpoints.add(checkpoint_path)
        if call_number == 1:
            raise RuntimeError("provider rejected content")
        return run_sol_adjudication(
            original_persona=original_persona,
            candidate_row=candidate_row,
            prompt=prompt,
            config=config,
            budget=budget,
            checkpoint_path=checkpoint_path,
            transport=_sol_transport(original_persona=original_persona),
            changed_fact_hints=changed_fact_hints,
            original_row=original_row,
        )


def test_no_raw_id_or_prose_in_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure messages do not echo private identifiers or persona prose."""
    paths = _fixture_paths(tmp_path, rows=2, secret=True)
    _patch_budget_path(tmp_path=tmp_path, monkeypatch=monkeypatch)

    with pytest.raises(adjudicate.PersonaReleaseAdjudicationError) as error:
        adjudicate.run_release_adjudication(
            paths=paths,
            execute=True,
            max_rows=None,
            workers=1,
            expected_original_sha256="f" * 64,
            expected_row_count=2,
            sol_runner=_RecordingRunner(),
        )

    message = str(error.value)
    assert "raw-secret-id" not in message
    assert "unik hemmelig prosatekst" not in message


def test_pilot_resume_processes_stable_prefix_then_full_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 32-row pilot writes reusable checkpoints for a later full run."""
    paths = _fixture_paths(tmp_path, rows=40)
    _patch_budget_path(tmp_path=tmp_path, monkeypatch=monkeypatch)
    runner = _RecordingRunner()

    pilot = adjudicate.run_release_adjudication(
        paths=paths,
        execute=True,
        max_rows=32,
        workers=3,
        expected_original_sha256=sha256_file(paths.original),
        expected_row_count=40,
        sol_runner=runner,
    )
    assert _reservation_count(path=tmp_path / "sol-budget.jsonl") == 32
    full = adjudicate.run_release_adjudication(
        paths=paths,
        execute=True,
        max_rows=None,
        workers=4,
        expected_original_sha256=sha256_file(paths.original),
        expected_row_count=40,
        sol_runner=runner,
    )

    assert pilot["processed"] == 32
    assert full["processed"] == 40
    assert len(runner.checkpoints) == 40
    assert _reservation_count(path=tmp_path / "sol-budget.jsonl") == 40
    status = json.loads((paths.output_dir / "status.json").read_text())
    assert len(status["processed"]) == 40
    assert all("persona_id" not in record for record in status["processed"])


def _reservation_count(*, path: Path) -> int:
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("type") == "reservation"
    )


def test_restricted_text_refuses_before_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sensitive identity terms in local prose fail closed before requests."""
    paths = _fixture_paths(tmp_path, rows=2, restricted=True)
    _patch_budget_path(tmp_path=tmp_path, monkeypatch=monkeypatch)
    runner = _RecordingRunner()

    with pytest.raises(adjudicate.PersonaReleaseAdjudicationError):
        adjudicate.run_release_adjudication(
            paths=paths,
            execute=True,
            max_rows=None,
            workers=1,
            expected_original_sha256=sha256_file(paths.original),
            expected_row_count=2,
            sol_runner=runner,
        )

    assert runner.calls == 0
    assert not paths.output_dir.exists()


def test_selected_rows_preflight_raw_code_before_first_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later unsafe selected row fails before any provider request."""
    paths = _fixture_paths(tmp_path, rows=3)
    candidate = pl.read_parquet(paths.candidate).with_columns(
        pl.Series("origin_country_code", ["SRC-777", "SRC-777", "SRC-777"]),
        pl.Series("job_title", ["lærer", "lærer SRC-777", "lærer"]),
    )
    candidate.write_parquet(paths.candidate)
    report = json.loads(paths.report.read_text(encoding="utf-8"))
    report["preview_sha256"] = adjudicate._frame_hash(frame=candidate)
    paths.report.write_text(json.dumps(report), encoding="utf-8")
    _patch_budget_path(tmp_path=tmp_path, monkeypatch=monkeypatch)
    runner = _RecordingRunner()

    with pytest.raises(adjudicate.PersonaReleaseAdjudicationError):
        adjudicate.run_release_adjudication(
            paths=paths,
            execute=True,
            max_rows=None,
            workers=1,
            expected_original_sha256=sha256_file(paths.original),
            expected_row_count=3,
            sol_runner=runner,
        )

    assert runner.calls == 0
    assert not paths.output_dir.exists()


def test_stale_input_and_report_fail_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changed source bindings are rejected before private output is created."""
    paths = _fixture_paths(tmp_path, rows=3)
    _patch_budget_path(tmp_path=tmp_path, monkeypatch=monkeypatch)

    with pytest.raises(adjudicate.PersonaReleaseAdjudicationError) as error:
        adjudicate.run_release_adjudication(
            paths=paths,
            execute=True,
            max_rows=None,
            workers=1,
            expected_original_sha256="0" * 64,
            expected_row_count=3,
            sol_runner=_RecordingRunner(),
        )

    assert "raw-id" not in str(error.value)
    assert not paths.output_dir.exists()

    report = json.loads(paths.report.read_text(encoding="utf-8"))
    report["preview_sha256"] = "1" * 64
    paths.report.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(adjudicate.PersonaReleaseAdjudicationError):
        adjudicate.run_release_adjudication(
            paths=paths,
            execute=False,
            max_rows=None,
            workers=1,
            expected_original_sha256=sha256_file(paths.original),
            expected_row_count=3,
            sol_runner=_RecordingRunner(),
        )


def test_tampered_checkpoint_with_intact_status_fails_before_new_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every processed checkpoint is revalidated before pending rows run."""
    paths = _fixture_paths(tmp_path, rows=4)
    _patch_budget_path(tmp_path=tmp_path, monkeypatch=monkeypatch)
    runner = _RecordingRunner()
    adjudicate.run_release_adjudication(
        paths=paths,
        execute=True,
        max_rows=2,
        workers=1,
        expected_original_sha256=sha256_file(paths.original),
        expected_row_count=4,
        sol_runner=runner,
    )
    ledger = tmp_path / "sol-budget.jsonl"
    assert _reservation_count(path=ledger) == 2
    status = json.loads((paths.output_dir / "status.json").read_text())
    checkpoint = paths.output_dir / status["processed"][0]["checkpoint"]
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    saved["response"] = saved["response"].replace("ok", "changed", 1)
    checkpoint.write_text(json.dumps(saved), encoding="utf-8")
    os.chmod(checkpoint, 0o600)

    with pytest.raises(adjudicate.PersonaReleaseAdjudicationError):
        adjudicate.run_release_adjudication(
            paths=paths,
            execute=True,
            max_rows=None,
            workers=1,
            expected_original_sha256=sha256_file(paths.original),
            expected_row_count=4,
            sol_runner=runner,
        )

    assert runner.calls == 2
    assert _reservation_count(path=ledger) == 2
