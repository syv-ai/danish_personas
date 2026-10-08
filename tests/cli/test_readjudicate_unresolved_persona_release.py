"""Offline contracts for the private unresolved Sol follow-up CLI."""

from __future__ import annotations

import json
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
    SolAdjudicationMode,
    SolAdjudicationResult,
    run_sol_adjudication,
)
from danish_personas.io import sha256_file
from scripts import adjudicate_persona_release as first_pass
from scripts import readjudicate_unresolved_persona_release as followup


def test_dry_run_selects_only_processed_unresolved_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pilot scope excludes parent-pending rows and caps at nine unresolved."""
    paths = _fixture_paths(tmp_path, rows=20)
    _patch_budget_path(tmp_path=tmp_path, monkeypatch=monkeypatch)
    dispositions = ["unresolved" for _ in range(10)] + ["consistent", "consistent"]
    _create_parent(paths=paths, dispositions=dispositions)

    summary = followup.run_unresolved_followup(
        paths=paths,
        execute=False,
        max_rows=9,
        workers=1,
        expected_original_sha256=sha256_file(paths.original),
        expected_row_count=20,
        sol_runner=_DispositionRunner(mode="unresolved_followup"),
    )

    assert summary["dry_run"] is True
    assert summary["selected"] == 9
    assert summary["total"] == 10
    assert summary["pending"] == 9
    assert not paths.output_dir.exists()


def test_parent_manifest_tamper_rejected_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parent manifest changes fail closed before any follow-up request."""
    paths = _fixture_paths(tmp_path, rows=4)
    _patch_budget_path(tmp_path=tmp_path, monkeypatch=monkeypatch)
    _create_parent(paths=paths, dispositions=["unresolved", "consistent"])
    manifest_path = paths.parent_output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selection"]["ordered_rows"] = 999
    _write_private_json(path=manifest_path, value=manifest)
    runner = _DispositionRunner(mode="unresolved_followup")

    with pytest.raises(followup.FollowupAdjudicationError):
        followup.run_unresolved_followup(
            paths=paths,
            execute=True,
            max_rows=1,
            workers=1,
            expected_original_sha256=sha256_file(paths.original),
            expected_row_count=4,
            sol_runner=runner,
        )

    assert runner.calls == 0
    assert not paths.output_dir.exists()


def test_tampered_parent_checkpoint_rejected_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every processed parent checkpoint is revalidated before requests."""
    paths = _fixture_paths(tmp_path, rows=4)
    _patch_budget_path(tmp_path=tmp_path, monkeypatch=monkeypatch)
    _create_parent(paths=paths, dispositions=["unresolved", "consistent"])
    status = first_pass._load_json_object(
        path=paths.parent_output_dir / "status.json", label="status"
    )
    record = first_pass._processed_records(status=status)[0]
    checkpoint_path = paths.parent_output_dir / record["checkpoint"]
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["response_sha256"] = "0" * 64
    _write_private_json(path=checkpoint_path, value=checkpoint)
    runner = _DispositionRunner(mode="unresolved_followup")

    with pytest.raises(followup.FollowupAdjudicationError):
        followup.run_unresolved_followup(
            paths=paths,
            execute=True,
            max_rows=1,
            workers=1,
            expected_original_sha256=sha256_file(paths.original),
            expected_row_count=4,
            sol_runner=runner,
        )

    assert runner.calls == 0
    assert not paths.output_dir.exists()


def test_pilot_resume_processes_remaining_without_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nine-row pilot can resume to all unresolved without touching the parent."""
    paths = _fixture_paths(tmp_path, rows=6)
    _patch_budget_path(tmp_path=tmp_path, monkeypatch=monkeypatch)
    _create_parent(
        paths=paths,
        dispositions=["unresolved", "unresolved", "unresolved", "unresolved"],
    )
    before = _tree_hashes(paths.parent_output_dir)
    runner = _DispositionRunner(mode="unresolved_followup")

    pilot = followup.run_unresolved_followup(
        paths=paths,
        execute=True,
        max_rows=2,
        workers=1,
        expected_original_sha256=sha256_file(paths.original),
        expected_row_count=6,
        sol_runner=runner,
    )
    resumed = followup.run_unresolved_followup(
        paths=paths,
        execute=True,
        max_rows=None,
        workers=1,
        expected_original_sha256=sha256_file(paths.original),
        expected_row_count=6,
        sol_runner=runner,
    )

    assert pilot["processed"] == 2
    assert resumed["processed"] == 4
    assert runner.calls == 4
    assert _tree_hashes(paths.parent_output_dir) == before
    status = first_pass._load_json_object(
        path=paths.output_dir / "status.json", label="status"
    )
    assert stat.S_IMODE(paths.output_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((paths.output_dir / "manifest.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((paths.output_dir / "status.json").stat().st_mode) == 0o600
    assert len(first_pass._processed_records(status=status)) == 4


def test_unresolved_followup_result_is_recorded_not_promoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second unresolved decision stays unresolved in follow-up status."""
    paths = _fixture_paths(tmp_path, rows=3)
    _patch_budget_path(tmp_path=tmp_path, monkeypatch=monkeypatch)
    _create_parent(paths=paths, dispositions=["unresolved"])

    summary = followup.run_unresolved_followup(
        paths=paths,
        execute=True,
        max_rows=1,
        workers=1,
        expected_original_sha256=sha256_file(paths.original),
        expected_row_count=3,
        sol_runner=_DispositionRunner(
            mode="unresolved_followup", dispositions=["unresolved"]
        ),
    )

    status = first_pass._load_json_object(
        path=paths.output_dir / "status.json", label="status"
    )
    [record] = first_pass._processed_records(status=status)
    assert summary["unresolved"] == 1
    assert summary["consistent"] == 0
    assert record["disposition"] == "unresolved"


def test_failure_status_is_clear_and_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Local validation failures record a generic status without prose or raw IDs."""
    paths = _fixture_paths(tmp_path, rows=2, secret=True)
    _patch_budget_path(tmp_path=tmp_path, monkeypatch=monkeypatch)
    _create_parent(paths=paths, dispositions=["unresolved"])
    runner = _DispositionRunner(mode="unresolved_followup", invalid=True)

    with pytest.raises(followup.FollowupAdjudicationError) as error:
        followup.run_unresolved_followup(
            paths=paths,
            execute=True,
            max_rows=1,
            workers=1,
            expected_original_sha256=sha256_file(paths.original),
            expected_row_count=2,
            sol_runner=runner,
        )

    status = first_pass._load_json_object(
        path=paths.output_dir / "status.json", label="status"
    )
    failure = status["last_failure"]
    assert isinstance(failure, dict)
    assert failure["processed"] == 0
    assert failure["pending"] == 1
    message = json.dumps(failure, ensure_ascii=False) + str(error.value)
    assert "raw-secret-id" not in message
    assert "unik hemmelig prosatekst" not in message


def _create_parent(paths: followup.FollowupPaths, dispositions: list[str]) -> None:
    first_pass.run_release_adjudication(
        paths=first_pass.ReleasePaths(
            original=paths.original,
            candidate=paths.candidate,
            report=paths.report,
            prompt=paths.prompt,
            output_dir=paths.parent_output_dir,
            registry=paths.registry,
        ),
        execute=True,
        max_rows=len(dispositions),
        workers=1,
        expected_original_sha256=sha256_file(paths.original),
        expected_row_count=_row_count(path=paths.original),
        sol_runner=_DispositionRunner(dispositions=dispositions, mode="default"),
    )


class _DispositionRunner:
    def __init__(
        self,
        *,
        dispositions: list[str] | None = None,
        mode: SolAdjudicationMode,
        invalid: bool = False,
    ) -> None:
        self.dispositions = dispositions or ["consistent"]
        self.mode = mode
        self.invalid = invalid
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
            index = self.calls
            self.calls += 1
            self.checkpoints.add(checkpoint_path)
        disposition = self.dispositions[min(index, len(self.dispositions) - 1)]
        result = run_sol_adjudication(
            original_persona=original_persona,
            candidate_row=candidate_row,
            prompt=prompt,
            config=config,
            budget=budget,
            checkpoint_path=checkpoint_path,
            transport=_sol_transport(
                original_persona=original_persona,
                disposition=disposition,
                expect_followup=self.mode == "unresolved_followup",
                invalid=self.invalid,
            ),
            changed_fact_hints=changed_fact_hints,
            original_row=original_row,
            adjudication_mode=self.mode,
        )
        assert stat.S_IMODE(checkpoint_path.stat().st_mode) == 0o600
        return result


def _sol_transport(
    *, original_persona: str, disposition: str, expect_followup: bool, invalid: bool
) -> httpx.MockTransport:
    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        user_payload = json.loads(body["messages"][1]["content"])
        if expect_followup:
            assert "unresolved_followup_rule" in user_payload
        else:
            assert "unresolved_followup_rule" not in user_payload
        content = (
            _invalid_response()
            if invalid
            else _response_content(
                original_persona=original_persona, disposition=disposition
            )
        )
        return httpx.Response(
            200,
            json={
                "id": "response-1",
                "model": "gpt-6-sol",
                "choices": [{"message": {"content": json.dumps(content)}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            },
        )

    return httpx.MockTransport(respond)


def _response_content(*, original_persona: str, disposition: str) -> dict[str, object]:
    if disposition == "unresolved":
        return {
            "disposition": "unresolved",
            "reason": "mangler sikker rettelse",
            "evidence": [],
            "patches": [],
        }
    return {
        "disposition": "consistent",
        "reason": "ok",
        "evidence": [
            {"field": "age", "kind": "negative_evidence", "quote": original_persona}
        ],
        "patches": [],
    }


def _invalid_response() -> dict[str, object]:
    return {"disposition": "consistent", "reason": "ok", "evidence": [], "patches": []}


def _fixture_paths(
    tmp_path: Path, *, rows: int, secret: bool = False
) -> followup.FollowupPaths:
    root = tmp_path / "audit"
    root.mkdir()
    original = root / "train.parquet"
    candidate = root / "candidate.parquet"
    report = root / "candidate.json"
    prompt = root / "prompt.md"
    registry = root / "models.json"
    original_frame = _frame(rows=rows, candidate=False, secret=secret)
    candidate_frame = _frame(rows=rows, candidate=True, secret=secret)
    original_frame.write_parquet(original)
    candidate_frame.write_parquet(candidate)
    report.write_text(
        json.dumps(
            {
                "row_count": rows,
                "preview_sha256": first_pass._frame_hash(frame=candidate_frame),
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
    return followup.FollowupPaths(
        original=original,
        candidate=candidate,
        report=report,
        prompt=prompt,
        parent_output_dir=root / "sol-adjudication-v2",
        output_dir=root / "sol-followup-pilot32",
        registry=registry,
    )


def _frame(*, rows: int, candidate: bool, secret: bool) -> pl.DataFrame:
    prefix = "raw-secret-id" if secret else "raw-id"
    prose = "unik hemmelig prosatekst" if secret else "almindelig prosatekst"
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


def _row_count(*, path: Path) -> int:
    return pl.read_parquet(path).height


def _patch_budget_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        proxy_budget, "USER_SOL_ADJUDICATION_BUDGET_PATH", tmp_path / "sol-budget.jsonl"
    )


def _write_private_json(*, path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    path.chmod(0o600)


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
