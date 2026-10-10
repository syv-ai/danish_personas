"""Run the private full-release Sol adjudication campaign."""

from __future__ import annotations

import collections.abc as c
import concurrent.futures as futures
import hashlib
import json
import os
import re
import stat
import typing as t
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import click
import httpx
import polars as pl

from danish_personas.cli_logging import configure_cli_logging
from danish_personas.environment import load_repository_environment
from danish_personas.generation.models import GenerationConfig
from danish_personas.generation.proxy_budget import (
    ADJUDICATION_MODEL_ENV,
    BASE_URL,
    SOL_ADJUDICATION_LEDGER_MAX_TOKENS,
    SOL_ADJUDICATION_MODEL,
    SOL_ADJUDICATION_PURPOSE,
    ProxyBudget,
    ProxyBudgetError,
    require_runtime_model,
)
from danish_personas.generation.sol_adjudication import (
    SOL_ALLOWED_FACT_FIELDS,
    SOL_MAX_OUTPUT_TOKENS,
    SolAdjudicationError,
    SolAdjudicationResponse,
    SolAdjudicationResult,
    preflight_sol_adjudication_payload,
    run_sol_adjudication,
)
from danish_personas.io import canonical_json, sha256_file, sha256_text

DEFAULT_ROOT = Path("/tmp/danish-personas-audit")
DEFAULT_ORIGINAL = DEFAULT_ROOT / "data/train-00000-of-00001.parquet"
DEFAULT_CANDIDATE = DEFAULT_ROOT / "prose-candidate-v5-merged-PROVISIONAL.parquet"
DEFAULT_REPORT = DEFAULT_CANDIDATE.with_suffix(".json")
DEFAULT_PROMPT = Path("config/persona-sol-adjudication-da.md")
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "sol-adjudication-v2"
DEFAULT_REGISTRY = Path.home() / ".pi" / "agent" / "models-store.json"
EXPECTED_ORIGINAL_SHA256 = (
    "c178e63d40046274bcc559bdd9322f32336656da980c1809f794d68449d6250f"
)
EXPECTED_RELEASE_ROWS = 100_000
CAMPAIGN = "persona-sol-adjudication-v2"
ID_FIELD = "persona_id"
PERSONA_FIELD = "persona"
MANIFEST_VERSION = 1
STATUS_VERSION = 1

_RESTRICTED_IDENTITY_TERMS = (
    r"sexual\s+orientation",
    r"seksu(?:el|al)\s+orientering",
    r"seksualitet",
    r"sexuality",
    r"homoseksuel",
    r"homosexual",
    r"biseksuel",
    r"bisexual",
    r"panseksuel",
    r"pansexual",
    r"aseksuel",
    r"asexual",
    r"heteroseksuel",
    r"heterosexual",
    r"lesbisk",
    r"lesbian",
    r"gay",
    r"queer",
    r"lgbtq?i?a?\+?",
    r"same[-\s]+sex",
    r"samkønnet",
    r"samkoennet",
    r"kønsidentitet",
    r"koensidentitet",
    r"gender\s+identity",
    r"transkønnet",
    r"transkoennet",
    r"transseksuel",
    r"transsexual",
    r"transgender",
    r"transperson",
    r"trans[-\s]?(?:mand|kvinde|man|woman)",
    r"non[-\s]?(?:binær|binaer|binary)",
    r"interkøn(?:net)?",
    r"interkoen(?:net)?",
    r"intersex",
    r"sex[-\s]+characteristics",
    r"kønskarakteristika",
    r"koenskarakteristika",
    r"variation\s+in\s+sex\s+characteristics",
    r"variation(?:er)?\s+i\s+kønskarakteristika",
    r"variation(?:er)?\s+i\s+koenskarakteristika",
)
_RESTRICTED_TEXT = re.compile(
    rf"(?<![\w])(?:{'|'.join(_RESTRICTED_IDENTITY_TERMS)})(?![\w])", re.IGNORECASE
)

FutureResult = t.TypeVar("FutureResult")
JSONScalar: t.TypeAlias = str | int | float | bool | None
JSONDocument: t.TypeAlias = (
    JSONScalar | list["JSONDocument"] | dict[str, "JSONDocument"]
)
SolRunner: t.TypeAlias = c.Callable[
    [
        str,
        dict[str, object],
        str,
        GenerationConfig,
        ProxyBudget,
        Path,
        httpx.BaseTransport,
        dict[str, dict[str, object]],
        dict[str, object],
    ],
    SolAdjudicationResult,
]


@click.command()
@click.option("--original", type=click.Path(path_type=Path), default=DEFAULT_ORIGINAL)
@click.option("--candidate", type=click.Path(path_type=Path), default=DEFAULT_CANDIDATE)
@click.option("--report", type=click.Path(path_type=Path), default=DEFAULT_REPORT)
@click.option("--prompt", type=click.Path(path_type=Path), default=DEFAULT_PROMPT)
@click.option(
    "--output-dir", type=click.Path(path_type=Path), default=DEFAULT_OUTPUT_DIR
)
@click.option("--registry", type=click.Path(path_type=Path), default=DEFAULT_REGISTRY)
@click.option("--max-rows", type=click.IntRange(min=1), default=None)
@click.option(
    "--workers", type=click.IntRange(min=1, max=4), default=1, show_default=True
)
@click.option("--run", "execute", is_flag=True, default=False)
def main(
    original: Path,
    candidate: Path,
    report: Path,
    prompt: Path,
    output_dir: Path,
    registry: Path,
    max_rows: int | None,
    workers: int,
    execute: bool,
) -> None:
    """Run or dry-run the private full-release Sol campaign.

    Raises:
        click.ClickException: If input, resume, budget, or proxy boundaries are unsafe.
    """
    configure_cli_logging()
    paths = ReleasePaths(
        original=original,
        candidate=candidate,
        report=report,
        prompt=prompt,
        output_dir=output_dir,
        registry=registry,
    )
    try:
        summary = run_release_adjudication(
            paths=paths,
            execute=execute,
            max_rows=max_rows,
            workers=workers,
            expected_original_sha256=EXPECTED_ORIGINAL_SHA256,
            expected_row_count=EXPECTED_RELEASE_ROWS,
            sol_runner=_run_sol_adjudication_adapter,
        )
    except (
        PersonaReleaseAdjudicationError,
        ProxyBudgetError,
        SolAdjudicationError,
        OSError,
    ) as exc:
        raise click.ClickException(_safe_error_message(exc=exc)) from exc
    click.echo(json.dumps(summary, ensure_ascii=False, sort_keys=True))


@dataclass(frozen=True)
class ReleasePaths:
    """Private Sol campaign inputs and output paths."""

    original: Path
    candidate: Path
    report: Path
    prompt: Path
    output_dir: Path
    registry: Path


def _safe_error_message(*, exc: Exception) -> str:
    if isinstance(exc, PersonaReleaseAdjudicationError):
        return str(exc)
    if isinstance(exc, ProxyBudgetError):
        return str(exc)
    if isinstance(exc, SolAdjudicationError):
        return "Sol adjudication rejected a row without storing provider content"
    return "Sol adjudication failed closed"


def run_release_adjudication(
    *,
    paths: ReleasePaths,
    execute: bool,
    max_rows: int | None,
    workers: int,
    expected_original_sha256: str,
    expected_row_count: int,
    sol_runner: SolRunner,
) -> dict[str, object]:
    """Run or dry-run the release Sol adjudication campaign.

    Args:
        paths: Immutable inputs and private output path.
        execute: If true, provider I/O is allowed. Defaults to dry-run in the CLI.
        max_rows: Optional ordered prefix bound for pilot processing.
        workers: Worker count from one to four.
        expected_original_sha256: Required SHA-256 of the frozen baseline parquet.
        expected_row_count: Required ordered release row count.
        sol_runner: Injectable per-row Sol runner for offline tests.

    Returns:
        Machine-readable progress summary without raw persona IDs or prose.

    Raises:
        PersonaReleaseAdjudicationError: If the run cannot continue safely.
    """
    _require_worker_count(workers=workers)
    inputs = load_release_inputs(
        paths=paths,
        expected_original_sha256=expected_original_sha256,
        expected_row_count=expected_row_count,
    )
    selection = select_release_rows(inputs=inputs)
    manifest = _campaign_manifest(paths=paths, inputs=inputs, selection=selection)
    scoped_rows = _scoped_rows(selection=selection, max_rows=max_rows)
    _preflight_selected_rows(rows=scoped_rows, prompt=inputs.prompt)
    if not execute:
        return _dry_run_summary(
            manifest=manifest, selection=selection, max_rows=max_rows, workers=workers
        )

    _prepare_private_output(output_dir=paths.output_dir)
    _write_or_check_manifest(path=paths.output_dir / "manifest.json", manifest=manifest)
    status_path = paths.output_dir / "status.json"
    status = _load_or_create_status(
        status_path=status_path, output_dir=paths.output_dir, manifest=manifest
    )
    config = _generation_config(prompt_path=paths.prompt)
    budget = _proxy_budget(paths=paths, inputs=inputs, manifest=manifest)
    _validate_processed_checkpoints(
        selection=selection,
        status=status,
        output_dir=paths.output_dir,
        prompt=inputs.prompt,
        config=config,
        budget=budget,
    )
    pending = _pending_rows(selection=selection, status=status, max_rows=max_rows)
    if pending:
        _process_pending(
            rows=pending,
            status=status,
            status_path=status_path,
            output_dir=paths.output_dir,
            prompt=inputs.prompt,
            config=config,
            budget=budget,
            workers=workers,
            sol_runner=sol_runner,
        )
    summary = _status_summary(status=status, max_rows=max_rows, workers=workers)
    if max_rows is None and summary["pending"] != 0:
        raise PersonaReleaseAdjudicationError("Sol campaign ended with pending rows")
    return summary


class PersonaReleaseAdjudicationError(Exception):
    """Raised when release adjudication must fail closed."""


def _bounded_total(*, total: int, max_rows: int | None) -> int:
    if max_rows is None:
        return total
    return min(total, max_rows)


def _hash_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _generation_config(*, prompt_path: Path) -> GenerationConfig:
    return GenerationConfig.model_validate(
        {
            "base_url": BASE_URL,
            "model": SOL_ADJUDICATION_MODEL,
            "api_key_env": None,
            "timeout_seconds": 120.0,
            "maximum_http_attempts": 5,
            "maximum_total_requests": None,
            "retry_backoff_seconds": 1.0,
            "maximum_rows_per_shard": 1,
            "max_tokens": None,
            "enable_thinking": None,
            "reasoning_effort": "none",
            "prompt": prompt_path,
            "origin_label_contract": Path("config/folk2-ieland-labels-da.yaml"),
        }
    )


def _load_or_create_status(
    *, status_path: Path, output_dir: Path, manifest: dict[str, JSONDocument]
) -> dict[str, object]:
    manifest_sha256 = _hash_json(manifest)
    if not status_path.exists():
        status: dict[str, object] = {
            "version": STATUS_VERSION,
            "manifest_sha256": manifest_sha256,
            "counts": {"consistent": 0, "patched": 0, "unresolved": 0},
            "processed": [],
        }
        _write_status(path=status_path, status=status)
        return status
    _require_private_file(path=status_path, label="status")
    status = _load_json_object(path=status_path, label="status")
    if status.get("version") != STATUS_VERSION:
        raise PersonaReleaseAdjudicationError("Status version does not match")
    if status.get("manifest_sha256") != manifest_sha256:
        raise PersonaReleaseAdjudicationError("Status manifest binding does not match")
    _validate_status(status=status, output_dir=output_dir)
    return status


def _load_json_object(*, path: Path, label: str) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PersonaReleaseAdjudicationError(f"{label} is not readable") from exc
    if not isinstance(document, dict):
        raise PersonaReleaseAdjudicationError(f"{label} must be a JSON object")
    return t.cast(dict[str, object], document)


def _require_private_file(*, path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise PersonaReleaseAdjudicationError(f"{label} is unsafe")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        raise PersonaReleaseAdjudicationError(f"{label} must be private")


def _validate_status(*, status: dict[str, object], output_dir: Path) -> None:
    processed = _processed_records(status=status)
    seen: set[str] = set()
    for record in processed:
        persona_hash = record["persona_hash"]
        if persona_hash in seen:
            raise PersonaReleaseAdjudicationError("Status contains duplicate rows")
        seen.add(persona_hash)
        checkpoint = _checkpoint_path(output_dir=output_dir, persona_hash=persona_hash)
        if record["checkpoint"] != _checkpoint_reference(
            output_dir=output_dir, path=checkpoint
        ):
            raise PersonaReleaseAdjudicationError(
                "Status checkpoint reference does not match"
            )
        if not checkpoint.exists():
            raise PersonaReleaseAdjudicationError(
                "Status references missing checkpoint"
            )
        _require_private_file(path=checkpoint, label="checkpoint")
        if sha256_file(checkpoint) != record["checkpoint_sha256"]:
            raise PersonaReleaseAdjudicationError(
                "Status checkpoint hash does not match"
            )
        if _checkpoint_response_hash(path=checkpoint) != record["response_sha256"]:
            raise PersonaReleaseAdjudicationError("Status response hash does not match")
    counts = status.get("counts")
    if counts != _counts_from_processed(processed=processed):
        raise PersonaReleaseAdjudicationError("Status counts are inconsistent")


def _checkpoint_path(*, output_dir: Path, persona_hash: str) -> Path:
    return output_dir / "checkpoints" / persona_hash[:2] / f"{persona_hash}.json"


def _checkpoint_reference(*, output_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(output_dir).as_posix()
    except ValueError as exc:
        raise PersonaReleaseAdjudicationError(
            "Checkpoint path is outside output"
        ) from exc


def _checkpoint_response_hash(*, path: Path) -> str:
    checkpoint = _load_json_object(path=path, label="checkpoint")
    response_sha256 = checkpoint.get("response_sha256")
    if not _is_sha256(value=response_sha256):
        raise PersonaReleaseAdjudicationError("Checkpoint response hash is invalid")
    return str(response_sha256)


def _is_sha256(*, value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _counts_from_processed(*, processed: list[dict[str, str]]) -> dict[str, int]:
    counts = {"consistent": 0, "patched": 0, "unresolved": 0}
    for record in processed:
        counts[record["disposition"]] += 1
    return counts


def _processed_records(*, status: dict[str, object]) -> list[dict[str, str]]:
    processed = status.get("processed")
    if not isinstance(processed, list):
        raise PersonaReleaseAdjudicationError("Status processed rows are invalid")
    records: list[dict[str, str]] = []
    for item in processed:
        if not isinstance(item, dict):
            raise PersonaReleaseAdjudicationError("Status processed rows are invalid")
        persona_hash = item.get("persona_hash")
        disposition = item.get("disposition")
        checkpoint = item.get("checkpoint")
        checkpoint_sha256 = item.get("checkpoint_sha256")
        response_sha256 = item.get("response_sha256")
        if (
            not isinstance(persona_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", persona_hash) is None
            or disposition not in {"consistent", "patched", "unresolved"}
            or not isinstance(checkpoint, str)
            or not _is_sha256(value=checkpoint_sha256)
            or not _is_sha256(value=response_sha256)
        ):
            raise PersonaReleaseAdjudicationError("Status processed rows are invalid")
        records.append(
            {
                "persona_hash": persona_hash,
                "disposition": str(disposition),
                "checkpoint": checkpoint,
                "checkpoint_sha256": str(checkpoint_sha256),
                "response_sha256": str(response_sha256),
            }
        )
    return records


def _write_status(*, path: Path, status: dict[str, object]) -> None:
    _write_private_json(path=path, value=t.cast(dict[str, JSONDocument], status))


def _write_private_json(*, path: Path, value: dict[str, JSONDocument]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _require_private_directory(path=path.parent, label="private parent directory")
    temporary = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _require_private_directory(*, path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise PersonaReleaseAdjudicationError(f"{label} is unsafe")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o700:
        raise PersonaReleaseAdjudicationError(f"{label} must be private")


def _prepare_private_output(*, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    _require_private_directory(path=output_dir, label="output directory")
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(mode=0o700, exist_ok=True)
    _require_private_directory(path=checkpoints, label="checkpoint directory")


def _require_worker_count(*, workers: int) -> None:
    if not 1 <= workers <= 4:
        raise PersonaReleaseAdjudicationError("Worker count must be between 1 and 4")


def _status_summary(
    *, status: dict[str, object], max_rows: int | None, workers: int
) -> dict[str, object]:
    processed = _processed_records(status=status)
    counts = _counts_from_processed(processed=processed)
    selected_total = _status_selected_total(status=status)
    selected = _bounded_total(total=selected_total, max_rows=max_rows)
    processed_in_scope = min(len(processed), selected)
    budget_summary = None
    return {
        "dry_run": False,
        "campaign": CAMPAIGN,
        "manifest_sha256": status["manifest_sha256"],
        "selected": selected,
        "total": selected_total,
        "pending": max(selected - processed_in_scope, 0),
        "processed": processed_in_scope,
        "consistent": counts["consistent"],
        "patched": counts["patched"],
        "unresolved": counts["unresolved"],
        "max_rows": max_rows,
        "workers": workers,
        "budget": budget_summary,
    }


def _status_selected_total(*, status: dict[str, object]) -> int:
    value = status.get("selected_total")
    if isinstance(value, int) and value >= 0:
        return value
    processed = _processed_records(status=status)
    return len(processed)


def _failing_transport() -> httpx.MockTransport:
    def respond(_request: httpx.Request) -> httpx.Response:
        raise RuntimeError("checkpoint resume attempted network I/O")

    return httpx.MockTransport(respond)


def _write_or_check_manifest(*, path: Path, manifest: dict[str, JSONDocument]) -> None:
    if path.exists():
        _require_private_file(path=path, label="manifest")
        saved = _load_json_object(path=path, label="manifest")
        if saved != manifest:
            raise PersonaReleaseAdjudicationError("Manifest binding does not match")
        return
    _write_private_json(path=path, value=manifest)


@dataclass(frozen=True)
class ReleaseInputs:
    """Loaded full-release frames and prompt text."""

    original_rows: list[dict[str, object]]
    candidate_rows: list[dict[str, object]]
    ordered_id_hashes: list[str]
    ordered_id_sha256: str
    ordered_id_hashes_sha256: str
    candidate_preview_sha256: str
    report_sha256: str
    prompt: str


def _proxy_budget(
    *, paths: ReleasePaths, inputs: ReleaseInputs, manifest: c.Mapping[str, object]
) -> ProxyBudget:
    del manifest
    schema_hash = sha256_text(
        canonical_json(SolAdjudicationResponse.provider_json_schema())
    )
    source_hash = sha256_text(
        canonical_json(
            {
                "candidate_preview_sha256": inputs.candidate_preview_sha256,
                "ordered_id_sha256": inputs.ordered_id_sha256,
                "ordered_id_hashes_sha256": inputs.ordered_id_hashes_sha256,
                "report_sha256": inputs.report_sha256,
            }
        )
    )
    return ProxyBudget(
        ledger_path=paths.output_dir / "ignored-sol-budget.jsonl",
        registry_path=paths.registry,
        campaign=CAMPAIGN,
        source_hash=source_hash,
        prompt_hash=sha256_file(paths.prompt),
        schema_hash=schema_hash,
        model=SOL_ADJUDICATION_MODEL,
        input_usd_per_million="2",
        output_usd_per_million="10",
        max_tokens=SOL_ADJUDICATION_LEDGER_MAX_TOKENS,
        cap_usd=Decimal("1"),
        uncapped=True,
        uncapped_purpose=SOL_ADJUDICATION_PURPOSE,
    )


def load_release_inputs(
    *, paths: ReleasePaths, expected_original_sha256: str, expected_row_count: int
) -> ReleaseInputs:
    """Load and validate immutable release inputs without provider I/O.

    Returns:
        Validated frames, prompt text, and private ordered-ID bindings.

    Raises:
        PersonaReleaseAdjudicationError: If an input binding is unsafe or stale.
    """
    original_sha256 = sha256_file(paths.original)
    if original_sha256 != expected_original_sha256:
        raise PersonaReleaseAdjudicationError(
            "Frozen original parquet SHA-256 mismatch"
        )
    original = pl.read_parquet(paths.original)
    candidate = pl.read_parquet(paths.candidate)
    _require_columns(frame=original, label="original")
    _require_columns(frame=candidate, label="candidate")
    _require_row_count(frame=original, expected=expected_row_count, label="original")
    _require_row_count(frame=candidate, expected=expected_row_count, label="candidate")
    original_ids = _ordered_ids(frame=original, label="original")
    candidate_ids = _ordered_ids(frame=candidate, label="candidate")
    if original_ids != candidate_ids:
        raise PersonaReleaseAdjudicationError("Ordered release IDs do not match")

    report = _load_json_object(path=paths.report, label="candidate report")
    preview_sha256 = _frame_hash(frame=candidate)
    if _json_string(
        document=report, key="preview_sha256", label="candidate report"
    ) != (preview_sha256):
        raise PersonaReleaseAdjudicationError(
            "Candidate report preview SHA-256 mismatch"
        )
    reported_count = report.get("row_count")
    if isinstance(reported_count, int) and reported_count != expected_row_count:
        raise PersonaReleaseAdjudicationError("Candidate report row count mismatch")
    source_hashes = report.get("source_hashes")
    if isinstance(source_hashes, dict):
        reported_original = source_hashes.get("original_v1_sha256")
        if isinstance(reported_original, str) and reported_original != original_sha256:
            raise PersonaReleaseAdjudicationError(
                "Candidate report original SHA mismatch"
            )

    prompt = paths.prompt.read_text(encoding="utf-8")
    _check_restricted_text(value=prompt, label="prompt")
    original_rows = original.to_dicts()
    candidate_rows = candidate.to_dicts()
    _check_rows_private(original_rows=original_rows, candidate_rows=candidate_rows)
    id_hashes = [sha256_text(persona_id) for persona_id in original_ids]
    return ReleaseInputs(
        original_rows=original_rows,
        candidate_rows=candidate_rows,
        ordered_id_hashes=id_hashes,
        ordered_id_sha256=sha256_text(canonical_json(original_ids)),
        ordered_id_hashes_sha256=sha256_text(canonical_json(id_hashes)),
        candidate_preview_sha256=preview_sha256,
        report_sha256=sha256_file(paths.report),
        prompt=prompt,
    )


def _check_restricted_text(*, value: str, label: str) -> None:
    if _RESTRICTED_TEXT.search(value):
        raise PersonaReleaseAdjudicationError(f"{label} contains restricted text")


def _check_rows_private(
    *, original_rows: list[dict[str, object]], candidate_rows: list[dict[str, object]]
) -> None:
    for original, candidate in zip(original_rows, candidate_rows, strict=True):
        original_persona = original.get(PERSONA_FIELD)
        candidate_persona = candidate.get(PERSONA_FIELD)
        if not isinstance(original_persona, str) or not isinstance(
            candidate_persona, str
        ):
            raise PersonaReleaseAdjudicationError("Persona prose is missing")
        _check_restricted_text(value=original_persona, label="original persona prose")
        _check_restricted_text(value=candidate_persona, label="candidate persona prose")


def _frame_hash(*, frame: pl.DataFrame) -> str:
    payload = {"columns": frame.columns, "rows": _jsonable(frame.to_dicts())}
    return sha256_text(canonical_json(payload))


def _jsonable(value: object) -> JSONDocument:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, c.Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, c.Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(child) for child in value]
    return str(value)


def _json_string(*, document: dict[str, object], key: str, label: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PersonaReleaseAdjudicationError(f"{label} has invalid checksum")
    return value


def _ordered_ids(*, frame: pl.DataFrame, label: str) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for value in frame.get_column(ID_FIELD).to_list():
        if not isinstance(value, str) or not value:
            raise PersonaReleaseAdjudicationError(f"{label} frame has invalid IDs")
        if value in seen:
            raise PersonaReleaseAdjudicationError(f"{label} frame has duplicate IDs")
        seen.add(value)
        ids.append(value)
    return ids


def _require_columns(*, frame: pl.DataFrame, label: str) -> None:
    missing = {ID_FIELD, PERSONA_FIELD} - set(frame.columns)
    if missing:
        raise PersonaReleaseAdjudicationError(f"{label} frame is missing columns")


def _require_row_count(*, frame: pl.DataFrame, expected: int, label: str) -> None:
    if frame.height != expected:
        raise PersonaReleaseAdjudicationError(f"{label} frame row count mismatch")


@dataclass(frozen=True)
class ReleaseRow:
    """One ordered release row selected for Sol adjudication."""

    index: int
    persona_hash: str
    original_row: dict[str, object]
    candidate_row: dict[str, object]
    changed_facts: dict[str, dict[str, object]]


def _preflight_selected_rows(*, rows: list[ReleaseRow], prompt: str) -> None:
    for row in rows:
        try:
            preflight_sol_adjudication_payload(
                original_persona=str(row.candidate_row[PERSONA_FIELD]),
                candidate_row=row.candidate_row,
                prompt=prompt,
                changed_fact_hints=row.changed_facts,
                original_row=row.original_row,
            )
        except SolAdjudicationError as exc:
            raise PersonaReleaseAdjudicationError(
                "Selected Sol row is privacy-unsafe for outbound adjudication"
            ) from exc


def _process_pending(
    *,
    rows: list[ReleaseRow],
    status: dict[str, object],
    status_path: Path,
    output_dir: Path,
    prompt: str,
    config: GenerationConfig,
    budget: ProxyBudget,
    workers: int,
    sol_runner: SolRunner,
) -> None:
    row_iter = iter(rows)
    future_map: dict[futures.Future[RowResult], ReleaseRow] = {}
    stop_exc: Exception | None = None
    executor = futures.ThreadPoolExecutor(max_workers=workers)
    try:
        _submit_futures(
            row_iter=row_iter,
            future_map=future_map,
            executor=executor,
            limit=workers,
            output_dir=output_dir,
            prompt=prompt,
            config=config,
            budget=budget,
            sol_runner=sol_runner,
        )
        while future_map:
            done, _ = futures.wait(future_map, return_when=futures.FIRST_COMPLETED)
            future = next(iter(done))
            row = future_map.pop(future)
            if future.cancelled():
                continue
            completed_exc = _record_completed_future(
                future=future, row=row, status=status, status_path=status_path
            )
            if completed_exc is not None:
                if stop_exc is None:
                    stop_exc = completed_exc
                _cancel_not_started(future_map=future_map)
                continue
            if stop_exc is None and not _has_completed_future(future_map=future_map):
                _submit_futures(
                    row_iter=row_iter,
                    future_map=future_map,
                    executor=executor,
                    limit=workers,
                    output_dir=output_dir,
                    prompt=prompt,
                    config=config,
                    budget=budget,
                    sol_runner=sol_runner,
                )
    finally:
        if stop_exc is not None:
            _cancel_not_started(future_map=future_map)
        executor.shutdown(wait=stop_exc is None, cancel_futures=stop_exc is not None)
    if stop_exc is not None:
        raise PersonaReleaseAdjudicationError(
            "Sol adjudication stopped before all selected rows were resolved"
        ) from stop_exc


def _cancel_not_started(
    *, future_map: dict[futures.Future[FutureResult], ReleaseRow]
) -> None:
    for future in future_map:
        future.cancel()


def _has_completed_future(
    *, future_map: dict[futures.Future[FutureResult], ReleaseRow]
) -> bool:
    return any(future.done() for future in future_map)


@dataclass(frozen=True)
class ReleaseSelection:
    """Deterministic full-campaign Sol selection."""

    rows: list[ReleaseRow]
    ordered_id_sha256: str
    ordered_id_hashes_sha256: str
    candidate_preview_sha256: str


def _campaign_manifest(
    *, paths: ReleasePaths, inputs: ReleaseInputs, selection: ReleaseSelection
) -> dict[str, JSONDocument]:
    schema_hash = sha256_text(
        canonical_json(SolAdjudicationResponse.provider_json_schema())
    )
    return {
        "version": MANIFEST_VERSION,
        "campaign": CAMPAIGN,
        "inputs": {
            "original": sha256_file(paths.original),
            "candidate": sha256_file(paths.candidate),
            "candidate_report": inputs.report_sha256,
            "candidate_preview": inputs.candidate_preview_sha256,
            "prompt": sha256_file(paths.prompt),
            "schema": schema_hash,
            "registry": sha256_file(paths.registry),
        },
        "model": SOL_ADJUDICATION_MODEL,
        "base_url": BASE_URL,
        "reasoning_effort": "none",
        "max_tokens": SOL_MAX_OUTPUT_TOKENS,
        "maximum_http_attempts": 5,
        "budget_purpose": SOL_ADJUDICATION_PURPOSE,
        "allowed_facts": sorted(SOL_ALLOWED_FACT_FIELDS),
        "selection": {
            "ordered_rows": len(selection.rows),
            "ordered_id_sha256": selection.ordered_id_sha256,
            "ordered_id_hashes_sha256": selection.ordered_id_hashes_sha256,
        },
    }


def _dry_run_summary(
    *,
    manifest: dict[str, JSONDocument],
    selection: ReleaseSelection,
    max_rows: int | None,
    workers: int,
) -> dict[str, object]:
    selected = _bounded_total(total=len(selection.rows), max_rows=max_rows)
    return {
        "dry_run": True,
        "run_required": True,
        "campaign": CAMPAIGN,
        "manifest_sha256": _hash_json(manifest),
        "selected": selected,
        "total": len(selection.rows),
        "pending": selected,
        "consistent": 0,
        "patched": 0,
        "unresolved": 0,
        "processed": 0,
        "max_rows": max_rows,
        "workers": workers,
    }


def _pending_rows(
    *, selection: ReleaseSelection, status: dict[str, object], max_rows: int | None
) -> list[ReleaseRow]:
    processed = {record["persona_hash"] for record in _processed_records(status=status)}
    rows = _scoped_rows(selection=selection, max_rows=max_rows)
    status["selected_total"] = len(selection.rows)
    return [row for row in rows if row.persona_hash not in processed]


def _scoped_rows(
    *, selection: ReleaseSelection, max_rows: int | None
) -> list[ReleaseRow]:
    return selection.rows[
        : _bounded_total(total=len(selection.rows), max_rows=max_rows)
    ]


def _validate_processed_checkpoints(
    *,
    selection: ReleaseSelection,
    status: dict[str, object],
    output_dir: Path,
    prompt: str,
    config: GenerationConfig,
    budget: ProxyBudget,
) -> None:
    rows_by_hash = {row.persona_hash: row for row in selection.rows}
    transport = _failing_transport()
    try:
        for record in _processed_records(status=status):
            row = rows_by_hash.get(record["persona_hash"])
            if row is None:
                raise PersonaReleaseAdjudicationError(
                    "Status references a row outside the current selection"
                )
            checkpoint_path = _checkpoint_path(
                output_dir=output_dir, persona_hash=row.persona_hash
            )
            try:
                result = run_sol_adjudication(
                    original_persona=str(row.candidate_row[PERSONA_FIELD]),
                    candidate_row=row.candidate_row,
                    prompt=prompt,
                    config=config,
                    budget=budget,
                    checkpoint_path=checkpoint_path,
                    transport=transport,
                    changed_fact_hints=row.changed_facts,
                    original_row=row.original_row,
                )
            except (RuntimeError, SolAdjudicationError) as exc:
                raise PersonaReleaseAdjudicationError(
                    "Processed Sol checkpoint does not match current inputs"
                ) from exc
            if result.disposition != record["disposition"]:
                raise PersonaReleaseAdjudicationError(
                    "Processed Sol checkpoint disposition does not match status"
                )
    finally:
        transport.close()


def select_release_rows(*, inputs: ReleaseInputs) -> ReleaseSelection:
    """Select every ordered release row for Sol adjudication.

    Returns:
        Full-campaign selection in immutable release order.
    """
    rows: list[ReleaseRow] = []
    for index, (original, candidate, persona_hash) in enumerate(
        zip(
            inputs.original_rows,
            inputs.candidate_rows,
            inputs.ordered_id_hashes,
            strict=True,
        )
    ):
        rows.append(
            ReleaseRow(
                index=index,
                persona_hash=persona_hash,
                original_row=original,
                candidate_row=candidate,
                changed_facts=_changed_facts(original=original, candidate=candidate),
            )
        )
    return ReleaseSelection(
        rows=rows,
        ordered_id_sha256=inputs.ordered_id_sha256,
        ordered_id_hashes_sha256=inputs.ordered_id_hashes_sha256,
        candidate_preview_sha256=inputs.candidate_preview_sha256,
    )


def _changed_facts(
    *, original: dict[str, object], candidate: dict[str, object]
) -> dict[str, dict[str, object]]:
    facts: dict[str, dict[str, object]] = {}
    for field in sorted((set(original) & set(candidate)) & SOL_ALLOWED_FACT_FIELDS):
        old = original[field]
        new = candidate[field]
        if old != new:
            facts[field] = {"old": old, "new": new}
    return facts


@dataclass(frozen=True)
class RowResult:
    """Sanitised result for one completed row."""

    persona_hash: str
    disposition: t.Literal["consistent", "patched", "unresolved"]
    checkpoint: str
    checkpoint_sha256: str
    response_sha256: str


def _record_completed_future(
    *,
    future: futures.Future[RowResult],
    row: ReleaseRow,
    status: dict[str, object],
    status_path: Path,
) -> Exception | None:
    try:
        result = future.result()
    except Exception as exc:
        return exc
    if result.persona_hash != row.persona_hash:
        return PersonaReleaseAdjudicationError("Completed row binding does not match")
    _record_status_result(status=status, result=result)
    _write_status(path=status_path, status=status)
    return None


def _record_status_result(*, status: dict[str, object], result: RowResult) -> None:
    processed = _processed_records(status=status)
    if any(record["persona_hash"] == result.persona_hash for record in processed):
        return
    processed.append(
        {
            "persona_hash": result.persona_hash,
            "disposition": result.disposition,
            "checkpoint": result.checkpoint,
            "checkpoint_sha256": result.checkpoint_sha256,
            "response_sha256": result.response_sha256,
        }
    )
    status["processed"] = processed
    status["counts"] = _counts_from_processed(processed=processed)


def _submit_futures(
    *,
    row_iter: c.Iterator[ReleaseRow],
    future_map: dict[futures.Future[RowResult], ReleaseRow],
    executor: futures.ThreadPoolExecutor,
    limit: int,
    output_dir: Path,
    prompt: str,
    config: GenerationConfig,
    budget: ProxyBudget,
    sol_runner: SolRunner,
) -> None:
    while len(future_map) < limit:
        try:
            row = next(row_iter)
        except StopIteration:
            return
        future_map[
            executor.submit(
                _run_one_row,
                row=row,
                output_dir=output_dir,
                prompt=prompt,
                config=config,
                budget=budget,
                sol_runner=sol_runner,
            )
        ] = row


def _run_one_row(
    *,
    row: ReleaseRow,
    output_dir: Path,
    prompt: str,
    config: GenerationConfig,
    budget: ProxyBudget,
    sol_runner: SolRunner,
) -> RowResult:
    transport = httpx.HTTPTransport()
    checkpoint_path = _checkpoint_path(
        output_dir=output_dir, persona_hash=row.persona_hash
    )
    try:
        result = sol_runner(
            str(row.candidate_row[PERSONA_FIELD]),
            row.candidate_row,
            prompt,
            config,
            budget,
            checkpoint_path,
            transport,
            row.changed_facts,
            row.original_row,
        )
    finally:
        transport.close()
    return RowResult(
        persona_hash=row.persona_hash,
        disposition=result.disposition,
        checkpoint=_checkpoint_reference(output_dir=output_dir, path=checkpoint_path),
        checkpoint_sha256=sha256_file(checkpoint_path),
        response_sha256=_checkpoint_response_hash(path=checkpoint_path),
    )


def _run_sol_adjudication_adapter(
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
    require_runtime_model(ADJUDICATION_MODEL_ENV)
    return run_sol_adjudication(
        original_persona=original_persona,
        candidate_row=candidate_row,
        prompt=prompt,
        config=config,
        budget=budget,
        checkpoint_path=checkpoint_path,
        transport=transport,
        changed_fact_hints=changed_fact_hints,
        original_row=original_row,
    )


if __name__ == "__main__":
    load_repository_environment()
    main()
