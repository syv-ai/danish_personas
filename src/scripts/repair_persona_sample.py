"""Run a bounded local-proxy batch for persona prose repair proposals."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import typing as t
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import click
import httpx
import polars as pl

from danish_personas.cli_logging import configure_cli_logging
from danish_personas.generation.models import GenerationConfig
from danish_personas.generation.prose_patch import ProsePatchResponse
from danish_personas.generation.proxy_budget import (
    BASE_URL,
    MODEL,
    ProxyBudget,
    ProxyBudgetError,
)
from danish_personas.generation.proxy_patch_runner import (
    _ALLOWED_FACTS,
    ProxyPatchError,
    ProxyPatchProposal,
    run_proxy_patch,
)
from danish_personas.io import canonical_json, sha256_file, sha256_text

LOGGER = logging.getLogger(__name__)

DEFAULT_ROOT = Path("/tmp/danish-personas-audit")
DEFAULT_ORIGINAL = DEFAULT_ROOT / "data/train-00000-of-00001.parquet"
DEFAULT_CANDIDATE = DEFAULT_ROOT / "attribute-candidate-v2.parquet"
DEFAULT_IDENTITY_SIDECAR = DEFAULT_ROOT / "paired-identity-v2.parquet"
DEFAULT_TRIAGE = DEFAULT_ROOT / "prose-triage-v1.json"
DEFAULT_REPORT = DEFAULT_ROOT / "attribute-candidate-v2.report.json"
DEFAULT_PROMPT = Path("config/persona-patch-da.md")
DEFAULT_OUTPUT_DIR = Path("/tmp/danish-personas-audit/persona-repair-proposals")
DEFAULT_REGISTRY = Path.home() / ".pi" / "agent" / "models-store.json"
EXPECTED_ORIGINAL_SHA256 = (
    "c178e63d40046274bcc559bdd9322f32336656da980c1809f794d68449d6250f"
)
CAMPAIGN = "persona-sample-proxy-repair-v1"
MAX_CAMPAIGN_USD = Decimal("10")
DEFAULT_MAX_ATTEMPTS = 100
DRY_RUN_LIMIT = 5
ID_FIELD = "persona_id"
PERSONA_FIELD = "persona"
IDENTITY_COLUMNS = (ID_FIELD, "gender", "partner_gender")
COMPANION_CODE_FIELDS = {
    "detailed_status_code": "detailed_status",
    "job_function_code": "job_function",
    "education_source_code": "education_level",
}
PRIORITY_FACTS = {
    "legal_status_detail": 0,
    "marital_status": 1,
    "education_level": 2,
    "job_function": 3,
    "job_title": 3,
    "detailed_status": 3,
    "labour_market_status": 3,
    "hobbies_and_interests": 4,
    "skills_and_expertise": 4,
    "current_relationship_status": 5,
    "age": 6,
}
SENSITIVE_IDENTITY_FIELDS = frozenset(
    {
        "sexual_orientation",
        "partner_sexual_orientation",
        "transgender",
        "partner_transgender",
        "variation_in_sex_characteristics",
        "partner_variation_in_sex_characteristics",
        "same_sex_partner_target",
        "partner_same_sex_target",
    }
)

JSONScalar: t.TypeAlias = str | int | float | bool | None
JSONValue: t.TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
PatchRunner: t.TypeAlias = t.Callable[
    [
        dict[str, t.Any],
        dict[str, t.Any],
        dict[str, dict[str, object]],
        str | None,
        str | None,
        str,
        GenerationConfig,
        ProxyBudget,
        Path,
        httpx.BaseTransport,
    ],
    ProxyPatchProposal,
]


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise RepairSampleError("Identity sidecar values must be strings or null")


class RepairSampleError(RuntimeError):
    """Raised when the sample repair CLI must fail closed."""


def _run_proxy_patch_adapter(
    row: dict[str, t.Any],
    candidate_row: dict[str, t.Any],
    changed_facts: dict[str, dict[str, object]],
    gender: str | None,
    partner_gender: str | None,
    prompt: str,
    config: GenerationConfig,
    budget: ProxyBudget,
    checkpoint_path: Path,
    transport: httpx.BaseTransport,
) -> ProxyPatchProposal:
    return run_proxy_patch(
        row=row,
        candidate_row=candidate_row,
        changed_facts=changed_facts,
        gender=gender,
        partner_gender=partner_gender,
        prompt=prompt,
        config=config,
        budget=budget,
        checkpoint_path=checkpoint_path,
        transport=transport,
    )


@click.command()
@click.option("--original", type=click.Path(path_type=Path), default=DEFAULT_ORIGINAL)
@click.option("--candidate", type=click.Path(path_type=Path), default=DEFAULT_CANDIDATE)
@click.option(
    "--identity-sidecar",
    type=click.Path(path_type=Path),
    default=DEFAULT_IDENTITY_SIDECAR,
)
@click.option("--triage", type=click.Path(path_type=Path), default=DEFAULT_TRIAGE)
@click.option("--report", type=click.Path(path_type=Path), default=DEFAULT_REPORT)
@click.option("--prompt", type=click.Path(path_type=Path), default=DEFAULT_PROMPT)
@click.option(
    "--output-dir", type=click.Path(path_type=Path), default=DEFAULT_OUTPUT_DIR
)
@click.option("--registry", type=click.Path(path_type=Path), default=DEFAULT_REGISTRY)
@click.option(
    "--max-attempts",
    type=click.IntRange(min=1, max=DEFAULT_MAX_ATTEMPTS),
    default=DEFAULT_MAX_ATTEMPTS,
    show_default=True,
)
@click.option("--dry-run", is_flag=True, default=False)
@click.option(
    "--cost-cap-usd", default=str(MAX_CAMPAIGN_USD), show_default=True, metavar="USD"
)
def repair_persona_sample(
    original: Path,
    candidate: Path,
    identity_sidecar: Path,
    triage: Path,
    report: Path,
    prompt: Path,
    output_dir: Path,
    registry: Path,
    max_attempts: int,
    dry_run: bool,
    cost_cap_usd: str,
) -> None:
    """Select and run a bounded batch of local proxy prose patch proposals.

    Raises:
        click.ClickException: If input, budget, or proxy boundaries are unsafe.
    """
    configure_cli_logging()
    paths = RepairPaths(
        original=original,
        candidate=candidate,
        identity_sidecar=identity_sidecar,
        triage=triage,
        report=report,
        prompt=prompt,
        output_dir=output_dir,
        registry=registry,
    )
    try:
        summary = run_repair_campaign(
            paths=paths,
            max_attempts=max_attempts,
            dry_run=dry_run,
            cost_cap_usd=_parse_cap(cost_cap_usd),
            expected_original_sha256=EXPECTED_ORIGINAL_SHA256,
            patch_runner=_run_proxy_patch_adapter,
        )
    except (RepairSampleError, ProxyPatchError, ProxyBudgetError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(summary, ensure_ascii=False, sort_keys=True))


@dataclass(frozen=True)
class RepairPaths:
    """Filesystem inputs and private output location."""

    original: Path
    candidate: Path
    identity_sidecar: Path
    triage: Path
    report: Path
    prompt: Path
    output_dir: Path
    registry: Path


def _parse_cap(value: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise click.BadParameter("cost cap must be a decimal USD amount") from exc


def run_repair_campaign(
    *,
    paths: RepairPaths,
    max_attempts: int,
    dry_run: bool,
    cost_cap_usd: Decimal,
    expected_original_sha256: str,
    patch_runner: PatchRunner,
) -> dict[str, object]:
    """Run or dry-run the persona repair campaign.

    Args:
        paths: Input and output paths.
        max_attempts: Maximum selected rows in this campaign.
        dry_run: If true, only emit private-ID-free selection metadata.
        cost_cap_usd: Campaign cap, which must not exceed 10 USD.
        expected_original_sha256: Required SHA-256 of the frozen baseline parquet.
        patch_runner: Injectable proxy patch runner for offline tests.

    Returns:
        Machine-readable progress summary without raw persona IDs or prose.
    """
    _require_attempt_limit(max_attempts=max_attempts)
    _require_campaign_cap(cost_cap_usd=cost_cap_usd)
    manifest = build_manifest(
        paths=paths,
        max_attempts=max_attempts,
        cost_cap_usd=cost_cap_usd,
        expected_original_sha256=expected_original_sha256,
    )
    loaded = load_repair_inputs(paths=paths)
    selected = select_eligible_repairs(inputs=loaded, max_attempts=max_attempts)
    if dry_run:
        return _dry_run_summary(selected=selected[:DRY_RUN_LIMIT], manifest=manifest)
    _prepare_private_output(paths.output_dir)
    status_path = paths.output_dir / "status.json"
    status = _load_or_create_status(
        status_path=status_path, manifest=manifest, total=len(selected)
    )
    config = _generation_config(prompt_path=paths.prompt, max_attempts=max_attempts)
    budget = _proxy_budget(
        paths=paths, prompt=loaded.prompt, manifest=manifest, cost_cap_usd=cost_cap_usd
    )
    transport = httpx.HTTPTransport()
    try:
        _process_selected(
            selected=selected,
            status=status,
            status_path=status_path,
            output_dir=paths.output_dir,
            prompt=loaded.prompt,
            config=config,
            budget=budget,
            transport=transport,
            patch_runner=patch_runner,
        )
    finally:
        transport.close()
    return _public_status_summary(status=status, status_path=status_path)


def _generation_config(*, prompt_path: Path, max_attempts: int) -> GenerationConfig:
    return GenerationConfig(
        base_url=BASE_URL,
        model=MODEL,
        api_key_env=None,
        timeout_seconds=120.0,
        maximum_http_attempts=1,
        maximum_total_requests=max_attempts,
        retry_backoff_seconds=0.0,
        maximum_rows_per_shard=1,
        max_tokens=None,
        enable_thinking=None,
        reasoning_effort="none",
        prompt=prompt_path,
        origin_label_contract=Path("config/folk2-ieland-labels-da.yaml"),
    )


def _load_or_create_status(
    *, status_path: Path, manifest: dict[str, JSONValue], total: int
) -> dict[str, t.Any]:
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("manifest") != manifest:
            raise RepairSampleError("status.json pins do not match current inputs")
        return _validate_status(status)
    status = {
        "manifest": manifest,
        "total": total,
        "processed": 0,
        "proposed": 0,
        "failed": 0,
        "skipped": 0,
        "attempted": 0,
        "processed_persona_ids": [],
    }
    _write_status(path=status_path, status=status)
    return status


def _validate_status(status: dict[str, t.Any]) -> dict[str, t.Any]:
    for key in ("total", "processed", "proposed", "failed", "skipped", "attempted"):
        if not isinstance(status.get(key), int) or status[key] < 0:
            raise RepairSampleError("status.json progress counters are invalid")
    _status_ids(status)
    return status


def _status_ids(status: dict[str, t.Any]) -> list[str]:
    ids = status.get("processed_persona_ids")
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise RepairSampleError("status.json processed IDs are invalid")
    return ids


def _write_status(*, path: Path, status: dict[str, t.Any]) -> None:
    content = json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_private_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output_dir, 0o700)
    if output_dir.stat().st_mode & 0o077:
        raise RepairSampleError("Output directory must be private (mode 0700)")


def _checkpoint_path(*, output_dir: Path, persona_hash: str) -> Path:
    return output_dir / "checkpoints" / persona_hash[:2] / f"{persona_hash}.json"


def _mark_processed(*, status: dict[str, t.Any], persona_id: str) -> None:
    ids = _status_ids(status)
    if persona_id not in ids:
        ids.append(persona_id)
    status["processed_persona_ids"] = ids
    status["processed"] = len(ids)


def _must_stop(exc: Exception) -> bool:
    if isinstance(exc, ProxyBudgetError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {401, 403, 429}
    if isinstance(exc, ProxyPatchError):
        return "changed" in str(exc).lower() or "pins" in str(exc).lower()
    return False


def _proxy_budget(
    *,
    paths: RepairPaths,
    prompt: str,
    manifest: dict[str, JSONValue],
    cost_cap_usd: Decimal,
) -> ProxyBudget:
    inputs = manifest["inputs"]
    if not isinstance(inputs, dict) or not isinstance(inputs.get("schema"), str):
        raise RepairSampleError("Manifest schema hash is malformed")
    schema_hash = inputs["schema"]
    return ProxyBudget(
        registry_path=paths.registry,
        campaign=CAMPAIGN,
        source_hash=sha256_text(canonical_json(manifest)),
        prompt_hash=sha256_text(prompt),
        schema_hash=schema_hash,
        cap_usd=cost_cap_usd,
    )


def _public_status_summary(
    *, status: dict[str, t.Any], status_path: Path
) -> dict[str, object]:
    return {
        "dry_run": False,
        "status_path": str(status_path),
        "total": status["total"],
        "processed": status["processed"],
        "proposed": status["proposed"],
        "failed": status["failed"],
        "skipped": status["skipped"],
        "attempted": status["attempted"],
    }


def _require_attempt_limit(*, max_attempts: int) -> None:
    if not 1 <= max_attempts <= DEFAULT_MAX_ATTEMPTS:
        raise RepairSampleError("max-attempts must be between 1 and 100")


def _require_campaign_cap(*, cost_cap_usd: Decimal) -> None:
    if cost_cap_usd <= 0 or cost_cap_usd > MAX_CAMPAIGN_USD:
        raise RepairSampleError("cost cap must be greater than 0 and at most 10 USD")


def build_manifest(
    *,
    paths: RepairPaths,
    max_attempts: int,
    cost_cap_usd: Decimal,
    expected_original_sha256: str,
) -> dict[str, JSONValue]:
    """Build a checksum manifest for inputs and pinned proxy settings.

    Returns:
        JSON-compatible manifest for resume and budget binding.

    Raises:
        RepairSampleError: If the frozen baseline checksum does not match.
    """
    original_sha256 = sha256_file(paths.original)
    if original_sha256 != expected_original_sha256:
        raise RepairSampleError("Frozen original parquet SHA-256 does not match")
    schema_hash = sha256_text(canonical_json(ProsePatchResponse.provider_json_schema()))
    hashes = {
        "original": original_sha256,
        "candidate": sha256_file(paths.candidate),
        "identity_sidecar": sha256_file(paths.identity_sidecar),
        "triage": sha256_file(paths.triage),
        "report": sha256_file(paths.report),
        "prompt": sha256_file(paths.prompt),
        "registry": sha256_file(paths.registry),
        "schema": schema_hash,
    }
    return {
        "campaign": CAMPAIGN,
        "inputs": hashes,
        "max_attempts": max_attempts,
        "cost_cap_usd": str(cost_cap_usd),
        "model": MODEL,
        "base_url": BASE_URL,
        "reasoning_effort": "none",
        "max_tokens": None,
        "maximum_http_attempts": 1,
    }


@dataclass(frozen=True)
class RepairInputs:
    """Loaded repair inputs, keyed by private persona ID in memory only."""

    original_rows: dict[str, dict[str, t.Any]]
    candidate_rows: dict[str, dict[str, t.Any]]
    identity_rows: dict[str, dict[str, t.Any]]
    triage_personas: dict[str, dict[str, t.Any]]
    unresolved_ids: frozenset[str]
    prompt: str


def load_repair_inputs(*, paths: RepairPaths) -> RepairInputs:
    """Load only the files and sidecar columns needed by the repair selector.

    Returns:
        Loaded rows and metadata keyed by persona ID.
    """
    original = pl.read_parquet(paths.original)
    candidate = pl.read_parquet(paths.candidate)
    identity = pl.read_parquet(paths.identity_sidecar, columns=list(IDENTITY_COLUMNS))
    _require_columns(
        frame=original, columns={ID_FIELD, PERSONA_FIELD}, label="original"
    )
    _require_columns(
        frame=candidate, columns={ID_FIELD, PERSONA_FIELD}, label="candidate"
    )
    _require_columns(frame=identity, columns=set(IDENTITY_COLUMNS), label="identity")
    return RepairInputs(
        original_rows=_rows_by_id(original, label="original"),
        candidate_rows=_rows_by_id(candidate, label="candidate"),
        identity_rows=_rows_by_id(identity, label="identity"),
        triage_personas=_load_triage_personas(paths.triage),
        unresolved_ids=_load_unresolved_ids(paths.report),
        prompt=paths.prompt.read_text(encoding="utf-8"),
    )


def _load_triage_personas(path: Path) -> dict[str, dict[str, t.Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not {"personas", "counts"}.issubset(document):
        raise RepairSampleError("Triage JSON must contain personas and counts")
    personas = document["personas"]
    if not isinstance(personas, dict):
        raise RepairSampleError("Triage personas must be keyed by persona ID")
    parsed: dict[str, dict[str, t.Any]] = {}
    for persona_id, entry in personas.items():
        if not isinstance(persona_id, str) or not isinstance(entry, dict):
            raise RepairSampleError("Triage persona entries are malformed")
        parsed[persona_id] = entry
    return parsed


def _load_unresolved_ids(path: Path) -> frozenset[str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise RepairSampleError("Attribute report must be a JSON object")
    unresolved = document.get("unresolved", {})
    if not isinstance(unresolved, dict):
        raise RepairSampleError("Attribute report unresolved field must be an object")
    return frozenset(key for key in unresolved if isinstance(key, str))


def _require_columns(*, frame: pl.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise RepairSampleError(f"Missing {label} columns: {sorted(missing)}")


def _rows_by_id(frame: pl.DataFrame, *, label: str) -> dict[str, dict[str, t.Any]]:
    ids = frame.get_column(ID_FIELD)
    if ids.n_unique() != frame.height:
        raise RepairSampleError(f"{label} persona IDs must be unique")
    rows: dict[str, dict[str, t.Any]] = {}
    for row in frame.to_dicts():
        persona_id = row.get(ID_FIELD)
        if not isinstance(persona_id, str) or not persona_id:
            raise RepairSampleError(f"{label} contains an invalid persona ID")
        rows[persona_id] = row
    return rows


@dataclass(frozen=True)
class EligibleRepair:
    """One row selected for a safe proxy patch attempt."""

    persona_id: str
    persona_hash: str
    original_row: dict[str, t.Any]
    candidate_row: dict[str, t.Any]
    changed_facts: dict[str, dict[str, object]]
    gender: str | None
    partner_gender: str | None


def _selection_key(repair: EligibleRepair) -> tuple[int, int, int, str]:
    fields = set(repair.changed_facts)
    priority = min(PRIORITY_FACTS.get(field, 99) for field in fields)
    return (0 if len(fields) == 1 else 1, priority, len(fields), repair.persona_id)


def _dry_run_summary(
    *, selected: list[EligibleRepair], manifest: dict[str, JSONValue]
) -> dict[str, object]:
    return {
        "dry_run": True,
        "selected": [
            {
                "persona_sha256": repair.persona_hash,
                "changed_fields": sorted(repair.changed_facts),
            }
            for repair in selected
        ],
        "selected_count": len(selected),
        "manifest_sha256": sha256_text(canonical_json(manifest)),
    }


def _process_selected(
    *,
    selected: list[EligibleRepair],
    status: dict[str, t.Any],
    status_path: Path,
    output_dir: Path,
    prompt: str,
    config: GenerationConfig,
    budget: ProxyBudget,
    transport: httpx.BaseTransport,
    patch_runner: PatchRunner,
) -> None:
    processed_ids = set(_status_ids(status))
    for repair in selected:
        if repair.persona_id in processed_ids:
            continue
        checkpoint = _checkpoint_path(
            output_dir=output_dir, persona_hash=repair.persona_hash
        )
        try:
            status["attempted"] += 1
            patch_runner(
                repair.original_row,
                repair.candidate_row,
                repair.changed_facts,
                repair.gender,
                repair.partner_gender,
                prompt,
                config,
                budget,
                checkpoint,
                transport,
            )
        except Exception as exc:
            status["failed"] += 1
            _mark_processed(status=status, persona_id=repair.persona_id)
            _write_status(path=status_path, status=status)
            if _must_stop(exc):
                raise
            LOGGER.warning("Persona repair proposal failed for one selected row")
            continue
        status["proposed"] += 1
        _mark_processed(status=status, persona_id=repair.persona_id)
        _write_status(path=status_path, status=status)


def select_eligible_repairs(
    *, inputs: RepairInputs, max_attempts: int
) -> list[EligibleRepair]:
    """Select deterministic privacy-safe repair rows from triage metadata.

    Returns:
        Eligible rows ordered by conservative prose-repair priority.
    """
    sample_fields = (
        "legal_status_detail",
        "marital_status",
        "education_level",
        "detailed_status",
        "hobbies_and_interests",
        "job_function",
        "job_title",
    )
    buckets: dict[str, list[EligibleRepair]] = {field: [] for field in sample_fields}
    for persona_id, entry in sorted(inputs.triage_personas.items()):
        eligible = _eligible_repair(persona_id=persona_id, entry=entry, inputs=inputs)
        if eligible is None:
            continue
        fields = set(eligible.changed_facts)
        if len(fields) == 1 and (field := next(iter(fields))) in buckets:
            buckets[field].append(eligible)
    selected: list[EligibleRepair] = []
    while len(selected) < max_attempts and any(buckets.values()):
        for field in sample_fields:
            if buckets[field]:
                selected.append(buckets[field].pop(0))
            if len(selected) >= max_attempts:
                break
    return selected


def _eligible_repair(
    *, persona_id: str, entry: dict[str, t.Any], inputs: RepairInputs
) -> EligibleRepair | None:
    if entry.get("classification") != "needs_prose_review_or_regeneration":
        return None
    if persona_id in inputs.unresolved_ids:
        return None
    original = inputs.original_rows.get(persona_id)
    candidate = inputs.candidate_rows.get(persona_id)
    identity = inputs.identity_rows.get(persona_id)
    if original is None or candidate is None or identity is None:
        return None
    if original.get(PERSONA_FIELD) != candidate.get(PERSONA_FIELD):
        return None
    changed_fields = _changed_fields(entry)
    if not changed_fields or changed_fields & SENSITIVE_IDENTITY_FIELDS:
        return None
    if not _changed_fields_are_supported(changed_fields):
        return None
    actual = _actual_changed_fields(original=original, candidate=candidate)
    if not changed_fields.issubset(actual) or not actual.issubset(changed_fields):
        return None
    fact_fields = sorted(changed_fields & _ALLOWED_FACTS)
    if not fact_fields:
        return None
    changed_facts = {
        field: {"old": original[field], "new": candidate[field]}
        for field in fact_fields
    }
    return EligibleRepair(
        persona_id=persona_id,
        persona_hash=sha256_text(persona_id),
        original_row=original,
        candidate_row=candidate,
        changed_facts=changed_facts,
        gender=None,
        partner_gender=None,
    )


def _actual_changed_fields(
    *, original: dict[str, t.Any], candidate: dict[str, t.Any]
) -> set[str]:
    fields = (set(original) & set(candidate)) - {PERSONA_FIELD}
    return {field for field in fields if original[field] != candidate[field]}


def _changed_fields(entry: dict[str, t.Any]) -> set[str]:
    fields = entry.get("changed_fields")
    if not isinstance(fields, list) or not all(
        isinstance(field, str) for field in fields
    ):
        raise RepairSampleError("Triage changed_fields must be a list of strings")
    return set(fields)


def _changed_fields_are_supported(changed_fields: set[str]) -> bool:
    for field in changed_fields:
        if field in _ALLOWED_FACTS:
            continue
        visible = COMPANION_CODE_FIELDS.get(field)
        if visible is None or visible not in changed_fields:
            return False
    return True


if __name__ == "__main__":
    repair_persona_sample()
