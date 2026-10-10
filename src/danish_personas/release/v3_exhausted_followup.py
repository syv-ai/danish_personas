"""Private offline evidence for exhausted targeted v3 follow-up rows."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import polars as pl

from ..generation.proxy_budget import (
    BASE_URL,
    SOL_ADJUDICATION_LEDGER_MAX_TOKENS,
    V3_TARGETED_FOLLOWUP_PURPOSE,
)
from ..generation.sol_adjudication import (
    SOL_ADJUDICATION_MODEL,
    SOL_ALLOWED_FACT_FIELDS,
    SolAdjudicationResponse,
    _request_id_prefix,
    _validated_payload,
)
from ..io import canonical_json, sha256_file, sha256_text
from .candidate_validation import validate_release_candidate

ROOT = Path("/tmp/danish-personas-audit/v3-private")
DEFAULT_BASE = ROOT / "campaign-long"
DEFAULT_FOLLOWUP = ROOT / "campaign-followup"
DEFAULT_CANDIDATE = ROOT / "demographics-ready.parquet"
DEFAULT_ORIGINAL = Path(
    "/tmp/danish-personas-audit/hf-v2-foreign-hotfix-remote/data/"
    "train-00000-of-00001.parquet"
)
DEFAULT_BUNDLE = Path(
    "/Users/saattrupdan/gitsky/syv-ai/danish_personas/data/processed/6e27b5c08fbeae79"
)
DEFAULT_PROMPT = Path("config/persona-sol-adjudication-da.md")
DEFAULT_LEDGER = Path.home() / ".danish-personas/proxy-v3-targeted-followup.jsonl"
DEFAULT_OUTPUT = ROOT / "editorial-acceptance"
ID_FIELD = "persona_id"
TEXT_FIELD = "persona"
CAMPAIGN = "persona-sol-adjudication-v3-targeted-followup"
ATTEMPTS = 10
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class ExhaustedFollowupError(Exception):
    """Raised when private evidence cannot be safely derived or verified."""


def write_evidence(
    *,
    base_dir: Path = DEFAULT_BASE,
    followup_dir: Path = DEFAULT_FOLLOWUP,
    candidate_path: Path = DEFAULT_CANDIDATE,
    original_path: Path = DEFAULT_ORIGINAL,
    bundle_dir: Path = DEFAULT_BUNDLE,
    prompt_path: Path = DEFAULT_PROMPT,
    ledger_path: Path = DEFAULT_LEDGER,
    output_dir: Path = DEFAULT_OUTPUT,
    run: bool = False,
) -> dict[str, object]:
    """Derive hashed best-effort evidence without model or provider interaction.

    Args:
        base_dir: Completed pinned base campaign directory.
        followup_dir: Incomplete targeted follow-up campaign directory.
        candidate_path: Pinned candidate Parquet.
        original_path: Pinned original Parquet.
        bundle_dir: Prepared source bundle for hard source gates.
        prompt_path: Pinned adjudication prompt.
        ledger_path: Existing durable request ledger, read-only.
        output_dir: New private evidence directory.
        run (optional): Whether to write evidence. Defaults to False.

    Returns:
        Aggregate-only result without identifiers, prose, or provider data.

    Raises:
        ExhaustedFollowupError: If any source or ledger binding is invalid.
    """
    evidence = _derive(
        base_dir=base_dir,
        followup_dir=followup_dir,
        candidate_path=candidate_path,
        original_path=original_path,
        bundle_dir=bundle_dir,
        prompt_path=prompt_path,
        ledger_path=ledger_path,
    )
    if not run:
        return {"dry_run": True, "pending_rows": evidence["row_count"]}
    output_dir = output_dir.resolve()
    if output_dir.exists():
        prior = output_dir / "evidence.json"
        if prior.is_file() and prior.read_bytes() == _json_bytes(evidence):
            return {"dry_run": False, "pending_rows": evidence["row_count"]}
        raise ExhaustedFollowupError("Evidence output already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_dir.mkdir(mode=0o700)
    _write_exclusive(path=output_dir / "evidence.json", content=_json_bytes(evidence))
    return {"dry_run": False, "pending_rows": evidence["row_count"]}


def verify_evidence(
    *,
    evidence_path: Path = DEFAULT_OUTPUT / "evidence.json",
    base_dir: Path = DEFAULT_BASE,
    followup_dir: Path = DEFAULT_FOLLOWUP,
    candidate_path: Path = DEFAULT_CANDIDATE,
    original_path: Path = DEFAULT_ORIGINAL,
    bundle_dir: Path = DEFAULT_BUNDLE,
    prompt_path: Path = DEFAULT_PROMPT,
    ledger_path: Path = DEFAULT_LEDGER,
) -> dict[str, int | str]:
    """Re-derive the evidence bindings and return aggregate-only counts.

    This is the composer-facing API. Success verifies current sources and ledger;
    it is not a model or human review.

    Args:
        evidence_path: Private evidence JSON produced by :func:`write_evidence`.
        base_dir: Completed pinned base campaign directory.
        followup_dir: Incomplete targeted follow-up campaign directory.
        candidate_path: Pinned candidate Parquet.
        original_path: Pinned original Parquet.
        bundle_dir: Prepared source bundle for hard source gates.
        prompt_path: Pinned adjudication prompt.
        ledger_path: Existing durable request ledger, read-only.

    Returns:
        Aggregate decision and attempt counts only.

    Raises:
        ExhaustedFollowupError: If evidence or bindings differ.
    """
    try:
        saved = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExhaustedFollowupError("Evidence document is unreadable") from exc
    current = _derive(
        base_dir=base_dir,
        followup_dir=followup_dir,
        candidate_path=candidate_path,
        original_path=original_path,
        bundle_dir=bundle_dir,
        prompt_path=prompt_path,
        ledger_path=ledger_path,
    )
    if saved != current:
        raise ExhaustedFollowupError("Evidence bindings do not match current inputs")
    rows_value = current.get("rows")
    if not isinstance(rows_value, list):
        raise ExhaustedFollowupError("Evidence rows are malformed")
    rows = rows_value
    return {
        "decision": "best_effort_retained_original",
        "rows": len(rows),
        "reservations": ATTEMPTS * len(rows),
        "observed_responses": sum(row["observed_responses"] for row in rows),
    }


def _derive(
    *,
    base_dir: Path,
    followup_dir: Path,
    candidate_path: Path,
    original_path: Path,
    bundle_dir: Path,
    prompt_path: Path,
    ledger_path: Path,
) -> dict[str, object]:
    paths = {
        "base_manifest": base_dir / "manifest.json",
        "base_status": base_dir / "status.json",
        "follow_manifest": followup_dir / "manifest.json",
        "follow_status": followup_dir / "status.json",
    }
    bm, bs, fm, fs = (_read_json(paths[k]) for k in paths)
    _check_campaigns(
        base_manifest=bm, base_status=bs, follow_manifest=fm, follow_status=fs
    )
    frame, original = pl.read_parquet(candidate_path), pl.read_parquet(original_path)
    ids, ordered = _ordered_hashes(
        frame=frame,
        original=original,
        base_manifest=bm,
        candidate_path=candidate_path,
        original_path=original_path,
    )
    follow_inputs = fm.get("input_hashes")
    expected = {
        "base_manifest_sha256": sha256_file(paths["base_manifest"]),
        "base_status_sha256": sha256_file(paths["base_status"]),
        "candidate_sha256": sha256_file(candidate_path),
        "original_sha256": sha256_file(original_path),
        "ordered_id_hashes_sha256": ordered,
        "prompt_sha256": sha256_file(prompt_path),
    }
    if not isinstance(follow_inputs, dict) or any(
        follow_inputs.get(k) != v for k, v in expected.items()
    ):
        raise ExhaustedFollowupError("Follow-up input hashes differ")
    selected, processed, pending = _coverage(
        base_status=bs, follow_status=fs, ids=ids, follow_inputs=follow_inputs
    )
    _require_gates(
        candidate_path=candidate_path,
        original_path=original_path,
        bundle_dir=bundle_dir,
        row_count=frame.height,
    )
    prompt = prompt_path.read_text(encoding="utf-8")
    schema_hash = sha256_text(
        canonical_json(SolAdjudicationResponse.provider_json_schema())
    )
    header, reservations, usages, ledger_hash = _ledger(ledger_path)
    model_hash = sha256_text(SOL_ADJUDICATION_MODEL)
    _check_ledger_header(
        header=header,
        prompt=prompt,
        schema_hash=schema_hash,
        model_hash=model_hash,
        follow_inputs=follow_inputs,
    )
    rows = _rows(
        pending=pending,
        ids=ids,
        frame=frame,
        original=original,
        selected=selected,
        prompt=prompt,
        schema_hash=schema_hash,
        model_hash=model_hash,
        header=header,
        reservations=reservations,
        usages=usages,
    )
    if set(processed) & set(pending) or set(processed) | set(pending) != set(selected):
        raise ExhaustedFollowupError("Follow-up pending coverage is not one-to-one")
    return {
        "contract": "v3-exhausted-followup-best-effort-v1",
        "decision": "best_effort_retained_original",
        "model_verdict": False,
        "human_certified": False,
        "source_gate_hard_pass": True,
        "generated_hard_failures": 0,
        "inputs": {
            "base_manifest_sha256": sha256_file(paths["base_manifest"]),
            "base_status_sha256": sha256_file(paths["base_status"]),
            "followup_manifest_sha256": sha256_file(paths["follow_manifest"]),
            "followup_status_sha256": sha256_file(paths["follow_status"]),
            "candidate_sha256": sha256_file(candidate_path),
            "original_sha256": sha256_file(original_path),
            "ordered_id_hashes_sha256": ordered,
            "queue_sha256": follow_inputs["queue_sha256"],
            "prepared_bundle_sha256": _bundle_hash(bundle_dir),
            "prompt_sha256": sha256_file(prompt_path),
            "schema_sha256": schema_hash,
            "model_sha256": model_hash,
            "ledger_sha256": ledger_hash,
            "ledger_header_sha256": sha256_text(canonical_json(header)),
        },
        "ledger_pins": {
            key: header[key]
            for key in (
                "campaign",
                "source_hash",
                "prompt_hash",
                "schema_hash",
                "uncapped_purpose",
            )
        },
        "row_count": len(rows),
        "rows": rows,
    }


def _check_campaigns(
    *,
    base_manifest: dict[str, object],
    base_status: dict[str, object],
    follow_manifest: dict[str, object],
    follow_status: dict[str, object],
) -> None:
    progress = base_status.get("progress")
    if (
        base_manifest.get("campaign") != "persona-sol-adjudication-v3-long"
        or base_status.get("campaign") != base_manifest.get("campaign")
        or base_status.get("input_hashes") != base_manifest.get("input_hashes")
        or not isinstance(progress, dict)
        or progress.get("pending") != 0
        or progress.get("completed") != base_manifest.get("selected_total")
    ):
        raise ExhaustedFollowupError("Pinned base campaign is not complete")
    if (
        follow_manifest.get("campaign") != CAMPAIGN
        or follow_status.get("campaign") != CAMPAIGN
        or follow_status.get("input_hashes") != follow_manifest.get("input_hashes")
        or not isinstance(follow_status.get("processed"), list)
    ):
        raise ExhaustedFollowupError("Follow-up campaign binding is invalid")


def _ordered_hashes(
    *,
    frame: pl.DataFrame,
    original: pl.DataFrame,
    base_manifest: dict[str, object],
    candidate_path: Path,
    original_path: Path,
) -> tuple[list[str], str]:
    if ID_FIELD not in frame.columns or TEXT_FIELD not in frame.columns:
        raise ExhaustedFollowupError("Candidate schema is incomplete")
    if frame[ID_FIELD].to_list() != original[ID_FIELD].to_list():
        raise ExhaustedFollowupError("Ordered source identity binding differs")
    ids = [sha256_text(value) for value in frame[ID_FIELD].to_list()]
    if len(set(ids)) != len(ids):
        raise ExhaustedFollowupError("Candidate contains duplicate identities")
    ordered = sha256_text(canonical_json(ids))
    expected = {
        "baseline_sha256": sha256_file(original_path),
        "candidate_sha256": sha256_file(candidate_path),
        "ordered_id_hashes_sha256": ordered,
    }
    base_inputs = base_manifest.get("input_hashes")
    if not isinstance(base_inputs, dict) or any(
        base_inputs.get(k) != v for k, v in expected.items()
    ):
        raise ExhaustedFollowupError("Base campaign source hashes differ")
    return ids, ordered


def _coverage(
    *,
    base_status: dict[str, object],
    follow_status: dict[str, object],
    ids: list[str],
    follow_inputs: dict[str, object],
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]], list[str]]:
    base_items = base_status.get("processed")
    follow_items = follow_status.get("processed")
    if not isinstance(base_items, list) or not isinstance(follow_items, list):
        raise ExhaustedFollowupError("Base campaign records are invalid")
    items = [
        item
        for item in base_items
        if isinstance(item, dict)
        and item.get("disposition")
        in {"unresolved", "validation_failed", "privacy_blocked"}
    ]
    if any(
        item.get("result_sha256")
        != sha256_text(f"{item.get('persona_hash')}:{item.get('disposition')}")
        for item in items
    ):
        raise ExhaustedFollowupError("Base result binding is invalid")
    selected = {item.get("persona_hash"): item for item in items}
    if len(selected) != len(items) or None in selected:
        raise ExhaustedFollowupError("Base campaign selection is not unique")
    processed: dict[str, dict[str, object]] = {}
    for item in follow_items:
        if not isinstance(item, dict) or not isinstance(item.get("persona_hash"), str):
            raise ExhaustedFollowupError("Follow-up processed records are invalid")
        digest = item["persona_hash"]
        if item.get("result_sha256") != sha256_text(
            f"{digest}:{item.get('disposition')}"
        ):
            raise ExhaustedFollowupError("Follow-up result binding is invalid")
        if digest in processed or digest not in selected:
            raise ExhaustedFollowupError("Follow-up coverage overlaps or is unexpected")
        processed[digest] = item
    pending = [digest for digest in selected if digest not in processed]
    progress = follow_status.get("progress")
    if (
        not isinstance(progress, dict)
        or follow_status.get("selected_total") != len(selected)
        or progress.get("total") != len(selected)
        or progress.get("completed") != len(processed)
        or progress.get("pending") != len(pending)
    ):
        raise ExhaustedFollowupError("Follow-up status counters are inconsistent")
    indexes = sorted(ids.index(digest) for digest in selected)
    if follow_inputs.get("queue_sha256") != sha256_text(canonical_json(indexes)):
        raise ExhaustedFollowupError("Follow-up queue binding differs")
    return selected, processed, pending


def _require_gates(
    *, candidate_path: Path, original_path: Path, bundle_dir: Path, row_count: int
) -> None:
    report = validate_release_candidate(
        candidate_path=candidate_path,
        original_path=original_path,
        bundle_dir=bundle_dir,
        expected_row_count=row_count,
    )
    if (
        any(report.source_support.unsupported_rows.values())
        or report.generated_fields.hard_failure_rows
    ):
        raise ExhaustedFollowupError("Candidate mandatory source/generated gate failed")


def _check_ledger_header(
    *,
    header: dict[str, object],
    prompt: str,
    schema_hash: str,
    model_hash: str,
    follow_inputs: dict[str, object],
) -> None:
    if (
        header.get("campaign") != CAMPAIGN
        or header.get("uncapped_purpose") != V3_TARGETED_FOLLOWUP_PURPOSE
        or header.get("source_hash") != sha256_text(canonical_json(follow_inputs))
        or header.get("prompt_hash") != sha256_text(prompt)
        or header.get("schema_hash") != schema_hash
        or header.get("base_url") != BASE_URL
        or header.get("max_tokens") != SOL_ADJUDICATION_LEDGER_MAX_TOKENS
        or header.get("input_usd_per_million") != "0.1"
        or header.get("output_usd_per_million") != "0.5"
        or header.get("uncapped") is not True
        or sha256_text(str(header.get("model", ""))) != model_hash
    ):
        raise ExhaustedFollowupError("Ledger header pins do not match")


def _rows(
    *,
    pending: list[str],
    ids: list[str],
    frame: pl.DataFrame,
    original: pl.DataFrame,
    selected: dict[str, dict[str, object]],
    prompt: str,
    schema_hash: str,
    model_hash: str,
    header: dict[str, object],
    reservations: list[dict[str, object]],
    usages: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current, before = frame.to_dicts(), original.to_dicts()
    for digest in sorted(pending, key=ids.index):
        index = ids.index(digest)
        row, prior = current[index], before[index]
        hints = {
            field: {"old": prior[field], "new": row[field]}
            for field in sorted((set(prior) & set(row)) & SOL_ALLOWED_FACT_FIELDS)
            if prior[field] != row[field]
        }
        payload = _validated_payload(
            original_persona=row[TEXT_FIELD],
            candidate_row=row,
            prompt=prompt,
            changed_fact_hints=hints,
            original_row=prior,
            adjudication_mode="unresolved_followup",
        )
        binding = {
            "checkpoint_version": 1,
            "original_persona_sha256": sha256_text(row[TEXT_FIELD]),
            "candidate_row_sha256": sha256_text(canonical_json(row)),
            "prompt_sha256": sha256_text(prompt),
            "schema_sha256": schema_hash,
            "payload_sha256": sha256_text(canonical_json(payload)),
            "model_sha256": model_hash,
            "base_url_sha256": sha256_text(str(header["base_url"])),
            "source_pin_sha256": str(header["source_hash"]),
        }
        prefix = _request_id_prefix(binding=binding)
        ids_for_row = [f"{prefix}-{n}" for n in range(1, ATTEMPTS + 1)]
        row_reservations = [
            r for r in reservations if r.get("request_id") in ids_for_row
        ]
        suffixes = sorted(
            _suffix(r.get("request_id"), prefix) for r in row_reservations
        )
        if suffixes != list(range(1, ATTEMPTS + 1)):
            raise ExhaustedFollowupError(
                "Pending row reservation sequence is incomplete"
            )
        row_usages = [r for r in usages if r.get("request_id") in ids_for_row]
        usage_by_id = {r.get("request_id"): r for r in row_usages}
        if len(usage_by_id) != len(row_usages) or len(row_usages) not in {9, 10}:
            raise ExhaustedFollowupError("Observed usage count is invalid")
        if any(
            not _SHA.fullmatch(str(r.get("response_sha256", ""))) for r in row_usages
        ):
            raise ExhaustedFollowupError("Usage response hashes are invalid")
        attempts = [
            {
                "suffix": n,
                "reservation_sha256": sha256_text(ids_for_row[n - 1]),
                "usage_request_sha256": sha256_text(ids_for_row[n - 1])
                if ids_for_row[n - 1] in usage_by_id
                else None,
                "response_sha256": usage_by_id[ids_for_row[n - 1]].get(
                    "response_sha256"
                )
                if ids_for_row[n - 1] in usage_by_id
                else None,
            }
            for n in range(1, ATTEMPTS + 1)
        ]
        rows.append(
            {
                "persona_hash": digest,
                "candidate_row_sha256": binding["candidate_row_sha256"],
                "candidate_prose_sha256": binding["original_persona_sha256"],
                "base_result_sha256": selected[digest].get("result_sha256"),
                "followup_binding_sha256": sha256_text(canonical_json(binding)),
                "request_prefix_sha256": sha256_text(prefix),
                "reserved_suffixes": suffixes,
                "observed_responses": len(row_usages),
                "attempts": attempts,
                "advisory_checks": [
                    "unresolved_followup_payload_reconstructed",
                    "source_gate_pass",
                    "generated_hard_failures_zero",
                ],
                "decision": "best_effort_retained_original",
            }
        )
    return rows


def _ledger(
    path: Path,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], str]:
    raw = path.read_bytes()
    try:
        lines = [json.loads(line) for line in raw.splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExhaustedFollowupError("Ledger is malformed") from exc
    if not lines or not isinstance(lines[0], dict) or lines[0].get("type") != "header":
        raise ExhaustedFollowupError("Ledger header is missing")
    records = lines[1:]
    if any(not isinstance(r, dict) for r in records):
        raise ExhaustedFollowupError("Ledger records are malformed")
    return (
        lines[0],
        [r for r in records if r.get("type") == "reservation"],
        [r for r in records if r.get("type") == "usage"],
        hashlib.sha256(raw).hexdigest(),
    )


def _bundle_hash(path: Path) -> str:
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else []
    if not files or not (path / "bundle-manifest.json").is_file():
        raise ExhaustedFollowupError("Prepared bundle manifest is missing")
    records = [
        {"path": p.relative_to(path).as_posix(), "sha256": sha256_file(p)}
        for p in files
    ]
    return sha256_text(canonical_json(records))


def _suffix(value: object, prefix: str) -> int:
    if not isinstance(value, str) or not value.startswith(prefix + "-"):
        return -1
    try:
        return int(value[len(prefix) + 1 :])
    except ValueError:
        return -1


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExhaustedFollowupError("Campaign evidence file is unreadable") from exc
    if not isinstance(value, dict):
        raise ExhaustedFollowupError("Campaign evidence document is malformed")
    return value


def _json_bytes(value: dict[str, object]) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _write_exclusive(*, path: Path, content: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as target:
        target.write(content)
        target.flush()
        os.fsync(target.fileno())
