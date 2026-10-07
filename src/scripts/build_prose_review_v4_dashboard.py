"""Build an offline dashboard for private v4 prose-review checkpoints."""

from __future__ import annotations

import argparse
import difflib
import html
import json
import logging
import os
import re
import tempfile
import typing as t
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from danish_personas.cli_logging import configure_cli_logging
from danish_personas.environment import load_repository_environment
from danish_personas.generation.prose_review import (
    ProseReviewError,
    ProseReviewResponse,
    ProseReviewResult,
    validate_prose_review,
)
from danish_personas.generation.proxy_budget import BASE_URL, MODEL
from danish_personas.generation.proxy_patch_runner import (
    _ALLOWED_FACTS,
    ProxyPatchError,
    _validated_input,
)
from danish_personas.generation.proxy_review_runner import (
    ProxyReviewError,
    _payload_with_null_detail_context,
    _verified_context_from_payload,
)
from danish_personas.io import canonical_json, sha256_file, sha256_text

LOGGER = logging.getLogger(__name__)

DEFAULT_ROOT = Path("/tmp/danish-personas-audit")
DEFAULT_ORIGINAL = DEFAULT_ROOT / "data/train-00000-of-00001.parquet"
DEFAULT_CANDIDATE = DEFAULT_ROOT / "attribute-candidate-v4.parquet"
DEFAULT_TRIAGE = DEFAULT_ROOT / "prose-triage-v1.json"
DEFAULT_PROMPT = Path("config/persona-review-da.md")
DEFAULT_REGISTRY = Path.home() / ".pi" / "agent" / "models-store.json"
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "persona-review-v4"
DEFAULT_STATUS = DEFAULT_OUTPUT_DIR / "status.json"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "manifest.json"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "prose-review-v4-dashboard.html"
DEFAULT_LIMIT = 50
CAMPAIGN = "persona-prose-review-v4"
H90_CAMPAIGN = "persona-prose-review-h90-v5"
ID_FIELD = "persona_id"
PERSONA_FIELD = "persona"
_CHECKPOINT_VERSION = 1
_HASH_RE = re.compile(r"[0-9a-f]{64}")
_FRACTION_BUCKETS = ((0.05, "small"), (0.15, "medium"))
_FRACTION_ORDER = {"small": 0, "medium": 1, "large": 2}

JSONScalar: t.TypeAlias = str | int | float | bool | None
JSONValue: t.TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


def main(argv: list[str] | None = None) -> int:
    """Run the dashboard command-line interface.

    Args:
        argv (optional):
            Command-line arguments. Defaults to ``sys.argv`` when omitted.

    Returns:
        Zero after a successful dashboard build.

    Raises:
        SystemExit:
            If the offline dashboard cannot be verified or written.
    """
    configure_cli_logging()
    parser = _argument_parser()
    args = parser.parse_args(argv)
    limit = None if args.all else args.limit
    paths = DashboardPaths(
        original=args.original,
        candidate=args.candidate,
        triage=args.triage,
        prompt=args.prompt,
        registry=args.registry,
        status=args.status,
        manifest=args.manifest,
        checkpoint_root=args.checkpoint_root,
        output=args.output,
    )
    try:
        summary = build_prose_review_v4_dashboard(paths=paths, limit=limit)
    except ProseReviewV4DashboardError as exc:
        LOGGER.error("Prose review v4 dashboard failed: %s", exc)
        raise SystemExit(1) from exc
    LOGGER.info(
        "Wrote private dashboard with %s patched cards from %s verified checkpoints",
        summary["patched_cards"],
        summary["verified_checkpoints"],
    )
    return 0


@dataclass(frozen=True)
class DashboardPaths:
    """Filesystem inputs for the v4 prose-review dashboard."""

    original: Path = DEFAULT_ORIGINAL
    candidate: Path = DEFAULT_CANDIDATE
    triage: Path = DEFAULT_TRIAGE
    prompt: Path = DEFAULT_PROMPT
    registry: Path = DEFAULT_REGISTRY
    status: Path = DEFAULT_STATUS
    manifest: Path = DEFAULT_MANIFEST
    checkpoint_root: Path = DEFAULT_OUTPUT_DIR
    output: Path = DEFAULT_OUTPUT


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an offline HTML dashboard for v4 prose-review checkpoints."
    )
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--triage", type=Path, default=DEFAULT_TRIAGE)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum patched proposal cards to include. Defaults to 50.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Include every locally verified patched proposal card.",
    )
    return parser


def build_prose_review_v4_dashboard(
    *, paths: DashboardPaths, limit: int | None = DEFAULT_LIMIT
) -> dict[str, JSONValue]:
    """Write a self-contained offline v4 prose-review dashboard.

    Args:
        paths:
            Input parquet, manifest, status, checkpoint, and output paths.
        limit (optional):
            Maximum patched proposal cards to render. ``None`` means all patched
            proposals. Defaults to 50.

    Returns:
        JSON-compatible build summary without private persona IDs.

    Raises:
        ProseReviewV4DashboardError:
            If sources, bindings, local validation, or file permissions are unsafe.
    """
    if limit is not None and limit < 1:
        raise ProseReviewV4DashboardError("Dashboard limit must be at least one")
    manifest = _load_json_object(path=paths.manifest, label="manifest.json")
    status = _load_json_object(path=paths.status, label="status.json")
    _verify_status_manifest(status=status, manifest=manifest)
    prompt = _verify_manifest_sources(manifest=manifest, paths=paths)
    hashes = _status_hashes(status=status)
    _verify_status_counts(status=status, hashes=hashes)
    original_rows = _rows_by_hash(path=paths.original, label="original")
    candidate_rows = _rows_by_hash(path=paths.candidate, label="candidate")
    decisions = _load_verified_decisions(
        hashes=hashes,
        original_rows=original_rows,
        candidate_rows=candidate_rows,
        prompt=prompt,
        manifest=manifest,
        checkpoint_root=paths.checkpoint_root,
    )
    _verify_decision_counts(status=status, decisions=decisions)
    patched = [decision for decision in decisions if decision.disposition == "patched"]
    cards = _select_patched_cards(decisions=patched, limit=limit)
    document = render_dashboard(
        status=status, manifest=manifest, decisions=decisions, cards=cards, limit=limit
    )
    _write_private_html(path=paths.output, content=document)
    return {
        "output": str(paths.output),
        "verified_checkpoints": len(decisions),
        "patched": len(patched),
        "patched_cards": len(cards),
        "unchanged_consistent": sum(
            decision.disposition == "unchanged_consistent" for decision in decisions
        ),
        "needs_manual_review": sum(
            decision.disposition == "needs_manual_review" for decision in decisions
        ),
    }


class ProseReviewV4DashboardError(RuntimeError):
    """Raised when the v4 dashboard cannot be built safely."""


def _load_json_object(*, path: Path, label: str) -> dict[str, JSONValue]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProseReviewV4DashboardError(f"{label} is missing") from exc
    except json.JSONDecodeError as exc:
        raise ProseReviewV4DashboardError(f"{label} is not valid JSON") from exc
    if not isinstance(document, dict):
        raise ProseReviewV4DashboardError(f"{label} must be a JSON object")
    return t.cast(dict[str, JSONValue], document)


def _checkpoint_path(*, checkpoint_root: Path, persona_hash: str) -> Path:
    roots = (checkpoint_root, checkpoint_root / "checkpoints")
    if checkpoint_root.name == "checkpoints":
        roots = (checkpoint_root,)
    candidates = tuple(
        root / persona_hash[:2] / f"{persona_hash}.json" for root in roots
    )
    for path in candidates:
        if path.exists():
            return path
    raise ProseReviewV4DashboardError("Processed checkpoint file is missing")


def _load_complete_checkpoint(*, path: Path) -> dict[str, JSONValue]:
    if path.suffix != ".json" or not path.is_file():
        raise ProseReviewV4DashboardError("Checkpoint path is not a complete JSON file")
    if path.stat().st_mode & 0o777 != 0o600:
        raise ProseReviewV4DashboardError("Checkpoint file must be private (mode 0600)")
    document = _load_json_object(path=path, label="checkpoint")
    return document


def _changed_facts_mapping(
    *, original: dict[str, JSONValue], candidate: dict[str, JSONValue]
) -> dict[str, dict[str, object]]:
    facts: dict[str, dict[str, object]] = {}
    for field in sorted((set(original) & set(candidate)) & _ALLOWED_FACTS):
        old = original[field]
        new = candidate[field]
        if old != new:
            facts[field] = {"old": old, "new": new}
    return facts


def _checkpoint_binding(
    *,
    original: dict[str, JSONValue],
    candidate: dict[str, JSONValue],
    changed_facts: dict[str, dict[str, object]],
    original_text: str,
    prompt: str,
    manifest: dict[str, JSONValue],
    payload: dict[str, object],
    campaign: str = CAMPAIGN,
) -> dict[str, str | int]:
    schema = ProseReviewResponse.provider_json_schema()
    binding: dict[str, str | int] = {
        "checkpoint_version": _CHECKPOINT_VERSION,
        "original_text_sha256": sha256_text(original_text),
        "row_sha256": sha256_text(canonical_json(original)),
        "candidate_row_sha256": sha256_text(canonical_json(candidate)),
        "changed_facts_sha256": sha256_text(canonical_json(changed_facts)),
        "prompt_sha256": sha256_text(prompt),
        "schema_sha256": sha256_text(canonical_json(schema)),
        "campaign_sha256": sha256_text(campaign),
        "model_sha256": sha256_text(MODEL),
        "base_url_sha256": sha256_text(BASE_URL),
        "source_pin_sha256": sha256_text(canonical_json(manifest)),
    }
    if "legal_status_detail_null_context" in payload:
        binding["payload_sha256"] = sha256_text(canonical_json(payload))
    return binding


def _checkpoint_response(*, checkpoint: dict[str, JSONValue]) -> dict[str, object]:
    return {
        "disposition": checkpoint.get("disposition"),
        "patches": checkpoint.get("patches"),
        "unchanged_evidence": checkpoint.get("unchanged_evidence"),
        "manual_review_reason": checkpoint.get("manual_review_reason"),
    }


def _checkpoint_uses_legacy_contextless_binding(
    *,
    checkpoint: dict[str, JSONValue],
    binding: dict[str, str | int],
    changed_facts: dict[str, dict[str, object]],
    original_text: str,
    pre_context_payload: dict[str, object],
    context: dict[str, object] | None,
) -> bool:
    if checkpoint.get("payload_sha256") is not None:
        return False
    if "payload_sha256" not in binding:
        return False
    if context is None:
        return False
    detail_change = changed_facts.get("legal_status_detail")
    if (
        detail_change is None
        or not isinstance(detail_change.get("old"), str)
        or detail_change.get("new") is not None
    ):
        return False
    expected_payload = {"persona": original_text, "changed_facts": changed_facts}
    if pre_context_payload != expected_payload:
        raise ProseReviewV4DashboardError("Legacy checkpoint payload is malformed")
    legacy_binding = _legacy_contextless_binding(binding=binding)
    expected_keys = _checkpoint_expected_keys(binding=legacy_binding)
    if set(checkpoint) != expected_keys:
        return False
    return True


def _checkpoint_expected_keys(*, binding: dict[str, str | int]) -> set[str]:
    return set(binding) | {
        "disposition",
        "changed_fraction",
        "proposed_text_sha256",
        "patches",
        "unchanged_evidence",
        "manual_review_reason",
        "unchanged_consistent_note",
        "checkpoint_sha256",
    }


def _legacy_contextless_binding(
    *, binding: dict[str, str | int]
) -> dict[str, str | int]:
    legacy_binding = dict(binding)
    legacy_binding.pop("payload_sha256", None)
    return legacy_binding


@dataclass(frozen=True)
class ReviewDecision:
    """One locally verified v4 checkpoint decision."""

    persona_hash: str
    short_id: str
    disposition: str
    changed_fields: tuple[str, ...]
    changed_facts: tuple[tuple[str, str, str], ...]
    changed_fraction: float
    original_text: str
    proposed_text: str
    patches: tuple[tuple[str, str], ...]
    unchanged_evidence: tuple[tuple[str, str, str], ...]
    manual_review_reason: str | None
    unchanged_consistent_note: str | None
    legacy_provisional: bool = False


def _load_verified_decisions(
    *,
    hashes: list[str],
    original_rows: dict[str, dict[str, JSONValue]],
    candidate_rows: dict[str, dict[str, JSONValue]],
    prompt: str,
    manifest: dict[str, JSONValue],
    checkpoint_root: Path,
) -> list[ReviewDecision]:
    decisions: list[ReviewDecision] = []
    for persona_hash in sorted(hashes):
        checkpoint_path = _checkpoint_path(
            checkpoint_root=checkpoint_root, persona_hash=persona_hash
        )
        checkpoint = _load_complete_checkpoint(path=checkpoint_path)
        original = original_rows.get(persona_hash)
        candidate = candidate_rows.get(persona_hash)
        if original is None:
            raise ProseReviewV4DashboardError("Checkpoint original row is missing")
        if candidate is None:
            raise ProseReviewV4DashboardError("Checkpoint candidate row is missing")
        decisions.append(
            _verify_checkpoint_decision(
                checkpoint=checkpoint,
                persona_hash=persona_hash,
                original=original,
                candidate=candidate,
                prompt=prompt,
                manifest=manifest,
            )
        )
    return decisions


def _verify_checkpoint_decision(
    *,
    checkpoint: dict[str, JSONValue],
    persona_hash: str,
    original: dict[str, JSONValue],
    candidate: dict[str, JSONValue],
    prompt: str,
    manifest: dict[str, JSONValue],
    campaign: str = CAMPAIGN,
) -> ReviewDecision:
    changed_facts = _changed_facts_mapping(original=original, candidate=candidate)
    if not changed_facts:
        raise ProseReviewV4DashboardError("Checkpoint has no allowlisted fact change")
    original_text = _row_text(row=original, label="original")
    try:
        _, pre_context_payload = _validated_input(
            original, candidate, changed_facts, None, None, prompt
        )
        payload = _payload_with_null_detail_context(
            payload=pre_context_payload,
            row=original,
            candidate_row=candidate,
            changed_facts=changed_facts,
        )
        context = _verified_context_from_payload(payload=payload)
    except (ProxyPatchError, ProxyReviewError) as exc:
        raise ProseReviewV4DashboardError(
            "Checkpoint inputs fail local safety checks"
        ) from exc
    binding = _checkpoint_binding(
        original=original,
        candidate=candidate,
        changed_facts=changed_facts,
        original_text=original_text,
        prompt=prompt,
        manifest=manifest,
        payload=payload,
        campaign=campaign,
    )
    legacy_provisional = _checkpoint_uses_legacy_contextless_binding(
        checkpoint=checkpoint,
        binding=binding,
        changed_facts=changed_facts,
        original_text=original_text,
        pre_context_payload=pre_context_payload,
        context=context,
    )
    validation_context = None if legacy_provisional else context
    if legacy_provisional:
        binding = _legacy_contextless_binding(binding=binding)
    _verify_checkpoint_keys(checkpoint=checkpoint, binding=binding)
    _verify_checkpoint_digest(checkpoint=checkpoint)
    for key, value in binding.items():
        if checkpoint.get(key) != value:
            raise ProseReviewV4DashboardError(f"Checkpoint binding mismatch: {key}")
    response = _checkpoint_response(checkpoint=checkpoint)
    try:
        result = validate_prose_review(
            original_text=original_text,
            changed_facts=changed_facts,
            response=response,
            verified_context=validation_context,
        )
    except ProseReviewError as exc:
        raise ProseReviewV4DashboardError(
            "Checkpoint decision fails local validation"
        ) from exc
    _verify_checkpoint_result(checkpoint=checkpoint, result=result)
    return _decision_from_result(
        persona_hash=persona_hash,
        changed_facts=changed_facts,
        result=result,
        legacy_provisional=legacy_provisional,
    )


def _decision_from_result(
    *,
    persona_hash: str,
    changed_facts: dict[str, dict[str, object]],
    result: ProseReviewResult,
    legacy_provisional: bool = False,
) -> ReviewDecision:
    facts = tuple(
        (field, _display_value(values["old"]), _display_value(values["new"]))
        for field, values in sorted(changed_facts.items())
    )
    return ReviewDecision(
        persona_hash=persona_hash,
        short_id=persona_hash[:12],
        disposition=result.disposition,
        changed_fields=tuple(field for field, _, _ in facts),
        changed_facts=facts,
        changed_fraction=result.changed_fraction,
        original_text=result.original_text,
        proposed_text=result.proposed_text,
        patches=tuple(
            (patch.old_excerpt, patch.new_excerpt) for patch in result.patches
        ),
        unchanged_evidence=tuple(
            (item.field, item.kind, item.quote) for item in result.unchanged_evidence
        ),
        manual_review_reason=result.manual_review_reason,
        unchanged_consistent_note=result.unchanged_consistent_note,
        legacy_provisional=legacy_provisional,
    )


def _display_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    try:
        return canonical_json(value)
    except TypeError:
        return str(value)


def _row_text(*, row: dict[str, JSONValue], label: str) -> str:
    value = row.get(PERSONA_FIELD)
    if not isinstance(value, str):
        raise ProseReviewV4DashboardError(f"{label} row has invalid prose")
    return value


def _verify_checkpoint_digest(*, checkpoint: dict[str, JSONValue]) -> None:
    digest = checkpoint.get("checkpoint_sha256")
    unsigned = {
        key: value for key, value in checkpoint.items() if key != "checkpoint_sha256"
    }
    if digest != sha256_text(canonical_json(unsigned)):
        raise ProseReviewV4DashboardError("Checkpoint checksum is invalid")


def _verify_checkpoint_keys(
    *, checkpoint: dict[str, JSONValue], binding: dict[str, str | int]
) -> None:
    if set(checkpoint) != _checkpoint_expected_keys(binding=binding):
        raise ProseReviewV4DashboardError("Checkpoint schema is malformed")


def _verify_checkpoint_result(
    *, checkpoint: dict[str, JSONValue], result: ProseReviewResult
) -> None:
    if checkpoint.get("disposition") != result.disposition:
        raise ProseReviewV4DashboardError("Checkpoint disposition changed")
    if checkpoint.get("changed_fraction") != result.changed_fraction:
        raise ProseReviewV4DashboardError("Checkpoint changed fraction changed")
    if checkpoint.get("proposed_text_sha256") != sha256_text(result.proposed_text):
        raise ProseReviewV4DashboardError("Checkpoint proposed text hash changed")
    if checkpoint.get("manual_review_reason") != result.manual_review_reason:
        raise ProseReviewV4DashboardError("Checkpoint manual reason changed")
    if checkpoint.get("unchanged_consistent_note") != result.unchanged_consistent_note:
        raise ProseReviewV4DashboardError("Checkpoint unchanged note changed")


def _rows_by_hash(*, path: Path, label: str) -> dict[str, dict[str, JSONValue]]:
    try:
        frame = pl.read_parquet(path)
    except Exception as exc:
        raise ProseReviewV4DashboardError(f"{label} parquet is not readable") from exc
    missing = {ID_FIELD, PERSONA_FIELD} - set(frame.columns)
    if missing:
        raise ProseReviewV4DashboardError(f"{label} parquet lacks required columns")
    rows: dict[str, dict[str, JSONValue]] = {}
    for row in frame.to_dicts():
        persona_id = row.get(ID_FIELD)
        persona = row.get(PERSONA_FIELD)
        if not isinstance(persona_id, str) or not persona_id:
            raise ProseReviewV4DashboardError(f"{label} parquet has invalid IDs")
        if not isinstance(persona, str):
            raise ProseReviewV4DashboardError(f"{label} parquet has invalid prose")
        persona_hash = sha256_text(persona_id)
        if persona_hash in rows:
            raise ProseReviewV4DashboardError(f"{label} parquet has duplicate hashes")
        rows[persona_hash] = t.cast(dict[str, JSONValue], row)
    return rows


def _select_patched_cards(
    *, decisions: list[ReviewDecision], limit: int | None
) -> list[ReviewDecision]:
    ordered = sorted(
        decisions,
        key=lambda item: (
            item.changed_fields,
            _fraction_bucket_index(item.changed_fraction),
            item.persona_hash,
        ),
    )
    if limit is None:
        return ordered
    buckets: dict[tuple[str, str], list[ReviewDecision]] = {}
    for decision in ordered:
        key = (
            ",".join(decision.changed_fields),
            _fraction_bucket(decision.changed_fraction),
        )
        buckets.setdefault(key, []).append(decision)
    selected: list[ReviewDecision] = []
    for key in sorted(buckets, key=lambda item: (item[0], _FRACTION_ORDER[item[1]])):
        if len(selected) >= limit:
            break
        selected.append(buckets[key].pop(0))
    ordered_keys = sorted(buckets, key=lambda item: (item[0], _FRACTION_ORDER[item[1]]))
    while len(selected) < limit and any(buckets.values()):
        for key in ordered_keys:
            bucket = buckets[key]
            if bucket:
                selected.append(bucket.pop(0))
            if len(selected) >= limit:
                break
    return selected


def _fraction_bucket(value: float) -> str:
    for boundary, label in _FRACTION_BUCKETS:
        if value < boundary:
            return label
    return "large"


def _fraction_bucket_index(value: float) -> int:
    return _FRACTION_ORDER[_fraction_bucket(value)]


def _status_hashes(*, status: dict[str, JSONValue]) -> list[str]:
    value = status.get("processed_persona_hashes")
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ProseReviewV4DashboardError("status.json processed hashes are malformed")
    hashes = list(value)
    invalid_hash = any(_HASH_RE.fullmatch(item) is None for item in hashes)
    if len(hashes) != len(set(hashes)) or invalid_hash:
        raise ProseReviewV4DashboardError("status.json processed hashes are invalid")
    return hashes


def _verify_decision_counts(
    *, status: dict[str, JSONValue], decisions: list[ReviewDecision]
) -> None:
    counts = Counter(decision.disposition for decision in decisions)
    for key in ("patched", "unchanged_consistent", "needs_manual_review"):
        if _status_int(status=status, key=key) != counts[key]:
            raise ProseReviewV4DashboardError(
                "status.json disposition counts are stale"
            )


def _status_int(*, status: dict[str, JSONValue], key: str) -> int:
    value = status.get(key)
    if not isinstance(value, int) or value < 0:
        raise ProseReviewV4DashboardError(f"status.json {key} is invalid")
    return value


def _verify_manifest_sources(
    *,
    manifest: dict[str, JSONValue],
    paths: DashboardPaths,
    expected_campaign: str = CAMPAIGN,
    candidate_key: str = "candidate_v4",
) -> str:
    if manifest.get("version") != 1 or manifest.get("campaign") != expected_campaign:
        raise ProseReviewV4DashboardError(
            "manifest.json is not the expected review campaign"
        )
    if manifest.get("model") != MODEL or manifest.get("base_url") != BASE_URL:
        raise ProseReviewV4DashboardError("manifest.json model or base URL changed")
    if manifest.get("allowed_facts") != sorted(_ALLOWED_FACTS):
        raise ProseReviewV4DashboardError("manifest.json allowlist changed")
    inputs = _manifest_inputs(manifest=manifest)
    expected_schema = sha256_text(
        canonical_json(ProseReviewResponse.provider_json_schema())
    )
    original_key = "original" if candidate_key == "candidate_v4" else "original_v4"
    _verify_input_hash(inputs=inputs, key=original_key, path=paths.original)
    _verify_input_hash(inputs=inputs, key=candidate_key, path=paths.candidate)
    if candidate_key == "candidate_h90_v5":
        _verify_input_hash(
            inputs=inputs,
            key="h90_report",
            path=_h90_report_path(candidate=paths.candidate),
        )
        _verify_h90_report_candidate(candidate=paths.candidate)
    _verify_input_hash(inputs=inputs, key="triage", path=paths.triage)
    _verify_input_hash(inputs=inputs, key="prompt", path=paths.prompt)
    _verify_input_hash(inputs=inputs, key="registry", path=paths.registry)
    if inputs.get("schema") != expected_schema:
        raise ProseReviewV4DashboardError("manifest.json schema hash changed")
    try:
        return paths.prompt.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProseReviewV4DashboardError("Prompt file is not readable") from exc


def _h90_report_path(*, candidate: Path) -> Path:
    return candidate.with_suffix(".report.json")


def _manifest_inputs(*, manifest: dict[str, JSONValue]) -> dict[str, str]:
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ProseReviewV4DashboardError("manifest.json input hashes are missing")
    parsed: dict[str, str] = {}
    for key, value in inputs.items():
        if isinstance(key, str) and isinstance(value, str):
            parsed[key] = value
    return parsed


def _verify_h90_report_candidate(*, candidate: Path) -> None:
    try:
        document = json.loads(
            _h90_report_path(candidate=candidate).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ProseReviewV4DashboardError("H90 report JSON is not readable") from exc
    if not isinstance(document, dict):
        raise ProseReviewV4DashboardError("H90 report JSON must be an object")
    for key in (
        "candidate_sha256",
        "candidate_v5_sha256",
        "candidate_h90_v5_sha256",
        "output_sha256",
        "parquet_sha256",
    ):
        value = document.get(key)
        if not isinstance(value, str):
            continue
        if _HASH_RE.fullmatch(value) is None:
            raise ProseReviewV4DashboardError("H90 report candidate SHA-256 malformed")
        if value != sha256_file(candidate):
            raise ProseReviewV4DashboardError("H90 report candidate SHA-256 mismatch")
        return
    raise ProseReviewV4DashboardError("H90 report lacks candidate SHA-256")


def _verify_input_hash(*, inputs: dict[str, str], key: str, path: Path) -> None:
    expected = inputs.get(key)
    if expected is None:
        raise ProseReviewV4DashboardError(f"manifest.json lacks {key} hash")
    try:
        actual = sha256_file(path)
    except OSError as exc:
        raise ProseReviewV4DashboardError(
            f"Manifest input is not readable: {key}"
        ) from exc
    if actual != expected:
        raise ProseReviewV4DashboardError(f"Manifest input hash mismatch: {key}")


def _verify_status_counts(*, status: dict[str, JSONValue], hashes: list[str]) -> None:
    processed = _status_int(status=status, key="processed")
    if processed != len(hashes):
        raise ProseReviewV4DashboardError("status.json processed hashes are incomplete")
    for key in (
        "reviewable",
        "patched",
        "unchanged_consistent",
        "needs_manual_review",
        "failed",
        "attempted",
        "pending",
    ):
        _status_int(status=status, key=key)


def _verify_status_manifest(
    *, status: dict[str, JSONValue], manifest: dict[str, JSONValue]
) -> None:
    if status.get("manifest") != manifest:
        raise ProseReviewV4DashboardError("status.json does not match manifest.json")


def _write_private_html(*, path: Path, content: str) -> None:
    if not path.parent.exists():
        raise ProseReviewV4DashboardError("Dashboard output directory must exist")
    if not path.parent.is_dir() or path.parent.stat().st_mode & 0o077:
        raise ProseReviewV4DashboardError("Dashboard output directory must be private")
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


def render_dashboard(
    *,
    status: dict[str, JSONValue],
    manifest: dict[str, JSONValue],
    decisions: list[ReviewDecision],
    cards: list[ReviewDecision],
    limit: int | None,
) -> str:
    """Render the complete offline dashboard document.

    Args:
        status:
            Private campaign status document.
        manifest:
            Private campaign manifest document.
        decisions:
            Locally verified checkpoint decisions.
        cards:
            Patched decisions selected for full proposal-card review.
        limit:
            Configured patched-card limit, or ``None`` for all.

    Returns:
        Self-contained HTML with no scripts, network references, or sidecars.
    """
    disposition_counts = Counter(decision.disposition for decision in decisions)
    processed = _status_int(status=status, key="processed")
    pending = _status_int(status=status, key="pending")
    failed = _status_int(status=status, key="failed")
    attempted = _status_int(status=status, key="attempted")
    limit_text = "all patched proposals" if limit is None else f"limit {limit}"
    card_html = "\n".join(
        _render_patched_card(index=index, decision=decision)
        for index, decision in enumerate(cards, start=1)
    )
    if not cards:
        card_html = (
            '<p class="empty">No locally verified patched checkpoints are available '
            "within this dashboard selection.</p>"
        )
    csp = (
        "default-src 'none'; style-src 'unsafe-inline'; img-src 'none'; "
        "script-src 'none'; base-uri 'none'; form-action 'none'; "
        "frame-ancestors 'none'"
    )
    styles = _styles()
    unchanged = _render_unchanged_summary(decisions=decisions)
    manual = _render_manual_summary(decisions=decisions)
    campaign = html.escape(_manifest_text(manifest=manifest, key="campaign"))
    legacy_count = sum(decision.legacy_provisional for decision in decisions)
    legacy_notice = _render_legacy_notice(count=legacy_count)
    notice = (
        "<strong>NOT accepted repairs.</strong> This private offline dashboard "
        "shows locally validated checkpoint proposals for human audit only. It "
        "does not approve dataset changes, publishing, or data-card edits."
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="{csp}">
<title>Private v4 prose-review dashboard</title>
<style>
{styles}
</style>
</head>
<body>
<main>
<h1>Private v4 prose-review dashboard</h1>
<p class="notice">{notice}</p>
<section class="counts" aria-label="Run counts">
<div class="count"><strong>{processed}</strong> processed</div>
<div class="count"><strong>{pending}</strong> pending</div>
<div class="count"><strong>{failed}</strong> failed</div>
<div class="count"><strong>{attempted}</strong> attempted</div>
<div class="count"><strong>{disposition_counts["patched"]}</strong> patched</div>
<div class="count"><strong>{len(cards)}</strong> cards, {html.escape(limit_text)}</div>
</section>
<section class="summary" aria-label="Campaign binding">
<h2>Verified local bindings</h2>
<p>Campaign: <code>{campaign}</code>. Each rendered checkpoint passed local schema,
source-row, candidate-row, prompt, schema, model, base URL, source-pin, checksum,
patch-application, and proposed-text hash validation. Raw persona IDs are not shown.</p>
{legacy_notice}
</section>
{unchanged}
{manual}
<section aria-label="Patched proposal cards">
<h2>Patched proposal cards</h2>
{card_html}
</section>
</main>
</body>
</html>
"""


def _manifest_text(*, manifest: dict[str, JSONValue], key: str) -> str:
    value = manifest.get(key)
    return value if isinstance(value, str) else ""


def _render_legacy_notice(*, count: int) -> str:
    if count == 0:
        return ""
    return (
        '<p class="warning"><strong>LEGACY PROVISIONAL:</strong> '
        f"{count} contextless historical checkpoint(s) were verified against the "
        "signed pre-context persona and changed-facts binding, then revalidated "
        "without source-verified context. They require renewed review before any "
        "dataset change.</p>"
    )


def _render_manual_summary(*, decisions: list[ReviewDecision]) -> str:
    manual = [
        decision
        for decision in decisions
        if decision.disposition == "needs_manual_review"
    ]
    reason_counts = Counter(
        decision.manual_review_reason or "unknown" for decision in manual
    )
    rows = "".join(
        f"<tr><td>{html.escape(reason)}</td><td>{count}</td></tr>"
        for reason, count in sorted(reason_counts.items())
    )
    if not rows:
        rows = '<tr><td colspan="2">No needs-manual-review decisions.</td></tr>'
    return f"""<section class="summary" aria-label="Needs-manual-review summary">
<h2>Needs-manual-review aggregate summary</h2>
<table><thead><tr><th>Reason</th><th>Count</th></tr></thead>
<tbody>{rows}</tbody></table>
</section>"""


def _render_patched_card(*, index: int, decision: ReviewDecision) -> str:
    facts = "".join(
        "<tr>"
        f"<td>{html.escape(field)}</td>"
        f"<td>{html.escape(old)}</td>"
        f"<td>{html.escape(new)}</td>"
        "</tr>"
        for field, old, new in decision.changed_facts
    )
    patches = "".join(
        "<li>"
        f"<strong>Old:</strong> {html.escape(old)}<br>"
        f"<strong>New:</strong> {html.escape(new)}"
        "</li>"
        for old, new in decision.patches
    )
    exact_diff = html.escape(
        "\n".join(
            difflib.unified_diff(
                decision.original_text.splitlines(),
                decision.proposed_text.splitlines(),
                fromfile="original-mistral",
                tofile="locally-validated-proposal",
                lineterm="",
            )
        )
    )
    legacy_label = ""
    if decision.legacy_provisional:
        legacy_label = (
            '<p class="warning"><strong>LEGACY PROVISIONAL:</strong> '
            "contextless historical checkpoint requiring renewed review.</p>"
        )
    return f"""<article class="card">
<header>
<h3>Patched sample {index}: {html.escape(decision.short_id)}</h3>
<p class="meta">Fields: {html.escape(", ".join(decision.changed_fields))}; changed
fraction: {decision.changed_fraction:.1%}</p>
{legacy_label}
</header>
<section aria-label="Allowlisted changed facts">
<h4>Allowlisted source/candidate fact changes</h4>
<table><thead><tr><th>Field</th><th>Original</th><th>Candidate v4</th></tr></thead>
<tbody>{facts}</tbody></table>
</section>
<section aria-label="Patch excerpts">
<h4>Locally validated exact patch excerpts</h4>
<ol>{patches}</ol>
</section>
<section class="compare" aria-label="Original and proposed prose">
<div class="panel"><h4>Original Mistral prose</h4>
<p class="prose">{html.escape(decision.original_text)}</p></div>
<div class="panel"><h4>Locally validated proposed prose</h4>
<p class="prose">{html.escape(decision.proposed_text)}</p></div>
</section>
<section aria-label="Exact unified diff">
<h4>Exact unified diff</h4>
<pre>{exact_diff}</pre>
</section>
</article>"""


def _render_unchanged_summary(*, decisions: list[ReviewDecision]) -> str:
    unchanged = [
        decision
        for decision in decisions
        if decision.disposition == "unchanged_consistent"
    ]
    evidence_counts: Counter[tuple[str, str]] = Counter()
    for decision in unchanged:
        for field, kind, _quote in decision.unchanged_evidence:
            evidence_counts[(field, kind)] += 1
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(field)}</td>"
        f"<td>{html.escape(kind)}</td>"
        f"<td>{count}</td>"
        "</tr>"
        for (field, kind), count in sorted(evidence_counts.items())
    )
    if not rows:
        rows = (
            '<tr><td colspan="3">No provisional unchanged-consistent '
            "decisions.</td></tr>"
        )
    return f"""<section class="summary" aria-label="Unchanged-consistent summary">
<h2>Provisional unchanged-consistent summary</h2>
<p>These counts are classifier outputs only, not independently certified repairs.</p>
<table><thead><tr><th>Field</th><th>Evidence kind</th><th>Count</th></tr></thead>
<tbody>{rows}</tbody></table>
</section>"""


def _styles() -> str:
    return """
:root { color-scheme: light; font-family: system-ui, sans-serif; }
body { margin: 0; background: #f8fafc; color: #172033; }
main { max-width: 1180px; margin: 0 auto; padding: 2rem; }
h1 { margin-bottom: 0.4rem; }
.notice { border: 2px solid #9a3412; background: #fff7ed; padding: 1rem; }
.warning { border-left: 4px solid #b45309; background: #fffbeb; padding: 0.75rem; }
.counts {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr));
  gap: 0.75rem;
  margin: 1.5rem 0;
}
.count, .summary, .card {
  background: white;
  border: 1px solid #cbd5e1;
  border-radius: 0.6rem;
  padding: 1rem;
  margin: 1rem 0;
}
.count strong { display: block; font-size: 1.6rem; }
.card header { background: #e2e8f0; margin: -1rem -1rem 1rem; padding: 1rem; }
.meta { color: #334155; }
.compare { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }
.panel { border: 1px solid #cbd5e1; border-radius: 0.5rem; padding: 0.8rem; }
.prose { white-space: pre-wrap; line-height: 1.55; }
pre { white-space: pre-wrap; background: #0f172a; color: #e2e8f0; padding: 1rem; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #cbd5e1; padding: 0.45rem; text-align: left; }
.empty { background: white; border: 1px dashed #64748b; padding: 1rem; }
@media (max-width: 760px) { .compare { grid-template-columns: 1fr; } }
""".strip()


if __name__ == "__main__":
    load_repository_environment()
    main()
