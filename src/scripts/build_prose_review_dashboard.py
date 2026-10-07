"""Build an offline review dashboard for provisional prose repair proposals."""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import tempfile
import typing as t
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from danish_personas.cli_logging import configure_cli_logging
from danish_personas.generation.prose_patch import (
    ProsePatchError,
    ProsePatchResponse,
    apply_patches,
)
from danish_personas.generation.proxy_patch_runner import _ALLOWED_FACTS
from danish_personas.io import canonical_json, sha256_file, sha256_text

LOGGER = logging.getLogger(__name__)

DEFAULT_ROOT = Path("/tmp/danish-personas-audit")
DEFAULT_ORIGINAL = DEFAULT_ROOT / "data/train-00000-of-00001.parquet"
DEFAULT_CANDIDATE = DEFAULT_ROOT / "attribute-candidate-v2.parquet"
DEFAULT_PROPOSALS_DIR = DEFAULT_ROOT / "persona-repair-proposals"
DEFAULT_STATUS = DEFAULT_PROPOSALS_DIR / "status.json"
DEFAULT_OUTPUT = DEFAULT_ROOT / "prose-review-dashboard.html"
DEFAULT_SAMPLE_LIMIT = 30
ID_FIELD = "persona_id"
PERSONA_FIELD = "persona"
FRACTION_BUCKETS = ((0.05, "small"), (0.15, "medium"))
FRACTION_BUCKET_ORDER = {"small": 0, "medium": 1, "large": 2}

JSONScalar: t.TypeAlias = str | int | float | bool | None
JSONValue: t.TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


class ReviewDashboardError(RuntimeError):
    """Raised when the dashboard cannot be built safely."""


@dataclass(frozen=True)
class DashboardPaths:
    """Filesystem inputs for the offline prose review dashboard."""

    original: Path = DEFAULT_ORIGINAL
    candidate: Path = DEFAULT_CANDIDATE
    status: Path = DEFAULT_STATUS
    checkpoint_root: Path = DEFAULT_PROPOSALS_DIR
    output: Path = DEFAULT_OUTPUT


@dataclass(frozen=True)
class ReviewProposal:
    """One verified provisional prose repair for human review."""

    persona_hash: str
    short_id: str
    changed_fields: tuple[str, ...]
    changed_facts: tuple[tuple[str, str, str], ...]
    changed_fraction: float
    original_text: str
    proposed_text: str
    evidence: tuple[dict[str, str], ...]


def main(argv: list[str] | None = None) -> int:
    """Run the dashboard command-line interface.

    Args:
        argv (optional):
            Command-line arguments. Defaults to ``sys.argv`` when omitted.

    Returns:
        Process exit code.
    """
    configure_cli_logging()
    parser = _argument_parser()
    args = parser.parse_args(argv)
    paths = DashboardPaths(
        original=args.original,
        candidate=args.candidate,
        status=args.status,
        checkpoint_root=args.checkpoint_root,
        output=args.output,
    )
    try:
        summary = build_prose_review_dashboard(
            paths=paths,
            sample_limit=args.sample_limit,
            expected_original_sha256=args.expected_original_sha256,
        )
    except ReviewDashboardError as exc:
        LOGGER.error("Prose review dashboard failed: %s", exc)
        return 1
    LOGGER.info(
        "Wrote %s with %s sampled proposals from %s verified proposals",
        summary["output"],
        summary["sampled"],
        summary["verified_proposals"],
    )
    return 0


def build_prose_review_dashboard(
    *,
    paths: DashboardPaths,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    expected_original_sha256: str | None = None,
) -> dict[str, JSONValue]:
    """Write a self-contained offline HTML dashboard.

    Args:
        paths:
            Input parquet, status, checkpoint, and output paths.
        sample_limit (optional):
            Maximum number of verified proposals to include. Defaults to 30.
        expected_original_sha256 (optional):
            Extra fail-closed baseline parquet checksum pin.

    Returns:
        JSON-compatible build summary without private persona IDs.

    Raises:
        ReviewDashboardError:
            If the job is not terminal, an input is tampered, or a checkpoint is
            malformed.
    """
    if sample_limit < 1:
        raise ReviewDashboardError("Sample limit must be at least one")
    status = _load_status(path=paths.status)
    _require_terminal_status(status=status)
    _verify_input_hashes(
        status=status,
        original=paths.original,
        candidate=paths.candidate,
        expected_original_sha256=expected_original_sha256,
    )
    original_rows = _rows_by_id(path=paths.original, label="original")
    candidate_rows = _rows_by_id(path=paths.candidate, label="candidate")
    proposals = _load_verified_proposals(
        status=status,
        original_rows=original_rows,
        candidate_rows=candidate_rows,
        checkpoint_root=paths.checkpoint_root,
    )
    sample = select_dashboard_sample(proposals=proposals, limit=sample_limit)
    html_document = render_dashboard(
        status=status,
        proposal_count=len(proposals),
        sample=sample,
        sample_limit=sample_limit,
    )
    _write_private_html(path=paths.output, content=html_document)
    return {
        "output": str(paths.output),
        "processed": _status_int(status=status, key="processed"),
        "total": _status_int(status=status, key="total"),
        "verified_proposals": len(proposals),
        "sampled": len(sample),
    }


def select_dashboard_sample(
    *, proposals: list[ReviewProposal], limit: int = DEFAULT_SAMPLE_LIMIT
) -> list[ReviewProposal]:
    """Return a deterministic stratified sample by changed field and patch size.

    Args:
        proposals:
            Verified proposals available for review.
        limit (optional):
            Maximum selected rows. Defaults to 30.

    Returns:
        Deterministically ordered review rows.
    """
    buckets: dict[tuple[str, str], list[ReviewProposal]] = {}
    for proposal in sorted(
        proposals,
        key=lambda item: (
            item.changed_fields,
            _fraction_bucket_index(item.changed_fraction),
            item.short_id,
        ),
    ):
        field_key = ",".join(proposal.changed_fields) or "unknown"
        key = (field_key, _fraction_bucket(proposal.changed_fraction))
        buckets.setdefault(key, []).append(proposal)
    ordered_keys = sorted(
        buckets, key=lambda item: (item[0], FRACTION_BUCKET_ORDER[item[1]])
    )
    selected: list[ReviewProposal] = []
    while len(selected) < limit and any(buckets.values()):
        for key in ordered_keys:
            bucket = buckets[key]
            if bucket:
                selected.append(bucket.pop(0))
            if len(selected) >= limit:
                break
    return selected


def render_dashboard(
    *,
    status: dict[str, JSONValue],
    proposal_count: int,
    sample: list[ReviewProposal],
    sample_limit: int,
) -> str:
    """Render the review dashboard as a complete offline HTML document.

    Args:
        status:
            Terminal private status document.
        proposal_count:
            Number of verified proposal checkpoints.
        sample:
            Sampled proposals for review.
        sample_limit:
            Configured sample cap.

    Returns:
        Self-contained HTML.
    """
    processed = _status_int(status=status, key="processed")
    total = _status_int(status=status, key="total")
    proposed = _status_int(status=status, key="proposed")
    failed = _status_int(status=status, key="failed")
    skipped = _status_int(status=status, key="skipped")
    cards = "\n".join(
        _render_proposal_card(index=index, proposal=proposal)
        for index, proposal in enumerate(sample, start=1)
    )
    empty = ""
    if not sample:
        empty = (
            '<p class="empty">No verified prose proposals were available for '
            "review after excluding missing or non-proposal checkpoints.</p>"
        )
    csp = (
        "default-src 'none'; style-src 'unsafe-inline'; img-src 'none'; "
        "script-src 'none'; base-uri 'none'; form-action 'none'; "
        "frame-ancestors 'none'"
    )
    styles = """
:root { color-scheme: light; font-family: system-ui, sans-serif; }
body { margin: 0; background: #f8fafc; color: #172033; }
main { max-width: 1180px; margin: 0 auto; padding: 2rem; }
h1 { margin-bottom: 0.4rem; }
.notice { border: 2px solid #9a3412; background: #fff7ed; padding: 1rem; }
.counts {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr));
  gap: 0.75rem;
  margin: 1.5rem 0;
}
.count {
  background: white;
  border: 1px solid #cbd5e1;
  border-radius: 0.6rem;
  padding: 0.8rem;
}
.count strong { display: block; font-size: 1.6rem; }
.card {
  background: white;
  border: 1px solid #cbd5e1;
  border-radius: 0.8rem;
  margin: 1.4rem 0;
  overflow: hidden;
}
.card header { background: #e2e8f0; padding: 0.9rem 1rem; }
.meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.5rem;
}
.badge {
  background: #1e3a8a;
  color: white;
  border-radius: 999px;
  padding: 0.15rem 0.55rem;
  font-size: 0.85rem;
}
.facts { display: grid; gap: 0.5rem; padding: 1rem; }
.fact { border-left: 4px solid #0369a1; padding-left: 0.65rem; }
.compare {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
  padding: 1rem;
}
.panel {
  border: 1px solid #cbd5e1;
  border-radius: 0.5rem;
  padding: 0.85rem;
}
.panel h3 { margin-top: 0; }
p.prose { white-space: pre-wrap; line-height: 1.55; }
mark { background: #fde68a; color: #111827; padding: 0.08rem 0.12rem; }
.empty { background: white; border: 1px dashed #64748b; padding: 1rem; }
@media (max-width: 760px) { .compare { grid-template-columns: 1fr; } }
""".strip()
    notice = (
        "<strong>Unreviewed proposals only.</strong> This offline file is for "
        "human inspection before any dataset, card, or publication step. Do not "
        "publish or merge these texts from the dashboard alone."
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="{csp}">
<title>Prose repair proposal review</title>
<style>
{styles}
</style>
</head>
<body>
<main>
<h1>Prose repair proposal review</h1>
<p class="notice">{notice}</p>
<section class="counts" aria-label="Run counts">
<div class="count"><strong>{processed}</strong> processed of {total}</div>
<div class="count"><strong>{proposed}</strong> status proposals</div>
<div class="count"><strong>{proposal_count}</strong> verified checkpoints</div>
<div class="count"><strong>{len(sample)}</strong> sampled, cap {sample_limit}</div>
<div class="count"><strong>{failed}</strong> failed</div>
<div class="count"><strong>{skipped}</strong> skipped</div>
</section>
{empty}
{cards}
</main>
</body>
</html>
"""


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an offline HTML review dashboard for prose proposals."
    )
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_PROPOSALS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-original-sha256",
        default=None,
        help="Optional extra SHA-256 pin for the original parquet baseline.",
    )
    parser.add_argument("--sample-limit", type=int, default=DEFAULT_SAMPLE_LIMIT)
    return parser


def _load_status(*, path: Path) -> dict[str, JSONValue]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReviewDashboardError("status.json is missing") from exc
    except json.JSONDecodeError as exc:
        raise ReviewDashboardError("status.json is not valid JSON") from exc
    if not isinstance(document, dict):
        raise ReviewDashboardError("status.json must be a JSON object")
    return document


def _require_terminal_status(*, status: dict[str, JSONValue]) -> None:
    processed = _status_int(status=status, key="processed")
    total = _status_int(status=status, key="total")
    if processed != total:
        raise ReviewDashboardError("Repair job is not terminal: processed != total")
    ids = status.get("processed_persona_ids")
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise ReviewDashboardError("status.json processed IDs are malformed")


def _verify_input_hashes(
    *,
    status: dict[str, JSONValue],
    original: Path,
    candidate: Path,
    expected_original_sha256: str | None,
) -> None:
    inputs = _status_inputs(status=status)
    original_sha = sha256_file(original)
    if (
        expected_original_sha256 is not None
        and original_sha != expected_original_sha256
    ):
        raise ReviewDashboardError("Original parquet does not match the expected hash")
    if inputs.get("original") != original_sha:
        raise ReviewDashboardError("Original parquet does not match status.json")
    candidate_sha = inputs.get("candidate")
    if isinstance(candidate_sha, str) and candidate_sha != sha256_file(candidate):
        raise ReviewDashboardError("Candidate parquet does not match status.json")
    schema_sha = inputs.get("schema")
    expected_schema = sha256_text(
        canonical_json(ProsePatchResponse.provider_json_schema())
    )
    if isinstance(schema_sha, str) and schema_sha != expected_schema:
        raise ReviewDashboardError("Patch schema does not match status.json")


def _rows_by_id(*, path: Path, label: str) -> dict[str, dict[str, JSONValue]]:
    frame = pl.read_parquet(path)
    missing = {ID_FIELD, PERSONA_FIELD} - set(frame.columns)
    if missing:
        raise ReviewDashboardError(f"{label} parquet lacks required columns")
    rows: dict[str, dict[str, JSONValue]] = {}
    for row in frame.to_dicts():
        persona_id = row.get(ID_FIELD)
        persona = row.get(PERSONA_FIELD)
        if not isinstance(persona_id, str) or not persona_id:
            raise ReviewDashboardError(f"{label} parquet has an invalid persona ID")
        if persona_id in rows:
            raise ReviewDashboardError(f"{label} parquet has duplicate persona IDs")
        if not isinstance(persona, str):
            raise ReviewDashboardError(f"{label} parquet has invalid persona prose")
        rows[persona_id] = t.cast(dict[str, JSONValue], row)
    return rows


def _load_verified_proposals(
    *,
    status: dict[str, JSONValue],
    original_rows: dict[str, dict[str, JSONValue]],
    candidate_rows: dict[str, dict[str, JSONValue]],
    checkpoint_root: Path,
) -> list[ReviewProposal]:
    proposals: list[ReviewProposal] = []
    expected_model = _status_model(status=status)
    expected_prompt = _status_inputs(status=status).get("prompt")
    for persona_id in t.cast(list[str], status["processed_persona_ids"]):
        persona_hash = sha256_text(persona_id)
        checkpoint_path = _checkpoint_path(
            checkpoint_root=checkpoint_root, persona_hash=persona_hash
        )
        if checkpoint_path is None:
            continue
        proposal = _load_checkpoint_proposal(
            checkpoint_path=checkpoint_path,
            persona_id=persona_id,
            persona_hash=persona_hash,
            original_rows=original_rows,
            candidate_rows=candidate_rows,
            expected_model=expected_model,
            expected_prompt_sha256=expected_prompt,
        )
        if proposal is not None:
            proposals.append(proposal)
    return proposals


def _load_checkpoint_proposal(
    *,
    checkpoint_path: Path,
    persona_id: str,
    persona_hash: str,
    original_rows: dict[str, dict[str, JSONValue]],
    candidate_rows: dict[str, dict[str, JSONValue]],
    expected_model: str | None,
    expected_prompt_sha256: str | None,
) -> ReviewProposal | None:
    checkpoint = _load_checkpoint(path=checkpoint_path)
    original = original_rows.get(persona_id)
    candidate = candidate_rows.get(persona_id)
    if original is None or candidate is None:
        raise ReviewDashboardError("Checkpoint persona is missing from inputs")
    original_text = _row_text(row=original, field=PERSONA_FIELD)
    if candidate.get(PERSONA_FIELD) != original_text:
        raise ReviewDashboardError("Candidate prose changed before review")
    evidence = _checkpoint_evidence(checkpoint=checkpoint)
    if not evidence:
        return None
    proposed = checkpoint.get("proposed_persona_text")
    if not isinstance(proposed, str):
        raise ReviewDashboardError("Checkpoint proposal text is malformed")
    recomputed = _apply_checkpoint_evidence(text=original_text, evidence=evidence)
    if recomputed != proposed:
        raise ReviewDashboardError("Checkpoint proposal does not match evidence")
    changed_facts = _changed_facts(original=original, candidate=candidate)
    if not changed_facts:
        raise ReviewDashboardError("Checkpoint has no changed facts to review")
    _verify_checkpoint_binding(
        checkpoint=checkpoint,
        original=original,
        candidate=candidate,
        old_text=original_text,
        changed_facts=changed_facts,
        expected_model=expected_model,
        expected_prompt_sha256=expected_prompt_sha256,
    )
    changed_fraction = _checkpoint_fraction(
        checkpoint=checkpoint, old_text=original_text, evidence=evidence
    )
    fields = tuple(field for field, _, _ in changed_facts)
    return ReviewProposal(
        persona_hash=persona_hash,
        short_id=persona_hash[:12],
        changed_fields=fields,
        changed_facts=changed_facts,
        changed_fraction=changed_fraction,
        original_text=original_text,
        proposed_text=proposed,
        evidence=evidence,
    )


def _render_proposal_card(*, index: int, proposal: ReviewProposal) -> str:
    original_spans, proposed_spans = _patch_spans(
        original_text=proposal.original_text, evidence=proposal.evidence
    )
    facts = "\n".join(
        _render_fact(field=field, old=old, new=new)
        for field, old, new in proposal.changed_facts
    )
    fields = ", ".join(html.escape(field) for field in proposal.changed_fields)
    fraction = f"{proposal.changed_fraction:.1%}"
    original = _highlight_text(text=proposal.original_text, spans=original_spans)
    proposed = _highlight_text(text=proposal.proposed_text, spans=proposed_spans)
    return f"""<article class="card">
<header>
<h2>Sample {index}: {html.escape(proposal.short_id)}</h2>
<div class="meta">
<span class="badge">Fields: {fields}</span>
<span class="badge">Patch fraction: {fraction}</span>
</div>
</header>
<section class="facts" aria-label="Changed facts">
{facts}
</section>
<section class="compare" aria-label="Original and proposed prose">
<div class="panel"><h3>Original prose</h3><p class="prose">{original}</p></div>
<div class="panel"><h3>Proposed prose</h3><p class="prose">{proposed}</p></div>
</section>
</article>"""


def _render_fact(*, field: str, old: str, new: str) -> str:
    return (
        '<div class="fact"><strong>'
        f"{html.escape(field)}</strong>: old "
        f"<code>{html.escape(old)}</code> -> new "
        f"<code>{html.escape(new)}</code></div>"
    )


def _load_checkpoint(*, path: Path) -> dict[str, JSONValue]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReviewDashboardError("Checkpoint JSON is invalid") from exc
    if not isinstance(document, dict):
        raise ReviewDashboardError("Checkpoint must be a JSON object")
    digest = document.get("checkpoint_sha256")
    unsigned = {
        key: value for key, value in document.items() if key != "checkpoint_sha256"
    }
    if digest != sha256_text(canonical_json(unsigned)):
        raise ReviewDashboardError("Checkpoint checksum is invalid")
    return document


def _checkpoint_path(*, checkpoint_root: Path, persona_hash: str) -> Path | None:
    candidates = (
        checkpoint_root / f"{persona_hash}.json",
        checkpoint_root / "checkpoints" / f"{persona_hash}.json",
        checkpoint_root / "checkpoints" / persona_hash[:2] / f"{persona_hash}.json",
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def _checkpoint_evidence(
    *, checkpoint: dict[str, JSONValue]
) -> tuple[dict[str, str], ...]:
    raw = checkpoint.get("evidence")
    if not isinstance(raw, list):
        raise ReviewDashboardError("Checkpoint evidence is malformed")
    evidence: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ReviewDashboardError("Checkpoint evidence entries are malformed")
        old = item.get("old_excerpt")
        new = item.get("new_excerpt")
        if not isinstance(old, str) or not isinstance(new, str):
            raise ReviewDashboardError("Checkpoint evidence excerpts are malformed")
        evidence.append({"old_excerpt": old, "new_excerpt": new})
    return tuple(evidence)


def _apply_checkpoint_evidence(
    *, text: str, evidence: tuple[dict[str, str], ...]
) -> str:
    try:
        return apply_patches(text, {"patches": list(evidence)})
    except ProsePatchError as exc:
        raise ReviewDashboardError("Checkpoint evidence fails local patching") from exc


def _verify_checkpoint_binding(
    *,
    checkpoint: dict[str, JSONValue],
    original: dict[str, JSONValue],
    candidate: dict[str, JSONValue],
    old_text: str,
    changed_facts: tuple[tuple[str, str, str], ...],
    expected_model: str | None,
    expected_prompt_sha256: str | None,
) -> None:
    payload = {
        "persona": old_text,
        "changed_facts": {
            field: {"old": original[field], "new": candidate[field]}
            for field, _, _ in changed_facts
        },
    }
    expected = {
        "source_sha256": sha256_text(old_text),
        "row_sha256": sha256_text(canonical_json(original)),
        "facts_sha256": sha256_text(canonical_json(payload)),
        "schema_sha256": sha256_text(
            canonical_json(ProsePatchResponse.provider_json_schema())
        ),
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ReviewDashboardError(f"Checkpoint binding mismatch: {key}")
    prompt = checkpoint.get("prompt_sha256")
    if not isinstance(prompt, str):
        raise ReviewDashboardError("Checkpoint prompt binding is missing")
    if expected_prompt_sha256 is not None and prompt != expected_prompt_sha256:
        raise ReviewDashboardError("Checkpoint prompt binding mismatch")
    model = checkpoint.get("model")
    if not isinstance(model, str):
        raise ReviewDashboardError("Checkpoint model binding is missing")
    if expected_model is not None and model != expected_model:
        raise ReviewDashboardError("Checkpoint model binding mismatch")


def _checkpoint_fraction(
    *,
    checkpoint: dict[str, JSONValue],
    old_text: str,
    evidence: tuple[dict[str, str], ...],
) -> float:
    recorded = checkpoint.get("changed_fraction")
    if not isinstance(recorded, int | float):
        raise ReviewDashboardError("Checkpoint changed fraction is malformed")
    actual = sum(
        max(len(item["old_excerpt"]), len(item["new_excerpt"])) for item in evidence
    ) / len(old_text)
    if abs(float(recorded) - actual) > 1e-12:
        raise ReviewDashboardError("Checkpoint changed fraction is invalid")
    return actual


def _changed_facts(
    *, original: dict[str, JSONValue], candidate: dict[str, JSONValue]
) -> tuple[tuple[str, str, str], ...]:
    facts: list[tuple[str, str, str]] = []
    for field in sorted((set(original) & set(candidate)) & _ALLOWED_FACTS):
        old = original[field]
        new = candidate[field]
        if old != new:
            facts.append((field, _display_value(old), _display_value(new)))
    return tuple(facts)


def _patch_spans(
    *, original_text: str, evidence: tuple[dict[str, str], ...]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    raw_spans = []
    for patch in evidence:
        old = patch["old_excerpt"]
        start = original_text.find(old)
        if start < 0 or original_text.find(old, start + 1) >= 0:
            raise ReviewDashboardError("Patch excerpt is not unique in original prose")
        raw_spans.append((start, start + len(old), patch["new_excerpt"]))
    raw_spans.sort(key=lambda item: item[0])
    original_spans: list[tuple[int, int]] = []
    proposed_spans: list[tuple[int, int]] = []
    offset = 0
    previous_end = -1
    for start, end, new in raw_spans:
        if start < previous_end:
            raise ReviewDashboardError("Patch excerpts overlap")
        original_spans.append((start, end))
        proposed_start = start + offset
        proposed_end = proposed_start + len(new)
        proposed_spans.append((proposed_start, proposed_end))
        offset += len(new) - (end - start)
        previous_end = end
    return original_spans, proposed_spans


def _highlight_text(*, text: str, spans: list[tuple[int, int]]) -> str:
    parts: list[str] = []
    position = 0
    for start, end in spans:
        parts.append(html.escape(text[position:start]))
        parts.append("<mark>")
        parts.append(html.escape(text[start:end]))
        parts.append("</mark>")
        position = end
    parts.append(html.escape(text[position:]))
    return "".join(parts)


def _write_private_html(*, path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
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


def _status_inputs(*, status: dict[str, JSONValue]) -> dict[str, str]:
    manifest = status.get("manifest")
    if not isinstance(manifest, dict):
        raise ReviewDashboardError("status.json manifest is missing")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ReviewDashboardError("status.json input hashes are missing")
    parsed: dict[str, str] = {}
    for key, value in inputs.items():
        if isinstance(key, str) and isinstance(value, str):
            parsed[key] = value
    return parsed


def _status_model(*, status: dict[str, JSONValue]) -> str | None:
    manifest = status.get("manifest")
    if not isinstance(manifest, dict):
        return None
    model = manifest.get("model")
    if model is None:
        return None
    if not isinstance(model, str):
        raise ReviewDashboardError("status.json model binding is invalid")
    return model


def _status_int(*, status: dict[str, JSONValue], key: str) -> int:
    value = status.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ReviewDashboardError(f"status.json {key} counter is invalid")
    return value


def _row_text(*, row: dict[str, JSONValue], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise ReviewDashboardError(f"Row field {field} is not text")
    return value


def _display_value(value: JSONValue) -> str:
    if value is None:
        return ""
    if isinstance(value, bool | int | float | str):
        return str(value)
    return canonical_json(value)


def _fraction_bucket(value: float) -> str:
    for threshold, label in FRACTION_BUCKETS:
        if value < threshold:
            return label
    return "large"


def _fraction_bucket_index(value: float) -> int:
    return FRACTION_BUCKET_ORDER[_fraction_bucket(value)]


if __name__ == "__main__":
    raise SystemExit(main())
