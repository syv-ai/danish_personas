"""Build a private offline before/after release comparison dashboard."""

from __future__ import annotations

import argparse
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
from danish_personas.io import canonical_json, sha256_file, sha256_text

LOGGER = logging.getLogger(__name__)

DEFAULT_ROOT = Path("/tmp/danish-personas-audit")
DEFAULT_ORIGINAL = DEFAULT_ROOT / "data/train-00000-of-00001.parquet"
DEFAULT_PREVIEW = DEFAULT_ROOT / "prose-candidate-v5-merged-PROVISIONAL.parquet"
DEFAULT_OUTPUT = DEFAULT_ROOT / "persona-dashboard-v5-BEFORE-AFTER-PROVISIONAL.html"
DEFAULT_V4_REVIEW_STATUS = DEFAULT_ROOT / "persona-review-v4/status.json"
DEFAULT_V4_VERIFY_STATUS = DEFAULT_ROOT / "persona-verify-v4/status.json"
DEFAULT_H90_REVIEW_STATUS = DEFAULT_ROOT / "persona-review-h90-v5/status.json"
DEFAULT_H90_VERIFY_STATUS = DEFAULT_ROOT / "persona-verify-h90-v5/status.json"
EXPECTED_ORIGINAL_SHA256 = (
    "c178e63d40046274bcc559bdd9322f32336656da980c1809f794d68449d6250f"
)
EXPECTED_RELEASE_ROWS = 100_000
MAX_CARDS = 100
ID_FIELD = "persona_id"
PERSONA_FIELD = "persona"
CHART_FIELDS = (
    ("sex", "Sex"),
    ("age_band", "Age band"),
    ("education_level", "Education"),
    ("legal_status_detail", "Legal status detail"),
    ("marital_status", "Broad marital status"),
)
DIFF_FIELDS = tuple(field for field, _label in CHART_FIELDS)
PUBLICATION_PATH_TOKENS = frozenset(
    {
        "card",
        "data",
        "dataset",
        "hf",
        "huggingface",
        "publication",
        "publish",
        "published",
        "train",
        "v1",
    }
)
RESTRICTED_FIELDS = frozenset(
    {
        "origin_country",
        "origin_country_code",
        "origin_country_resolution",
        "origin_contract_sha256",
        "partner_same_sex_target",
        "partner_sexual_orientation",
        "partner_transgender",
        "partner_variation_in_sex_characteristics",
        "same_sex_partner_target",
        "sexual_orientation",
        "transgender",
        "variation_in_sex_characteristics",
    }
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")

JSONScalar: t.TypeAlias = str | int | float | bool | None
JSONValue: t.TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class DashboardPaths:
    """Filesystem inputs for the private release comparison dashboard."""

    original: Path = DEFAULT_ORIGINAL
    preview: Path = DEFAULT_PREVIEW
    v4_review_status: Path = DEFAULT_V4_REVIEW_STATUS
    v4_verify_status: Path = DEFAULT_V4_VERIFY_STATUS
    h90_review_status: Path = DEFAULT_H90_REVIEW_STATUS
    h90_verify_status: Path = DEFAULT_H90_VERIFY_STATUS
    output: Path = DEFAULT_OUTPUT
    private_root: Path = DEFAULT_ROOT


@dataclass(frozen=True)
class ChangedCard:
    """One hash-selected changed-prose comparison card."""

    persona_hash: str
    original_prose: str
    preview_prose: str
    field_diffs: list[tuple[str, str, str]]


class ReleaseComparisonDashboardError(RuntimeError):
    """Raised when the private dashboard cannot be built safely."""


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
    paths = DashboardPaths(
        original=args.original,
        preview=args.preview,
        v4_review_status=args.v4_review_status,
        v4_verify_status=args.v4_verify_status,
        h90_review_status=args.h90_review_status,
        h90_verify_status=args.h90_verify_status,
        output=args.output,
        private_root=args.private_root,
    )
    try:
        summary = build_release_comparison_dashboard(
            paths=paths,
            expected_original_sha256=args.expected_original_sha256,
            expected_row_count=args.expected_row_count,
            card_limit=args.card_limit,
        )
    except ReleaseComparisonDashboardError as exc:
        LOGGER.error("Release comparison dashboard failed: %s", exc)
        raise SystemExit(1) from exc
    LOGGER.warning(
        "PROVISIONAL, NOT RELEASE-READY: wrote local inspection dashboard only to %s",
        summary["output"],
    )
    LOGGER.warning(
        "Publication allowed: false; report SHA-256: %s", summary["report_sha256"]
    )
    return 0


def build_release_comparison_dashboard(
    *,
    paths: DashboardPaths,
    expected_original_sha256: str = EXPECTED_ORIGINAL_SHA256,
    expected_row_count: int = EXPECTED_RELEASE_ROWS,
    card_limit: int = MAX_CARDS,
) -> dict[str, JSONValue]:
    """Write a self-contained private before/after dashboard.

    Args:
        paths:
            Input Parquet, status, report, and output locations.
        expected_original_sha256 (optional):
            Pinned SHA-256 for the published v1 input.
        expected_row_count (optional):
            Required ordered row count. Defaults to 100,000.
        card_limit (optional):
            Maximum changed-prose cards to render. Defaults to 100.

    Returns:
        JSON-compatible build summary without raw persona IDs or prose.

    Raises:
        ReleaseComparisonDashboardError:
            If inputs, report bindings, or write targets are unsafe.
    """
    if card_limit < 1 or card_limit > MAX_CARDS:
        raise ReleaseComparisonDashboardError("Card limit must be between 1 and 100")
    _guard_private_output(path=paths.output, private_root=paths.private_root)
    _guard_read_path(path=paths.original, label="original parquet")
    _guard_read_path(path=paths.preview, label="preview parquet")

    original_sha256 = sha256_file(paths.original)
    if original_sha256 != expected_original_sha256:
        raise ReleaseComparisonDashboardError("Original v1 SHA-256 mismatch")

    original = pl.read_parquet(paths.original)
    preview = pl.read_parquet(paths.preview)
    ordered_ids = _validate_ordered_frames(
        original=original, preview=preview, expected_row_count=expected_row_count
    )
    report_path = paths.preview.with_suffix(".json")
    report = _load_json_object(path=report_path, label="preview report")
    report_sha256 = sha256_file(report_path)
    preview_sha256 = _frame_hash(frame=preview)
    _validate_report(
        report=report,
        preview_sha256=preview_sha256,
        original_sha256=original_sha256,
        expected_row_count=expected_row_count,
    )
    metrics = _review_metrics(paths=paths, report=report)
    charts = _chart_counts(original=original, preview=preview)
    cards = _changed_cards(
        original=original,
        preview=preview,
        ordered_ids=ordered_ids,
        report=report,
        limit=card_limit,
    )
    document = render_dashboard(
        charts=charts,
        cards=cards,
        metrics=metrics,
        report=report,
        report_sha256=report_sha256,
        preview_sha256=preview_sha256,
        card_limit=card_limit,
    )
    _write_new_private_html(path=paths.output, content=document)
    return {
        "output": str(paths.output),
        "report_sha256": report_sha256,
        "preview_sha256": preview_sha256,
        "row_count": expected_row_count,
        "changed_prose_count": _json_int(
            document=report, key="final_changed_prose_count", label="preview report"
        ),
        "cards": len(cards),
        "status_sha256": {
            "v4_review": metrics["v4_review_status_sha256"],
            "v4_verify": metrics["v4_verify_status_sha256"],
            "h90_review": metrics["h90_review_status_sha256"],
            "h90_verify": metrics["h90_verify_status_sha256"],
        },
        "release_ready": False,
        "publication_allowed": False,
    }


def render_dashboard(
    *,
    charts: dict[str, dict[str, Counter[str]]],
    cards: list[ChangedCard],
    metrics: dict[str, int | str],
    report: dict[str, JSONValue],
    report_sha256: str,
    preview_sha256: str,
    card_limit: int,
) -> str:
    """Render the complete offline dashboard document.

    Returns:
        Self-contained HTML with no scripts, network references, or sidecars.
    """
    csp = (
        "default-src 'none'; style-src 'unsafe-inline'; img-src 'none'; "
        "script-src 'none'; base-uri 'none'; form-action 'none'; "
        "frame-ancestors 'none'"
    )
    chart_html = "\n".join(
        _render_chart(field=field, label=label, counts=charts[field])
        for field, label in CHART_FIELDS
    )
    card_html = "\n".join(
        _render_card(index=index, card=card) for index, card in enumerate(cards, 1)
    )
    if not cards:
        card_html = '<p class="empty">No changed-prose cards were selected.</p>'
    styles = _styles()
    metrics_html = _render_metrics(metrics=metrics)
    final_changed = _json_int(
        document=report, key="final_changed_prose_count", label="preview report"
    )
    release_ready = str(report.get("release_ready")).lower()
    publication_allowed = str(report.get("publication_allowed")).lower()
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="{csp}">
<title>Private v5 before/after provisional dashboard</title>
<style>
{styles}
</style>
</head>
<body>
<header class="banner">PROVISIONAL, NOT RELEASE-READY - LOCAL INSPECTION ONLY</header>
<main>
<h1>Private v5 before/after provisional dashboard</h1>
<p class="notice"><strong>Do not publish.</strong> This static offline page compares
original published v1 rows with a local merged v5 PROVISIONAL preview. It does not
edit the published card, publish data, upload files, contact providers, or certify
release readiness.</p>
<section class="counts" aria-label="Release guardrails">
<div class="count"><strong>{release_ready}</strong> release_ready</div>
<div class="count"><strong>{publication_allowed}</strong> publication_allowed</div>
<div class="count"><strong>{final_changed}</strong> actual changed prose</div>
<div class="count"><strong>{len(cards)}</strong> hash-selected cards,
limit {card_limit}</div>
</section>
<section class="summary" aria-label="Verified local bindings">
<h2>Verified local bindings</h2>
<p>Report SHA-256: <code>{html.escape(report_sha256)}</code>. Preview canonical frame
SHA-256: <code>{html.escape(preview_sha256)}</code>. The adjacent report must state
<code>release_ready=false</code>, <code>publication_allowed=false</code>, and match the
preview frame hash before this page is written.</p>
<p>Official H90 is synthetic missingness in aggregate source support, not an observed
person split. Source support is not prose certification, and accepted local verifier
outputs remain provisional.</p>
</section>
{metrics_html}
<section aria-label="Demographic before and after charts">
<h2>Demographic count summaries</h2>
{chart_html}
</section>
<section aria-label="Changed prose cards">
<h2>Deterministic changed-prose cards</h2>
<p>Cards are selected by sorted SHA-256 persona ID hash. Only the hash prefix is shown;
raw persona IDs and full-dataset prose are not embedded.</p>
{card_html}
</section>
</main>
</body>
</html>
"""


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument(
        "--v4-review-status", type=Path, default=DEFAULT_V4_REVIEW_STATUS
    )
    parser.add_argument(
        "--v4-verify-status", type=Path, default=DEFAULT_V4_VERIFY_STATUS
    )
    parser.add_argument(
        "--h90-review-status", type=Path, default=DEFAULT_H90_REVIEW_STATUS
    )
    parser.add_argument(
        "--h90-verify-status", type=Path, default=DEFAULT_H90_VERIFY_STATUS
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--private-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--expected-original-sha256", default=EXPECTED_ORIGINAL_SHA256)
    parser.add_argument("--expected-row-count", type=int, default=EXPECTED_RELEASE_ROWS)
    parser.add_argument("--card-limit", type=int, default=MAX_CARDS)
    return parser


def _validate_ordered_frames(
    *, original: pl.DataFrame, preview: pl.DataFrame, expected_row_count: int
) -> list[str]:
    for label, frame in (("original", original), ("preview", preview)):
        missing = {ID_FIELD, PERSONA_FIELD} - set(frame.columns)
        if missing:
            raise ReleaseComparisonDashboardError(
                f"{label} frame is missing required columns: {sorted(missing)}"
            )
        if frame.height != expected_row_count:
            raise ReleaseComparisonDashboardError(
                f"{label} frame row count does not match {expected_row_count}"
            )
    original_ids = _ids(frame=original, label="original")
    preview_ids = _ids(frame=preview, label="preview")
    if original_ids != preview_ids:
        raise ReleaseComparisonDashboardError(
            "Original and preview IDs differ in order"
        )
    if len(set(original_ids)) != len(original_ids):
        raise ReleaseComparisonDashboardError("Persona IDs must be unique")
    return original_ids


def _validate_report(
    *,
    report: dict[str, JSONValue],
    preview_sha256: str,
    original_sha256: str,
    expected_row_count: int,
) -> None:
    if report.get("release_ready") is not False:
        raise ReleaseComparisonDashboardError(
            "Preview report must set release_ready=false"
        )
    if report.get("publication_allowed") is not False:
        raise ReleaseComparisonDashboardError(
            "Preview report must set publication_allowed=false"
        )
    if _json_string(document=report, key="preview_sha256", label="preview report") != (
        preview_sha256
    ):
        raise ReleaseComparisonDashboardError("Preview report frame SHA-256 mismatch")
    if _json_int(document=report, key="row_count", label="preview report") != (
        expected_row_count
    ):
        raise ReleaseComparisonDashboardError("Preview report row count mismatch")
    source_hashes = _json_object(
        document=report, key="source_hashes", label="preview report"
    )
    reported_original = _json_string(
        document=source_hashes, key="original_v1_sha256", label="source hashes"
    )
    if reported_original != original_sha256:
        raise ReleaseComparisonDashboardError(
            "Preview report original SHA-256 mismatch"
        )


def _review_metrics(
    *, paths: DashboardPaths, report: dict[str, JSONValue]
) -> dict[str, int | str]:
    v4_review = _load_json_object(path=paths.v4_review_status, label="v4 review status")
    v4_verify = _load_json_object(
        path=paths.v4_verify_status, label="v4 verifier status"
    )
    h90_review = _load_json_object(
        path=paths.h90_review_status, label="H90 review status"
    )
    h90_verify = _load_json_object(
        path=paths.h90_verify_status, label="H90 verifier status"
    )
    _verify_optional_status_hash(
        report=report, key="h90_second_status_sha256", path=paths.h90_verify_status
    )
    v4_accepted = _json_int(
        document=report, key="v4_accepted_prose_count", label="preview report"
    )
    overlap = _json_int(
        document=report, key="v4_excluded_overlap_count", label="preview report"
    )
    h90_accepted = _json_int(
        document=report, key="h90_accepted_count", label="preview report"
    )
    actual = _json_int(
        document=report, key="final_changed_prose_count", label="preview report"
    )
    if v4_accepted - overlap + h90_accepted != actual:
        raise ReleaseComparisonDashboardError(
            "Changed-prose arithmetic is inconsistent"
        )
    return {
        "v4_manual_review": _status_int(
            status=v4_review, key="needs_manual_review", label="v4 review status"
        ),
        "v4_no_changed_fact": _status_int(
            status=v4_review, key="no_changed_fact", label="v4 review status"
        ),
        "v4_verifier_rejected": _status_int(
            status=v4_verify, key="rejected", label="v4 verifier status"
        ),
        "h90_manual_review": _status_int(
            status=h90_review, key="needs_manual_review", label="H90 review status"
        ),
        "h90_verifier_rejected": _status_int(
            status=h90_verify, key="rejected", label="H90 verifier status"
        ),
        "v4_accepted": v4_accepted,
        "v4_overlap": overlap,
        "h90_accepted": h90_accepted,
        "actual_changed": actual,
        "v4_review_status_sha256": sha256_file(paths.v4_review_status),
        "v4_verify_status_sha256": sha256_file(paths.v4_verify_status),
        "h90_review_status_sha256": sha256_file(paths.h90_review_status),
        "h90_verify_status_sha256": sha256_file(paths.h90_verify_status),
    }


def _verify_optional_status_hash(
    *, report: dict[str, JSONValue], key: str, path: Path
) -> None:
    source_hashes = report.get("source_hashes")
    if not isinstance(source_hashes, dict) or key not in source_hashes:
        return
    reported = source_hashes.get(key)
    if reported != sha256_file(path):
        raise ReleaseComparisonDashboardError(f"Status SHA-256 mismatch for {path}")


def _chart_counts(
    *, original: pl.DataFrame, preview: pl.DataFrame
) -> dict[str, dict[str, Counter[str]]]:
    charts: dict[str, dict[str, Counter[str]]] = {}
    for field, _label in CHART_FIELDS:
        if field not in original.columns or field not in preview.columns:
            raise ReleaseComparisonDashboardError(f"Missing chart field: {field}")
        charts[field] = {
            "before": _value_counts(frame=original, field=field),
            "after": _value_counts(frame=preview, field=field),
        }
    return charts


def _changed_cards(
    *,
    original: pl.DataFrame,
    preview: pl.DataFrame,
    ordered_ids: list[str],
    report: dict[str, JSONValue],
    limit: int,
) -> list[ChangedCard]:
    original_rows = original.to_dicts()
    preview_rows = preview.to_dicts()
    changed: list[ChangedCard] = []
    for persona_id, before, after in zip(ordered_ids, original_rows, preview_rows):
        before_prose = _string_value(before.get(PERSONA_FIELD), field=PERSONA_FIELD)
        after_prose = _string_value(after.get(PERSONA_FIELD), field=PERSONA_FIELD)
        if before_prose == after_prose:
            continue
        persona_hash = sha256_text(persona_id)
        changed.append(
            ChangedCard(
                persona_hash=persona_hash,
                original_prose=_redact_id(text=before_prose, persona_id=persona_id),
                preview_prose=_redact_id(text=after_prose, persona_id=persona_id),
                field_diffs=_field_diffs(
                    before=before, after=after, persona_id=persona_id
                ),
            )
        )
    expected_changed = _json_int(
        document=report, key="final_changed_prose_count", label="preview report"
    )
    if len(changed) != expected_changed:
        raise ReleaseComparisonDashboardError(
            "Changed-prose count does not match report"
        )
    return sorted(changed, key=lambda card: card.persona_hash)[:limit]


def _field_diffs(
    *, before: dict[str, object], after: dict[str, object], persona_id: str
) -> list[tuple[str, str, str]]:
    diffs: list[tuple[str, str, str]] = []
    for field in DIFF_FIELDS:
        if field in RESTRICTED_FIELDS or field == ID_FIELD or field == PERSONA_FIELD:
            continue
        before_value = _display_value(before.get(field))
        after_value = _display_value(after.get(field))
        if before_value == after_value:
            continue
        diffs.append(
            (
                field,
                _redact_id(text=before_value, persona_id=persona_id),
                _redact_id(text=after_value, persona_id=persona_id),
            )
        )
    return diffs


def _render_metrics(*, metrics: dict[str, int | str]) -> str:
    rows = (
        ("v4 manual review", metrics["v4_manual_review"]),
        ("v4 no changed fact", metrics["v4_no_changed_fact"]),
        ("v4 verifier rejected", metrics["v4_verifier_rejected"]),
        ("H90 manual review", metrics["h90_manual_review"]),
        ("H90 verifier rejected", metrics["h90_verifier_rejected"]),
        ("v4 accepted prose", metrics["v4_accepted"]),
        ("v4/H90 overlap excluded", metrics["v4_overlap"]),
        ("H90 accepted prose", metrics["h90_accepted"]),
        ("actual changed prose", metrics["actual_changed"]),
    )
    count_html = "\n".join(
        f'<div class="count"><strong>{value}</strong> {html.escape(label)}</div>'
        for label, value in rows
    )
    return f"""<section class="summary" aria-label="Review metrics">
<h2>Aggregate review metrics from pinned local reports/statuses</h2>
<section class="counts">{count_html}</section>
<p>Arithmetic check: v4 accepted minus overlap plus H90 accepted equals actual changed
prose. Status SHA-256 pins are retained in the local build summary, not rendered with
raw persona identifiers.</p>
</section>"""


def _render_chart(*, field: str, label: str, counts: dict[str, Counter[str]]) -> str:
    before = counts["before"]
    after = counts["after"]
    values = sorted(set(before) | set(after))
    max_count = max([*before.values(), *after.values(), 1])
    rows = []
    for value in values:
        before_count = before[value]
        after_count = after[value]
        before_width = int((before_count / max_count) * 100)
        after_width = int((after_count / max_count) * 100)
        rows.append(
            "<tr>"
            f"<th>{html.escape(value)}</th>"
            f'<td>{before_count}<div class="bar before" '
            f'style="width:{before_width}%"></div></td>'
            f'<td>{after_count}<div class="bar after" '
            f'style="width:{after_width}%"></div></td>'
            f"<td>{after_count - before_count:+d}</td>"
            "</tr>"
        )
    rows_html = "\n".join(rows)
    return f"""<article class="chart">
<h3>{html.escape(label)} <code>{html.escape(field)}</code></h3>
<table>
<thead><tr><th>Value</th><th>Before</th><th>After</th><th>Delta</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
</article>"""


def _render_card(*, index: int, card: ChangedCard) -> str:
    diff_rows = "\n".join(
        "<tr>"
        f"<th>{html.escape(field)}</th>"
        f"<td>{html.escape(before)}</td>"
        f"<td>{html.escape(after)}</td>"
        "</tr>"
        for field, before, after in card.field_diffs
    )
    if not diff_rows:
        diff_rows = (
            '<tr><td colspan="3">No displayed structured fields changed.</td></tr>'
        )
    return f"""<article class="card">
<h3>Changed prose card {index}: hash prefix {html.escape(card.persona_hash[:12])}</h3>
<section class="compare" aria-label="Original and preview prose">
<div class="panel"><h4>Original v1 prose</h4>
<p class="prose">{html.escape(card.original_prose)}</p></div>
<div class="panel"><h4>Provisional v5 preview prose</h4>
<p class="prose">{html.escape(card.preview_prose)}</p></div>
</section>
<table class="diffs">
<thead><tr><th>Field</th><th>Before</th><th>After</th></tr></thead>
<tbody>{diff_rows}</tbody>
</table>
</article>"""


def _styles() -> str:
    return """
:root { color-scheme: light; font-family: system-ui, sans-serif; }
body { margin: 0; background: #f7f4ee; color: #191714; }
.banner { background: #8a1f11; color: #fff; font-weight: 800; padding: 1rem;
  position: sticky; top: 0; z-index: 10; text-align: center; letter-spacing: .04em; }
main { max-width: 1180px; margin: 0 auto; padding: 1.5rem; }
.notice, .summary, .chart, .card { background: #fffaf0; border: 1px solid #d8cbb7;
  border-radius: .75rem; margin: 1rem 0; padding: 1rem; }
.counts { display: grid; gap: .75rem; grid-template-columns: repeat(auto-fit,
  minmax(180px, 1fr)); margin: 1rem 0; }
.count { background: #fff; border: 1px solid #ddcfbb; border-radius: .5rem;
  padding: .75rem; }
.count strong { display: block; font-size: 1.4rem; }
table { border-collapse: collapse; width: 100%; }
th, td { border-top: 1px solid #e5d9c8; padding: .45rem; text-align: left;
  vertical-align: top; }
.bar { height: .45rem; min-width: 2px; margin-top: .25rem; border-radius: 999px; }
.before { background: #6f8db9; }
.after { background: #b9825b; }
.compare { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit,
  minmax(280px, 1fr)); }
.panel { background: #fff; border: 1px solid #e5d9c8; border-radius: .5rem;
  padding: .75rem; }
.prose { white-space: pre-wrap; }
code { background: #eee2cf; padding: .1rem .25rem; border-radius: .25rem; }
.empty { color: #5b5348; font-style: italic; }
""".strip()


def _guard_private_output(*, path: Path, private_root: Path) -> None:
    if path.suffix.lower() != ".html":
        raise ReleaseComparisonDashboardError("Dashboard output must be an HTML file")
    _refuse_symlink(path=private_root, label="private root")
    _refuse_symlink(path=path, label="dashboard output")
    _refuse_symlink(path=path.parent, label="dashboard output directory")
    root_resolved = private_root.resolve(strict=False)
    path_resolved = path.resolve(strict=False)
    if not _is_relative_to(path_resolved, root_resolved):
        raise ReleaseComparisonDashboardError(
            "Refusing output outside the private path"
        )
    if _publication_like(path=path):
        raise ReleaseComparisonDashboardError("Refusing publication-like output path")
    if path.exists() or path.is_symlink():
        raise ReleaseComparisonDashboardError(
            f"Refusing to overwrite existing file: {path}"
        )
    if path.parent.exists():
        mode = path.parent.stat().st_mode & 0o777
        if not path.parent.is_dir() or mode != 0o700:
            raise ReleaseComparisonDashboardError(
                "Dashboard output directory must exist with mode 0700"
            )
        return
    path.parent.mkdir(parents=True, mode=0o700)
    os.chmod(path.parent, 0o700)


def _guard_read_path(*, path: Path, label: str) -> None:
    _refuse_symlink(path=path, label=label)
    if not path.is_file():
        raise ReleaseComparisonDashboardError(f"Missing {label}: {path}")


def _write_new_private_html(*, path: Path, content: str) -> None:
    if path.exists() or path.is_symlink():
        raise ReleaseComparisonDashboardError(
            f"Refusing to overwrite existing file: {path}"
        )
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json_object(*, path: Path, label: str) -> dict[str, JSONValue]:
    _guard_read_path(path=path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseComparisonDashboardError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise ReleaseComparisonDashboardError(f"{label} is not a JSON object")
    return t.cast(dict[str, JSONValue], value)


def _frame_hash(*, frame: pl.DataFrame) -> str:
    payload = {"columns": frame.columns, "rows": frame.to_dicts()}
    return sha256_text(canonical_json(payload))


def _ids(*, frame: pl.DataFrame, label: str) -> list[str]:
    ids: list[str] = []
    for value in frame.get_column(ID_FIELD).to_list():
        if not isinstance(value, str) or not value.strip():
            raise ReleaseComparisonDashboardError(
                f"{label} frame has invalid persona ID"
            )
        ids.append(value)
    return ids


def _value_counts(*, frame: pl.DataFrame, field: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for value in frame.get_column(field).to_list():
        counts[_display_value(value)] += 1
    return counts


def _json_object(
    *, document: dict[str, JSONValue], key: str, label: str
) -> dict[str, JSONValue]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise ReleaseComparisonDashboardError(f"{label} missing object key: {key}")
    return t.cast(dict[str, JSONValue], value)


def _json_string(*, document: dict[str, JSONValue], key: str, label: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise ReleaseComparisonDashboardError(f"{label} missing SHA-256 key: {key}")
    return value


def _json_int(*, document: dict[str, JSONValue], key: str, label: str) -> int:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReleaseComparisonDashboardError(f"{label} missing integer key: {key}")
    return value


def _status_int(*, status: dict[str, JSONValue], key: str, label: str) -> int:
    value = status.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReleaseComparisonDashboardError(
            f"{label} missing non-negative count: {key}"
        )
    return value


def _string_value(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ReleaseComparisonDashboardError(f"Field must be string: {field}")
    return value


def _display_value(value: object) -> str:
    if value is None:
        return "(missing)"
    if isinstance(value, str):
        return value if value else "(blank)"
    return str(value)


def _redact_id(*, text: str, persona_id: str) -> str:
    return text.replace(persona_id, "[redacted persona ID]")


def _publication_like(*, path: Path) -> bool:
    lowered_parts = {part.lower() for part in path.parts}
    stem_tokens = set(path.stem.lower().replace("-", "_").split("_"))
    return bool(
        lowered_parts & PUBLICATION_PATH_TOKENS
        or stem_tokens & (PUBLICATION_PATH_TOKENS)
    )


def _refuse_symlink(*, path: Path, label: str) -> None:
    if path.is_symlink():
        raise ReleaseComparisonDashboardError(f"Refusing symlink for {label}: {path}")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
