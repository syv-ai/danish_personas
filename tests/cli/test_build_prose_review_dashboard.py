"""Offline contracts for the prose review dashboard builder."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from danish_personas.generation.prose_patch import ProsePatchResponse, apply_patches
from danish_personas.io import canonical_json, sha256_file, sha256_text
from scripts import build_prose_review_dashboard as dashboard

TEXT_TEMPLATE = (
    "Anna bor i Aarhus og arbejder med gamle oplysninger i hverdagen. "
    "Hun beskrives som rolig, grundig og hjælpsom, og hun bruger fritiden "
    "på cykelture, madlavning og samtaler med naboer. Teksten er syntetisk "
    "og indeholder kun generelle detaljer til en afgrænset test af prose. "
    "Familien omtales neutralt uden navne, adresser eller arbejdssteder."
)


def test_builds_private_offline_dashboard_and_excludes_non_proposals(
    tmp_path: Path,
) -> None:
    """Only terminal, verified proposal checkpoints reach the sample dashboard."""
    paths = _write_fixture(tmp_path, count=4)
    _write_status(paths=paths, ids=["pid-0", "pid-1", "pid-2", "pid-3"])
    _write_checkpoint(paths=paths, persona_id="pid-0", field="job_title")
    _write_checkpoint(paths=paths, persona_id="pid-1", field="education_level")
    _write_checkpoint(paths=paths, persona_id="pid-2", field="job_title", evidence=[])

    summary = dashboard.build_prose_review_dashboard(paths=paths, sample_limit=5)

    content = paths.output.read_text(encoding="utf-8")
    assert summary["verified_proposals"] == 2
    assert summary["sampled"] == 2
    assert "pid-0" not in content
    assert "pid-1" not in content
    assert sha256_text("pid-0")[:12] in content
    assert "Unreviewed proposals only" in content
    assert "education_level" in content
    assert "job_title" in content
    assert "<script" not in content
    assert (paths.output.stat().st_mode & 0o777) == 0o600
    assert (paths.output.parent.stat().st_mode & 0o777) == 0o700


class FixturePaths(dashboard.DashboardPaths):
    """Typed alias for synthetic test paths."""


def _write_checkpoint(
    *,
    paths: FixturePaths,
    persona_id: str,
    field: str,
    old_excerpt: str = "gamle oplysninger",
    new_excerpt: str = "nye oplysninger",
    evidence: list[dict[str, str]] | None = None,
) -> Path:
    original = _row(paths=paths, persona_id=persona_id, source="original")
    candidate = _row(paths=paths, persona_id=persona_id, source="candidate")
    changed_facts = {
        name: {"old": original[name], "new": candidate[name]}
        for name in sorted(("education_level", "job_title"))
        if original[name] != candidate[name]
    }
    patch_evidence = (
        [{"old_excerpt": old_excerpt, "new_excerpt": new_excerpt}]
        if evidence is None
        else evidence
    )
    proposed = (
        apply_patches(original["persona"], {"patches": patch_evidence})
        if patch_evidence
        else original["persona"]
    )
    fraction = 0.0
    if patch_evidence:
        fraction = sum(
            max(len(item["old_excerpt"]), len(item["new_excerpt"]))
            for item in patch_evidence
        ) / len(original["persona"])
    payload = {"persona": original["persona"], "changed_facts": changed_facts}
    document = {
        "source_sha256": sha256_text(original["persona"]),
        "row_sha256": sha256_text(canonical_json(original)),
        "facts_sha256": sha256_text(canonical_json(payload)),
        "prompt_sha256": "0" * 64,
        "schema_sha256": sha256_text(
            canonical_json(ProsePatchResponse.provider_json_schema())
        ),
        "model": "synthetic-model",
        "proposed_persona_text": proposed,
        "changed_fraction": fraction,
        "evidence": patch_evidence,
    }
    document["checkpoint_sha256"] = sha256_text(canonical_json(document))
    persona_hash = sha256_text(persona_id)
    path = paths.checkpoint_root / f"{persona_hash}.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


def _row(*, paths: FixturePaths, persona_id: str, source: str) -> dict[str, str]:
    path = paths.original if source == "original" else paths.candidate
    rows = pl.read_parquet(path).to_dicts()
    for row in rows:
        if row["persona_id"] == persona_id:
            return {key: str(value) for key, value in row.items()}
    raise AssertionError(f"Missing persona {persona_id}")


def _write_fixture(
    tmp_path: Path,
    *,
    count: int,
    old_value: str = "gammel titel",
    new_value: str = "ny titel",
    marker: str = "gamle oplysninger",
) -> FixturePaths:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    original_path = root / "original.parquet"
    candidate_path = root / "candidate.parquet"
    status_path = root / "proposals" / "status.json"
    output_path = root / "dashboard.html"
    original_rows = []
    candidate_rows = []
    for index in range(count):
        persona_id = f"pid-{index}"
        original_rows.append(
            {
                "persona_id": persona_id,
                "persona": TEXT_TEMPLATE.replace("gamle oplysninger", marker),
                "job_title": old_value,
                "education_level": "kort uddannelse",
            }
        )
        candidate_rows.append(
            {
                "persona_id": persona_id,
                "persona": TEXT_TEMPLATE.replace("gamle oplysninger", marker),
                "job_title": new_value,
                "education_level": "lang uddannelse",
            }
        )
    pl.DataFrame(original_rows).write_parquet(original_path)
    pl.DataFrame(candidate_rows).write_parquet(candidate_path)
    paths = FixturePaths(
        original=original_path,
        candidate=candidate_path,
        status=status_path,
        checkpoint_root=status_path.parent,
        output=output_path,
    )
    paths.checkpoint_root.mkdir()
    return paths


def _write_status(
    *, paths: FixturePaths, ids: list[str], total: int | None = None
) -> None:
    status = {
        "manifest": {
            "inputs": {
                "original": sha256_file(paths.original),
                "candidate": sha256_file(paths.candidate),
                "schema": sha256_text(
                    canonical_json(ProsePatchResponse.provider_json_schema())
                ),
            },
            "model": "synthetic-model",
        },
        "total": len(ids) if total is None else total,
        "processed": len(ids),
        "proposed": len(ids),
        "failed": 0,
        "skipped": 0,
        "attempted": len(ids),
        "processed_persona_ids": ids,
    }
    paths.status.parent.mkdir(exist_ok=True)
    paths.status.write_text(json.dumps(status), encoding="utf-8")


def test_html_escapes_prose_and_changed_fact_labels(tmp_path: Path) -> None:
    """Prose, facts, and highlighted spans are escaped before HTML rendering."""
    paths = _write_fixture(
        tmp_path,
        count=1,
        old_value="gammel <script>alert(1)</script>",
        new_value="ny <img src=x onerror=alert(2)>",
        marker="gamle <script>alert(3)</script> oplysninger",
    )
    _write_status(paths=paths, ids=["pid-0"])
    _write_checkpoint(
        paths=paths,
        persona_id="pid-0",
        field="job_title",
        old_excerpt="gamle <script>alert(3)</script> oplysninger",
        new_excerpt="nye <b>ufarlige</b> oplysninger",
    )

    dashboard.build_prose_review_dashboard(paths=paths)

    content = paths.output.read_text(encoding="utf-8")
    assert "<script>alert" not in content
    assert "<img src=x" not in content
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in content
    escaped_mark = (
        "<mark>gamle &lt;script&gt;alert(3)&lt;/script&gt; oplysninger</mark>"
    )
    assert escaped_mark in content
    assert "<mark>nye &lt;b&gt;ufarlige&lt;/b&gt; oplysninger</mark>" in content


def test_non_terminal_status_is_refused(tmp_path: Path) -> None:
    """Dashboards are only built once the repair job has reached a terminal state."""
    paths = _write_fixture(tmp_path, count=1)
    _write_status(paths=paths, ids=[], total=1)

    with pytest.raises(dashboard.ReviewDashboardError, match="processed != total"):
        dashboard.build_prose_review_dashboard(paths=paths)


def test_sampling_round_robins_fields_and_patch_fraction(tmp_path: Path) -> None:
    """The deterministic sample covers strata before taking later same-stratum rows."""
    proposals = [
        _proposal(short_id="a1", field="education_level", fraction=0.01),
        _proposal(short_id="a2", field="education_level", fraction=0.01),
        _proposal(short_id="b1", field="education_level", fraction=0.20),
        _proposal(short_id="c1", field="job_title", fraction=0.01),
        _proposal(short_id="d1", field="job_title", fraction=0.20),
    ]

    sample = dashboard.select_dashboard_sample(proposals=proposals, limit=4)

    assert [item.short_id for item in sample] == ["a1", "b1", "c1", "d1"]


def _proposal(
    *, short_id: str, field: str, fraction: float
) -> dashboard.ReviewProposal:
    return dashboard.ReviewProposal(
        persona_hash=short_id * 6,
        short_id=short_id,
        changed_fields=(field,),
        changed_facts=((field, "old", "new"),),
        changed_fraction=fraction,
        original_text=TEXT_TEMPLATE,
        proposed_text=TEXT_TEMPLATE.replace("gamle", "nye", 1),
        evidence=({"old_excerpt": "gamle", "new_excerpt": "nye"},),
    )


def test_shared_output_directory_is_not_repermissioned(tmp_path: Path) -> None:
    """Reject an unsafe output path without changing its parent permissions."""
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)
    with pytest.raises(dashboard.ReviewDashboardError, match="must be private"):
        dashboard._write_private_html(path=shared / "review.html", content="private")
    assert shared.stat().st_mode & 0o777 == 0o755
    assert not (shared / "review.html").exists()


def test_tampered_checkpoint_fails_closed_without_output(tmp_path: Path) -> None:
    """Checksum or evidence tampering aborts the dashboard build."""
    paths = _write_fixture(tmp_path, count=1)
    _write_status(paths=paths, ids=["pid-0"])
    checkpoint = _write_checkpoint(paths=paths, persona_id="pid-0", field="job_title")
    document = json.loads(checkpoint.read_text(encoding="utf-8"))
    document["proposed_persona_text"] = "tampered " + document["proposed_persona_text"]
    checkpoint.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(dashboard.ReviewDashboardError, match="checksum"):
        dashboard.build_prose_review_dashboard(paths=paths)

    assert not paths.output.exists()
