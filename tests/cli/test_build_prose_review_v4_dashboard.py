"""Offline contracts for the private v4 prose-review dashboard builder."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from danish_personas.generation.prose_review import validate_prose_review
from danish_personas.io import canonical_json, sha256_file, sha256_text
from scripts import build_prose_review_v4_dashboard as dashboard

TEXT = (
    "Maja er rolig og nysgerrig. Hun holder af lange gåture ved kysten og bruger "
    "weekenderne på at besøge familie og venner. På arbejdet omtales hun med "
    "Før ændring i de syntetiske noter, og hun er grundig, hjælpsom og glad for "
    "at samarbejde med kolleger. Hun tager gerne imod nye opgaver, men sørger "
    "for at planlægge dem i et tempo, der giver plads til fordybelse. Når hun "
    "har fri, læser hun ofte en roman eller laver mad med sæsonens grøntsager. "
    "Hun sætter pris på nærværende samtaler og små udflugter."
)


class FixturePaths(dashboard.DashboardPaths):
    """Typed alias for synthetic v4 dashboard paths."""


def test_builds_private_offline_v4_dashboard_without_provider_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Patched cards and aggregate summaries come from local checkpoints only."""
    paths, manifest = _write_fixture(tmp_path, count=3)
    hashes = [sha256_text(f"pid-{index}") for index in range(3)]
    _write_checkpoint(
        paths=paths, manifest=manifest, persona_id="pid-0", disposition="patched"
    )
    _write_checkpoint(
        paths=paths,
        manifest=manifest,
        persona_id="pid-1",
        disposition="unchanged_consistent",
    )
    _write_checkpoint(
        paths=paths,
        manifest=manifest,
        persona_id="pid-2",
        disposition="needs_manual_review",
    )
    _write_status(paths=paths, manifest=manifest, hashes=hashes)

    def fail_provider_call(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dashboard attempted provider I/O")

    monkeypatch.setattr(
        "danish_personas.generation.client.OpenAIClient.complete", fail_provider_call
    )

    summary = dashboard.build_prose_review_v4_dashboard(paths=paths, limit=10)

    content = paths.output.read_text(encoding="utf-8")
    assert summary["verified_checkpoints"] == 3
    assert summary["patched_cards"] == 1
    assert "NOT accepted repairs" in content
    assert "Provisional unchanged-consistent summary" in content
    assert "Needs-manual-review aggregate summary" in content
    assert "pid-0" not in content
    assert hashes[0][:12] in content
    assert "Exact unified diff" in content
    assert "<script" not in content
    assert (paths.output.stat().st_mode & 0o777) == 0o600
    assert (paths.output.parent.stat().st_mode & 0o777) == 0o700


def test_html_escapes_malicious_prose_and_fact_values(tmp_path: Path) -> None:
    """Raw prose, diffs, patches, and fact values are escaped before HTML."""
    malicious_old = '<img src=x onerror="alert(1)">'
    malicious_new = '<script>alert("patched")</script>'
    paths, manifest = _write_fixture(
        tmp_path,
        count=1,
        marker=malicious_old,
        old_value='<svg onload="old()">',
        new_value='<script>alert("fact")</script>',
    )
    persona_hash = sha256_text("pid-0")
    _write_checkpoint(
        paths=paths,
        manifest=manifest,
        persona_id="pid-0",
        disposition="patched",
        old_excerpt=malicious_old,
        new_excerpt=malicious_new,
    )
    _write_status(paths=paths, manifest=manifest, hashes=[persona_hash])

    dashboard.build_prose_review_v4_dashboard(paths=paths, limit=None)

    content = paths.output.read_text(encoding="utf-8")
    assert malicious_old not in content
    assert malicious_new not in content
    assert '<script>alert("fact")</script>' not in content
    assert "&lt;img src=x" in content
    assert "&lt;script&gt;alert" in content
    assert "<script" not in content


def test_stale_checkpoint_fails_closed_without_output(tmp_path: Path) -> None:
    """A checkpoint bound to an older candidate parquet is refused."""
    paths, manifest = _write_fixture(tmp_path, count=1)
    persona_hash = sha256_text("pid-0")
    _write_checkpoint(paths=paths, manifest=manifest, persona_id="pid-0")
    _write_status(paths=paths, manifest=manifest, hashes=[persona_hash])
    pl.DataFrame(
        [
            {
                "persona_id": "pid-0",
                "persona": TEXT,
                "job_title": "endnu nyere titel",
                "education_level": "lang uddannelse",
            }
        ]
    ).write_parquet(paths.candidate)
    inputs = manifest["inputs"]
    assert isinstance(inputs, dict)
    inputs["candidate_v4"] = sha256_file(paths.candidate)
    _write_json(paths.manifest, manifest)
    _write_status(paths=paths, manifest=manifest, hashes=[persona_hash])

    with pytest.raises(dashboard.ProseReviewV4DashboardError, match="binding mismatch"):
        dashboard.build_prose_review_v4_dashboard(paths=paths)

    assert not paths.output.exists()


def test_file_permissions_are_private_and_unsafe_output_parent_is_refused(
    tmp_path: Path,
) -> None:
    """The dashboard writes 0600 files and does not relax shared directories."""
    paths, manifest = _write_fixture(tmp_path, count=1)
    persona_hash = sha256_text("pid-0")
    _write_checkpoint(paths=paths, manifest=manifest, persona_id="pid-0")
    _write_status(paths=paths, manifest=manifest, hashes=[persona_hash])

    unsafe = tmp_path / "shared"
    unsafe.mkdir(mode=0o755)
    unsafe_paths = FixturePaths(
        original=paths.original,
        candidate=paths.candidate,
        triage=paths.triage,
        prompt=paths.prompt,
        registry=paths.registry,
        status=paths.status,
        manifest=paths.manifest,
        checkpoint_root=paths.checkpoint_root,
        output=unsafe / "dashboard.html",
    )

    with pytest.raises(dashboard.ProseReviewV4DashboardError, match="private"):
        dashboard.build_prose_review_v4_dashboard(paths=unsafe_paths)

    assert (unsafe.stat().st_mode & 0o777) == 0o755
    dashboard.build_prose_review_v4_dashboard(paths=paths)
    assert (paths.output.stat().st_mode & 0o777) == 0o600


def test_missing_original_candidate_or_proposed_hash_fails_closed(
    tmp_path: Path,
) -> None:
    """Missing source rows or proposed-text bindings abort the build."""
    paths, manifest = _write_fixture(tmp_path, count=1)
    missing_hash = sha256_text("missing-private-id")
    _write_status(paths=paths, manifest=manifest, hashes=[missing_hash])

    with pytest.raises(dashboard.ProseReviewV4DashboardError, match="missing"):
        dashboard.build_prose_review_v4_dashboard(paths=paths)

    persona_hash = sha256_text("pid-0")
    checkpoint = _write_checkpoint(paths=paths, manifest=manifest, persona_id="pid-0")
    document = json.loads(checkpoint.read_text(encoding="utf-8"))
    del document["proposed_text_sha256"]
    document["checkpoint_sha256"] = sha256_text(canonical_json(document))
    checkpoint.write_text(json.dumps(document), encoding="utf-8")
    checkpoint.chmod(0o600)
    _write_status(paths=paths, manifest=manifest, hashes=[persona_hash])

    with pytest.raises(dashboard.ProseReviewV4DashboardError, match="schema"):
        dashboard.build_prose_review_v4_dashboard(paths=paths)


def _write_fixture(
    tmp_path: Path,
    *,
    count: int,
    marker: str = "Før ændring",
    old_value: str = "gammel titel",
    new_value: str = "ny titel",
) -> tuple[FixturePaths, dict[str, dashboard.JSONValue]]:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    output_dir = root / "persona-review-v4"
    output_dir.mkdir(mode=0o700)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(mode=0o700)
    original = root / "original.parquet"
    candidate = root / "candidate.parquet"
    triage = root / "triage.json"
    prompt = root / "prompt.md"
    registry = root / "registry.json"
    rows = []
    candidates = []
    for index in range(count):
        persona_id = f"pid-{index}"
        text = TEXT.replace("Før ændring", marker)
        rows.append(
            {
                "persona_id": persona_id,
                "persona": text,
                "job_title": old_value,
                "education_level": "kort uddannelse",
            }
        )
        candidates.append(
            {
                "persona_id": persona_id,
                "persona": text,
                "job_title": new_value,
                "education_level": "lang uddannelse",
            }
        )
    pl.DataFrame(rows).write_parquet(original)
    pl.DataFrame(candidates).write_parquet(candidate)
    triage.write_text('{"rows": []}\n', encoding="utf-8")
    prompt.write_text("Review prompt for offline tests.\n", encoding="utf-8")
    registry.write_text('{"models": []}\n', encoding="utf-8")
    paths = FixturePaths(
        original=original,
        candidate=candidate,
        triage=triage,
        prompt=prompt,
        registry=registry,
        status=output_dir / "status.json",
        manifest=output_dir / "manifest.json",
        checkpoint_root=output_dir,
        output=output_dir / "dashboard.html",
    )
    manifest = _manifest(paths=paths)
    _write_json(paths.manifest, manifest)
    return paths, manifest


def _manifest(*, paths: FixturePaths) -> dict[str, dashboard.JSONValue]:
    return {
        "version": 1,
        "campaign": dashboard.CAMPAIGN,
        "inputs": {
            "original": sha256_file(paths.original),
            "candidate_v4": sha256_file(paths.candidate),
            "triage": sha256_file(paths.triage),
            "prompt": sha256_file(paths.prompt),
            "registry": sha256_file(paths.registry),
            "schema": sha256_text(
                canonical_json(dashboard.ProseReviewResponse.provider_json_schema())
            ),
        },
        "model": dashboard.MODEL,
        "base_url": dashboard.BASE_URL,
        "allowed_facts": sorted(dashboard._ALLOWED_FACTS),
    }


def _write_status(
    *, paths: FixturePaths, manifest: dict[str, dashboard.JSONValue], hashes: list[str]
) -> None:
    status = {
        "version": 1,
        "manifest": manifest,
        "reviewable": len(hashes),
        "patched": _count_disposition(
            paths=paths, hashes=hashes, disposition="patched"
        ),
        "unchanged_consistent": _count_disposition(
            paths=paths, hashes=hashes, disposition="unchanged_consistent"
        ),
        "needs_manual_review": _count_disposition(
            paths=paths, hashes=hashes, disposition="needs_manual_review"
        ),
        "failed": 0,
        "attempted": len(hashes),
        "processed": len(hashes),
        "pending": 0,
        "processed_persona_hashes": hashes,
    }
    _write_json(paths.status, status)


def _count_disposition(
    *, paths: FixturePaths, hashes: list[str], disposition: str
) -> int:
    count = 0
    for persona_hash in hashes:
        path = (
            paths.checkpoint_root
            / "checkpoints"
            / persona_hash[:2]
            / f"{persona_hash}.json"
        )
        if path.exists():
            document = json.loads(path.read_text(encoding="utf-8"))
            count += document.get("disposition") == disposition
    return count


def _write_checkpoint(
    *,
    paths: FixturePaths,
    manifest: dict[str, dashboard.JSONValue],
    persona_id: str,
    disposition: str = "patched",
    old_excerpt: str = "Før ændring",
    new_excerpt: str = "Efter ændring",
) -> Path:
    original = _row(path=paths.original, persona_id=persona_id)
    candidate = _row(path=paths.candidate, persona_id=persona_id)
    changed_facts = dashboard._changed_facts_mapping(
        original=original, candidate=candidate
    )
    prompt = paths.prompt.read_text(encoding="utf-8")
    _, payload = dashboard._validated_input(
        original, candidate, changed_facts, None, None, prompt
    )
    payload = dashboard._payload_with_null_detail_context(
        payload=payload,
        row=original,
        candidate_row=candidate,
        changed_facts=changed_facts,
    )
    response = _response(
        disposition=disposition,
        old_excerpt=old_excerpt,
        new_excerpt=new_excerpt,
        changed_facts=changed_facts,
    )
    result = validate_prose_review(
        original_text=str(original["persona"]),
        changed_facts=changed_facts,
        response=response,
    )
    binding = dashboard._checkpoint_binding(
        original=original,
        candidate=candidate,
        changed_facts=changed_facts,
        original_text=str(original["persona"]),
        prompt=prompt,
        manifest=manifest,
        payload=payload,
    )
    document: dict[str, dashboard.JSONValue] = {
        **binding,
        "disposition": result.disposition,
        "changed_fraction": result.changed_fraction,
        "proposed_text_sha256": sha256_text(result.proposed_text),
        "patches": [
            {"old_excerpt": patch.old_excerpt, "new_excerpt": patch.new_excerpt}
            for patch in result.patches
        ],
        "unchanged_evidence": [
            {"field": item.field, "kind": item.kind, "quote": item.quote}
            for item in result.unchanged_evidence
        ],
        "manual_review_reason": result.manual_review_reason,
        "unchanged_consistent_note": result.unchanged_consistent_note,
    }
    document["checkpoint_sha256"] = sha256_text(canonical_json(document))
    persona_hash = sha256_text(persona_id)
    parent = paths.checkpoint_root / "checkpoints" / persona_hash[:2]
    parent.mkdir(mode=0o700, exist_ok=True)
    path = parent / f"{persona_hash}.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    path.chmod(0o600)
    return path


def _response(
    *,
    disposition: str,
    old_excerpt: str,
    new_excerpt: str,
    changed_facts: dict[str, dict[str, object]],
) -> dict[str, object]:
    if disposition == "patched":
        return {
            "disposition": "patched",
            "patches": [{"old_excerpt": old_excerpt, "new_excerpt": new_excerpt}],
            "unchanged_evidence": [],
            "manual_review_reason": None,
        }
    if disposition == "unchanged_consistent":
        return {
            "disposition": "unchanged_consistent",
            "patches": [],
            "unchanged_evidence": [
                {"field": field, "kind": "fact_not_stated", "quote": ""}
                for field in changed_facts
            ],
            "manual_review_reason": None,
        }
    return {
        "disposition": "needs_manual_review",
        "patches": [],
        "unchanged_evidence": [],
        "manual_review_reason": "ambiguous",
    }


def _row(*, path: Path, persona_id: str) -> dict[str, dashboard.JSONValue]:
    for row in pl.read_parquet(path).to_dicts():
        if row["persona_id"] == persona_id:
            return row
    raise AssertionError("missing fixture row")


def _write_json(path: Path, value: object) -> None:
    content = json.dumps(value, ensure_ascii=False, sort_keys=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)
