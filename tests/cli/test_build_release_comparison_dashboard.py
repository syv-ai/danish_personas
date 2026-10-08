"""Offline contracts for the private release comparison dashboard."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import polars as pl
import pytest

from danish_personas.io import sha256_file
from scripts import build_release_comparison_dashboard as dashboard


def test_builds_counts_cards_and_escapes_private_content(tmp_path: Path) -> None:
    """The dashboard is static, escaped, and free of raw IDs or sidecar fields."""
    paths = _write_fixture(tmp_path)

    summary = dashboard.build_release_comparison_dashboard(
        paths=paths,
        expected_original_sha256=sha256_file(paths.original),
        expected_row_count=4,
        card_limit=2,
    )

    content = paths.output.read_text(encoding="utf-8")
    first_hash = dashboard.sha256_text("raw-persona-id-0")[:12]
    assert summary["changed_prose_count"] == 2
    assert summary["cards"] == 2
    assert "PROVISIONAL, NOT RELEASE-READY" in content
    assert "Demographic count summaries" in content
    assert "Legal status detail" in content
    assert "male" in content
    assert "female" in content
    assert first_hash in content
    assert "raw-persona-id" not in content
    assert "same_sex_partner_target" not in content
    assert "secret-sidecar" not in content
    assert "origin_country_code" not in content
    assert "DK-secret" not in content
    assert '<script>alert("x")</script>' not in content
    assert "&lt;script&gt;alert" in content
    assert '<img src=x onerror="alert(1)">' not in content
    assert "&lt;img src=x" in content
    assert "Official H90 is synthetic missingness" in content
    assert "Source support is not prose certification" in content
    assert "<script" not in content
    assert paths.output.stat().st_mode & 0o777 == 0o600
    assert paths.output.parent.stat().st_mode & 0o777 == 0o700


def test_source_hash_mismatch_and_altered_report_fail(tmp_path: Path) -> None:
    """Source and report bindings fail closed before an HTML file is written."""
    paths = _write_fixture(tmp_path)
    report_path = paths.preview.with_suffix(".json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source_hashes"]["original_v1_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(dashboard.ReleaseComparisonDashboardError, match="mismatch"):
        dashboard.build_release_comparison_dashboard(
            paths=paths,
            expected_original_sha256=sha256_file(paths.original),
            expected_row_count=4,
        )
    assert not paths.output.exists()

    paths = _write_fixture(tmp_path / "altered")
    report_path = paths.preview.with_suffix(".json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["release_ready"] = True
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(
        dashboard.ReleaseComparisonDashboardError, match="release_ready"
    ):
        dashboard.build_release_comparison_dashboard(
            paths=paths,
            expected_original_sha256=sha256_file(paths.original),
            expected_row_count=4,
        )
    assert not paths.output.exists()


def test_refuses_unsafe_output_paths(tmp_path: Path) -> None:
    """Output must be new, private, non-symlinked, and not publication-like."""
    paths = _write_fixture(tmp_path)
    paths.output.write_text("exists", encoding="utf-8")

    with pytest.raises(dashboard.ReleaseComparisonDashboardError, match="overwrite"):
        dashboard.build_release_comparison_dashboard(
            paths=paths,
            expected_original_sha256=sha256_file(paths.original),
            expected_row_count=4,
        )

    fresh = _write_fixture(tmp_path / "fresh")
    outside = dashboard.DashboardPaths(
        original=fresh.original,
        preview=fresh.preview,
        v4_review_status=fresh.v4_review_status,
        v4_verify_status=fresh.v4_verify_status,
        h90_review_status=fresh.h90_review_status,
        h90_verify_status=fresh.h90_verify_status,
        output=tmp_path / "outside.html",
        private_root=fresh.private_root,
    )
    with pytest.raises(dashboard.ReleaseComparisonDashboardError, match="outside"):
        dashboard.build_release_comparison_dashboard(
            paths=outside,
            expected_original_sha256=sha256_file(fresh.original),
            expected_row_count=4,
        )

    public = _replace_output(
        paths=fresh, output=fresh.private_root / "published-card.html"
    )
    with pytest.raises(
        dashboard.ReleaseComparisonDashboardError, match="publication-like"
    ):
        dashboard.build_release_comparison_dashboard(
            paths=public,
            expected_original_sha256=sha256_file(fresh.original),
            expected_row_count=4,
        )

    shared = fresh.private_root / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)
    shared_paths = _replace_output(paths=fresh, output=shared / "dashboard.html")
    with pytest.raises(dashboard.ReleaseComparisonDashboardError, match="0700"):
        dashboard.build_release_comparison_dashboard(
            paths=shared_paths,
            expected_original_sha256=sha256_file(fresh.original),
            expected_row_count=4,
        )

    symlink_paths = _symlink_output_paths(tmp_path=tmp_path / "symlink")
    if symlink_paths is not None:
        with pytest.raises(dashboard.ReleaseComparisonDashboardError, match="symlink"):
            dashboard.build_release_comparison_dashboard(
                paths=symlink_paths,
                expected_original_sha256=sha256_file(symlink_paths.original),
                expected_row_count=4,
            )


def test_build_does_not_use_network_or_provider_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic local files are sufficient for the whole dashboard build."""
    paths = _write_fixture(tmp_path)

    def fail_socket(*_args: object, **_kwargs: object) -> socket.socket:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", fail_socket)

    dashboard.build_release_comparison_dashboard(
        paths=paths,
        expected_original_sha256=sha256_file(paths.original),
        expected_row_count=4,
    )
    assert paths.output.exists()


def _write_fixture(tmp_path: Path) -> dashboard.DashboardPaths:
    root = tmp_path / "private-audit"
    root.mkdir(parents=True, mode=0o700)
    root.chmod(0o700)
    data_dir = root / "data"
    data_dir.mkdir()
    original = data_dir / "train-00000-of-00001.parquet"
    preview = root / "prose-candidate-v5-merged-PROVISIONAL.parquet"
    v4_review_status = root / "persona-review-v4" / "status.json"
    v4_verify_status = root / "persona-verify-v4" / "status.json"
    h90_review_status = root / "persona-review-h90-v5" / "status.json"
    h90_verify_status = root / "persona-verify-h90-v5" / "status.json"
    output = root / "persona-dashboard-v5-BEFORE-AFTER-PROVISIONAL.html"

    original_frame = _frame(original=True)
    preview_frame = _frame(original=False)
    original_frame.write_parquet(original)
    preview_frame.write_parquet(preview)

    _write_json(v4_review_status, {"needs_manual_review": 5, "no_changed_fact": 1})
    _write_json(v4_verify_status, {"rejected": 3})
    _write_json(h90_review_status, {"needs_manual_review": 2})
    _write_json(h90_verify_status, {"rejected": 1})
    _write_json(
        preview.with_suffix(".json"),
        {
            "row_count": 4,
            "release_ready": False,
            "publication_allowed": False,
            "preview_sha256": dashboard._frame_hash(frame=preview_frame),
            "v4_accepted_prose_count": 2,
            "v4_excluded_overlap_count": 1,
            "h90_accepted_count": 1,
            "final_changed_prose_count": 2,
            "source_hashes": {
                "original_v1_sha256": sha256_file(original),
                "h90_second_status_sha256": sha256_file(h90_verify_status),
            },
        },
    )
    return dashboard.DashboardPaths(
        original=original,
        preview=preview,
        v4_review_status=v4_review_status,
        v4_verify_status=v4_verify_status,
        h90_review_status=h90_review_status,
        h90_verify_status=h90_verify_status,
        output=output,
        private_root=root,
    )


def _frame(*, original: bool) -> pl.DataFrame:
    prose = [
        'Original prose mentions raw-persona-id-0 and <img src=x onerror="alert(1)">.',
        "Unchanged prose for raw-persona-id-1.",
        "Original text for raw-persona-id-2.",
        "Unchanged prose for raw-persona-id-3.",
    ]
    if not original:
        prose[0] = (
            'Preview prose redacts raw-persona-id-0 and <script>alert("x")</script>.'
        )
        prose[2] = "Preview text for raw-persona-id-2."
    education = ["basic", "medium", "long", "basic"]
    if not original:
        education[0] = "long"
    return pl.DataFrame(
        {
            "persona_id": [f"raw-persona-id-{index}" for index in range(4)],
            "persona": prose,
            "sex": ["male", "female", "female", "male"],
            "age_band": ["30-39", "40-49", "30-39", "50-59"],
            "education_level": education,
            "legal_status_detail": ["married", None, "separated", None],
            "marital_status": ["married", "unmarried", "married", "widowed"],
            "same_sex_partner_target": ["secret-sidecar"] * 4,
            "origin_country_code": ["DK-secret"] * 4,
        }
    )


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )


def _symlink_output_paths(tmp_path: Path) -> dashboard.DashboardPaths | None:
    paths = _write_fixture(tmp_path)
    symlink_path = paths.private_root / "symlink-dashboard.html"
    try:
        symlink_path.symlink_to(paths.private_root / "target.html")
    except NotImplementedError, OSError:
        return None
    return _replace_output(paths=paths, output=symlink_path)


def _replace_output(
    *, paths: dashboard.DashboardPaths, output: Path
) -> dashboard.DashboardPaths:
    return dashboard.DashboardPaths(
        original=paths.original,
        preview=paths.preview,
        v4_review_status=paths.v4_review_status,
        v4_verify_status=paths.v4_verify_status,
        h90_review_status=paths.h90_review_status,
        h90_verify_status=paths.h90_verify_status,
        output=output,
        private_root=paths.private_root,
    )
