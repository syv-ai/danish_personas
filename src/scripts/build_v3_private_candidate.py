"""Compose a private candidate from a completed v3 review campaign."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

import click
import httpx
import polars as pl

from danish_personas.generation.models import GenerationConfig
from danish_personas.generation.proxy_budget import (
    SOL_ADJUDICATION_LEDGER_MAX_TOKENS,
    SOL_ADJUDICATION_MODEL,
    V3_ADJUDICATION_PURPOSE,
    ProxyBudget,
)
from danish_personas.generation.sol_adjudication import run_sol_adjudication
from danish_personas.io import canonical_json, sha256_file, sha256_text
from danish_personas.release.candidate_validation import validate_release_candidate
from scripts import adjudicate_persona_v3 as campaign

ROOT = Path("/tmp/danish-personas-audit")
DEFAULT_ORIGINAL = (
    ROOT / "hf-v2-foreign-hotfix-remote/data/train-00000-of-00001.parquet"
)
DEFAULT_CANDIDATE = ROOT / "v3-private/demographics-ready.parquet"
DEFAULT_REPAIR = ROOT / "v3-private/demographics-ready.manifest.json"
DEFAULT_CAMPAIGN = ROOT / "v3-private/campaign"
DEFAULT_OUTPUT = ROOT / "v3-private/composed-candidate.parquet"
DEFAULT_PATCHES = ROOT / "v3-private/composed-candidate.vetted.json"
DEFAULT_REPORT = ROOT / "v3-private/composed-candidate.report.json"
BUNDLE = Path("data/processed/6e27b5c08fbeae79")


class ComposeError(RuntimeError):
    """Raised when campaign evidence or composition is invalid."""


@click.command()
@click.option("--original", type=click.Path(path_type=Path), default=DEFAULT_ORIGINAL)
@click.option("--candidate", type=click.Path(path_type=Path), default=DEFAULT_CANDIDATE)
@click.option(
    "--repair-manifest", type=click.Path(path_type=Path), default=DEFAULT_REPAIR
)
@click.option(
    "--campaign-dir", type=click.Path(path_type=Path), default=DEFAULT_CAMPAIGN
)
@click.option("--output", type=click.Path(path_type=Path), default=DEFAULT_OUTPUT)
@click.option(
    "--patch-manifest", type=click.Path(path_type=Path), default=DEFAULT_PATCHES
)
@click.option("--report", type=click.Path(path_type=Path), default=DEFAULT_REPORT)
def main(
    original: Path,
    candidate: Path,
    repair_manifest: Path,
    campaign_dir: Path,
    output: Path,
    patch_manifest: Path,
    report: Path,
) -> None:
    """Compose after review completes.

    Raises:
        click.ClickException: If campaign validation or composition fails.
    """
    try:
        result = compose(
            original,
            candidate,
            repair_manifest,
            campaign_dir,
            output,
            patch_manifest,
            report,
        )
    except (ComposeError, OSError, ValueError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result, sort_keys=True))


def compose(  # noqa: C901, PLR0912
    original: Path,
    candidate: Path,
    repair_path: Path,
    campaign_dir: Path,
    output: Path,
    patch_path: Path,
    report_path: Path,
) -> dict[str, object]:
    """Validate campaign evidence and write a private release candidate.

    Returns:
        Aggregate composition summary containing counts and checksums.

    Raises:
        ComposeError: If campaign evidence or release gates are invalid.
    """
    paths = (output, patch_path, report_path)
    if len(set(paths)) != len(paths) or any(path.exists() for path in paths):
        raise ComposeError("Output paths must be distinct and not already exist")
    original_frame, candidate_frame = (
        pl.read_parquet(original),
        pl.read_parquet(candidate),
    )
    if (
        original_frame.height != campaign.EXPECTED_ROWS
        or candidate_frame.height != campaign.EXPECTED_ROWS
    ):
        raise ComposeError("Input row count mismatch")
    ids = original_frame[campaign.ID_FIELD].to_list()
    if ids != candidate_frame[campaign.ID_FIELD].to_list() or any(
        not isinstance(i, str) for i in ids
    ):
        raise ComposeError("Ordered identities do not match")
    hashes = [sha256_text(i) for i in ids]
    ordered_hash = sha256_text(canonical_json(hashes))
    repair = _json(repair_path)
    if (
        repair.get("private") is not True
        or repair.get("published_sha256") != sha256_file(original)
        or repair.get("candidate_sha256") != sha256_file(candidate)
    ):
        raise ComposeError("Repair manifest input binding mismatch")
    changed_indexes = repair.get("changed_row_indexes")
    actual_changed = [
        index
        for index, (before, after) in enumerate(
            zip(original_frame.to_dicts(), candidate_frame.to_dicts(), strict=True)
        )
        if before != after
    ]
    if changed_indexes != actual_changed:
        raise ComposeError("Repair manifest row binding mismatch")
    manifest = _json(campaign_dir / "manifest.json")
    status = _json(campaign_dir / "status.json")
    inputs = manifest.get("input_hashes")
    if not isinstance(inputs, dict) or manifest.get("campaign") != campaign.CAMPAIGN:
        raise ComposeError("Campaign manifest schema mismatch")
    expected = {
        "baseline_sha256": sha256_file(original),
        "candidate_sha256": sha256_file(candidate),
        "repair_sha256": sha256_file(repair_path),
        "ordered_id_hashes_sha256": ordered_hash,
    }
    if any(inputs.get(key) != value for key, value in expected.items()):
        raise ComposeError("Campaign input checksum binding mismatch")
    if (
        status.get("campaign") != campaign.CAMPAIGN
        or status.get("version") != campaign.VERSION
        or status.get("input_hashes") != inputs
        or status.get("selected_total") != manifest.get("selected_total")
        or status.get("progress", {}).get("pending") != 0
        or status.get("progress", {}).get("completed") != manifest.get("selected_total")
        or status.get("progress", {}).get("total") != manifest.get("selected_total")
    ):
        raise ComposeError("Campaign is incomplete or status binding is invalid")
    processed = status.get("processed")
    if not isinstance(processed, list) or len(processed) != manifest.get(
        "selected_total"
    ):
        raise ComposeError("Campaign processed rows do not cover selection")
    by_hash = {digest: index for index, digest in enumerate(hashes)}
    if len(by_hash) != len(hashes):
        raise ComposeError("Duplicate identities")
    prompt_path = campaign.PROMPT
    prompt = prompt_path.read_text(encoding="utf-8")
    config = GenerationConfig.model_validate(
        {
            "base_url": campaign.BASE_URL,
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
    budget = ProxyBudget(
        ledger_path=Path("/dev/null"),
        registry_path=campaign.REGISTRY,
        campaign=campaign.CAMPAIGN,
        source_hash=sha256_text(canonical_json(inputs)),
        prompt_hash=inputs["prompt_sha256"],
        schema_hash=inputs["schema_sha256"],
        model=SOL_ADJUDICATION_MODEL,
        input_usd_per_million="0.1",
        output_usd_per_million="0.5",
        max_tokens=SOL_ADJUDICATION_LEDGER_MAX_TOKENS,
        cap_usd=Decimal("1"),
        uncapped=True,
        uncapped_purpose=V3_ADJUDICATION_PURPOSE,
    )
    original_rows, rows = original_frame.to_dicts(), candidate_frame.to_dicts()
    dispositions: Counter[str] = Counter()
    seen: set[str] = set()
    for item in processed:
        digest, disposition = item.get("persona_hash"), item.get("disposition")
        if (
            digest not in by_hash
            or digest in seen
            or disposition
            not in {
                "consistent",
                "patched",
                "unresolved",
                "privacy_blocked",
                "validation_failed",
            }
        ):
            raise ComposeError("Campaign contains invalid disposition or identity")
        seen.add(digest)
        if item.get("result_sha256") != sha256_text(f"{digest}:{disposition}"):
            raise ComposeError("Campaign result digest mismatch")
        dispositions[disposition] += 1
        index = by_hash[digest]
        if disposition == "validation_failed":
            campaign._verify_validation_failure(
                output_dir=campaign_dir, digest=digest, input_hashes=inputs
            )
            continue
        if disposition == "privacy_blocked":
            continue
        checkpoint = item.get("checkpoint")
        checkpoint_hash, response_hash = (
            item.get("checkpoint_sha256"),
            item.get("response_sha256"),
        )
        cp = campaign_dir / checkpoint if isinstance(checkpoint, str) else Path("/")
        if (
            not cp.is_relative_to(campaign_dir)
            or not cp.is_file()
            or sha256_file(cp) != checkpoint_hash
        ):
            raise ComposeError("Checkpoint binding mismatch")
        doc = _json(cp)
        if doc.get("response_sha256") != response_hash:
            raise ComposeError("Checkpoint response binding mismatch")
        old, row = original_rows[index], rows[index]
        hints = {
            field: {"old": old[field], "new": row[field]}
            for field in sorted(
                (set(old) & set(row)) & campaign.SOL_ALLOWED_FACT_FIELDS
            )
            if old[field] != row[field]
        }
        verified = run_sol_adjudication(
            original_persona=row[campaign.TEXT_FIELD],
            candidate_row=row,
            prompt=prompt,
            config=config,
            budget=budget,
            checkpoint_path=cp,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(500, request=request)
            ),
            changed_fact_hints=hints,
            original_row=old,
        )
        if verified.disposition != disposition:
            raise ComposeError("Verified disposition mismatch")
        if disposition == "patched":
            row[campaign.TEXT_FIELD] = verified.proposed_text
    selected_indexes = sorted(by_hash[digest] for digest in seen)
    if inputs.get("queue_sha256") != sha256_text(canonical_json(selected_indexes)):
        raise ComposeError("Completed rows do not match the pinned campaign queue")
    expected_counts = {
        name: dispositions[name]
        for name in (
            "consistent",
            "patched",
            "unresolved",
            "privacy_blocked",
            "validation_failed",
        )
    }
    if status.get("counts") != expected_counts:
        raise ComposeError("Campaign disposition totals mismatch")
    final = pl.DataFrame(rows, schema=candidate_frame.schema)
    if any(
        a != b
        for a, b in zip(final.to_dicts(), candidate_frame.to_dicts(), strict=True)
        if {k: v for k, v in a.items() if k != campaign.TEXT_FIELD}
        != {k: v for k, v in b.items() if k != campaign.TEXT_FIELD}
    ):
        raise ComposeError("Non-prose candidate field changed")
    _check_pinned_marginals(original_frame, candidate_frame, final)
    parquet = final.write_parquet()
    candidate_hash = hashlib.sha256(parquet).hexdigest()
    allowed = sorted(
        sha256_text(i)
        for i, a, b in zip(
            ids,
            original_frame[campaign.TEXT_FIELD].to_list(),
            final[campaign.TEXT_FIELD].to_list(),
            strict=True,
        )
        if a != b
    )
    patch_manifest = {
        "version": 1,
        "status": "vetted",
        "allowed_persona_id_hashes": allowed,
        "candidate_sha256": candidate_hash,
        "original_sha256": sha256_file(original),
    }
    _write_new(output, parquet)
    _write_new(patch_path, (canonical_json(patch_manifest) + "\n").encode())
    validation = validate_release_candidate(
        candidate_path=output,
        original_path=original,
        bundle_dir=BUNDLE,
        vetted_patch_manifest_path=patch_path,
    )
    if not validation.passes_hard_gates:
        output.unlink(missing_ok=True)
        patch_path.unlink(missing_ok=True)
        raise ComposeError("Release candidate failed mandatory validation gates")
    summary = {
        "campaign_dispositions": dict(dispositions),
        "patched_rows": dispositions["patched"],
        "preserved_unresolved": dispositions["unresolved"],
        "preserved_privacy_blocked": dispositions["privacy_blocked"],
        "preserved_validation_failed": dispositions["validation_failed"],
        "candidate_sha256": candidate_hash,
        "validation_passes_hard_gates": validation.passes_hard_gates,
    }
    _write_new(report_path, (canonical_json(summary) + "\n").encode())
    return summary


def _check_pinned_marginals(
    original: pl.DataFrame, candidate: pl.DataFrame, final: pl.DataFrame
) -> None:
    for frame in (candidate, final):
        counts = Counter(
            (row["age"], row["labour_market_status"])
            for row in frame.to_dicts()
            if row.get("age") in (20, 22)
        )
        base = Counter(
            (row["age"], row["labour_market_status"])
            for row in original.to_dicts()
            if row.get("age") in (20, 22)
        )
        if counts != base:
            raise ComposeError("Pinned exact-age status totals drifted")
    if "origin_country_code" not in original.columns or "sex" not in original.columns:
        raise ComposeError("Published origin marginal fields are unavailable")
    if "age" not in original.columns:
        raise ComposeError("Published exact-age field is unavailable")
    baseline = Counter(
        zip(
            original["origin_country_code"].to_list(),
            original["age"].to_list(),
            original["sex"].to_list(),
            strict=True,
        )
    )
    for frame in (candidate, final):
        if (
            Counter(
                zip(
                    frame["origin_country_code"].to_list(),
                    frame["age"].to_list(),
                    frame["sex"].to_list(),
                    strict=True,
                )
            )
            != baseline
        ):
            raise ComposeError("Published origin age-sex marginals drifted")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ComposeError("Expected JSON object")
    return value


def _write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o600)


if __name__ == "__main__":
    main()
