"""Compose a private candidate from a completed v3 review campaign."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from typing import Any

import click
import httpx
import polars as pl

from danish_personas.environment import load_repository_environment
from danish_personas.generation.models import GenerationConfig
from danish_personas.generation.proxy_budget import (
    SOL_ADJUDICATION_LEDGER_MAX_TOKENS,
    SOL_ADJUDICATION_MODEL,
    V3_LONG_ADJUDICATION_PURPOSE,
    V3_TARGETED_FOLLOWUP_PURPOSE,
    ProxyBudget,
)
from danish_personas.generation.sol_adjudication import (
    SolAdjudicationResponse,
    run_sol_adjudication,
)
from danish_personas.io import canonical_json, sha256_file, sha256_text
from danish_personas.release.candidate_validation import validate_release_candidate
from scripts import adjudicate_persona_v3 as campaign

ROOT = Path("/tmp/danish-personas-audit")
DEFAULT_ORIGINAL = (
    ROOT / "hf-v2-foreign-hotfix-remote/data/train-00000-of-00001.parquet"
)
DEFAULT_CANDIDATE = ROOT / "v3-private/demographics-ready.parquet"
DEFAULT_REPAIR = ROOT / "v3-private/demographics-ready.manifest.json"
DEFAULT_CAMPAIGN = ROOT / "v3-private/campaign-long"
DEFAULT_FOLLOWUP = ROOT / "v3-private/campaign-followup"
DEFAULT_OUTPUT = ROOT / "v3-private/composed-candidate.parquet"
DEFAULT_PATCHES = ROOT / "v3-private/composed-candidate.vetted.json"
DEFAULT_REPORT = ROOT / "v3-private/composed-candidate.report.json"
DEFAULT_PREHOTFIX = ROOT / "hf-v2-final-verify/data/train-00000-of-00001.parquet"
PREHOTFIX_SHA256 = "02d96101eaa0d4e21485e7ef788489a86d45e06933910d32e205175a40cc675b"


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
@click.option(
    "--followup-dir", type=click.Path(path_type=Path), default=DEFAULT_FOLLOWUP
)
@click.option("--output", type=click.Path(path_type=Path), default=DEFAULT_OUTPUT)
@click.option(
    "--patch-manifest", type=click.Path(path_type=Path), default=DEFAULT_PATCHES
)
@click.option("--report", type=click.Path(path_type=Path), default=DEFAULT_REPORT)
@click.option(
    "--bundle",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--prehotfix",
    type=click.Path(path_type=Path),
    default=DEFAULT_PREHOTFIX,
    show_default=True,
)
def main(
    original: Path,
    candidate: Path,
    repair_manifest: Path,
    campaign_dir: Path,
    followup_dir: Path,
    output: Path,
    patch_manifest: Path,
    report: Path,
    bundle: Path,
    prehotfix: Path,
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
            followup_dir,
            output,
            patch_manifest,
            report,
            bundle,
            prehotfix,
        )
    except (ComposeError, OSError, ValueError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result, sort_keys=True))


def compose(  # noqa: C901, PLR0912
    original: Path,
    candidate: Path,
    repair_path: Path,
    campaign_dir: Path,
    followup_dir: Path,
    output: Path,
    patch_path: Path,
    report_path: Path,
    bundle_dir: Path,
    prehotfix_path: Path,
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
    if not bundle_dir.is_dir():
        raise ComposeError("Source bundle path is unavailable")
    if sha256_file(prehotfix_path) != PREHOTFIX_SHA256:
        raise ComposeError("Pinned pre-hotfix v2 baseline checksum mismatch")
    original_frame, candidate_frame, prehotfix_frame = (
        pl.read_parquet(original),
        pl.read_parquet(candidate),
        pl.read_parquet(prehotfix_path),
    )
    _check_prehotfix_alignment(original_frame, prehotfix_frame)
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
    base_unresolved = {
        item.get("persona_hash")
        for item in processed
        if item.get("disposition")
        in {"unresolved", "validation_failed", "privacy_blocked"}
    }
    followup_manifest = _json(followup_dir / "manifest.json")
    followup_status = _json(followup_dir / "status.json")
    followup_items = followup_status.get("processed")
    if (
        followup_manifest.get("campaign")
        != "persona-sol-adjudication-v3-targeted-followup"
        or followup_status.get("campaign") != followup_manifest.get("campaign")
        or followup_status.get("input_hashes") != followup_manifest.get("input_hashes")
        or followup_status.get("progress", {}).get("pending") != 0
        or followup_status.get("progress", {}).get("completed")
        != followup_manifest.get("selected_total")
        or not isinstance(followup_items, list)
    ):
        raise ComposeError("Targeted follow-up is incomplete or unbound")
    followup_by_hash = {item.get("persona_hash"): item for item in followup_items}
    if (
        len(followup_by_hash) != len(followup_items)
        or set(followup_by_hash) != base_unresolved
    ):
        raise ComposeError("Follow-up does not cover exactly the unresolved base rows")
    follow_inputs = followup_manifest.get("input_hashes")
    if not isinstance(follow_inputs, dict):
        raise ComposeError("Follow-up input bindings are missing")
    follow_expected = {
        "base_manifest_sha256": sha256_file(campaign_dir / "manifest.json"),
        "base_status_sha256": sha256_file(campaign_dir / "status.json"),
        "candidate_sha256": sha256_file(candidate),
        "original_sha256": sha256_file(original),
        "ordered_id_hashes_sha256": ordered_hash,
    }
    if any(follow_inputs.get(key) != value for key, value in follow_expected.items()):
        raise ComposeError("Follow-up source bindings do not match composition inputs")
    if follow_inputs.get("queue_sha256") != sha256_text(
        canonical_json(sorted(hashes.index(digest) for digest in base_unresolved))
    ):
        raise ComposeError("Follow-up queue binding mismatch")
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
            "timeout_seconds": 3600.0,
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
        schema_hash=sha256_text(
            canonical_json(SolAdjudicationResponse.provider_json_schema())
        ),
        model=SOL_ADJUDICATION_MODEL,
        input_usd_per_million="0.1",
        output_usd_per_million="0.5",
        max_tokens=SOL_ADJUDICATION_LEDGER_MAX_TOKENS,
        cap_usd=Decimal("1"),
        uncapped=True,
        uncapped_purpose=V3_LONG_ADJUDICATION_PURPOSE,
    )
    follow_budget = ProxyBudget(
        registry_path=campaign.REGISTRY,
        campaign=followup_manifest["campaign"],
        source_hash=sha256_text(canonical_json(follow_inputs)),
        prompt_hash=follow_inputs["prompt_sha256"],
        schema_hash=sha256_text(
            canonical_json(SolAdjudicationResponse.provider_json_schema())
        ),
        model=SOL_ADJUDICATION_MODEL,
        input_usd_per_million="0.1",
        output_usd_per_million="0.5",
        max_tokens=SOL_ADJUDICATION_LEDGER_MAX_TOKENS,
        cap_usd=Decimal("1"),
        uncapped=True,
        uncapped_purpose=V3_TARGETED_FOLLOWUP_PURPOSE,
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
        if disposition in {"unresolved", "privacy_blocked", "validation_failed"}:
            follow = followup_by_hash[digest]
            follow_disposition = follow.get("disposition")
            if follow.get("base_result_sha256") != item.get("result_sha256"):
                raise ComposeError("Follow-up does not bind its base disposition")
            if disposition == "privacy_blocked":
                if follow_disposition != "privacy_blocked_local_only":
                    raise ComposeError("Privacy-blocked row was not retained locally")
            elif follow_disposition not in {
                "patched",
                "consistent",
                "unresolved",
                "editorial_retention_no_valid_verdict",
            }:
                raise ComposeError("Follow-up disposition cannot be accepted")
            if follow_disposition == "editorial_retention_no_valid_verdict":
                if follow.get("evidence_sha256") not in {
                    sha256_text("SolAdjudicationError"),
                    sha256_text("ProxyBudgetError"),
                }:
                    raise ComposeError(
                        "Editorial retention lacks bounded failure evidence"
                    )
        if disposition == "validation_failed":
            campaign._verify_validation_failure(
                output_dir=campaign_dir, digest=digest, input_hashes=inputs
            )
            follow = followup_by_hash[digest]
            if follow.get("disposition") in {"patched", "consistent", "unresolved"}:
                old, row = original_rows[index], rows[index]
                follow_cp_name = follow.get("checkpoint")
                follow_cp = (
                    followup_dir / follow_cp_name
                    if isinstance(follow_cp_name, str)
                    else Path("/")
                )
                if (
                    not follow_cp.is_relative_to(followup_dir)
                    or not follow_cp.is_file()
                    or sha256_file(follow_cp) != follow.get("checkpoint_sha256")
                ):
                    raise ComposeError("Follow-up checkpoint binding mismatch")
                follow_doc = _json(follow_cp)
                if follow_doc.get("response_sha256") != follow.get("response_sha256"):
                    raise ComposeError("Follow-up response binding mismatch")
                hints = {
                    field: {"old": old[field], "new": row[field]}
                    for field in sorted(
                        (set(old) & set(row)) & campaign.SOL_ALLOWED_FACT_FIELDS
                    )
                    if old[field] != row[field]
                }
                verified_follow = run_sol_adjudication(
                    original_persona=row[campaign.TEXT_FIELD],
                    candidate_row=row,
                    prompt=prompt,
                    config=config,
                    budget=follow_budget,
                    checkpoint_path=follow_cp,
                    transport=httpx.MockTransport(
                        lambda request: httpx.Response(500, request=request)
                    ),
                    changed_fact_hints=hints,
                    original_row=old,
                    adjudication_mode="unresolved_followup",
                )
                if verified_follow.disposition != follow["disposition"]:
                    raise ComposeError("Follow-up disposition verification failed")
                if follow["disposition"] == "patched":
                    if verified_follow.proposed_text == row[campaign.TEXT_FIELD]:
                        raise ComposeError("Follow-up patch produced no text change")
                    row[campaign.TEXT_FIELD] = verified_follow.proposed_text
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
            if (
                not isinstance(verified.proposed_text, str)
                or verified.proposed_text == row[campaign.TEXT_FIELD]
            ):
                raise ComposeError(
                    "Patched verdict did not produce an exact text change"
                )
            row[campaign.TEXT_FIELD] = verified.proposed_text
        if disposition in {"unresolved", "validation_failed"}:
            follow = followup_by_hash[digest]
            if follow.get("disposition") in {"patched", "consistent", "unresolved"}:
                follow_cp_name = follow.get("checkpoint")
                follow_cp = (
                    followup_dir / follow_cp_name
                    if isinstance(follow_cp_name, str)
                    else Path("/")
                )
                if (
                    not follow_cp.is_relative_to(followup_dir)
                    or not follow_cp.is_file()
                    or sha256_file(follow_cp) != follow.get("checkpoint_sha256")
                ):
                    raise ComposeError("Follow-up checkpoint binding mismatch")
                follow_doc = _json(follow_cp)
                if follow_doc.get("response_sha256") != follow.get("response_sha256"):
                    raise ComposeError("Follow-up response binding mismatch")
                follow_verified = run_sol_adjudication(
                    original_persona=row[campaign.TEXT_FIELD],
                    candidate_row=row,
                    prompt=prompt,
                    config=config,
                    budget=follow_budget,
                    checkpoint_path=follow_cp,
                    transport=httpx.MockTransport(
                        lambda request: httpx.Response(500, request=request)
                    ),
                    changed_fact_hints=hints,
                    original_row=old,
                    adjudication_mode="unresolved_followup",
                )
                if follow_verified.disposition != follow["disposition"]:
                    raise ComposeError("Follow-up disposition verification failed")
                if follow["disposition"] == "patched":
                    if follow_verified.proposed_text == row[campaign.TEXT_FIELD]:
                        raise ComposeError("Follow-up patch produced no text change")
                    row[campaign.TEXT_FIELD] = follow_verified.proposed_text
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
    _check_pinned_marginals(original_frame, candidate_frame, final, prehotfix_frame)
    buffer = BytesIO()
    final.write_parquet(file=buffer)
    parquet = buffer.getvalue()
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
        bundle_dir=bundle_dir,
        vetted_patch_manifest_path=patch_path,
    )
    if not validation.passes_hard_gates:
        output.unlink(missing_ok=True)
        patch_path.unlink(missing_ok=True)
        raise ComposeError("Release candidate failed mandatory validation gates")
    written_final = pl.read_parquet(output)
    text_changes = sum(
        before != after
        for before, after in zip(
            candidate_frame[campaign.TEXT_FIELD].to_list(),
            written_final[campaign.TEXT_FIELD].to_list(),
            strict=True,
        )
    )
    fixed_rows = sum(
        before != after
        for before, after in zip(
            original_frame.to_dicts(), written_final.to_dicts(), strict=True
        )
    )
    followup_counts = Counter(item["disposition"] for item in followup_items)
    summary = {
        "final_outcomes": {"good": final.height - fixed_rows, "fixed": fixed_rows},
        "historical_review_evidence": {
            "initial_dispositions": dict(dispositions),
            "followup_dispositions": dict(followup_counts),
        },
        "prose_patched_rows": text_changes,
        "candidate_sha256": candidate_hash,
        "validation_passes_hard_gates": validation.passes_hard_gates,
    }
    _write_new(report_path, (canonical_json(summary) + "\n").encode())
    return summary


def _check_prehotfix_alignment(
    published: pl.DataFrame, prehotfix: pl.DataFrame
) -> None:
    if (
        published.height != campaign.EXPECTED_ROWS
        or prehotfix.height != campaign.EXPECTED_ROWS
    ):
        raise ComposeError("Pre-hotfix baseline row count mismatch")
    allowed = {"detailed_status_code", "detailed_status"}
    published_rows, prehotfix_rows = published.to_dicts(), prehotfix.to_dicts()
    if any(
        {key: value for key, value in old.items() if key not in allowed}
        != {key: value for key, value in new.items() if key not in allowed}
        for old, new in zip(published_rows, prehotfix_rows, strict=True)
    ):
        raise ComposeError("Pre-hotfix baseline differs beyond detailed status fields")


def _check_pinned_marginals(
    original: pl.DataFrame,
    candidate: pl.DataFrame,
    final: pl.DataFrame,
    prehotfix: pl.DataFrame,
) -> None:
    fields = (
        "age",
        "sex",
        "labour_market_status",
        "detailed_status_code",
        "detailed_status",
    )
    if any(field not in prehotfix.columns for field in fields):
        raise ComposeError("Pre-hotfix detailed status marginals are unavailable")
    baseline_counts = Counter(
        tuple(row[field] for field in fields) for row in prehotfix.to_dicts()
    )
    for frame in (candidate, final):
        if (
            any(field not in frame.columns for field in fields)
            or Counter(
                tuple(row[field] for field in fields) for row in frame.to_dicts()
            )
            != baseline_counts
        ):
            raise ComposeError("Pinned exact-age detailed status totals drifted")
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
    load_repository_environment()
    main()
