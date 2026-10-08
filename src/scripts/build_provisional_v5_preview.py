"""Build an offline merged provisional v5 preview."""

from __future__ import annotations

import json
from pathlib import Path

import click

from danish_personas.cli_logging import configure_cli_logging
from danish_personas.environment import load_repository_environment
from danish_personas.generation.proxy_budget import ProxyBudgetError
from danish_personas.generation.proxy_patch_verifier import ProxyPatchVerificationError
from danish_personas.release.provisional_v5_preview import (
    DEFAULT_ORIGINAL,
    DEFAULT_OUTPUT,
    DEFAULT_V4_PROSE_PREVIEW,
    DEFAULT_V4_STRUCTURED,
    DEFAULT_V5_H90,
    ProvisionalV5PreviewError,
    build_provisional_v5_preview,
)
from scripts import verify_persona_patches as verify
from scripts.build_prose_review_v4_dashboard import ProseReviewV4DashboardError


@click.command()
@click.option("--original", type=click.Path(path_type=Path), default=DEFAULT_ORIGINAL)
@click.option(
    "--v4-structured", type=click.Path(path_type=Path), default=DEFAULT_V4_STRUCTURED
)
@click.option("--v5-h90", type=click.Path(path_type=Path), default=DEFAULT_V5_H90)
@click.option(
    "--v4-prose-preview",
    type=click.Path(path_type=Path),
    default=DEFAULT_V4_PROSE_PREVIEW,
)
@click.option(
    "--v4-triage", type=click.Path(path_type=Path), default=verify.DEFAULT_TRIAGE
)
@click.option(
    "--v4-first-prompt",
    type=click.Path(path_type=Path),
    default=verify.DEFAULT_FIRST_PASS_PROMPT,
)
@click.option(
    "--verify-prompt",
    type=click.Path(path_type=Path),
    default=verify.DEFAULT_VERIFY_PROMPT,
)
@click.option(
    "--registry", type=click.Path(path_type=Path), default=verify.DEFAULT_REGISTRY
)
@click.option(
    "--v4-first-status",
    type=click.Path(path_type=Path),
    default=verify.DEFAULT_FIRST_PASS_STATUS,
)
@click.option(
    "--v4-first-manifest",
    type=click.Path(path_type=Path),
    default=verify.DEFAULT_FIRST_PASS_MANIFEST,
)
@click.option(
    "--v4-first-checkpoint-root",
    type=click.Path(path_type=Path),
    default=verify.DEFAULT_FIRST_PASS_DIR,
)
@click.option(
    "--v4-second-status",
    type=click.Path(path_type=Path),
    default=verify.DEFAULT_OUTPUT_DIR / "status.json",
)
@click.option(
    "--v4-second-manifest",
    type=click.Path(path_type=Path),
    default=verify.DEFAULT_OUTPUT_DIR / "manifest.json",
)
@click.option(
    "--v4-second-checkpoint-root",
    type=click.Path(path_type=Path),
    default=verify.DEFAULT_OUTPUT_DIR,
)
@click.option(
    "--h90-triage", type=click.Path(path_type=Path), default=verify.DEFAULT_H90_TRIAGE
)
@click.option(
    "--h90-first-prompt",
    type=click.Path(path_type=Path),
    default=verify.DEFAULT_H90_FIRST_PASS_PROMPT,
)
@click.option(
    "--h90-first-status",
    type=click.Path(path_type=Path),
    default=verify.DEFAULT_H90_FIRST_PASS_STATUS,
)
@click.option(
    "--h90-first-manifest",
    type=click.Path(path_type=Path),
    default=verify.DEFAULT_H90_FIRST_PASS_MANIFEST,
)
@click.option(
    "--h90-first-checkpoint-root",
    type=click.Path(path_type=Path),
    default=verify.DEFAULT_H90_FIRST_PASS_DIR,
)
@click.option(
    "--h90-second-status",
    type=click.Path(path_type=Path),
    default=verify.DEFAULT_H90_OUTPUT_DIR / "status.json",
)
@click.option(
    "--h90-second-manifest",
    type=click.Path(path_type=Path),
    default=verify.DEFAULT_H90_OUTPUT_DIR / "manifest.json",
)
@click.option(
    "--h90-second-checkpoint-root",
    type=click.Path(path_type=Path),
    default=verify.DEFAULT_H90_OUTPUT_DIR,
)
@click.option("--output", type=click.Path(path_type=Path), default=DEFAULT_OUTPUT)
@click.option("--write", "write_output", is_flag=True, default=False)
def main(
    original: Path,
    v4_structured: Path,
    v5_h90: Path,
    v4_prose_preview: Path,
    v4_triage: Path,
    v4_first_prompt: Path,
    verify_prompt: Path,
    registry: Path,
    v4_first_status: Path,
    v4_first_manifest: Path,
    v4_first_checkpoint_root: Path,
    v4_second_status: Path,
    v4_second_manifest: Path,
    v4_second_checkpoint_root: Path,
    h90_triage: Path,
    h90_first_prompt: Path,
    h90_first_status: Path,
    h90_first_manifest: Path,
    h90_first_checkpoint_root: Path,
    h90_second_status: Path,
    h90_second_manifest: Path,
    h90_second_checkpoint_root: Path,
    output: Path,
    write_output: bool,
) -> None:
    """Build or dry-run the merged provisional v5 preview.

    Raises:
        click.ClickException:
            If inputs, checkpoints, or write targets are unsafe.
    """
    configure_cli_logging()
    try:
        summary = build_provisional_v5_preview(
            original=original,
            v4_structured=v4_structured,
            v5_h90=v5_h90,
            v4_prose_preview=v4_prose_preview,
            v4_triage=v4_triage,
            v4_first_prompt=v4_first_prompt,
            verify_prompt=verify_prompt,
            registry=registry,
            v4_first_status=v4_first_status,
            v4_first_manifest=v4_first_manifest,
            v4_first_checkpoint_root=v4_first_checkpoint_root,
            v4_second_status=v4_second_status,
            v4_second_manifest=v4_second_manifest,
            v4_second_checkpoint_root=v4_second_checkpoint_root,
            h90_triage=h90_triage,
            h90_first_prompt=h90_first_prompt,
            h90_first_status=h90_first_status,
            h90_first_manifest=h90_first_manifest,
            h90_first_checkpoint_root=h90_first_checkpoint_root,
            h90_second_status=h90_second_status,
            h90_second_manifest=h90_second_manifest,
            h90_second_checkpoint_root=h90_second_checkpoint_root,
            output=output,
            write_output=write_output,
        )
    except (
        ProvisionalV5PreviewError,
        ProseReviewV4DashboardError,
        verify.PatchVerificationCampaignError,
        ProxyBudgetError,
        ProxyPatchVerificationError,
        ValueError,
        OSError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    load_repository_environment()
    main()
