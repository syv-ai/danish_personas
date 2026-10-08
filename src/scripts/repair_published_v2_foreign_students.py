"""Build a private release-only correction for the published v2 sample."""

from __future__ import annotations

import logging
from pathlib import Path

import click
import polars as pl

from danish_personas.environment import load_repository_environment
from danish_personas.release.foreign_student_repair import repair_published_v2

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option("--published-v2", required=True, type=click.Path(path_type=Path))
@click.option("--pre-hotfix-v2", required=True, type=click.Path(path_type=Path))
@click.option("--bundle", required=True, type=click.Path(path_type=Path))
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--private-manifest", required=True, type=click.Path(path_type=Path))
def main(
    published_v2: Path,
    pre_hotfix_v2: Path,
    bundle: Path,
    output: Path,
    private_manifest: Path,
) -> None:
    """Repair the release candidate without exposing row-level information.

    Raises:
        click.ClickException: If inputs fail a release repair gate.
    """
    load_repository_environment()
    try:
        summary = repair_published_v2(
            published_v2_path=published_v2,
            pre_hotfix_v2_path=pre_hotfix_v2,
            bundle_dir=bundle,
            output_path=output,
            private_manifest_path=private_manifest,
        )
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(click.format_filename(str(output)))
    click.echo(f"Repair summary: {summary}")


if __name__ == "__main__":
    main()
