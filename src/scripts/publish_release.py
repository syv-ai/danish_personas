"""Publish a validated persona pilot as a committed release artefact."""

import logging
import shutil
from pathlib import Path

import click

from danish_personas.generation.models import PilotManifest
from danish_personas.io import sha256_file, write_json
from danish_personas.models import ValidationReport

RELEASE_ROOT = Path("data/releases")


@click.command()
@click.option("--pilot", "pilot_dir", type=click.Path(path_type=Path), required=True)
@click.option("--name", required=True, help="Release directory name.")
@click.option(
    "--reviewed-by",
    "reviewed_by",
    required=True,
    help="Who completed the human review the privacy register requires.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=RELEASE_ROOT,
    show_default=True,
)
def main(pilot_dir: Path, name: str, reviewed_by: str, output_dir: Path) -> None:
    """Copy a passed pilot into the tracked release directory with its provenance.

    Generated personas are otherwise ignored by Git. A release is a deliberate act, so
    this refuses a pilot whose own validation did not pass, and records who reviewed it.

    Raises:
        click.ClickException:
            If the pilot is missing, incomplete, or did not pass validation.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest_path = pilot_dir / "pilot-manifest.json"
    report_path = pilot_dir / "pilot-validation-report.json"
    if not manifest_path.is_file() or not report_path.is_file():
        message = f"Not a completed pilot: {pilot_dir}"
        raise click.ClickException(message)
    manifest = PilotManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    report = ValidationReport.model_validate_json(
        report_path.read_text(encoding="utf-8")
    )
    if not report.passed:
        message = f"Pilot {manifest.pilot_id} did not pass validation"
        raise click.ClickException(message)
    source = pilot_dir / manifest.output_file
    release_dir = output_dir / name
    release_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, release_dir / "personas.parquet")
    shutil.copy2(manifest_path, release_dir / "pilot-manifest.json")
    shutil.copy2(report_path, release_dir / "pilot-validation-report.json")
    write_json(
        path=release_dir / "release-manifest.json",
        payload={
            "name": name,
            "pilot_id": manifest.pilot_id,
            "model": manifest.model,
            "rows": manifest.generated_rows,
            "skipped_persona_ids": manifest.skipped_persona_ids,
            "reviewed_by": reviewed_by,
            "files": {
                path.name: sha256_file(path)
                for path in sorted(release_dir.glob("*"))
                if path.name != "release-manifest.json"
            },
        },
    )
    logging.info("Published %s rows to %s", manifest.generated_rows, release_dir)


if __name__ == "__main__":
    main()
