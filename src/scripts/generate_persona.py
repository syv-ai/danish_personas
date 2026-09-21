"""Generate and emit one validated Danish persona."""

import logging
import secrets
from pathlib import Path

import click
import polars as pl

from danish_personas.generation.pipeline import generate_personas
from danish_personas.generation.report import validate_persona_run
from danish_personas.io import sha256_file
from danish_personas.models import FrozenSampleManifest

DEFAULT_SAMPLE_DIR = Path("data/runs/statistical/55fb89fb303a67f0")
LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_SAMPLE_DIR / "text-development-seeds.parquet",
    show_default=True,
)
@click.option(
    "--sample-manifest",
    type=click.Path(path_type=Path),
    default=DEFAULT_SAMPLE_DIR / "text-development-seeds.manifest.json",
    show_default=True,
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=Path("config/config.yaml"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/personas"),
    show_default=True,
)
@click.option(
    "--offset",
    type=click.IntRange(min=0),
    default=None,
    help="Frozen-sample row to generate; omit to select one randomly.",
)
def main(
    input_path: Path,
    sample_manifest: Path,
    config_path: Path,
    output_dir: Path,
    offset: int | None,
) -> None:
    """Generate exactly one persona and write only its text to stdout.

    Raises:
        ValueError:
            If the completed run does not contain one persona.
        click.ClickException:
            If generation, upstream, guard, or validation checks fail.
    """
    _configure_logging()
    try:
        resolved_offset = _resolve_offset(
            input_path=input_path, sample_manifest_path=sample_manifest, offset=offset
        )
        run_dir = generate_personas(
            input_path=input_path,
            sample_manifest_path=sample_manifest,
            config_path=config_path,
            output_dir=output_dir,
            rows=1,
            offset=resolved_offset,
        )
        report = validate_persona_run(run_dir=run_dir)
        if not report.passed:
            raise ValueError("Generated persona failed validation")
        output = pl.read_parquet(run_dir / "generated-personas.parquet")
        if output.height != 1 or "persona" not in output.columns:
            raise ValueError(
                "Validated persona run did not contain exactly one persona"
            )
        persona = output.item(row=0, column="persona")
        if not isinstance(persona, str):
            raise ValueError("Validated persona text was not a string")
    except Exception as error:
        raise click.ClickException(str(error)) from error
    click.echo(persona)


def _configure_logging() -> None:
    """Configure diagnostics on stderr without polluting the persona output."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for logger_name in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _resolve_offset(
    *, input_path: Path, sample_manifest_path: Path, offset: int | None
) -> int:
    """Resolve an omitted offset from the current frozen sample locally.

    Args:
        input_path:
            Frozen sample Parquet file.
        sample_manifest_path:
            Frozen sample provenance manifest.
        offset (optional):
            Explicit zero-based sample offset, or ``None`` to sample one.

    Returns:
        Explicit or randomly selected zero-based sample offset.

    Raises:
        ValueError:
            If the manifest and sample do not agree on the row count or checksum.
    """
    if offset is not None:
        return offset

    manifest = FrozenSampleManifest.model_validate_json(
        sample_manifest_path.read_text(encoding="utf-8")
    )
    if manifest.data_file != Path(input_path.name):
        raise ValueError("Frozen sample filename does not match its manifest")
    row_count = pl.read_parquet(input_path).height
    if row_count != manifest.rows:
        raise ValueError("Frozen sample row count does not match its manifest")
    if sha256_file(input_path) != manifest.sha256:
        raise ValueError("Frozen sample checksum does not match its manifest")

    selected_offset = secrets.randbelow(row_count)
    LOGGER.info(
        "Selected frozen sample offset %s; pass --offset %s to replay it",
        selected_offset,
        selected_offset,
    )
    return selected_offset


if __name__ == "__main__":
    main()
