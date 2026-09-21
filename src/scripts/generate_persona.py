"""Generate and emit one validated Danish persona."""

import logging
from pathlib import Path

import click
import polars as pl

from danish_personas.generation.pipeline import generate_personas
from danish_personas.generation.report import validate_persona_run


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path),
    default=Path("data/text-development-seeds.parquet"),
    show_default=True,
)
@click.option(
    "--sample-manifest",
    type=click.Path(path_type=Path),
    default=Path("data/text-development-seeds.manifest.json"),
    show_default=True,
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=Path("config/generation.local.yaml"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/personas"),
    show_default=True,
)
@click.option("--offset", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--live", is_flag=True, help="Explicitly authorise model requests.")
def main(
    input_path: Path,
    sample_manifest: Path,
    config_path: Path,
    output_dir: Path,
    offset: int,
    live: bool,
) -> None:
    """Generate exactly one persona and write only its text to stdout.

    Raises:
        ValueError:
            If the completed run does not contain one persona.
        click.ClickException:
            If generation, upstream, guard, or validation checks fail.
    """
    _configure_logging()
    if not live:
        raise click.ClickException(
            "Persona generation requires explicit --live approval"
        )
    try:
        run_dir = generate_personas(
            input_path=input_path,
            sample_manifest_path=sample_manifest,
            config_path=config_path,
            output_dir=output_dir,
            rows=1,
            offset=offset,
            live=live,
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


if __name__ == "__main__":
    main()
