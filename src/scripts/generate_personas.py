"""Run guarded two-stage persona generation against an OpenAI-compatible API."""

import logging
from pathlib import Path

import click

from danish_personas.generation.pipeline import generate_personas


@click.command()
@click.option("--input", "input_path", type=click.Path(path_type=Path), required=True)
@click.option("--sample-manifest", type=click.Path(path_type=Path), required=True)
@click.option("--config", "config_path", type=click.Path(path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--rows", type=click.IntRange(min=1), required=True)
@click.option("--live", is_flag=True, help="Explicitly authorise model requests.")
def main(
    input_path: Path,
    sample_manifest: Path,
    config_path: Path,
    output_dir: Path,
    rows: int,
    live: bool,
) -> None:
    """Plan or execute a smoke persona-generation run.

    Raises:
        click.ClickException:
            If an upstream, guard, API, schema, or safety check fails.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run_dir = generate_personas(
            input_path=input_path,
            sample_manifest_path=sample_manifest,
            config_path=config_path,
            output_dir=output_dir,
            rows=rows,
            live=live,
        )
    except Exception as error:
        raise click.ClickException(str(error)) from error
    logging.info("Persona generation run: %s", run_dir)


if __name__ == "__main__":
    main()
