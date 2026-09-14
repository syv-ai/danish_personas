"""Run source or demographic validation gates."""

import logging
from pathlib import Path

import click

from danish_personas.generation.report import validate_persona_run
from danish_personas.validation.checks import validate_demographics, validate_sources


@click.group()
def main() -> None:
    """Validate non-LLM pipeline artefacts."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@main.command()
@click.option("--run", "run_dir", type=click.Path(path_type=Path), required=True)
@click.option("--bundle", "bundle_dir", type=click.Path(path_type=Path), required=True)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=Path("config/validation.yaml"),
    show_default=True,
)
@click.option(
    "--categories",
    "categories_path",
    type=click.Path(path_type=Path),
    default=Path("config/categories.yaml"),
    show_default=True,
)
def demographics(
    run_dir: Path, bundle_dir: Path, config_path: Path, categories_path: Path
) -> None:
    """Validate a generated demographic and OCEAN run.

    Raises:
        click.ClickException:
            If any mandatory demographic validation gate fails.
    """
    report = validate_demographics(
        run_dir=run_dir,
        bundle_dir=bundle_dir,
        validation_config_path=config_path,
        categories_path=categories_path,
    )
    if not report.passed:
        raise click.ClickException("Demographic validation failed")


@main.command()
@click.option("--run", "run_dir", type=click.Path(path_type=Path), required=True)
def personas(run_dir: Path) -> None:
    """Validate a generated persona smoke run.

    Raises:
        click.ClickException:
            If any mandatory persona validation gate fails.
    """
    report = validate_persona_run(run_dir=run_dir)
    if not report.passed:
        raise click.ClickException("Persona validation failed")


@main.command()
@click.option("--bundle", "bundle_dir", type=click.Path(path_type=Path), required=True)
def sources(bundle_dir: Path) -> None:
    """Validate a prepared source bundle.

    Raises:
        click.ClickException:
            If any mandatory source validation gate fails.
    """
    report = validate_sources(bundle_dir=bundle_dir)
    if not report.passed:
        raise click.ClickException("Source validation failed")


if __name__ == "__main__":
    main()
