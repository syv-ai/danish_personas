"""Guarded placeholder for the deferred LLM generation phase."""

from pathlib import Path

import click

from danish_personas.io import load_yaml


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=Path("config/generation.yaml"),
    show_default=True,
)
def main(config_path: Path) -> None:
    """Refuse LLM generation until Phase 3 is explicitly enabled.

    Raises:
        click.ClickException:
            Always, because LLM generation is outside the implemented scope.
    """
    config = load_yaml(path=config_path)
    if config.get("llm_generation_enabled") is not True:
        raise click.ClickException(
            "LLM generation is disabled until the Phase 2 validation gate passes"
        )
    raise click.ClickException("No LLM provider is implemented in the non-LLM scope")


if __name__ == "__main__":
    main()
