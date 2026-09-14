"""Generate deterministic demographics and OCEAN traits without an LLM."""

import logging
from pathlib import Path

import click

from danish_personas.sampling.generator import generate_records


@click.command()
@click.option("--bundle", "bundle_dir", type=click.Path(path_type=Path), required=True)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=Path("config/sampling.yaml"),
    show_default=True,
)
@click.option("--rows", type=click.IntRange(min=1), required=True)
@click.option("--seed", type=int, required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
def main(
    bundle_dir: Path, config_path: Path, rows: int, seed: int, output_dir: Path
) -> None:
    """Generate one reproducible Phase 2 run."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_dir = generate_records(
        bundle_dir=bundle_dir,
        sampling_config_path=config_path,
        output_dir=output_dir,
        rows=rows,
        seed=seed,
    )
    logging.info("Generated run: %s", run_dir)


if __name__ == "__main__":
    main()
