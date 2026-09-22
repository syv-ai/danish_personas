"""Build and optionally publish a persona dataset."""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from danish_personas.cli_logging import configure_cli_logging
from danish_personas.environment import load_repository_environment
from danish_personas.generation.config import persist_effective_generation_config
from danish_personas.generation.pilot import run_pilot
from danish_personas.hydra_cli import enable_hydra_cli
from danish_personas.publishing import upload_dataset
from danish_personas.script_config import (
    BuildDatasetConfig,
    load_llm_config,
    load_script_config,
)
from danish_personas.workflows import prepare_standard_sample

LOGGER = logging.getLogger(__name__)
enable_hydra_cli()


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config: DictConfig) -> None:
    """Build a dataset and optionally upload it.

    Raises:
        SystemExit:
            If configuration, generation, or upload fails.
    """
    configure_cli_logging()
    try:
        _run(config=config)
    except Exception as error:
        LOGGER.error("%s", error)
        raise SystemExit(1) from error


def _run(*, config: DictConfig) -> None:
    """Execute a dataset build from a composed Hydra configuration."""
    script_config = load_script_config(
        config, section="build_dataset", model=BuildDatasetConfig
    )
    llm_config = load_llm_config(config)
    config_path = persist_effective_generation_config(
        config=llm_config, output_dir=script_config.output_dir
    )
    LOGGER.info("Starting persona dataset build for %s row(s)", script_config.rows)
    input_path = script_config.input
    if input_path is None:
        input_path, sample_manifest = prepare_standard_sample()
    else:
        sample_manifest = input_path.with_suffix(".manifest.json")

    progress = tqdm(total=script_config.rows, unit="row", file=sys.stderr)
    try:
        LOGGER.info("Starting bounded persona generation")
        pilot_dir = run_pilot(
            input_path=input_path,
            sample_manifest_path=sample_manifest,
            config_path=config_path,
            output_dir=script_config.output_dir,
            rows=script_config.rows,
            batch_size=5,
            concurrency=script_config.concurrency,
            delay_between_batches=0.0,
            maximum_total_requests=script_config.request_limit,
            input_price_per_million=0.0,
            output_price_per_million=0.0,
            progress_callback=progress.update,
        )
        output_path = _merged_output_path(pilot_dir=pilot_dir)
        if script_config.hf_repo is not None:
            LOGGER.info("Generation complete; uploading dataset")
            upload_dataset(repo_id=script_config.hf_repo, parquet_path=output_path)
    finally:
        progress.close()
    LOGGER.info("Persona dataset build completed")
    sys.stdout.write(f"{output_path}\n")


def _merged_output_path(*, pilot_dir: Path) -> Path:
    """Return the merged Parquet path.

    Raises:
        ValueError:
            If the merged output file is missing.
    """
    output_path = pilot_dir / "generated-personas.parquet"
    if not output_path.is_file():
        raise ValueError("Generated dataset Parquet file is missing")
    return output_path


if __name__ == "__main__":
    load_repository_environment()
    main()
