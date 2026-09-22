"""Build, validate, and optionally publish a persona dataset."""

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
from danish_personas.generation.report import validate_persona_pilot
from danish_personas.hydra_cli import enable_hydra_cli
from danish_personas.release.packager import package_release
from danish_personas.release.upload import upload_release
from danish_personas.release.verifier import verify_release
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
    """Build a validated dataset, optionally packaging and uploading its release.

    Raises:
        SystemExit:
            If configuration, generation, validation, or release checks fail.
    """
    configure_cli_logging()
    try:
        _run(config=config)
    except Exception as error:
        LOGGER.error("%s", error)
        raise SystemExit(1) from error


def _run(*, config: DictConfig) -> None:
    """Execute a dataset build from a composed Hydra configuration.

    Raises:
        ValueError:
            If generation, validation, or release checks fail.
    """
    script_config = load_script_config(
        config, section="build_dataset", model=BuildDatasetConfig
    )
    script_config.validate_release_evidence()
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
        LOGGER.info("Generation complete; validating persona dataset")
        report = validate_persona_pilot(pilot_dir=pilot_dir)
        if not report.passed:
            raise ValueError("Persona dataset failed validation")
        output_path = _merged_output_path(pilot_dir=pilot_dir)
        if script_config.hf_repo is not None:
            assert script_config.attestation is not None
            assert script_config.policy is not None
            assert script_config.dataset_card is not None
            assert script_config.licence is not None
            LOGGER.info("Validation passed; packaging release")
            result = package_release(
                pilot_dir=pilot_dir,
                attestation_path=script_config.attestation,
                policy_path=script_config.policy,
                dataset_card_path=script_config.dataset_card,
                licence_path=script_config.licence,
                repository_root=Path.cwd(),
                output_parent=script_config.output_dir / "releases",
            )
            verify_release(
                release_dir=result.path, expected_manifest_sha256=result.manifest_sha256
            )
            LOGGER.info("Release package verified; uploading dataset")
            upload_release(repo_id=script_config.hf_repo, release_dir=result.path)
    finally:
        progress.close()
    LOGGER.info("Persona dataset build completed")
    sys.stdout.write(f"{output_path}\n")


def _merged_output_path(*, pilot_dir: Path) -> Path:
    """Return the validated merged Parquet path.

    Raises:
        ValueError:
            If the merged output file is missing.
    """
    output_path = pilot_dir / "generated-personas.parquet"
    if not output_path.is_file():
        raise ValueError("Validated pilot output Parquet file is missing")
    return output_path


if __name__ == "__main__":
    load_repository_environment()
    main()
