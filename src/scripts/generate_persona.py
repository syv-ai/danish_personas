"""Generate and emit one schema-valid persona."""

import logging
import secrets
import sys
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig

from danish_personas.cli_logging import configure_cli_logging
from danish_personas.environment import load_repository_environment
from danish_personas.generation.config import persist_effective_generation_config
from danish_personas.generation.pipeline import generate_personas
from danish_personas.generation.policy import ChecksumValidationPolicy
from danish_personas.hydra_cli import enable_hydra_cli
from danish_personas.io import sha256_file
from danish_personas.models import FrozenSampleManifest
from danish_personas.script_config import (
    GeneratePersonaConfig,
    load_llm_config,
    load_script_config,
)
from danish_personas.workflows import prepare_standard_sample

LOGGER = logging.getLogger(__name__)
enable_hydra_cli()


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config: DictConfig) -> None:
    """Generate exactly one schema-valid persona and write only its text to stdout.

    Raises:
        SystemExit:
            If configuration or schema-only generation fails.
    """
    configure_cli_logging()
    try:
        _run(config=config)
    except Exception as error:
        LOGGER.error("%s", error)
        raise SystemExit(1) from error


def _run(*, config: DictConfig) -> None:
    """Execute single-persona generation from a composed Hydra configuration.

    Raises:
        ValueError:
            If generation produces a response that fails schema parsing.
    """
    script_config = load_script_config(
        config, section="generate_persona", model=GeneratePersonaConfig
    )
    llm_config = load_llm_config(config)
    config_path = persist_effective_generation_config(
        config=llm_config, output_dir=script_config.output_dir
    )
    LOGGER.info("Loading and validating persona inputs")
    input_path = script_config.input
    if input_path is None:
        input_path, sample_manifest = prepare_standard_sample(
            checksum_policy=ChecksumValidationPolicy.IGNORE
        )
    else:
        sample_manifest = input_path.with_suffix(".manifest.json")
    sampled_offset = _sample_offset(
        input_path=input_path,
        sample_manifest_path=sample_manifest,
        checksum_policy=ChecksumValidationPolicy.IGNORE,
    )
    invocation_output_dir = script_config.output_dir / secrets.token_hex(16)
    run_dir = generate_personas(
        input_path=input_path,
        sample_manifest_path=sample_manifest,
        config_path=config_path,
        output_dir=invocation_output_dir,
        rows=1,
        offset=sampled_offset,
        checksum_policy=ChecksumValidationPolicy.IGNORE,
    )
    LOGGER.info("Provider generation finished; reading schema-valid output")
    output = pl.read_parquet(run_dir / "generated-personas.parquet")
    if output.height != 1 or "persona" not in output.columns:
        raise ValueError("Schema-only persona run did not contain exactly one persona")
    persona = output.item(row=0, column="persona")
    if not isinstance(persona, str):
        raise ValueError("Schema-only persona text was not a string")
    LOGGER.info("Schema parsing passed; emitting persona text")
    sys.stdout.write(f"{persona}\n")


def _sample_offset(
    *,
    input_path: Path,
    sample_manifest_path: Path,
    checksum_policy: ChecksumValidationPolicy = ChecksumValidationPolicy.STRICT,
) -> int:
    """Sample one row from the current frozen demographic sample.

    Args:
        input_path:
            Frozen sample Parquet file.
        sample_manifest_path:
            Frozen sample provenance manifest.
        checksum_policy:
            Whether persisted checksum comparisons are strict. Defaults to strict.

    Returns:
        Randomly selected zero-based sample offset.

    Raises:
        ValueError:
            If the manifest and sample do not agree on the row count or checksum.
    """
    manifest = FrozenSampleManifest.model_validate_json(
        sample_manifest_path.read_text(encoding="utf-8"),
        context={"checksum_policy": checksum_policy},
    )
    if manifest.data_file != Path(input_path.name):
        raise ValueError("Frozen sample filename does not match its manifest")
    row_count = pl.read_parquet(input_path).height
    if row_count != manifest.rows:
        raise ValueError("Frozen sample row count does not match its manifest")
    if (
        checksum_policy.validates_checksums
        and sha256_file(input_path) != manifest.sha256
    ):
        raise ValueError("Frozen sample checksum does not match its manifest")
    return secrets.randbelow(row_count)


if __name__ == "__main__":
    load_repository_environment()
    main()
