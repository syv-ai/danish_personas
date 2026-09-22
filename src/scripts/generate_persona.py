"""Generate and emit one validated Danish persona."""

import logging
import secrets
from pathlib import Path

import click
import polars as pl

from danish_personas.cli_logging import configure_cli_logging
from danish_personas.generation.pipeline import generate_personas
from danish_personas.generation.policy import ChecksumValidationPolicy
from danish_personas.generation.report import validate_persona_run
from danish_personas.io import sha256_file
from danish_personas.models import FrozenSampleManifest
from danish_personas.workflows import prepare_standard_sample

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Frozen sample Parquet path. Prepare the standard sample when omitted.",
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
def main(input_path: Path | None, config_path: Path, output_dir: Path) -> None:
    """Generate exactly one persona and write only its text to stdout.

    Raises:
        ValueError:
            If the completed run does not contain one persona.
        click.ClickException:
            If generation, upstream, guard, or validation checks fail.
    """
    configure_cli_logging()
    LOGGER.info("Loading and validating persona inputs")
    try:
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
        invocation_output_dir = output_dir / secrets.token_hex(16)
        run_dir = generate_personas(
            input_path=input_path,
            sample_manifest_path=sample_manifest,
            config_path=config_path,
            output_dir=invocation_output_dir,
            rows=1,
            offset=sampled_offset,
            checksum_policy=ChecksumValidationPolicy.IGNORE,
        )
        LOGGER.info("Provider generation finished; validating generated output")
        report = validate_persona_run(
            run_dir=run_dir, checksum_policy=ChecksumValidationPolicy.IGNORE
        )
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
        LOGGER.info("Persona validation passed; emitting validated text")
    except Exception as error:
        raise click.ClickException(str(error)) from error
    click.echo(persona)


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
    main()
