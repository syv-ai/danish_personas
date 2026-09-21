"""Build, validate, and optionally publish a persona dataset."""

import logging
import sys
from pathlib import Path

import click
from tqdm import tqdm

from danish_personas.cli_logging import configure_cli_logging
from danish_personas.generation.pilot import run_pilot
from danish_personas.generation.report import validate_persona_pilot
from danish_personas.release.packager import package_release
from danish_personas.release.upload import upload_release
from danish_personas.release.verifier import verify_release
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
    default=Path("data/persona-datasets"),
    show_default=True,
)
@click.option("--rows", type=click.IntRange(min=1), required=True)
@click.option("--concurrency", type=click.IntRange(min=1, max=8), default=4)
@click.option("--request-limit", type=click.IntRange(min=1), required=True)
@click.option("--input-price-per-million", type=click.FloatRange(min=0), required=True)
@click.option("--output-price-per-million", type=click.FloatRange(min=0), required=True)
@click.option(
    "--hf-repo", default=None, help="Dataset repository to publish after review."
)
@click.option("--attestation", type=click.Path(path_type=Path), default=None)
@click.option("--policy", type=click.Path(path_type=Path), default=None)
@click.option("--dataset-card", type=click.Path(path_type=Path), default=None)
@click.option("--licence", type=click.Path(path_type=Path), default=None)
def main(
    input_path: Path | None,
    config_path: Path,
    output_dir: Path,
    rows: int,
    concurrency: int,
    request_limit: int,
    input_price_per_million: float,
    output_price_per_million: float,
    hf_repo: str | None,
    attestation: Path | None,
    policy: Path | None,
    dataset_card: Path | None,
    licence: Path | None,
) -> None:
    """Build a validated dataset, optionally packaging and uploading its release.

    Raises:
        ValueError:
            If generation or validation produces an invalid dataset.
        click.ClickException:
            If approval, generation, validation, packaging, or upload fails.
    """
    configure_cli_logging()
    LOGGER.info("Starting persona dataset build for %s row(s)", rows)
    if hf_repo is not None:
        _require_release_options(
            attestation=attestation,
            policy=policy,
            dataset_card=dataset_card,
            licence=licence,
        )
    if input_path is None:
        try:
            input_path, sample_manifest = prepare_standard_sample()
        except Exception as error:
            raise click.ClickException(str(error)) from error
    else:
        sample_manifest = input_path.with_suffix(".manifest.json")

    progress = tqdm(total=rows, unit="row", file=sys.stderr)
    try:
        LOGGER.info("Starting bounded persona generation")
        pilot_dir = run_pilot(
            input_path=input_path,
            sample_manifest_path=sample_manifest,
            config_path=config_path,
            output_dir=output_dir,
            rows=rows,
            batch_size=5,
            concurrency=concurrency,
            delay_between_batches=0.0,
            maximum_total_requests=request_limit,
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
            progress_callback=progress.update,
        )
        LOGGER.info("Generation complete; validating persona dataset")
        report = validate_persona_pilot(pilot_dir=pilot_dir)
        if not report.passed:
            raise ValueError("Persona dataset failed validation")
        output_path = _merged_output_path(pilot_dir=pilot_dir)
        if hf_repo is not None:
            assert attestation is not None
            assert policy is not None
            assert dataset_card is not None
            assert licence is not None
            LOGGER.info("Validation passed; packaging release")
            result = package_release(
                pilot_dir=pilot_dir,
                attestation_path=attestation,
                policy_path=policy,
                dataset_card_path=dataset_card,
                licence_path=licence,
                repository_root=Path.cwd(),
                output_parent=output_dir / "releases",
            )
            verify_release(
                release_dir=result.path, expected_manifest_sha256=result.manifest_sha256
            )
            LOGGER.info("Release package verified; uploading dataset")
            upload_release(repo_id=hf_repo, release_dir=result.path)
    except Exception as error:
        raise click.ClickException(str(error)) from error
    finally:
        progress.close()
    LOGGER.info("Persona dataset build completed")
    click.echo(output_path)


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


def _require_release_options(
    *,
    attestation: Path | None,
    policy: Path | None,
    dataset_card: Path | None,
    licence: Path | None,
) -> None:
    """Require every release-boundary input when publication is requested."""
    missing = [
        name
        for name, value in (
            ("--attestation", attestation),
            ("--policy", policy),
            ("--dataset-card", dataset_card),
            ("--licence", licence),
        )
        if value is None
    ]
    if missing:
        raise click.ClickException("--hf-repo requires: " + ", ".join(missing))


if __name__ == "__main__":
    main()
