"""Build, validate, and optionally publish a persona dataset."""

import logging
import sys
from pathlib import Path

import click
from tqdm import tqdm

from danish_personas.generation.pilot import run_pilot
from danish_personas.generation.report import validate_persona_pilot
from danish_personas.release.packager import package_release
from danish_personas.release.upload import upload_release
from danish_personas.release.verifier import verify_release

DEFAULT_SAMPLE_DIR = Path("data/runs/statistical/55fb89fb303a67f0")


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_SAMPLE_DIR / "text-development-seeds.parquet",
    show_default=True,
)
@click.option(
    "--sample-manifest",
    type=click.Path(path_type=Path),
    default=DEFAULT_SAMPLE_DIR / "text-development-seeds.manifest.json",
    show_default=True,
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=Path("config.yaml"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/persona-datasets"),
    show_default=True,
)
@click.option("--rows", type=click.IntRange(min=1), required=True)
@click.option("--batch-size", type=click.IntRange(min=1, max=5), default=5)
@click.option("--concurrency", type=click.IntRange(min=1, max=8), default=4)
@click.option("--delay-between-batches", type=click.FloatRange(min=0), default=0.0)
@click.option("--maximum-total-requests", type=click.IntRange(min=1), required=True)
@click.option("--input-price-per-million", type=click.FloatRange(min=0), required=True)
@click.option("--output-price-per-million", type=click.FloatRange(min=0), required=True)
@click.option(
    "--hf-repo", default=None, help="Dataset repository to publish after review."
)
@click.option(
    "--attestation", "--attestation-path", type=click.Path(path_type=Path), default=None
)
@click.option(
    "--policy",
    "--release-policy",
    "--policy-path",
    type=click.Path(path_type=Path),
    default=None,
)
@click.option(
    "--dataset-card",
    "--dataset-card-path",
    type=click.Path(path_type=Path),
    default=None,
)
@click.option(
    "--licence",
    "--license",
    "--licence-path",
    type=click.Path(path_type=Path),
    default=None,
)
@click.option("--repository-root", type=click.Path(path_type=Path), default=None)
@click.option(
    "--release-output-parent",
    "--release-output-dir",
    "--output-parent",
    type=click.Path(path_type=Path),
    default=None,
)
def main(
    input_path: Path,
    sample_manifest: Path,
    config_path: Path,
    output_dir: Path,
    rows: int,
    batch_size: int,
    concurrency: int,
    delay_between_batches: float,
    maximum_total_requests: int,
    input_price_per_million: float,
    output_price_per_million: float,
    hf_repo: str | None,
    attestation: Path | None,
    policy: Path | None,
    dataset_card: Path | None,
    licence: Path | None,
    repository_root: Path | None,
    release_output_parent: Path | None,
) -> None:
    """Build a validated dataset, optionally packaging and uploading its release.

    Raises:
        ValueError:
            If generation or validation produces an invalid dataset.
        click.ClickException:
            If approval, generation, validation, packaging, or upload fails.
    """
    _configure_logging()
    release_paths: tuple[Path, Path, Path, Path, Path, Path] | None = None
    if hf_repo is not None:
        release_paths = _require_release_options(
            attestation=attestation,
            policy=policy,
            dataset_card=dataset_card,
            licence=licence,
            repository_root=repository_root,
            release_output_parent=release_output_parent,
        )

    progress = tqdm(total=rows, unit="row", file=sys.stderr)
    try:
        pilot_dir = run_pilot(
            input_path=input_path,
            sample_manifest_path=sample_manifest,
            config_path=config_path,
            output_dir=output_dir,
            rows=rows,
            batch_size=batch_size,
            concurrency=concurrency,
            delay_between_batches=delay_between_batches,
            maximum_total_requests=maximum_total_requests,
            input_price_per_million=input_price_per_million,
            output_price_per_million=output_price_per_million,
            progress_callback=progress.update,
        )
        report = validate_persona_pilot(pilot_dir=pilot_dir)
        if not report.passed:
            raise ValueError("Persona dataset failed validation")
        output_path = _merged_output_path(pilot_dir=pilot_dir)
        if hf_repo is not None:
            assert release_paths is not None
            (
                attestation_path,
                policy_path,
                dataset_card_path,
                licence_path,
                repository_path,
                release_parent,
            ) = release_paths
            result = package_release(
                pilot_dir=pilot_dir,
                attestation_path=attestation_path,
                policy_path=policy_path,
                dataset_card_path=dataset_card_path,
                licence_path=licence_path,
                repository_root=repository_path,
                output_parent=release_parent,
            )
            verify_release(
                release_dir=result.path, expected_manifest_sha256=result.manifest_sha256
            )
            upload_release(repo_id=hf_repo, release_dir=result.path)
    except Exception as error:
        raise click.ClickException(str(error)) from error
    finally:
        progress.close()
    click.echo(output_path)


def _configure_logging() -> None:
    """Configure diagnostics on stderr while keeping stdout machine-readable."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for logger_name in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


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
    repository_root: Path | None,
    release_output_parent: Path | None,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    """Require every release-boundary input when publication is requested.

    Returns:
        The six paths required by the release packager.

    Raises:
        click.ClickException:
            If a release-boundary input is missing.
    """
    missing = [
        name
        for name, value in (
            ("--attestation", attestation),
            ("--policy", policy),
            ("--dataset-card", dataset_card),
            ("--licence", licence),
            ("--repository-root", repository_root),
            ("--release-output-parent", release_output_parent),
        )
        if value is None
    ]
    if missing:
        raise click.ClickException("--hf-repo requires: " + ", ".join(missing))
    assert (
        attestation is not None
        and policy is not None
        and dataset_card is not None
        and licence is not None
        and repository_root is not None
        and release_output_parent is not None
    )
    return (
        attestation,
        policy,
        dataset_card,
        licence,
        repository_root,
        release_output_parent,
    )


if __name__ == "__main__":
    main()
