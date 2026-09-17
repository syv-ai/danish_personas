"""Click commands for offline release packaging and verification."""

from pathlib import Path

import click

from .packager import package_release
from .verifier import verify_release


@click.group(name="release")
def release() -> None:
    """Package and verify an offline release."""


@release.command(name="package")
@click.option("--pilot-dir", type=click.Path(path_type=Path), required=True)
@click.option(
    "--attestation",
    "--attestation-path",
    "attestation_path",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option(
    "--policy",
    "--policy-path",
    "policy_path",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option(
    "--dataset-card",
    "--dataset-card-path",
    "dataset_card_path",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option(
    "--licence",
    "--licence-path",
    "licence_path",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option("--repository-root", type=click.Path(path_type=Path), required=True)
@click.option("--output-parent", type=click.Path(path_type=Path), required=True)
def package_command(
    pilot_dir: Path,
    attestation_path: Path,
    policy_path: Path,
    dataset_card_path: Path,
    licence_path: Path,
    repository_root: Path,
    output_parent: Path,
) -> None:
    """Package a validated pilot without network or upload operations.

    Raises:
        ClickException:
            If packaging fails.
    """
    try:
        result = package_release(
            pilot_dir=pilot_dir,
            attestation_path=attestation_path,
            policy_path=policy_path,
            dataset_card_path=dataset_card_path,
            licence_path=licence_path,
            repository_root=repository_root,
            output_parent=output_parent,
        )
    except Exception as error:
        raise click.ClickException(str(error)) from error
    click.echo(result.path)
    click.echo(result.manifest_sha256)


@release.command(name="verify")
@click.option("--release-dir", type=click.Path(path_type=Path), required=True)
@click.option("--expected-manifest-sha256", required=True)
def verify_command(release_dir: Path, expected_manifest_sha256: str) -> None:
    """Verify a relocated release using its externally retained digest.

    Raises:
        ClickException:
            If verification fails.
    """
    try:
        manifest = verify_release(
            release_dir=release_dir, expected_manifest_sha256=expected_manifest_sha256
        )
    except Exception as error:
        raise click.ClickException(str(error)) from error
    click.echo(manifest.release_id)
