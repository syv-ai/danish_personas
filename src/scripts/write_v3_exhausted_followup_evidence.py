"""Write or verify offline private evidence for exhausted v3 follow-up rows."""

from __future__ import annotations

import json
from pathlib import Path

import click

from danish_personas.environment import load_repository_environment
from danish_personas.release.v3_exhausted_followup import (
    DEFAULT_BASE,
    DEFAULT_BUNDLE,
    DEFAULT_CANDIDATE,
    DEFAULT_FOLLOWUP,
    DEFAULT_LEDGER,
    DEFAULT_ORIGINAL,
    DEFAULT_OUTPUT,
    DEFAULT_PROMPT,
    ExhaustedFollowupError,
    verify_evidence,
    write_evidence,
)


@click.command()
@click.option("--base-dir", type=click.Path(path_type=Path), default=DEFAULT_BASE)
@click.option(
    "--followup-dir", type=click.Path(path_type=Path), default=DEFAULT_FOLLOWUP
)
@click.option("--candidate", type=click.Path(path_type=Path), default=DEFAULT_CANDIDATE)
@click.option("--original", type=click.Path(path_type=Path), default=DEFAULT_ORIGINAL)
@click.option(
    "--bundle", "bundle_dir", type=click.Path(path_type=Path), default=DEFAULT_BUNDLE
)
@click.option(
    "--prompt", "prompt_path", type=click.Path(path_type=Path), default=DEFAULT_PROMPT
)
@click.option(
    "--ledger", "ledger_path", type=click.Path(path_type=Path), default=DEFAULT_LEDGER
)
@click.option("--output-dir", type=click.Path(path_type=Path), default=DEFAULT_OUTPUT)
@click.option("--run", "execute", is_flag=True, default=False)
@click.option("--verify", "verify_only", is_flag=True, default=False)
def main(
    base_dir: Path,
    followup_dir: Path,
    candidate: Path,
    original: Path,
    bundle_dir: Path,
    prompt_path: Path,
    ledger_path: Path,
    output_dir: Path,
    execute: bool,
    verify_only: bool,
) -> None:
    """Read campaign inputs only; write evidence only with ``--run``.

    Raises:
        click.ClickException: If source evidence or output state is invalid.
    """
    try:
        arguments = {
            "base_dir": base_dir,
            "followup_dir": followup_dir,
            "candidate_path": candidate,
            "original_path": original,
            "bundle_dir": bundle_dir,
            "prompt_path": prompt_path,
            "ledger_path": ledger_path,
        }
        result = (
            verify_evidence(evidence_path=output_dir / "evidence.json", **arguments)
            if verify_only
            else write_evidence(output_dir=output_dir, run=execute, **arguments)
        )
    except (ExhaustedFollowupError, OSError, ValueError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    load_repository_environment()
    main()
