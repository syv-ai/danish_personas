"""Command-line interface for building and reading Danish persona datasets."""

import functools
import logging
import shutil
import typing as t
from pathlib import Path

import click
import polars as pl

from .generation.models import GenerationConfig
from .io import load_yaml_model, write_yaml
from .models import SamplingConfig
from .sources.restore import RAW_DIRECTORY, restore_snapshots
from .workflow import (
    CATEGORIES_CONFIG,
    brief_lines,
    create_demographic_run,
    create_personas,
    export_record,
    freeze_persona_input,
    frozen_sample_offset,
    load_run_frame,
    prepare_source_bundle,
    read_pointer,
    select_fields,
    summarise_run,
    write_pointer,
)

BUNDLE_ROOT = Path("data/processed")
EXPORT_ROOT = Path("data/exports")
GENERATION_CONFIG = Path("config/generation.local.yaml")
TEMPLATE_CONFIG = Path("config/generation.yaml")
LOCK_CONFIG = Path("config/sources.lock.yaml")
PERSONA_ROOT = Path("data/persona-smoke")
PILOT_ROOT = Path("data/persona-pilot")
RAW_DIR = Path("data") / RAW_DIRECTORY
RUNS_ROOT = Path("data/runs")
SAMPLING_CONFIG = Path("config/sampling.yaml")
STATE_DIR = Path("data/.state")


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Build and read Danish persona datasets.

    Start with 'demographics' to build the dataset, then 'brief' to read plain
    records from it, or 'generate' to add LLM-written persona text.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _authorise(rows: int, yes: bool) -> None:
    click.echo(f"Live run: {rows} personas, two model stages each.")
    if not yes:
        click.confirm("Authorise paid requests?", abort=True)


def _current_run(run_dir: Path | None) -> Path:
    if run_dir is not None:
        return run_dir
    recorded = read_pointer(STATE_DIR / "run")
    if recorded is None:
        message = "No dataset yet. Build one with 'personas demographics'."
        raise ValueError(message)
    return recorded


def _export_rows(
    frame: pl.DataFrame, run_dir: Path, rows: int, fields: list[str]
) -> None:
    for record in frame.head(rows).iter_rows(named=True):
        json_path, markdown_path = export_record(
            record=select_fields(record=record, fields=fields),
            run_dir=run_dir,
            output_dir=EXPORT_ROOT,
        )
        click.echo(f"Exported {json_path} and {markdown_path}")


def _names(fields: str) -> list[str]:
    return [name.strip() for name in fields.split(",") if name.strip()]


@main.command()
@click.option("--yes", is_flag=True, help="Delete without confirmation.")
def clean(yes: bool) -> None:
    """Delete generated datasets and personas, keeping the raw sources."""
    targets = [RUNS_ROOT, PERSONA_ROOT, PILOT_ROOT, STATE_DIR]
    existing = [target for target in targets if target.exists()]
    if not existing:
        click.echo("Nothing to delete.")
        return
    listed = ", ".join(str(target) for target in existing)
    if not yes:
        click.confirm(f"Delete {listed}?", abort=True)
    for target in existing:
        shutil.rmtree(target)
    click.echo(f"Removed {listed}.")


def _bundle() -> Path:
    recorded = read_pointer(STATE_DIR / "bundle")
    if recorded is not None and recorded.is_dir():
        return recorded
    if not RAW_DIR.is_dir():
        restore_snapshots()
    bundle_dir = prepare_source_bundle(
        raw_dir=RAW_DIR,
        bundle_root=BUNDLE_ROOT,
        lock_path=LOCK_CONFIG,
        categories_path=CATEGORIES_CONFIG,
    )
    write_pointer(STATE_DIR / "bundle", bundle_dir)
    return bundle_dir


def guarded(command: t.Callable[..., None]) -> t.Callable[..., None]:
    """Turn pipeline failures into readable command-line errors.

    Args:
        command:
            Command implementation.

    Returns:
        Command that reports errors without a traceback.
    """

    @functools.wraps(command)
    def wrapper(*args, **kwargs) -> None:
        try:
            command(*args, **kwargs)
        except click.ClickException:
            raise
        except Exception as error:
            raise click.ClickException(str(error)) from error

    return wrapper


@main.command()
@click.option("--rows", type=click.IntRange(min=1), default=None, help="Records.")
@click.option("--seed", type=int, default=None, help="Deterministic sampling seed.")
@click.option("--name", default="local", show_default=True, help="Dataset name.")
@click.option(
    "--skip-validation",
    "skip_validation",
    is_flag=True,
    help="Keep a dataset that fails its gates; never use it as release data.",
)
@guarded
def demographics(
    rows: int | None, seed: int | None, name: str, skip_validation: bool
) -> None:
    """Build the statistical dataset, and validate it against the sources."""
    config = load_yaml_model(path=SAMPLING_CONFIG, model=SamplingConfig)
    rows = rows if rows is not None else config.statistical_rows
    seed = seed if seed is not None else config.seed
    run_dir = create_demographic_run(
        bundle_dir=_bundle(),
        sampling_config_path=SAMPLING_CONFIG,
        runs_root=RUNS_ROOT,
        run_name=name,
        rows=rows,
        seed=seed,
        validate=not skip_validation,
    )
    write_pointer(STATE_DIR / "run", run_dir)
    click.echo(f"Dataset: {run_dir}   rows={rows} seed={seed}")
    click.echo("Read it with 'personas brief'.")


RUN_OPTION = click.option(
    "--run",
    "run_dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Read this run instead of the current one.",
)
FIELDS_OPTION = click.option(
    "--fields", default="", help="Comma-separated fields instead of the defaults."
)
EXPORT_OPTION = click.option(
    "--export", "to_export", is_flag=True, help="Also write JSON and Markdown."
)


@main.command()
@RUN_OPTION
@FIELDS_OPTION
@EXPORT_OPTION
@click.option("--rows", default=3, show_default=True, help="Records to print.")
@guarded
def brief(rows: int, fields: str, to_export: bool, run_dir: Path | None) -> None:
    """Print plain persona records: age, region, education, job type."""
    resolved = _current_run(run_dir)
    frame = load_run_frame(resolved)
    selected = _names(fields)
    click.echo(f"Run: {resolved}   Records: {frame.height}")
    for line in brief_lines(frame=frame, rows=rows, fields=selected):
        click.echo(line)
    if to_export:
        _export_rows(frame=frame, run_dir=resolved, rows=rows, fields=selected)


@main.command()
@RUN_OPTION
@FIELDS_OPTION
@EXPORT_OPTION
@click.option("--rows", default=1, show_default=True, help="Personas to generate.")
@click.option("--persona-id", "persona_id", default=None, help="Record to describe.")
@click.option("--offset", type=click.IntRange(min=0), default=0, help="Sample row.")
@click.option("--live", is_flag=True, help="Authorise paid model requests.")
@click.option("--yes", is_flag=True, help="Skip the live-run confirmation.")
@guarded
def generate(
    rows: int,
    fields: str,
    to_export: bool,
    run_dir: Path | None,
    persona_id: str | None,
    offset: int,
    live: bool,
    yes: bool,
) -> None:
    """Add LLM-written text to records, and keep their statistical fields."""
    resolved = _current_run(run_dir)
    sample_path = freeze_persona_input(run_dir=resolved)
    if persona_id is not None:
        offset = frozen_sample_offset(sample_path=sample_path, persona_id=persona_id)
        rows = 1
    if live:
        _authorise(rows=rows, yes=yes)
    persona_run = create_personas(
        run_dir=resolved,
        config_path=GENERATION_CONFIG,
        output_root=PERSONA_ROOT,
        rows=rows,
        offset=offset,
        live=live,
    )
    if not live:
        click.echo(f"Planned {rows} personas in {persona_run}; nothing was generated.")
        click.echo("Add --live to run it, after 'personas setup-llm'.")
        return
    write_pointer(STATE_DIR / "personas", persona_run)
    click.echo(f"Personas: {persona_run}")
    frame = load_run_frame(persona_run)
    for line in brief_lines(frame=frame, rows=rows, fields=_names(fields)):
        click.echo(line)
    if to_export:
        _export_rows(frame=frame, run_dir=persona_run, rows=rows, fields=_names(fields))


@main.command()
@guarded
def runs() -> None:
    """List the current dataset, the generated personas, and earlier runs."""
    for name in ("bundle", "run", "personas"):
        click.echo(f"{name:<10} {read_pointer(STATE_DIR / name) or '-'}")
    click.echo("earlier")
    for root in (RUNS_ROOT, PERSONA_ROOT, PILOT_ROOT):
        for manifest in sorted(root.glob("**/*manifest.json")):
            click.echo(f"  {manifest.parent}")


@main.command("setup-llm")
@click.option("--base-url", "base_url", required=True, help="OpenAI-compatible URL.")
@click.option("--model", required=True, help="Model identifier.")
@click.option("--api-key-env", "api_key_env", default=None, help="Token variable.")
def setup_llm(base_url: str, model: str, api_key_env: str | None) -> None:
    """Point the ignored local generation config at one provider."""
    source = GENERATION_CONFIG if GENERATION_CONFIG.is_file() else TEMPLATE_CONFIG
    config = load_yaml_model(path=source, model=GenerationConfig)
    write_yaml(
        path=GENERATION_CONFIG,
        payload=config.model_copy(
            update={
                "llm_generation_enabled": True,
                "base_url": base_url,
                "model": model,
                "api_key_env": api_key_env,
            }
        ),
    )
    click.echo(f"Configured {GENERATION_CONFIG} for {model}")
    if api_key_env:
        click.echo(f"The token is read from {api_key_env}, or from .env.")


@main.command()
@RUN_OPTION
@click.option("--top", type=click.IntRange(min=1), default=6, show_default=True)
@guarded
def summary(run_dir: Path | None, top: int) -> None:
    """Print the distributions of a dataset: region, education, job type, OCEAN."""
    resolved = _current_run(run_dir)
    click.echo(f"Run: {resolved}")
    for line in summarise_run(frame=load_run_frame(resolved), top=top):
        click.echo(line)


if __name__ == "__main__":
    main()
