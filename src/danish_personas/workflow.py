"""Local orchestration of the demographic and persona pipelines.

The functions here chain the individual pipeline stages, record the exact path each
stage produced, and read finished runs back. They keep the makefile thin: every target
is one command, and no stage path is recovered by parsing log output.
"""

import logging
import typing as t
from pathlib import Path

import polars as pl

from .generation.models import GenerationManifest, PilotManifest
from .generation.pipeline import generate_personas
from .generation.report import validate_persona_run
from .io import write_json
from .models import RunManifest
from .sampling.freeze import freeze_sample
from .sampling.generator import generate_records
from .sources.prepare import prepare_bundle
from .validation.checks import validate_demographics, validate_sources

LOGGER = logging.getLogger(__name__)

ATTRIBUTE_COLUMNS = (
    "cultural_context",
    "skills_and_expertise",
    "hobbies_and_interests",
    "career_goals_and_ambitions",
)
BRIEF_COLUMNS = (
    "age",
    "sex",
    "region",
    "origin",
    "origin_region",
    "marital_status",
    "education_level",
    "labour_market_status",
    "detailed_status",
)
BREAKDOWN_COLUMNS = (
    "sex",
    "age_band",
    "marital_status",
    "region",
    "origin",
    "origin_region",
    "education_level",
    "labour_market_status",
    "detailed_status",
)
DESCRIPTION_COLUMNS = (
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "visual_persona",
    "persona",
)
OCEAN_TRAITS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)
CATEGORIES_CONFIG = Path("config/categories.yaml")
ORIGIN_REGIONS_CONFIG = Path("config/origin-regions.yaml")
DEVELOPMENT_SAMPLE_ROWS = 1000
SEED_FILE_NAME = "text-development-seeds.parquet"
VALIDATION_CONFIG = Path("config/validation.yaml")


def brief_lines(frame: pl.DataFrame, rows: int, fields: t.Sequence[str]) -> list[str]:
    """Render the first records of a run as short, readable blocks.

    Args:
        frame:
            Records sorted by persona identifier.
        rows:
            Number of records to render.
        fields:
            Field names to render; an empty sequence uses the brief columns.

    Returns:
        Report lines.
    """
    selected = fields or [name for name in BRIEF_COLUMNS if name in frame.columns]
    lines: list[str] = []
    for record in frame.head(rows).iter_rows(named=True):
        lines.append(f"\n{record['persona_id']}")
        lines.extend(
            f"  {column:<24} {render_value(record[column])}" for column in selected
        )
    return lines


def render_value(value: object) -> str:
    """Render one record field as readable text.

    Args:
        value:
            Field value from a record.

    Returns:
        Human-readable representation.

    Examples:
        >>> render_value(["løb", "kor"])
        'løb, kor'
        >>> render_value(None)
        '-'
        >>> render_value(47.254114840330914)
        '47.3'
    """
    if value is None:
        return "-"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def create_demographic_run(
    bundle_dir: Path,
    sampling_config_path: Path,
    runs_root: Path,
    run_name: str,
    rows: int,
    seed: int,
    validate: bool = True,
    validation_config_path: Path = VALIDATION_CONFIG,
    categories_path: Path = CATEGORIES_CONFIG,
) -> Path:
    """Generate and validate one deterministic demographic run.

    Args:
        bundle_dir:
            Prepared source bundle.
        sampling_config_path:
            Sampling configuration file.
        runs_root:
            Root directory holding named run collections.
        run_name:
            Sub-directory selecting one run collection.
        rows:
            Number of records to generate.
        seed:
            Deterministic sampling seed.
        validate:
            Whether the mandatory validation gates must pass.
        validation_config_path:
            Validation thresholds.
        categories_path:
            Canonical category mappings.

    Returns:
        Validated run directory.

    Raises:
        ValueError:
            If the generated run fails a mandatory validation gate.
    """
    run_dir = generate_records(
        bundle_dir=bundle_dir,
        sampling_config_path=sampling_config_path,
        output_dir=runs_root / run_name,
        rows=rows,
        seed=seed,
    )
    if not validate:
        LOGGER.warning(
            "Skipped validation for %s; the run is not release data", run_dir
        )
        return run_dir
    report = validate_demographics(
        run_dir=run_dir,
        bundle_dir=bundle_dir,
        validation_config_path=validation_config_path,
        categories_path=categories_path,
    )
    if not report.passed:
        failed = ", ".join(
            metric.name for metric in report.metrics if not metric.passed
        )
        message = (
            f"Demographic validation failed for {run_dir}: {failed}. The thresholds "
            "are calibrated for the statistical row count, so small datasets fail "
            "them; use the default size, or --skip-validation for a quick look."
        )
        raise ValueError(message)
    return run_dir


def create_personas(
    run_dir: Path,
    config_path: Path,
    output_root: Path,
    rows: int,
    offset: int,
    live: bool,
) -> Path:
    """Generate complete personas for a slice of a frozen sample.

    Args:
        run_dir:
            Demographic run holding the frozen persona input.
        config_path:
            Local generation configuration.
        output_root:
            Root directory for persona runs.
        rows:
            Number of personas to generate.
        offset:
            Zero-based position within the ordered frozen sample.
        live:
            Whether network calls are explicitly authorised.

    Returns:
        Planned or completed persona run directory.

    Raises:
        ValueError:
            If the dataset is unvalidated, or a live run fails its persona gate.
    """
    if not (run_dir / "validation-report.json").is_file():
        message = (
            f"{run_dir} has no validation report, so it cannot seed personas. Build a "
            "validated dataset with 'personas demographics' without --skip-validation."
        )
        raise ValueError(message)
    sample_path = run_dir / SEED_FILE_NAME
    persona_run = generate_personas(
        input_path=sample_path,
        sample_manifest_path=sample_path.with_suffix(".manifest.json"),
        config_path=config_path,
        output_dir=output_root,
        rows=rows,
        offset=offset,
        live=live,
    )
    if not live:
        return persona_run
    report = validate_persona_run(run_dir=persona_run)
    if not report.passed:
        message = f"Persona validation failed for {persona_run}"
        raise ValueError(message)
    return persona_run


def export_record(
    record: dict[str, object], run_dir: Path, output_dir: Path
) -> tuple[Path, Path]:
    """Write one record as JSON and Markdown in a single directory.

    Args:
        record:
            Record to export.
        run_dir:
            Run the record came from.
        output_dir:
            Destination directory.

    Returns:
        Paths of the written JSON and Markdown files.
    """
    generated_columns = set(ATTRIBUTE_COLUMNS) | set(DESCRIPTION_COLUMNS)
    payload: dict[str, object] = {
        "source_run": str(run_dir),
        "statistical_inputs": {
            column: value
            for column, value in record.items()
            if column not in generated_columns
        },
        "generated": {
            column: value
            for column, value in record.items()
            if column in generated_columns
        },
    }
    identifier = str(record["persona_id"])
    json_path = output_dir / f"{identifier}.json"
    write_json(path=json_path, payload=payload)
    markdown_path = output_dir / f"{identifier}.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(payload=payload), encoding="utf-8")
    return json_path, markdown_path


def render_markdown(payload: t.Mapping[str, object]) -> str:
    """Render an exported record as a Markdown document.

    Args:
        payload:
            Export payload with statistical inputs and generated fields.

    Returns:
        Markdown document text.
    """
    inputs = t.cast(t.Mapping[str, object], payload["statistical_inputs"])
    generated = t.cast(t.Mapping[str, object], payload["generated"])
    identifier = inputs.get("persona_id", "unknown")
    lines = [f"# Persona {identifier}", "", f"Source run: `{payload['source_run']}`"]
    lines.extend(
        ["", "## Statistical inputs", "", "| Field | Value |", "| --- | --- |"]
    )
    lines.extend(
        f"| {column} | {render_value(value)} |" for column, value in inputs.items()
    )
    if generated:
        lines.append("\n## Generated")
        for column, value in generated.items():
            lines.extend(["", f"### {column}", "", render_value(value)])
    return "\n".join(lines) + "\n"


def freeze_persona_input(run_dir: Path) -> Path:
    """Freeze a stratified persona input sample unless it already exists.

    Args:
        run_dir:
            Validated demographic run.

    Returns:
        Frozen sample file, capped at the size of the run.
    """
    output_path = run_dir / SEED_FILE_NAME
    if output_path.is_file():
        return output_path
    return freeze_sample(
        run_dir=run_dir, rows=DEVELOPMENT_SAMPLE_ROWS, output=output_path
    )


def frozen_sample_offset(sample_path: Path, persona_id: str) -> int:
    """Find the position of one persona inside a frozen sample.

    Args:
        sample_path:
            Frozen sample file.
        persona_id:
            Persona identifier to locate.

    Returns:
        Zero-based position in the identifier-sorted sample.

    Raises:
        ValueError:
            If the identifier is absent from the frozen sample.
    """
    identifiers = (
        pl.read_parquet(sample_path, columns=["persona_id"])
        .sort("persona_id")
        .get_column("persona_id")
        .to_list()
    )
    try:
        return identifiers.index(persona_id)
    except ValueError as error:
        message = f"Frozen sample holds no persona {persona_id}"
        raise ValueError(message) from error


def load_run_frame(run_dir: Path) -> pl.DataFrame:
    """Read the records of a demographic, persona, or pilot run.

    Args:
        run_dir:
            Run directory holding a manifest and its Parquet data.

    Returns:
        Records sorted by persona identifier.

    Raises:
        ValueError:
            If the directory holds no readable run data.
    """
    data_file = _manifest_data_file(run_dir=run_dir)
    if data_file is None:
        message = f"No run manifest with readable Parquet data in {run_dir}"
        raise ValueError(message)
    return pl.read_parquet(run_dir / data_file).sort("persona_id")


def _manifest_data_file(run_dir: Path) -> Path | None:
    manifest_path = run_dir / "run-manifest.json"
    if manifest_path.is_file():
        return RunManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        ).data_file
    manifest_path = run_dir / "generation-manifest.json"
    if manifest_path.is_file():
        manifest = GenerationManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        return Path(manifest.output_file.name)
    manifest_path = run_dir / "pilot-manifest.json"
    if manifest_path.is_file():
        pilot = PilotManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        return Path(pilot.output_file.name)
    return None


def prepare_source_bundle(
    raw_dir: Path,
    bundle_root: Path,
    lock_path: Path,
    categories_path: Path,
    origin_regions_path: Path = ORIGIN_REGIONS_CONFIG,
) -> Path:
    """Build and validate an offline source bundle.

    Args:
        raw_dir:
            Restored raw snapshot directory.
        bundle_root:
            Root destination for prepared bundles.
        lock_path:
            Resolved source lock.
        categories_path:
            Canonical category mappings.
        origin_regions_path:
            Country-of-origin groupings.

    Returns:
        Validated bundle directory.

    Raises:
        ValueError:
            If the prepared bundle fails source validation.
    """
    bundle_dir = prepare_bundle(
        lock_path=lock_path,
        categories_path=categories_path,
        raw_dir=raw_dir,
        output_dir=bundle_root,
        origin_regions_path=origin_regions_path,
    )
    if not validate_sources(bundle_dir=bundle_dir).passed:
        message = f"Source validation failed for {bundle_dir}"
        raise ValueError(message)
    return bundle_dir


def read_pointer(path: Path) -> Path | None:
    """Read a recorded stage pointer.

    Args:
        path:
            Pointer file.

    Returns:
        Recorded directory, or None when the pointer is absent.
    """
    if not path.is_file():
        return None
    return Path(path.read_text(encoding="utf-8").strip())


def select_fields(
    record: dict[str, object], fields: t.Sequence[str]
) -> dict[str, object]:
    """Keep the persona identifier and the requested fields of a record.

    Args:
        record:
            One record from a run.
        fields:
            Field names to keep; an empty sequence keeps every field.

    Returns:
        The reduced record.

    Raises:
        ValueError:
            If a requested field is absent from the record.

    Examples:
        >>> select_fields({"persona_id": "a", "age": 30, "sex": "male"}, ["age"])
        {'persona_id': 'a', 'age': 30}
    """
    if not fields:
        return record
    unknown = [name for name in fields if name not in record]
    if unknown:
        message = f"Unknown fields: {', '.join(unknown)}"
        raise ValueError(message)
    selected: dict[str, object] = {"persona_id": record["persona_id"]}
    selected.update({name: record[name] for name in fields})
    return selected


def summarise_run(frame: pl.DataFrame, top: int) -> list[str]:
    """Render the demographic, labour, and personality shares of a run.

    Args:
        frame:
            Records to summarise.
        top:
            Maximum number of values per breakdown.

    Returns:
        Report lines.
    """
    generated = [column for column in ATTRIBUTE_COLUMNS if column in frame.columns]
    kind = "complete personas" if generated else "demographic briefs"
    lines = [f"Records: {frame.height}   Columns: {frame.width}", f"Content: {kind}"]
    for column in BREAKDOWN_COLUMNS:
        if column not in frame.columns:
            continue
        lines.append(f"\n  {column}")
        lines.extend(share_lines(frame=frame, column=column, limit=top))
    ages = frame.get_column("age")
    lines.append(f"\n  age\n    mean {ages.mean():.1f}   median {ages.median():.0f}")
    lines.append("\n  ocean (mean score)")
    lines.extend(
        f"    {trait:<20} {frame.get_column(f'{trait}_score').mean():5.1f}"
        for trait in OCEAN_TRAITS
    )
    if generated:
        interests = frame.get_column("hobbies_and_interests").explode().drop_nulls()
        lines.append("\n  most common interests")
        lines.extend(
            share_lines(
                frame=interests.to_frame(), column="hobbies_and_interests", limit=top
            )
        )
    return lines


def share_lines(frame: pl.DataFrame, column: str, limit: int) -> list[str]:
    """Render the most frequent values of one column as shares.

    Args:
        frame:
            Records to summarise.
        column:
            Column to count.
        limit:
            Maximum number of values to render.

    Returns:
        Formatted lines, most frequent first.
    """
    counts = frame.get_column(column).value_counts(sort=True).head(limit)
    lines: list[str] = []
    for value, count in counts.iter_rows():
        share = 100.0 * count / frame.height
        bar = "#" * round(share / 2.5)
        lines.append(f"    {str(value)[:28]:<28} {share:5.1f}%  {count:>7}  {bar}")
    return lines


def write_pointer(path: Path, value: Path) -> None:
    """Record the exact path one pipeline stage produced.

    Args:
        path:
            Pointer file.
        value:
            Directory the stage produced.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{value}\n", encoding="utf-8")
