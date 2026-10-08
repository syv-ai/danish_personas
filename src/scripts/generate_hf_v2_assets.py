"""Generate private aggregate PNG assets for the Hugging Face v2 release."""

from __future__ import annotations

import collections
import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import click
import polars as pl

from danish_personas.cli_logging import configure_cli_logging
from danish_personas.environment import load_repository_environment
from danish_personas.io import sha256_file

EXPECTED_ROW_COUNT = 100_000
TOP_CATEGORY_LIMIT = 25
PNG_DPI = 160
MISSING_LABEL = "(not recorded)"

OCEAN_FIELDS: tuple[tuple[str, str], ...] = (
    ("openness_score", "Openness"),
    ("conscientiousness_score", "Conscientiousness"),
    ("extraversion_score", "Extraversion"),
    ("agreeableness_score", "Agreeableness"),
    ("neuroticism_score", "Neuroticism"),
)

CATEGORY_CHARTS: tuple["CategoryChart", ...] = (
    ("age-distribution.png", "Age distribution", "age"),
    ("age-band-distribution.png", "Age-band distribution", "age_band"),
    ("sex-distribution.png", "Sex distribution", "sex"),
    (
        "marital-status-distribution.png",
        "Marital-status distribution",
        "marital_status",
    ),
    ("region-distribution.png", "Region distribution", "region"),
    ("municipality-distribution.png", "Municipality distribution", "municipality"),
    (
        "origin-country-distribution.png",
        "Origin-country distribution",
        "origin_country_da",
    ),
    ("education-distribution.png", "Education distribution", "education_level"),
    (
        "labour-market-status-distribution.png",
        "Labour-market status distribution",
        "labour_market_status",
    ),
    ("job-function-distribution.png", "Job-function distribution", "job_function"),
    (
        "current-relationship-status.png",
        "Current relationship status",
        "current_relationship_status",
    ),
)

REQUIRED_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys(
        [
            *(field for _, _, field in CATEGORY_CHARTS),
            "partner_gender",
            *(field for field, _ in OCEAN_FIELDS),
        ]
    )
)


@dataclass(frozen=True)
class AssetSummary:
    """Summary of generated release assets."""

    input: str
    input_sha256: str
    output_dir: str
    rows: int
    charts: list[str]


class AssetGenerationError(Exception):
    """Raised when Hugging Face v2 release assets cannot be generated safely."""


CategoryChart = tuple[str, str, str]
CountItems = list[tuple[str, int]]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Local, already downloaded Hugging Face v2 dataset Parquet path.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Private directory for the 13 aggregate PNG assets.",
)
@click.option(
    "--input-sha256",
    required=False,
    help="Optional expected SHA-256 for the downloaded Parquet input.",
)
def main(input_path: Path, output_dir: Path, input_sha256: str | None) -> None:
    """Generate the fixed aggregate PNG asset tree for the v2 release.

    Raises:
        click.ClickException:
            If the local input, output directory, or existing assets are unsafe.
    """
    configure_cli_logging()
    try:
        _prepare_matplotlib()
        summary = generate_hf_v2_assets(
            input_path=input_path,
            output_dir=output_dir,
            expected_input_sha256=input_sha256,
        )
    except (
        AssetGenerationError,
        ImportError,
        OSError,
        pl.exceptions.PolarsError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(summary.__dict__, ensure_ascii=False, sort_keys=True))


def generate_hf_v2_assets(
    *, input_path: Path, output_dir: Path, expected_input_sha256: str | None = None
) -> AssetSummary:
    """Generate all fixed Hugging Face v2 aggregate PNG assets.

    Args:
        input_path:
            Local downloaded Parquet dataset to aggregate.
        output_dir:
            Private directory that receives PNG assets.
        expected_input_sha256 (optional):
            Expected lower-case SHA-256 for the input Parquet file. Defaults to ``None``.

    Returns:
        Generated asset summary.
    """
    digest = _validate_input_file(
        input_path=input_path, expected_sha256=expected_input_sha256
    )
    frame = _read_guarded_frame(input_path=input_path)
    rendered = _render_assets(frame=frame)
    _prepare_output_dir(output_dir=output_dir)
    _write_assets(output_dir=output_dir, rendered=rendered)
    return AssetSummary(
        input=str(input_path),
        input_sha256=digest,
        output_dir=str(output_dir),
        rows=frame.height,
        charts=sorted(rendered),
    )


def _prepare_matplotlib() -> None:
    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required; run with "
            "`uv run --with matplotlib python src/scripts/"
            "generate_hf_v2_assets.py ...`"
        ) from exc
    matplotlib.use("Agg", force=True)


def _validate_input_file(*, input_path: Path, expected_sha256: str | None) -> str:
    if "://" in str(input_path):
        raise AssetGenerationError("Input must be a local Parquet path, not a URL")
    if not input_path.is_file():
        raise AssetGenerationError("Input Parquet path is missing or not a file")
    if input_path.suffix.casefold() != ".parquet":
        raise AssetGenerationError("Input must be a Parquet file")
    digest = sha256_file(input_path)
    if expected_sha256 is not None:
        expected = expected_sha256.strip().casefold()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise AssetGenerationError("Expected input SHA-256 is not valid hex")
        if digest != expected:
            raise AssetGenerationError("Input Parquet SHA-256 does not match")
    return digest


def _read_guarded_frame(*, input_path: Path) -> pl.DataFrame:
    scan = pl.scan_parquet(input_path)
    schema = scan.collect_schema()
    missing = sorted(set(REQUIRED_COLUMNS).difference(schema.names()))
    if missing:
        fields = ", ".join(missing)
        raise AssetGenerationError(f"Input Parquet lacks required fields: {fields}")
    row_count = int(scan.select(pl.len()).collect().item())
    if row_count != EXPECTED_ROW_COUNT:
        raise AssetGenerationError(
            f"Input Parquet must contain exactly {EXPECTED_ROW_COUNT:,} rows; "
            f"found {row_count:,}"
        )
    return pl.read_parquet(input_path, columns=list(REQUIRED_COLUMNS))


def _render_assets(*, frame: pl.DataFrame) -> dict[str, bytes]:
    rendered: dict[str, bytes] = {}
    for filename, title, field in CATEGORY_CHARTS:
        counts = _category_counts(frame=frame, field=field)
        rendered[filename] = _category_chart_png(
            title=title, counts=counts, collapse_high_cardinality=field != "age"
        )
    relationship_counts = _relationship_pair_counts(frame=frame)
    rendered["partner-relationship-pair.png"] = _category_chart_png(
        title="Partner relationship pair",
        counts=relationship_counts,
        collapse_high_cardinality=False,
    )
    rendered["ocean-distribution.png"] = _ocean_chart_png(frame=frame)
    return rendered


def _category_counts(*, frame: pl.DataFrame, field: str) -> CountItems:
    values = [_normalise_category(value=value) for value in frame.get_column(field)]
    counts = collections.Counter(values)
    if field == "age":
        return sorted(counts.items(), key=lambda item: _age_key(value=item[0]))
    if field == "age_band":
        return sorted(counts.items(), key=lambda item: _age_band_key(value=item[0]))
    return sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold()))


def _relationship_pair_counts(*, frame: pl.DataFrame) -> CountItems:
    pairs: list[str] = []
    for row in frame.select(
        ["sex", "partner_gender", "current_relationship_status"]
    ).iter_rows(named=True):
        status = _normalise_category(value=row["current_relationship_status"])
        if status.casefold() != "partnered":
            continue
        sex = _gender_label(value=row["sex"])
        partner = _gender_label(value=row["partner_gender"])
        if sex is None or partner is None:
            continue
        if sex == partner == "man":
            pairs.append("man-man")
        elif sex == partner == "woman":
            pairs.append("woman-woman")
        else:
            pairs.append("man-woman")
    counts = collections.Counter(pairs)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _category_chart_png(
    *, title: str, counts: CountItems, collapse_high_cardinality: bool
) -> bytes:
    import matplotlib.pyplot as plt

    collapsed, horizontal = _collapse_counts(
        counts=counts, collapse_high_cardinality=collapse_high_cardinality
    )
    labels = [label for label, _ in collapsed] or ["No records"]
    values = [count for _, count in collapsed] or [0]
    height = max(4.0, 0.34 * len(labels) + 1.4) if horizontal else 5.2
    fig, ax = plt.subplots(figsize=(10.5, height))
    if horizontal:
        ax.barh(labels, values, color="#386cb0")
        ax.invert_yaxis()
        ax.set_xlabel("Records")
    else:
        ax.bar(labels, values, color="#386cb0")
        ax.set_ylabel("Records")
        ax.tick_params(axis="x", labelrotation=45)
    ax.set_title(title)
    ax.grid(axis="x" if horizontal else "y", alpha=0.25)
    fig.tight_layout()
    return _figure_png(fig=fig, pyplot=plt)


def _ocean_chart_png(*, frame: pl.DataFrame) -> bytes:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10.5, 12.0), sharex=True)
    for axis, (field, label) in zip(axes, OCEAN_FIELDS, strict=True):
        values = [float(value) for value in frame.get_column(field).drop_nulls()]
        axis.hist(values, bins=12, color="#386cb0", edgecolor="white")
        axis.set_ylabel("Records")
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("Score")
    fig.suptitle("OCEAN score distributions")
    fig.tight_layout()
    return _figure_png(fig=fig, pyplot=plt)


def _collapse_counts(
    *, counts: CountItems, collapse_high_cardinality: bool
) -> tuple[CountItems, bool]:
    if not collapse_high_cardinality or len(counts) <= TOP_CATEGORY_LIMIT:
        return counts, False
    top = counts[:TOP_CATEGORY_LIMIT]
    other = sum(count for _, count in counts[TOP_CATEGORY_LIMIT:])
    return [*top, ("Other", other)], True


def _figure_png(*, fig: object, pyplot: object) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=PNG_DPI, metadata={"Software": "matplotlib"})
    pyplot.close(fig)
    return buffer.getvalue()


def _normalise_category(*, value: object) -> str:
    if value is None:
        return MISSING_LABEL
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    label = str(value).strip()
    return label or MISSING_LABEL


def _gender_label(*, value: object) -> str | None:
    normalised = _normalise_category(value=value).casefold()
    if normalised in {"male", "man", "m", "mand"}:
        return "man"
    if normalised in {"female", "woman", "f", "kvinde"}:
        return "woman"
    return None


def _age_key(*, value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 10_000, value


def _age_band_key(*, value: str) -> tuple[int, str]:
    match = re.search(r"\d+", value)
    if match is None:
        return 10_000, value
    return int(match.group(0)), value


def _prepare_output_dir(*, output_dir: Path) -> None:
    existed = output_dir.exists()
    if output_dir.is_symlink():
        raise AssetGenerationError("Output directory must not be a symbolic link")
    if existed and not output_dir.is_dir():
        raise AssetGenerationError("Output path exists and is not a directory")
    if not existed:
        output_dir.mkdir(parents=True, mode=0o700)
        os.chmod(output_dir, 0o700)
    if output_dir.stat().st_mode & 0o077:
        raise AssetGenerationError("Output directory must be private")


def _write_assets(*, output_dir: Path, rendered: dict[str, bytes]) -> None:
    expected = {filename for filename, _, _ in CATEGORY_CHARTS}
    expected.update({"partner-relationship-pair.png", "ocean-distribution.png"})
    if set(rendered) != expected:
        raise AssetGenerationError(
            "Rendered asset set does not match the v2 asset tree"
        )
    for filename, content in rendered.items():
        path = output_dir / filename
        if path.is_symlink():
            raise AssetGenerationError(
                f"Existing output is a symbolic link: {path.name}"
            )
        if path.exists() and not path.is_file():
            raise AssetGenerationError(f"Existing output is not a file: {path.name}")
        if path.exists() and sha256_file(path) != _sha256_bytes(content=content):
            raise AssetGenerationError(f"Existing output differs: {path.name}")
    for filename, content in rendered.items():
        _write_private_bytes(path=output_dir / filename, content=content)


def _write_private_bytes(*, path: Path, content: bytes) -> None:
    digest = _sha256_bytes(content=content)
    if path.is_symlink():
        raise AssetGenerationError(f"Existing output is a symbolic link: {path.name}")
    if path.exists():
        if not path.is_file():
            raise AssetGenerationError(f"Existing output is not a file: {path.name}")
        if sha256_file(path) != digest:
            raise AssetGenerationError(f"Existing output differs: {path.name}")
        os.chmod(path, 0o600)
        return
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_bytes(*, content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


if __name__ == "__main__":
    load_repository_environment()
    main()
