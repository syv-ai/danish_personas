"""Build a self-contained interactive dashboard for generated personas."""

import hashlib
import html
import json
import re
from pathlib import Path

import click
import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.offline import get_plotlyjs
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from danish_personas.io import load_yaml_model
from danish_personas.models import CategoryConfig

DISTRIBUTIONS: tuple[tuple[str, str, str], ...] = (
    ("first_name", "First names", "synthetic"),
    ("age", "Ages", "folk_age_sampling.parquet"),
    ("age_band", "Age bands", "folk_age_sampling.parquet"),
    ("sex", "Sex", "folk_age_sampling.parquet"),
    ("marital_status", "Marital status", "folk_marital_sampling.parquet"),
    ("education_level", "Education", "ras209_joint_unpooled.parquet"),
    ("labour_market_status", "Labour-market status", "ras209_joint_unpooled.parquet"),
    ("region", "Regions", "folk_age_sampling.parquet"),
    ("municipality", "Municipalities", "folk_age_sampling.parquet"),
    (
        "origin_country_da",
        "Origin-country labels",
        "folk2_origin_country_marginal.parquet",
    ),
    ("job_function", "Job functions", "job_function_sex_marginal.parquet"),
    ("current_relationship_status", "Current relationship status", "synthetic"),
    ("partner_gender", "Partner gender", "synthetic"),
)
OCEAN_FIELDS: tuple[tuple[str, str], ...] = (
    ("openness_score", "Openness"),
    ("conscientiousness_score", "Conscientiousness"),
    ("extraversion_score", "Extraversion"),
    ("agreeableness_score", "Agreeableness"),
    ("neuroticism_score", "Neuroticism"),
)
COLOUR_FIELDS: tuple[tuple[str, str], ...] = (
    ("sex", "Sex"),
    ("age_band", "Age band"),
    ("region", "Region"),
    ("education_level", "Education"),
    ("labour_market_status", "Labour-market status"),
)
TOKEN_RE = re.compile(r"[\wæøå]+", re.IGNORECASE)
CATEGORY_CONFIG_PATH = Path(__file__).parents[2] / "config" / "categories.yaml"
EMBEDDING_DECIMALS = 12


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Generated personas Parquet file.",
)
@click.option(
    "--bundle",
    "bundle_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Prepared bundle directory containing normalised target files.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Output self-contained HTML file.",
)
def main(input_path: Path, bundle_path: Path, output_path: Path) -> None:
    """Build one offline, interactive HTML persona dashboard.

    Raises:
        click.ClickException:
            If the input is empty or the dashboard cannot be rendered.
    """
    try:
        frame = pl.read_parquet(input_path)
        if frame.is_empty():
            raise click.ClickException("The generated personas file is empty")
        document = build_dashboard(
            frame=frame, bundle_path=bundle_path, input_path=input_path
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(document, encoding="utf-8")
    except click.ClickException:
        raise
    except (OSError, ValueError, KeyError, pl.exceptions.PolarsError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(output_path)


def build_dashboard(
    *, frame: pl.DataFrame, bundle_path: Path, input_path: Path | None = None
) -> str:
    """Render a complete dashboard document from generated records.

    Args:
        frame:
            Generated persona records, including their full persona text.
        bundle_path:
            Prepared bundle directory used for descriptive DST overlays.
        input_path (optional):
            Input path used in the provenance section. Defaults to ``None``.

    Returns:
        A single HTML document with all JavaScript and data embedded.
    """
    _require_columns(frame, {"persona", "persona_id"})
    target_cache = load_dst_targets(bundle_path=bundle_path, frame=frame)
    sections = [_overview_cards(frame=frame)]
    sections.extend(
        _distribution_chart(
            frame=frame,
            field=field,
            title=title,
            source=source,
            target=target_cache.get(field),
        )
        for field, title, source in DISTRIBUTIONS
        if field in frame.columns
    )
    sections.append(_ocean_chart(frame=frame))
    sections.append(_embedding_chart(frame=frame))
    sections.append(
        _provenance(
            frame=frame,
            bundle_path=bundle_path,
            input_path=input_path,
            targets=target_cache,
        )
    )
    data_json = json.dumps(
        [_json_safe_row(row) for row in frame.to_dicts()],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    safe_data_json = (
        data_json.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    # The bundled Plotly configuration contains a dormant CDN fallback URL. Remove
    # it so the standalone artefact has no network address, even in its JS payload.
    plotly_js = get_plotlyjs().replace("https://cdn.plot.ly", "")
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Generated persona dashboard</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;margin:0;background:#f5f7fa;color:#17202a}"
        ".wrap{max-width:1400px;margin:auto;padding:24px}.grid{display:grid;"
        "grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}"
        ".card,.chart{background:white;border-radius:10px;padding:18px;"
        "box-shadow:0 1px 4px #0002}.chart{min-height:390px}.wide{grid-column:1/-1}"
        ".metric{font-size:2rem;font-weight:700}.muted{color:#586674;font-size:.9rem}"
        ".source{font-size:.8rem;color:#586674}.plot{height:390px}"
        f"</style><script>{plotly_js}</script></head>"
        '<body><main class="wrap"><h1>Generated persona dashboard</h1>'
        + "".join(sections)
        + (
            f'<script type="application/json" id="persona-data">'
            f"{safe_data_json}</script>"
        )
        + "</main></body></html>"
    )


def _distribution_chart(
    *,
    frame: pl.DataFrame,
    field: str,
    title: str,
    source: str,
    target: dict[str, float] | None,
) -> str:
    """Create one interactive generated-versus-target distribution chart.

    Returns:
        An HTML section containing the interactive chart and semantic source label.
    """
    generated_counts = _generated_distribution(frame=frame, field=field)
    labels = list(generated_counts)
    if target:
        labels.extend(label for label in target if label not in labels)
    generated = [generated_counts.get(label, 0.0) for label in labels]
    figure = go.Figure()
    figure.add_bar(name="Generated", x=labels, y=generated)
    note = "Synthetic-only; no authoritative target available."
    if target:
        overlay = [target.get(label, 0.0) for label in labels]
        figure.add_scatter(name="DST target", x=labels, y=overlay, mode="lines+markers")
        note = (
            "DST target overlay is descriptive: the frozen sample is stratified and "
            "this is not a statistical acceptance test."
        )
    figure.update_layout(
        title=title,
        yaxis_title="Proportion",
        xaxis_title=title,
        margin={"l": 50, "r": 20, "t": 55, "b": 100},
        legend={"orientation": "h"},
    )
    return _chart_card(
        title=title, figure=figure, source=(f"Semantic source: {source}. " + note)
    )


def _chart_card(
    *, title: str, figure: go.Figure, source: str, wide: bool = False
) -> str:
    """Serialise a Plotly figure into a self-contained page section.

    Returns:
        An HTML chart section without an external JavaScript dependency.
    """
    figure_html = figure.to_html(
        full_html=False, include_plotlyjs=False, config={"responsive": True}
    )
    class_name = "chart wide" if wide else "chart"
    return (
        f'<section class="{class_name}"><h2>{html.escape(title)}</h2>'
        f'<div class="plot">{figure_html}</div><p class="source">'
        f"{html.escape(source)}</p></section>"
    )


def _generated_distribution(*, frame: pl.DataFrame, field: str) -> dict[str, float]:
    """Normalise generated values over the population eligible for a chart.

    Returns:
        Generated proportions in descending count order.
    """
    source = _recorded_job_functions(frame=frame) if field == "job_function" else frame
    if source.is_empty():
        return {}
    values = source.get_column(field).fill_null("(not recorded)").cast(pl.String)
    counts = values.value_counts(sort=True).sort("count", descending=True)
    return {
        str(value): float(count) / source.height
        for value, count in zip(
            counts.get_column(field).to_list(), counts["count"].to_list(), strict=True
        )
    }


def _recorded_job_functions(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Select records with a non-blank job-function label.

    Returns:
        Job-function-eligible generated records.
    """
    if "job_function" not in frame.columns:
        return frame.head(0)
    return frame.filter(
        pl.col("job_function").is_not_null()
        & (pl.col("job_function").cast(pl.String).str.strip_chars() != "")
    )


def _embedding_chart(*, frame: pl.DataFrame) -> str:
    """Create an LSA embedding scatter with demographic colour controls.

    Returns:
        An HTML section containing the interactive embedding chart.
    """
    coordinates = persona_embedding(frame=frame)
    figure = go.Figure()
    colour_options = [
        (field, label) for field, label in COLOUR_FIELDS if field in frame.columns
    ]
    all_traces: list[list[int]] = []
    if colour_options:
        for colour_field, _ in colour_options:
            groups = (
                frame.get_column(colour_field)
                .fill_null("(not recorded)")
                .cast(pl.String)
            )
            for category in sorted(set(groups.to_list())):
                indices = [
                    index for index, value in enumerate(groups) if value == category
                ]
                all_traces.append(indices)
                _add_embedding_trace(
                    figure=figure,
                    frame=frame,
                    coordinates=coordinates,
                    indices=indices,
                    name=f"{colour_field}: {category}",
                    visible=colour_field == colour_options[0][0],
                )
        buttons = _colour_buttons(
            frame=frame, colour_options=colour_options, trace_count=len(all_traces)
        )
    else:
        _add_embedding_trace(
            figure=figure,
            frame=frame,
            coordinates=coordinates,
            indices=list(range(frame.height)),
            name="All personas",
            visible=True,
        )
        buttons = []
    figure.update_layout(
        title="Persona text embedding (TF-IDF + SciPy LSA)",
        xaxis_title="LSA dimension 1",
        yaxis_title="LSA dimension 2",
        updatemenus=(
            [{"buttons": buttons, "x": 0, "y": 1.15, "xanchor": "left"}]
            if buttons
            else []
        ),
        margin={"l": 50, "r": 20, "t": 90, "b": 50},
    )
    return _chart_card(
        title="Persona text embedding",
        figure=figure,
        source=(
            "Semantic source: full persona prose. Word unigram/bigram TF-IDF and "
            "SciPy truncated SVD/LSA; colour selector changes demographic grouping."
        ),
        wide=True,
    )


def _add_embedding_trace(
    *,
    figure: go.Figure,
    frame: pl.DataFrame,
    coordinates: list[tuple[float, float]],
    indices: list[int],
    name: str,
    visible: bool,
) -> None:
    """Add one consistently configured persona embedding trace."""
    figure.add_scatter(
        x=[coordinates[index][0] for index in indices],
        y=[coordinates[index][1] for index in indices],
        mode="markers",
        name=name,
        customdata=[_hover_row(frame, index) for index in indices],
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>"
            "%{customdata[2]}<br>%{customdata[3]}<br>%{customdata[4]}<br>"
            "%{customdata[5]}<extra></extra>"
        ),
        visible=visible,
    )


def _hover_row(frame: pl.DataFrame, index: int) -> list[str]:
    """Build the privacy-conscious but complete persona hover payload.

    Returns:
        Hover fields for a persona point.
    """
    row = frame.row(index, named=True)
    name = str(row.get("first_name", "(unnamed)"))
    demographics = ", ".join(
        f"{label}: {row.get(field, '(not recorded)')}"
        for field, label in (
            ("age", "Age"),
            ("sex", "Sex"),
            ("education_level", "Education"),
            ("origin_country_da", "Origin"),
        )
    )
    location = (
        f"{row.get('municipality', '(unknown municipality)')}, {row.get('region', '')}"
    )
    job = str(
        row.get("job_title")
        or row.get("job_function")
        or row.get("labour_market_status", "")
    )
    return [
        name,
        demographics,
        f"Location: {location}",
        f"Work: {job}",
        "Persona:",
        _wrap_text(str(row.get("persona", ""))),
    ]


def _wrap_text(value: str, width: int = 90) -> str:
    """Insert HTML line breaks into hover prose without altering its content.

    Returns:
        Wrapped text suitable for Plotly hover labels.
    """
    return "<br>".join(
        value[index : index + width] for index in range(0, len(value), width)
    )


def _colour_buttons(
    *, frame: pl.DataFrame, colour_options: list[tuple[str, str]], trace_count: int
) -> list[dict[str, object]]:
    """Build trace-visibility controls for available colour fields.

    Returns:
        Plotly update-menu button definitions.
    """
    buttons: list[dict[str, object]] = []
    offset = 0
    for colour_field, label in colour_options:
        group_count = len(
            set(
                frame.get_column(colour_field)
                .fill_null("(not recorded)")
                .cast(pl.String)
                .to_list()
            )
        )
        visible = [False] * trace_count
        for index in range(offset, offset + group_count):
            visible[index] = True
        buttons.append(
            {"label": label, "method": "update", "args": [{"visible": visible}]}
        )
        offset += group_count
    return buttons


def persona_embedding(*, frame: pl.DataFrame) -> list[tuple[float, float]]:
    """Return deterministic two-dimensional unigram/bigram TF-IDF LSA coordinates."""
    texts = [
        str(value) for value in frame.get_column("persona").fill_null("").to_list()
    ]
    matrix = _tfidf_matrix(texts=texts)
    if matrix.shape[1] == 0:
        return [(0.0, 0.0) for _ in texts]
    return _lsa_coordinates(matrix=matrix)


def _lsa_coordinates(*, matrix: csr_matrix) -> list[tuple[float, float]]:
    """Project TF-IDF rows through deterministic SciPy truncated SVD.

    Returns:
        Two-dimensional coordinates for each input row.
    """
    if min(matrix.shape) <= 1:
        return [(float(matrix[index].sum()), 0.0) for index in range(matrix.shape[0])]
    maximum_components = min(matrix.shape) - 1
    components = min(3, maximum_components)
    while True:
        left, singular, _ = svds(matrix, k=components, solver="arpack", random_state=0)
        order = np.argsort(singular)[::-1]
        singular = singular[order]
        left = left[:, order]
        if not _tie_reaches_component_boundary(
            singular=singular, shape=matrix.shape, maximum=maximum_components
        ):
            break
        components += 1
    left = _canonicalise_tied_subspaces(
        left=left, singular=singular, shape=matrix.shape
    )
    tolerance = (
        np.finfo(float).eps * max(matrix.shape) * singular[0] if singular.size else 0.0
    )
    singular[singular <= tolerance] = 0.0
    coordinates = left * singular
    for component in range(coordinates.shape[1]):
        pivot = int(np.argmax(np.abs(coordinates[:, component])))
        if coordinates[pivot, component] < 0:
            coordinates[:, component] *= -1
    coordinates = np.round(coordinates, decimals=EMBEDDING_DECIMALS)
    coordinates[coordinates == 0.0] = 0.0
    result = [tuple(float(value) for value in row) for row in coordinates]
    return [(row[0], row[1] if len(row) > 1 else 0.0) for row in result]


def _canonicalise_tied_subspaces(
    *, left: np.ndarray, singular: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Choose row-index anchored bases for numerically tied singular values.

    A singular-vector basis is otherwise only defined up to an orthogonal rotation
    when singular values are tied.  Projecting successive standard row anchors into
    each tied left-singular subspace gives a basis independent of the ARPACK basis.

    Returns:
        Left singular vectors with tied subspaces in canonical bases.
    """
    if singular.size < 2:
        return left

    scale = max(1.0, float(singular[0]))
    tie_tolerance = 8.0 * np.finfo(float).eps * max(shape) * scale
    canonical = left.copy()
    start = 0
    while start < singular.size:
        end = start + 1
        while (
            end < singular.size
            and abs(singular[end] - singular[start]) <= tie_tolerance
        ):
            end += 1
        if end - start > 1:
            canonical[:, start:end] = _anchor_subspace(
                basis=left[:, start:end], tolerance=tie_tolerance
            )
        start = end
    return canonical


def _anchor_subspace(*, basis: np.ndarray, tolerance: float) -> np.ndarray:
    """Construct an orthonormal subspace basis from fixed row-index anchors.

    Returns:
        An orthonormal basis selected by the lowest available row indices.

    Raises:
        ValueError:
            If the supplied basis does not span the requested dimension.
    """
    dimension = basis.shape[1]
    vectors: list[np.ndarray] = []
    for row_index in range(basis.shape[0]):
        vector = basis @ basis[row_index, :]
        for previous in vectors:
            vector -= np.dot(previous, vector) * previous
        norm = float(np.linalg.norm(vector))
        if norm <= tolerance:
            continue
        vectors.append(vector / norm)
        if len(vectors) == dimension:
            break
    if len(vectors) != dimension:
        raise ValueError("Unable to construct a basis for a tied singular subspace")
    return np.column_stack(vectors)


def _tie_reaches_component_boundary(
    *, singular: np.ndarray, shape: tuple[int, int], maximum: int
) -> bool:
    """Report whether another component may belong to the final tied subspace.

    Returns:
        Whether the final computed singular value is tied at the truncation boundary.
    """
    if singular.size < 2 or singular.size >= maximum:
        return False
    scale = max(1.0, float(singular[0]))
    tolerance = 8.0 * np.finfo(float).eps * max(shape) * scale
    return singular[-1] > tolerance and abs(singular[-1] - singular[-2]) <= tolerance


def _tfidf_matrix(*, texts: list[str]) -> csr_matrix:
    """Build a row-normalised sparse word unigram/bigram TF-IDF matrix.

    Returns:
        Sparse TF-IDF matrix with one row per persona.
    """
    vocabulary: dict[str, int] = {}
    documents: list[list[str]] = []
    for text in texts:
        tokens = [token.casefold() for token in TOKEN_RE.findall(text)]
        terms = tokens + [f"{left} {right}" for left, right in zip(tokens, tokens[1:])]
        documents.append(terms)
        for term in terms:
            if term not in vocabulary:
                vocabulary[term] = len(vocabulary)
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for row_index, terms in enumerate(documents):
        counts: dict[int, float] = {}
        for term in terms:
            column = vocabulary[term]
            counts[column] = counts.get(column, 0.0) + 1.0
        for column, value in counts.items():
            rows.append(row_index)
            cols.append(column)
            values.append(value)
    matrix = csr_matrix(
        (values, (rows, cols)), shape=(len(texts), len(vocabulary)), dtype=float
    )
    document_frequency = np.asarray((matrix > 0).sum(axis=0)).ravel()
    matrix = matrix.multiply(
        np.log((1.0 + len(texts)) / (1.0 + document_frequency)) + 1.0
    ).tocsr()
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    return matrix.multiply((1.0 / norms)[:, None]).tocsr()


def _json_safe_row(row: dict[str, object]) -> dict[str, object]:
    """Convert Polars scalar values to JSON-compatible values.

    Returns:
        A row containing only JSON-compatible scalar values.
    """
    result: dict[str, object] = {}
    for key, value in row.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            result[key] = value
        else:
            result[key] = str(value)
    return result


def _ocean_chart(*, frame: pl.DataFrame) -> str:
    """Create the interactive OCEAN score chart.

    Returns:
        An HTML section containing the interactive chart and source label.
    """
    figure = go.Figure()
    available = [
        (field, label) for field, label in OCEAN_FIELDS if field in frame.columns
    ]
    for field, label in available:
        figure.add_box(
            y=frame.get_column(field).to_list(),
            name=label,
            boxmean=True,
            showlegend=False,
        )
    figure.update_layout(
        title="OCEAN scores",
        yaxis_title="Score (20-80)",
        margin={"l": 50, "r": 20, "t": 55, "b": 50},
    )
    return _chart_card(
        title="OCEAN scores",
        figure=figure,
        source=(
            "Semantic source: deterministic synthetic personality tendencies; "
            "synthetic-only."
        ),
    )


def _overview_cards(*, frame: pl.DataFrame) -> str:
    """Render compact overview metrics.

    Returns:
        An HTML overview section.
    """
    municipality_count = (
        frame.get_column("municipality").n_unique()
        if "municipality" in frame.columns
        else "-"
    )
    mean_age = frame.get_column("age").mean() if "age" in frame.columns else None
    mean_age_text = f"{mean_age:.1f}" if mean_age is not None else "-"
    return (
        '<section class="grid"><div class="card"><div class="muted">Personas</div>'
        f'<div class="metric">{frame.height}</div></div>'
        '<div class="card"><div class="muted">Municipalities</div>'
        f'<div class="metric">{municipality_count}</div></div>'
        '<div class="card"><div class="muted">Mean age</div>'
        f'<div class="metric">{mean_age_text}</div></div></section>'
    )


def _provenance(
    *,
    frame: pl.DataFrame,
    bundle_path: Path,
    input_path: Path | None,
    targets: dict[str, dict[str, float]],
) -> str:
    """Render checksums, source labels, and the stratification caveat.

    Returns:
        An HTML provenance section.
    """
    input_checksum = (
        _file_checksum(input_path) if input_path is not None else _frame_checksum(frame)
    )
    bundle_manifest = bundle_path / "bundle-manifest.json"
    manifest_checksum = (
        _file_checksum(bundle_manifest) if bundle_manifest.exists() else "missing"
    )
    target_names = ", ".join(sorted(targets)) or "none found"
    return (
        '<section class="card wide"><h2>Provenance and semantics</h2><p>'
        f"Input Parquet SHA-256: <code>{input_checksum}</code><br>"
        f"Bundle manifest SHA-256: <code>{manifest_checksum}</code><br>"
        f"Target overlays loaded: {html.escape(target_names)}<br>"
        "Semantic-source labels identify whether a chart is synthetic or from a "
        'normalised Statistics Denmark prepared file.</p><p class="muted">'
        "The frozen sample is stratified; DST overlays are descriptive comparisons, "
        "not a statistical acceptance test.</p></section>"
    )


def _file_checksum(path: Path) -> str:
    """Return a file's SHA-256 checksum."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_checksum(frame: pl.DataFrame) -> str:
    """Hash a stable JSON representation of the supplied records.

    Returns:
        SHA-256 checksum of the canonical record representation.
    """
    payload = json.dumps(
        [_json_safe_row(row) for row in frame.to_dicts()],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_columns(frame: pl.DataFrame, columns: set[str]) -> None:
    """Fail with a useful message when the generated contract is incomplete.

    Raises:
        ValueError:
            If one or more required columns are absent.
    """
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(
            f"Generated personas are missing columns: {', '.join(sorted(missing))}"
        )


def load_dst_targets(
    *, bundle_path: Path, frame: pl.DataFrame | None = None
) -> dict[str, dict[str, float]]:
    """Load equivalent Statistics Denmark marginal targets.

    Missing normalised files are ignored so that synthetic-only development bundles
    remain useful. Education source categories are pooled to the generated contract.
    The job-function marginal is included only when generated records permit the
    official sex-conditionals to be mixed over an equivalent eligible denominator.

    Args:
        bundle_path:
            Prepared bundle directory.
        frame (optional):
            Generated records used to condition job-function targets. Defaults to
            ``None``, which omits that target.

    Returns:
        Target proportions by generated semantic field and displayed label.
    """
    normalized = bundle_path / "normalized"
    targets: dict[str, dict[str, float]] = {}
    specifications = (
        ("folk_age_sampling", ("age", "age_band", "sex", "region", "municipality")),
        ("folk_marital_sampling", ("marital_status",)),
        ("folk2_origin_country_marginal", ("origin_country_da",)),
    )
    for stem, fields in specifications:
        source = _read_optional_target(normalized=normalized, stem=stem)
        if source is None:
            continue
        for field in fields:
            target_field = _target_column(field=field, columns=source.columns)
            if target_field is not None:
                targets[field] = _normalise_counts(
                    source=source, value_column=target_field
                )

    ras209 = _read_optional_target(normalized=normalized, stem="ras209_joint_unpooled")
    if ras209 is not None:
        categories = load_yaml_model(CATEGORY_CONFIG_PATH, CategoryConfig)
        targets["education_level"] = _pooled_education_target(
            source=ras209, pooling=categories.education_pooling
        )
        targets["labour_market_status"] = _normalise_counts(
            source=ras209, value_column="labour_market_status"
        )

    job_source = _read_optional_target(
        normalized=normalized, stem="job_function_sex_marginal"
    )
    if job_source is not None and frame is not None:
        job_target = _conditioned_job_function_target(source=job_source, frame=frame)
        if job_target is not None:
            targets["job_function"] = job_target
    return targets


def _conditioned_job_function_target(
    *, source: pl.DataFrame, frame: pl.DataFrame
) -> dict[str, float] | None:
    """Mix official sex-conditionals using generated eligible sex shares.

    Returns:
        An equivalent job-function target, or ``None`` when equivalence cannot be
        established from the available columns and recorded values.
    """
    required_source = {"job_function", "sex", "count"}
    required_generated = {"job_function", "sex"}
    if not required_source.issubset(source.columns) or not required_generated.issubset(
        frame.columns
    ):
        return None
    eligible = _recorded_job_functions(frame=frame)
    if eligible.is_empty():
        return None
    generated_sexes = eligible.get_column("sex")
    if generated_sexes.null_count() or any(
        not str(value).strip() for value in generated_sexes.to_list()
    ):
        return None
    sex_counts = eligible.group_by("sex").len()
    total_eligible = float(eligible.height)
    target: dict[str, float] = {}
    for row in sex_counts.to_dicts():
        sex = row["sex"]
        stratum = source.filter(pl.col("sex") == sex)
        conditional = _normalise_counts(source=stratum, value_column="job_function")
        if not conditional:
            return None
        weight = float(row["len"]) / total_eligible
        for label, proportion in conditional.items():
            target[label] = target.get(label, 0.0) + weight * proportion
    return target


def _normalise_counts(*, source: pl.DataFrame, value_column: str) -> dict[str, float]:
    """Aggregate duplicate target rows and convert counts to proportions.

    Returns:
        Normalised counts keyed by the displayed target value.
    """
    grouped = source.group_by(value_column).agg(pl.col("count").sum())
    total = float(grouped.get_column("count").sum())
    if total <= 0:
        return {}
    return {
        str(row[value_column]): float(row["count"]) / total
        for row in grouped.to_dicts()
        if row[value_column] is not None
    }


def _pooled_education_target(
    *, source: pl.DataFrame, pooling: dict[str, str]
) -> dict[str, float]:
    """Pool unpooled RAS209 education labels to generated categories.

    Returns:
        Normalised pooled education proportions.
    """
    pooled = source.with_columns(
        pl.col("education_level").replace_strict(pooling).alias("education_level")
    )
    return _normalise_counts(source=pooled, value_column="education_level")


def _read_optional_target(*, normalized: Path, stem: str) -> pl.DataFrame | None:
    """Read an optional count target.

    Returns:
        The target frame, or ``None`` when absent or lacking counts.
    """
    path = normalized / f"{stem}.parquet"
    if not path.exists():
        return None
    source = pl.read_parquet(path)
    return source if "count" in source.columns else None


def _target_column(*, field: str, columns: list[str]) -> str | None:
    """Find a semantic target column, allowing official naming variants.

    Returns:
        The matching source column, or ``None`` when no match exists.
    """
    candidates = {
        "origin_country_da": ("origin_country_da", "origin_country"),
        "job_function": ("job_function", "job_function_code"),
    }.get(field, (field,))
    return next((candidate for candidate in candidates if candidate in columns), None)


if __name__ == "__main__":
    main()
