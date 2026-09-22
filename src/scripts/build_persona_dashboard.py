"""Build a self-contained interactive dashboard for generated personas."""

import html
import json
import logging
import sys
from pathlib import Path

import httpx
import hydra
import numpy as np
import plotly.graph_objects as go
import polars as pl
from omegaconf import DictConfig
from plotly.offline import get_plotlyjs
from umap import UMAP

from danish_personas.cli_logging import configure_cli_logging
from danish_personas.environment import load_repository_environment
from danish_personas.hydra_cli import enable_hydra_cli
from danish_personas.io import load_yaml_model
from danish_personas.models import CategoryConfig
from danish_personas.script_config import PersonaDashboardConfig, load_script_config

LOGGER = logging.getLogger(__name__)

DISTRIBUTIONS: tuple[tuple[str, str, str], ...] = (
    ("age", "Ages", "folk_age_sampling.parquet"),
    ("age_band", "Age bands", "folk_age_sampling.parquet"),
    ("sex", "Sex", "folk_age_sampling.parquet"),
    ("marital_status", "Marital status", "folk_marital_sampling.parquet"),
    ("education_level", "Education", "ras209_joint_unpooled.parquet"),
    ("labour_market_status", "Labour-market status", "ras209_joint_unpooled.parquet"),
    ("region", "Regions", "folk_age_sampling.parquet"),
    (
        "municipality",
        "Municipalities",
        "ras209_joint_unpooled.parquet (RAS209 municipality marginal)",
    ),
    (
        "origin_country_da",
        "Origin-country labels",
        "folk2_origin_country_marginal.parquet",
    ),
    ("job_function", "Job functions", "job_function_sex_marginal.parquet"),
    ("current_relationship_status", "Current relationship status", "synthetic"),
    ("relationship_pair", "Partner relationship pair", "synthetic"),
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
CATEGORY_CONFIG_PATH = Path(__file__).parents[2] / "config" / "categories.yaml"
DEFAULT_EMBEDDING_BASE_URL = "http://127.0.0.1:18080/v1"
DEFAULT_EMBEDDING_MODEL = "jina-embeddings-v5-text-small-clustering"
DEFAULT_EMBEDDING_BATCH_SIZE = 32
EMBEDDING_DECIMALS = 12
NOT_STATED = "not_stated"
DOMESTIC_ORIGIN = "danmark"
DOMESTIC_ORIGIN_LABELS = frozenset({"danmark", "denmark"})
ORIGIN_ELIGIBILITY_COLUMN = "eligible_for_sampling"
HORIZONTAL_FIELDS = frozenset(
    {
        "municipality",
        "origin_country_da",
        "job_function",
        "education_level",
        "labour_market_status",
        "marital_status",
        "relationship_pair",
    }
)


enable_hydra_cli()


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config: DictConfig) -> None:
    """Build one offline, interactive HTML persona dashboard.

    Raises:
        SystemExit:
            If configuration, input validation, or rendering fails.
    """
    configure_cli_logging()
    try:
        _run(config=config)
    except Exception as error:
        LOGGER.error("%s", error)
        raise SystemExit(1) from error


def _run(*, config: DictConfig) -> None:
    """Build a dashboard from a composed Hydra configuration.

    Raises:
        ValueError:
            If the selected input is empty or cannot be rendered.
    """
    script_config = load_script_config(
        config, section="persona_dashboard", model=PersonaDashboardConfig
    )
    frame = pl.read_parquet(script_config.input)
    if frame.is_empty():
        raise ValueError("The generated personas file is empty")
    document = build_dashboard(
        frame=frame,
        bundle_path=script_config.bundle,
        input_path=script_config.input,
        embedding_base_url=script_config.embedding_base_url,
        embedding_model=script_config.embedding_model,
        embedding_batch_size=script_config.embedding_batch_size,
    )
    script_config.output.parent.mkdir(parents=True, exist_ok=True)
    script_config.output.write_text(document, encoding="utf-8")
    sys.stdout.write(f"{script_config.output}\n")


def build_dashboard(
    *,
    frame: pl.DataFrame,
    bundle_path: Path,
    input_path: Path | None = None,
    embedding_base_url: str = DEFAULT_EMBEDDING_BASE_URL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    embedding_client: httpx.Client | None = None,
) -> str:
    """Render a complete dashboard document from generated records.

    Args:
        frame:
            Generated persona records, including their full persona text.
        bundle_path:
            Prepared bundle directory used for descriptive DST overlays.
        input_path (optional):
            Deprecated input path accepted for caller compatibility. Defaults to
            ``None``.
        embedding_base_url (optional):
            Base URL for the local OpenAI-compatible embeddings service. Defaults to
            ``DEFAULT_EMBEDDING_BASE_URL``.
        embedding_model (optional):
            Embedding model alias. Defaults to ``DEFAULT_EMBEDDING_MODEL``.
        embedding_batch_size (optional):
            Maximum number of texts per request. Defaults to
            ``DEFAULT_EMBEDDING_BATCH_SIZE``.
        embedding_client (optional):
            HTTP client used by offline callers and tests. Defaults to ``None``.

    Returns:
        A single HTML document with all JavaScript and data embedded.
    """
    _require_columns(frame, {"persona", "persona_id"})
    origin_threshold = _validate_dashboard_origins(frame=frame, bundle_path=bundle_path)
    target_cache = load_dst_targets(
        bundle_path=bundle_path, frame=frame, minimum_source_count=origin_threshold
    )
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
        if _chart_field_available(frame=frame, field=field)
    )
    sections.append(_ocean_chart(frame=frame))
    sections.append(
        _embedding_chart(
            frame=frame,
            base_url=embedding_base_url,
            model=embedding_model,
            batch_size=embedding_batch_size,
            http_client=embedding_client,
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


def _chart_field_available(*, frame: pl.DataFrame, field: str) -> bool:
    """Report whether a regular or derived distribution can be displayed.

    Returns:
        Whether the chart has the columns needed to produce values.
    """
    if field == "relationship_pair":
        return {"sex", "partner_gender"}.issubset(frame.columns)
    return field in frame.columns


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
    if target and field in {"education_level", "origin_country_da"}:
        excluded = (
            {NOT_STATED} if field == "education_level" else DOMESTIC_ORIGIN_LABELS
        )
        target = {
            label: value
            for label, value in target.items()
            if label.casefold() not in excluded
        }
        total_target = sum(target.values())
        if total_target > 0:
            target = {label: value / total_target for label, value in target.items()}
    if target:
        labels.extend(label for label in target if label not in labels)
    if field == "age":
        labels.sort(key=int)
    generated = [generated_counts.get(label, 0.0) for label in labels]
    horizontal = field in HORIZONTAL_FIELDS
    figure = go.Figure()
    if horizontal:
        figure.add_bar(name="Generated", y=labels, x=generated, orientation="h")
    else:
        figure.add_bar(name="Generated", x=labels, y=generated)
    note = "Synthetic-only; no authoritative target available."
    if target:
        overlay = [target.get(label, 0.0) for label in labels]
        if horizontal:
            figure.add_scatter(
                name="DST target", y=labels, x=overlay, mode="lines+markers"
            )
        else:
            figure.add_scatter(
                name="DST target", x=labels, y=overlay, mode="lines+markers"
            )
        note = (
            "Overlay is the threshold-eligible DST universe. Small persona "
            "outputs/shards cannot generally reproduce the full marginal; "
            "merged/frozen outputs are the meaningful comparison, not a statistical "
            "acceptance test."
        )
    chart_height = min(6_000, max(390, 130 + 22 * len(labels))) if horizontal else 390
    figure.update_layout(
        height=chart_height,
        yaxis_title="Category" if horizontal else "Proportion",
        xaxis_title="Proportion" if horizontal else title,
        margin=(
            {"l": 210, "r": 25, "t": 85, "b": 55}
            if horizontal
            else {"l": 55, "r": 25, "t": 85, "b": 105}
        ),
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
    )
    return _chart_card(
        title=title,
        figure=figure,
        source=(f"Semantic source: {source}. " + note),
        height=chart_height,
    )


def _chart_card(
    *, title: str, figure: go.Figure, source: str, wide: bool = False, height: int = 390
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
        f'<div class="plot" style="height:{height}px">{figure_html}</div>'
        '<p class="source">'
        f"{html.escape(source)}</p></section>"
    )


def _generated_distribution(*, frame: pl.DataFrame, field: str) -> dict[str, float]:
    """Normalise generated values over the population eligible for a chart.

    Returns:
        Generated proportions in descending count order.
    """
    if field == "relationship_pair":
        values = _relationship_pairs(frame=frame)
        if not values:
            return {}
        counts: dict[str, float] = {}
        for value in values:
            counts[value] = counts.get(value, 0.0) + 1.0
        total = float(len(values))
        return {label: count / total for label, count in counts.items()}
    source = _recorded_job_functions(frame=frame) if field == "job_function" else frame
    if source.is_empty():
        return {}
    values = source.get_column(field).fill_null("(not recorded)").cast(pl.String)
    if field == "education_level":
        values = values.filter(values.str.to_lowercase() != NOT_STATED)
    elif field == "origin_country_da":
        values = values.filter(values.str.to_lowercase() != DOMESTIC_ORIGIN)
    if values.is_empty():
        return {}
    counts = values.value_counts(sort=True).sort("count", descending=True)
    total = float(values.len())
    return {
        str(value): float(count) / total
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


def _relationship_pairs(*, frame: pl.DataFrame) -> list[str]:
    """Return partnered records as one of the three relationship pair labels."""
    required = {"sex", "partner_gender"}
    if not required.issubset(frame.columns):
        return []
    partnered = frame
    if "current_relationship_status" in frame.columns:
        partnered = frame.filter(
            pl.col("current_relationship_status").cast(pl.String).str.to_lowercase()
            == "partnered"
        )
    pairs: list[str] = []
    for row in partnered.select(["sex", "partner_gender"]).iter_rows(named=True):
        sex = _gender_label(row["sex"])
        partner = _gender_label(row["partner_gender"])
        if sex is None or partner is None:
            continue
        if sex == partner == "man":
            pairs.append("man-man")
        elif sex == partner == "woman":
            pairs.append("woman-woman")
        else:
            pairs.append("man-woman")
    return pairs


def _gender_label(value: object) -> str | None:
    """Map a generated sex value to the relationship chart's labels.

    Returns:
        The normalised gender label, or ``None`` for an unknown value.
    """
    normalised = str(value).strip().casefold() if value is not None else ""
    if normalised in {"male", "man", "m", "mand"}:
        return "man"
    if normalised in {"female", "woman", "f", "kvinde"}:
        return "woman"
    return None


def _embedding_chart(
    *,
    frame: pl.DataFrame,
    base_url: str,
    model: str,
    batch_size: int,
    http_client: httpx.Client | None,
) -> str:
    """Create an embedding scatter with demographic colour controls.

    Returns:
        An HTML section containing the interactive embedding chart.
    """
    coordinates = persona_embedding(
        frame=frame,
        base_url=base_url,
        model=model,
        batch_size=batch_size,
        http_client=http_client,
    )
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
        xaxis_title="UMAP dimension 1",
        yaxis_title="UMAP dimension 2",
        updatemenus=(
            [{"buttons": buttons, "x": 0, "y": 1.15, "xanchor": "left"}]
            if buttons
            else []
        ),
        showlegend=False,
        margin={"l": 55, "r": 25, "t": 105, "b": 55},
    )
    return _chart_card(
        title="Persona text embedding",
        figure=figure,
        source=(
            "Semantic source: full persona prose embedded by the configured local "
            f"OpenAI-compatible model ({model}), then reduced with "
            "deterministic two-dimensional UMAP; colour selector changes demographic "
            "grouping."
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
    pronouns = row.get("pronouns") or row.get("pronoun")
    label = f"Pronouns: {pronouns}" if pronouns else "Unnamed persona"
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
        label,
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


def persona_embedding(
    *,
    frame: pl.DataFrame,
    base_url: str = DEFAULT_EMBEDDING_BASE_URL,
    model: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    http_client: httpx.Client | None = None,
) -> list[tuple[float, float]]:
    """Embed persona prose through a local OpenAI-compatible service and reduce it.

    Args:
        frame:
            Generated records containing a ``persona`` column.
        base_url (optional):
            Service base URL. Defaults to ``DEFAULT_EMBEDDING_BASE_URL``.
        model (optional):
            Embedding model alias. Defaults to ``DEFAULT_EMBEDDING_MODEL``.
        batch_size (optional):
            Number of texts per request. Defaults to ``DEFAULT_EMBEDDING_BATCH_SIZE``.
        http_client (optional):
            HTTP client, useful for offline tests. Defaults to ``None``.

    Returns:
        Deterministic two-dimensional UMAP coordinates in input row order.

    Raises:
        ValueError:
            If the service response is malformed or the batch size is invalid.
    """
    if batch_size < 1:
        raise ValueError("Embedding batch size must be at least one")
    texts = [
        str(value) for value in frame.get_column("persona").fill_null("").to_list()
    ]
    owns_client = http_client is None
    client = http_client or httpx.Client(timeout=120.0)
    try:
        vectors = OpenAIEmbeddingClient(
            base_url=base_url, model=model, batch_size=batch_size, http_client=client
        ).embed(texts=texts)
    finally:
        if owns_client:
            client.close()
    return _umap_coordinates(vectors=vectors)


class OpenAIEmbeddingClient:
    """Small client for an OpenAI-compatible ``/v1/embeddings`` endpoint."""

    def __init__(
        self, *, base_url: str, model: str, batch_size: int, http_client: httpx.Client
    ) -> None:
        """Initialise a client with a caller-owned HTTP transport."""
        self.endpoint = _embedding_endpoint(base_url)
        self.model = model
        self.batch_size = batch_size
        self.http_client = http_client

    def embed(self, *, texts: list[str]) -> list[list[float]]:
        """Return one vector per supplied text, requesting bounded batches.

        Raises:
            ValueError:
                If a response does not contain one valid vector per input text.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = self.http_client.post(
                self.endpoint, json={"input": batch, "model": self.model}
            )
            response.raise_for_status()
            try:
                payload = response.json()
            except (TypeError, ValueError) as error:
                raise ValueError("Embedding service returned invalid JSON") from error
            vectors.extend(_vectors_from_response(payload=payload, expected=len(batch)))
        return vectors


def _embedding_endpoint(base_url: str) -> str:
    """Resolve a service base URL to its OpenAI embeddings route.

    Returns:
        The complete embeddings endpoint URL.
    """
    clean = base_url.rstrip("/")
    if clean.endswith("/v1/embeddings"):
        return clean
    if clean.endswith("/v1"):
        return f"{clean}/embeddings"
    return f"{clean}/v1/embeddings"


def _vectors_from_response(*, payload: object, expected: int) -> list[list[float]]:
    """Validate and order an OpenAI embeddings response.

    Returns:
        Validated vectors in request order.

    Raises:
        ValueError:
            If the payload does not contain matching, equal-length vectors.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("Embedding service response has no data list")
    data = payload["data"]
    if len(data) != expected:
        raise ValueError(
            f"Embedding service returned {len(data)} vectors; expected {expected}"
        )
    indices = [
        item.get("index")
        if isinstance(item, dict)
        and isinstance(item.get("index"), int)
        and not isinstance(item.get("index"), bool)
        else None
        for item in data
    ]
    expected_indices = set(range(expected))
    if any(index is None for index in indices) or set(indices) != expected_indices:
        raise ValueError(
            "Embedding service returned invalid indices; expected each integer index "
            "from 0 to batch size minus 1"
        )
    ordered = sorted(data, key=lambda item: item["index"])
    vectors: list[list[float]] = []
    for item in ordered:
        vector = item.get("embedding") if isinstance(item, dict) else None
        if (
            not isinstance(vector, list)
            or not vector
            or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in vector
            )
        ):
            raise ValueError("Embedding service returned an invalid vector")
        vectors.append([float(value) for value in vector])
    dimensions = len(vectors[0]) if vectors else 0
    if any(len(vector) != dimensions for vector in vectors):
        raise ValueError("Embedding service returned vectors with different dimensions")
    return vectors


def _umap_coordinates(*, vectors: list[list[float]]) -> list[tuple[float, float]]:
    """Return deterministic two-dimensional UMAP coordinates.

    Returns:
        Two-dimensional coordinates in input order. Empty and constant input is
        represented by the origin, while two distinct points use a deterministic
        line because UMAP requires at least two neighbours.
    """
    if not vectors:
        return []
    matrix = np.asarray(vectors, dtype=float)
    if matrix.shape[1] == 0 or not np.any(matrix - matrix[0]):
        return [(0.0, 0.0) for _ in vectors]
    if len(vectors) == 1:
        return [(0.0, 0.0)]
    if len(vectors) == 2:
        return [(0.0, 0.0), (1.0, 0.0)]

    reducer = UMAP(
        n_neighbors=min(15, len(vectors) - 1),
        n_components=2,
        random_state=0,
        transform_seed=0,
        init="random",
    )
    coordinates = np.asarray(reducer.fit_transform(matrix), dtype=float)
    coordinates = np.round(coordinates, decimals=EMBEDDING_DECIMALS)
    coordinates[coordinates == 0.0] = 0.0
    return [(float(row[0]), float(row[1])) for row in coordinates]


def _json_safe_row(row: dict[str, object]) -> dict[str, object]:
    """Convert Polars scalar values to JSON-compatible values.

    Returns:
        A row containing only JSON-compatible scalar values.
    """
    result: dict[str, object] = {}
    for key, value in row.items():
        if key in {"first_name", "partner_first_name"}:
            continue
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
        yaxis_title="Score (20-80)", margin={"l": 55, "r": 25, "t": 55, "b": 50}
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


def _validate_dashboard_origins(
    *, frame: pl.DataFrame, bundle_path: Path
) -> int | None:
    """Validate generated origin rows before embedding them in dashboard HTML.

    Synthetic-only development frames may omit all origin fields, but a frame and
    bundle that claim to contain origins must use a current, threshold-bound source
    bundle. Full rows are embedded in the output, so rejecting a bad row is safer
    than rendering it with a warning.

    Returns:
        The bundle-bound origin eligibility threshold, or ``None`` for a synthetic-only
        dashboard.

    """
    origin_fields = {"origin_country_code", "origin_country", "origin_country_da"}
    present_fields = origin_fields & set(frame.columns)
    target_path = bundle_path / "normalized" / "folk2_origin_country_marginal.parquet"
    if not target_path.exists() and present_fields <= {"origin_country_da"}:
        return None
    _require_dashboard_origin_shape(
        present_fields=present_fields, target_exists=target_path.exists()
    )
    threshold = _dashboard_origin_threshold(bundle_path=bundle_path)
    source = _dashboard_origin_source(target_path=target_path)
    _validate_origin_threshold(source=source, minimum_source_count=threshold)
    _validate_dashboard_origin_rows(
        frame=frame, source=source, minimum_source_count=threshold
    )
    return threshold


def _dashboard_origin_source(*, target_path: Path) -> pl.DataFrame:
    """Load the complete origin target required for generated-row validation.

    Returns:
        The validated origin target.

    Raises:
        ValueError:
            If the target is missing or lacks required columns.
    """
    source = _read_origin_target(normalized=target_path.parent)
    if source is None:
        raise ValueError("Dashboard origin target is missing")
    required = {
        "origin_country_code",
        "origin_country",
        "origin_country_da",
        "count",
        ORIGIN_ELIGIBILITY_COLUMN,
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Dashboard origin target is missing columns: {missing}")
    return source


def _read_origin_target(*, normalized: Path) -> pl.DataFrame | None:
    """Read and validate the origin-country dashboard target.

    Returns:
        The target frame, or ``None`` when the target is absent.

    Raises:
        ValueError:
            If an existing target does not carry a non-null Boolean eligibility
            column or a count column.
    """
    path = normalized / "folk2_origin_country_marginal.parquet"
    if not path.exists():
        return None
    source = pl.read_parquet(path)
    eligibility = source.schema.get(ORIGIN_ELIGIBILITY_COLUMN)
    if (
        eligibility != pl.Boolean
        or source.get_column(ORIGIN_ELIGIBILITY_COLUMN).null_count()
    ):
        raise ValueError(
            "Origin-country target "
            f"{path} must contain a non-null Boolean "
            f"{ORIGIN_ELIGIBILITY_COLUMN!r} column; legacy or malformed targets "
            "are unsupported"
        )
    if "count" not in source.columns:
        raise ValueError(f"Origin-country target {path} is missing its 'count' column")
    return source


def _dashboard_origin_threshold(*, bundle_path: Path) -> int:
    """Read the current bundle's FOLK2 eligibility threshold.

    Returns:
        The non-negative threshold bound into the manifest.

    Raises:
        ValueError:
            If the manifest is absent, malformed, legacy, or has no valid threshold.
    """
    manifest_path = bundle_path / "bundle-manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "Origin dashboard data requires a readable current bundle-manifest.json"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("Origin dashboard bundle manifest is malformed")
    schema = payload.get("prepared_bundle_schema_version")
    if schema != 8:
        raise ValueError(
            f"Origin dashboard data requires prepared bundle schema 8; found {schema!r}"
        )
    threshold = payload.get("minimum_source_count")
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 0:
        raise ValueError(
            "Origin dashboard bundle manifest has an invalid minimum_source_count"
        )
    return threshold


def _require_dashboard_origin_shape(
    *, present_fields: set[str], target_exists: bool
) -> None:
    """Reject incomplete origin claims instead of rendering full bad rows.

    Raises:
        ValueError:
            If generated origin fields are incomplete for the selected bundle.
    """
    origin_fields = {"origin_country_code", "origin_country", "origin_country_da"}
    if present_fields != origin_fields:
        if not target_exists and not present_fields:
            raise ValueError(
                "Generated frame claims origin data, but its target is missing"
            )
        raise ValueError(
            "Dashboard origin data must contain origin_country_code, "
            "origin_country, and origin_country_da together"
        )


def _validate_dashboard_origin_rows(
    *, frame: pl.DataFrame, source: pl.DataFrame, minimum_source_count: int
) -> None:
    """Reject unknown, mismatched, or ineligible generated origin rows.

    Raises:
        ValueError:
            If any generated row has an unknown, mismatched, or ineligible origin.
    """
    expected = {
        str(row["origin_country_code"]): (
            row["origin_country"],
            row["origin_country_da"],
            bool(row[ORIGIN_ELIGIBILITY_COLUMN]),
            int(row["count"]),
        )
        for row in source.to_dicts()
    }
    invalid_rows: list[str] = []
    for index, row in enumerate(
        frame.select(
            "origin_country_code", "origin_country", "origin_country_da"
        ).to_dicts()
    ):
        code = row["origin_country_code"]
        details = expected.get(str(code)) if code is not None else None
        if details is None:
            invalid_rows.append(f"row {index}: unknown origin code {code!r}")
            continue
        english, danish, eligible, count = details
        if row["origin_country"] != english or row["origin_country_da"] != danish:
            invalid_rows.append(
                f"row {index}: origin labels do not match code {code!r}"
            )
        if not eligible or count < minimum_source_count:
            invalid_rows.append(f"row {index}: origin code {code!r} is ineligible")
    if invalid_rows:
        raise ValueError(
            "Generated dashboard origin rows are invalid: "
            + "; ".join(invalid_rows[:3])
        )


def _validate_origin_threshold(
    *, source: pl.DataFrame, minimum_source_count: int
) -> None:
    """Ensure origin flags are derived from the bundle-bound threshold.

    Raises:
        ValueError:
            If counts are invalid or eligibility flags do not match the threshold.
    """
    try:
        expected = source.get_column("count") >= minimum_source_count
    except (TypeError, pl.exceptions.PolarsError) as error:
        raise ValueError("Dashboard origin target has invalid count values") from error
    actual = source.get_column(ORIGIN_ELIGIBILITY_COLUMN)
    if actual.to_list() != expected.to_list():
        raise ValueError(
            "Dashboard origin eligibility does not match bundle minimum_source_count"
        )


def load_dst_targets(
    *,
    bundle_path: Path,
    frame: pl.DataFrame | None = None,
    minimum_source_count: int | None = None,
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
        minimum_source_count (optional):
            Bundle-bound FOLK2 eligibility threshold. Defaults to ``None``; a current
            bundle manifest is used automatically, while manifest-less synthetic target
            fixtures retain their existing flags for direct inspection.

    Returns:
        Target proportions by generated semantic field and displayed label.
    """
    normalized = bundle_path / "normalized"
    targets: dict[str, dict[str, float]] = {}
    minimum_source_count = _resolve_origin_threshold(
        bundle_path=bundle_path,
        normalized=normalized,
        minimum_source_count=minimum_source_count,
    )
    origin_target = _origin_dashboard_target(
        normalized=normalized, minimum_source_count=minimum_source_count
    )
    if origin_target is not None:
        targets["origin_country_da"] = origin_target

    specifications = (
        ("folk_age_sampling", ("age", "age_band", "sex", "region")),
        ("folk_marital_sampling", ("marital_status",)),
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
        # The unpooled joint is retained as an audit artefact, but undisclosed
        # education rows are outside every dashboard target's eligible universe.
        ras209 = _eligible_ras209_dashboard_rows(source=ras209)
        categories = load_yaml_model(CATEGORY_CONFIG_PATH, CategoryConfig)
        targets["education_level"] = _pooled_education_target(
            source=ras209, pooling=categories.education_pooling
        )
        targets["labour_market_status"] = _normalise_counts(
            source=ras209, value_column="labour_market_status"
        )
        if "municipality" in ras209.columns:
            targets["municipality"] = _normalise_counts(
                source=ras209, value_column="municipality"
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
    if value_column == "education_level":
        grouped = grouped.filter(
            pl.col(value_column).cast(pl.String).str.to_lowercase() != NOT_STATED
        )
    total = float(grouped.get_column("count").sum()) if grouped.height else 0.0
    if total <= 0:
        return {}
    return {
        str(row[value_column]): float(row["count"]) / total
        for row in grouped.to_dicts()
        if row[value_column] is not None
    }


def _eligible_ras209_dashboard_rows(*, source: pl.DataFrame) -> pl.DataFrame:
    """Exclude RAS209 undisclosed education rows from all dashboard targets.

    The source file itself is never rewritten: this returns a filtered view used only
    for descriptive overlays. Both the official H90 code and its normalised semantic
    label are checked because prepared bundles expose one or both representations.

    Returns:
        A filtered frame for dashboard target derivation.
    """
    eligible = source
    if "education_source_code" in eligible.columns:
        eligible = eligible.filter(
            pl.col("education_source_code").cast(pl.String).str.to_uppercase() != "H90"
        )
    if "education_level" in eligible.columns:
        eligible = eligible.filter(
            pl.col("education_level").cast(pl.String).str.to_lowercase() != NOT_STATED
        )
    return eligible


def _origin_dashboard_target(
    *, normalized: Path, minimum_source_count: int | None = None
) -> dict[str, float] | None:
    """Load the eligible, non-Danish origin target for display.

    Returns:
        Normalised target proportions, or ``None`` when the target is absent or
        has no recognised origin value column.
    """
    source = _read_origin_target(normalized=normalized)
    if source is None:
        return None
    if minimum_source_count is not None:
        _validate_origin_threshold(
            source=source, minimum_source_count=minimum_source_count
        )
    source = source.filter(pl.col(ORIGIN_ELIGIBILITY_COLUMN))
    value_column = _target_column(field="origin_country_da", columns=source.columns)
    if value_column is None:
        return None
    return _normalise_counts(
        source=_without_domestic_origin(source=source, value_column=value_column),
        value_column=value_column,
    )


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


def _without_domestic_origin(
    *, source: pl.DataFrame, value_column: str
) -> pl.DataFrame:
    """Remove Danish origin from a target before normalising its denominator.

    Returns:
        A target frame without Danish origin rows.
    """
    return source.filter(
        pl.col(value_column).is_not_null()
        & ~pl.col(value_column)
        .cast(pl.String)
        .str.to_lowercase()
        .is_in(DOMESTIC_ORIGIN_LABELS)
    )


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


def _resolve_origin_threshold(
    *, bundle_path: Path, normalized: Path, minimum_source_count: int | None
) -> int | None:
    """Resolve the explicit or manifest-bound origin threshold.

    Returns:
        The threshold, or ``None`` when no origin target is present.
    """
    if minimum_source_count is not None:
        return minimum_source_count
    if not (normalized / "folk2_origin_country_marginal.parquet").exists():
        return None
    if not (bundle_path / "bundle-manifest.json").exists():
        return None
    return _dashboard_origin_threshold(bundle_path=bundle_path)


if __name__ == "__main__":
    load_repository_environment()
    main()
