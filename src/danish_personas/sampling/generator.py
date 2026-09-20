"""Deterministic demographic and OCEAN record generation."""

import hashlib
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from ..io import canonical_json, load_yaml_model, sha256_file, sha256_text, write_json
from ..ladders import SAMPLED_ATTRIBUTES, Ladder
from ..models import (
    DISCO_TWO_DIGIT_CODES,
    ELIGIBLE_JOB_FUNCTION_STATUS_CODES,
    SAMPLER_SCHEMA_VERSION,
    BundleManifest,
    RunManifest,
    SamplingConfig,
)
from ..sources.bundle import verify_prepared_bundle

LOGGER = logging.getLogger(__name__)
TRAITS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)
Distribution = tuple[list[dict[str, object]], np.ndarray]


@dataclass(frozen=True)
class LadderLevel:
    """One built back-off level and the key columns that address it."""

    name: str
    key_columns: tuple[str, ...]
    distributions: dict[tuple[str, ...], Distribution]


LadderIndex = list[LadderLevel]


def generate_records(
    bundle_dir: Path, sampling_config_path: Path, output_dir: Path, rows: int, seed: int
) -> Path:
    """Generate deterministic demographic and OCEAN records.

    Args:
        bundle_dir:
            Prepared source bundle.
        sampling_config_path:
            Sampling configuration file.
        output_dir:
            Root destination for generated runs.
        rows:
            Number of records to generate.
        seed:
            Reproducible random seed.

    Returns:
        Generated run directory.

    Raises:
        ValueError:
            If an existing run fails checksum verification or a distribution is absent.
    """
    config = load_yaml_model(path=sampling_config_path, model=SamplingConfig)
    bundle_manifest_path = bundle_dir / "bundle-manifest.json"
    bundle = verify_prepared_bundle(bundle_dir=bundle_dir)
    run_id = sha256_text(
        f"{SAMPLER_SCHEMA_VERSION}:{bundle.bundle_id}:"
        f"{sha256_file(sampling_config_path)}:{rows}:{seed}"
    )[:16]
    run_dir = output_dir / run_id
    manifest_path = run_dir / "run-manifest.json"
    if manifest_path.exists():
        manifest = RunManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if manifest.sampler_schema_version != SAMPLER_SCHEMA_VERSION:
            message = "Generated run uses an unsupported sampler schema version"
            raise ValueError(message)
        data_path = run_dir / manifest.data_file
        if sha256_file(data_path) != manifest.data_sha256:
            message = f"Generated run checksum mismatch: {data_path}"
            raise ValueError(message)
        LOGGER.info("Reusing deterministic run %s", run_id)
        return run_dir

    source_dir = bundle_dir / "normalized"
    joint_frame = pl.read_parquet(source_dir / "ras209_sampling.parquet")
    geography = pl.read_parquet(source_dir / "geography_hierarchy.parquet")
    geography_lookup = {
        str(row["municipality_code"]): row
        for row in geography.select(
            "municipality_code", "municipality", "region_code", "region"
        ).iter_rows(named=True)
    }
    demographic_seed, ocean_seed, origin_seed, job_function_seed = (
        np.random.SeedSequence(seed).spawn(4)
    )
    demographic_rng = np.random.default_rng(demographic_seed)
    ocean_rng = np.random.default_rng(ocean_seed)
    origin_rng = np.random.default_rng(origin_seed)
    job_function_rng = np.random.default_rng(job_function_seed)

    sampled_joint = _municipality_quota_sample(
        frame=joint_frame, rows=rows, rng=demographic_rng
    )
    origin_frame = pl.read_parquet(source_dir / "folk2_origin_country_marginal.parquet")
    origin_frame = _ensure_origin_danish_labels(frame=origin_frame, bundle=bundle)
    sampled_origin = _origin_quota_sample(
        frame=origin_frame, rows=rows, rng=origin_rng, require_danish=True
    )
    job_function_frame = pl.read_parquet(
        source_dir / "job_function_sex_marginal.parquet"
    )
    ladders = {
        attribute.resolution_column: _ladder_index(
            frame=pl.read_parquet(source_dir / attribute.prepared_file),
            ladder=attribute.ladder,
            payload_columns=list(attribute.payload_columns),
            smoothing=config.smoothing,
        )
        for attribute in SAMPLED_ATTRIBUTES
    }
    records = _build_records(
        sampled_joint=sampled_joint,
        ladders=ladders,
        rng=demographic_rng,
        country=config.country,
        seed=seed,
        geography_lookup=geography_lookup,
    )
    _add_ocean(records=records, config=config, rng=ocean_rng)
    _attach_origin(records=records, sampled_origin=sampled_origin)
    _attach_job_functions(
        records=records, marginal=job_function_frame, rng=job_function_rng
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    data_path = run_dir / "structured-records.parquet"
    frame = pl.DataFrame(records).cast(
        {"job_function_code": pl.String, "job_function": pl.String}
    )
    frame.write_parquet(data_path, compression="zstd")
    manifest = RunManifest(
        run_id=run_id,
        sampler_schema_version=SAMPLER_SCHEMA_VERSION,
        created_at=_now(),
        bundle_id=bundle.bundle_id,
        bundle_manifest_sha256=sha256_file(bundle_manifest_path),
        sampling_config_sha256=sha256_file(sampling_config_path),
        origin_labels_contract_path=bundle.origin_labels_contract_path,
        origin_labels_contract_version=bundle.origin_labels_contract_version,
        origin_labels_contract_sha256=bundle.origin_labels_contract_sha256,
        origin_labels_contract_content=bundle.origin_labels_contract_content,
        rows=rows,
        seed=seed,
        data_file=Path(data_path.name),
        data_sha256=sha256_file(data_path),
        logical_content_sha256=_logical_checksum(frame=frame),
        llm_calls=0,
    )
    write_json(path=manifest_path, payload=manifest)
    LOGGER.info("Generated %s non-LLM records in run %s", f"{rows:,}", run_id)
    return run_dir


def _add_ocean(
    records: list[dict[str, object]], config: SamplingConfig, rng: np.random.Generator
) -> None:
    ocean = config.ocean
    scores = rng.normal(
        loc=ocean.mean, scale=ocean.standard_deviation, size=(len(records), len(TRAITS))
    )
    scores = np.clip(scores, ocean.minimum, ocean.maximum)
    boundaries = np.asarray(ocean.label_boundaries)
    for row_index, record in enumerate(records):
        for trait_index, trait in enumerate(TRAITS):
            score = float(scores[row_index, trait_index])
            label_index = int(np.searchsorted(boundaries, score, side="right"))
            record[f"{trait}_score"] = score
            record[f"{trait}_label"] = ocean.labels[label_index]


def _attach_job_functions(
    records: list[dict[str, object]], marginal: pl.DataFrame, rng: np.random.Generator
) -> None:
    """Allocate LONS20 job functions within sex to eligible employees only."""
    _validate_job_function_marginal(frame=marginal)
    for record in records:
        record["job_function_code"] = None
        record["job_function"] = None
        record["job_function_resolution"] = "not_applicable"
    for sex in ("female", "male"):
        eligible_indices = [
            index
            for index, record in enumerate(records)
            if record["sex"] == sex
            and str(record["detailed_status_code"])
            in ELIGIBLE_JOB_FUNCTION_STATUS_CODES
        ]
        sampled = _job_function_quota_sample(
            frame=marginal.filter(pl.col("sex") == sex),
            rows=len(eligible_indices),
            rng=rng,
        )
        for index, job_function in zip(
            eligible_indices, sampled.iter_rows(named=True), strict=True
        ):
            records[index]["job_function_code"] = job_function["job_function_code"]
            records[index]["job_function"] = job_function["job_function"]
            records[index]["job_function_resolution"] = "lons20_sex_marginal"


def _job_function_quota_sample(
    frame: pl.DataFrame, rows: int, rng: np.random.Generator
) -> pl.DataFrame:
    """Return exact largest-remainder LONS20 quotas with code-sorted ties."""
    selected = frame.select("job_function_code", "job_function", "sex", "count").sort(
        "job_function_code"
    )
    if rows == 0:
        return selected.head(0)
    weights = selected.get_column("count").to_numpy().astype(np.float64)
    expected = weights / weights.sum() * rows
    allocations = np.floor(expected).astype(np.int64)
    remainder = rows - int(allocations.sum())
    fractions = expected - allocations
    allocations[np.argsort(-fractions, kind="stable")[:remainder]] += 1
    indices = np.repeat(np.arange(selected.height), allocations)
    rng.shuffle(indices)
    return selected[indices]


def _validate_job_function_marginal(frame: pl.DataFrame) -> None:
    required = {"job_function_code", "job_function", "sex", "count"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"LONS20 marginal is missing columns: {missing}")
    selected = frame.select(sorted(required))
    if selected.height != len(DISCO_TWO_DIGIT_CODES) * 2:
        raise ValueError("LONS20 marginal must contain 42 codes for both sexes")
    if selected.null_count().sum_horizontal().item() > 0:
        raise ValueError("LONS20 marginal contains null values")
    if set(selected.get_column("job_function_code")) != set(DISCO_TWO_DIGIT_CODES):
        raise ValueError("LONS20 marginal contains a wrong DISCO-08 hierarchy level")
    if set(selected.get_column("sex")) != {"female", "male"}:
        raise ValueError("LONS20 marginal must contain female and male distributions")
    if selected.select("job_function_code", "sex").n_unique() != selected.height:
        raise ValueError("LONS20 marginal contains duplicate code-sex cells")
    mappings = selected.select("job_function_code", "job_function").unique()
    if mappings.height != len(DISCO_TWO_DIGIT_CODES):
        raise ValueError("LONS20 marginal contains inconsistent code-label pairs")
    if any(not str(label).strip() for label in mappings.get_column("job_function")):
        raise ValueError("LONS20 marginal contains blank labels")
    counts = selected.get_column("count")
    if counts.dtype not in {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    } or bool((counts <= 0).any()):
        raise ValueError("LONS20 marginal counts must be positive integers")


def _attach_origin(
    records: list[dict[str, object]], sampled_origin: pl.DataFrame
) -> None:
    """Attach an independent origin-country draw without touching other fields.

    Raises:
        ValueError:
            If the origin sample does not match the record count.
    """
    if len(records) != sampled_origin.height:
        message = "Origin sample size must equal the requested record count"
        raise ValueError(message)
    for record, origin in zip(
        records, sampled_origin.iter_rows(named=True), strict=True
    ):
        record["origin_country_code"] = origin["origin_country_code"]
        record["origin_country"] = origin["origin_country"]
        record["origin_country_da"] = origin["origin_country_da"]


def _build_records(
    sampled_joint: pl.DataFrame,
    ladders: dict[str, LadderIndex],
    rng: np.random.Generator,
    country: str,
    seed: int,
    geography_lookup: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    """Draw the remaining fields for every quota-sampled joint row.

    Args:
        sampled_joint:
            One row per record from the RAS209 joint backbone.
        ladders:
            Built ladder per resolution column.
        rng:
            Random generator, consumed once per drawn field.
        country:
            Fixed country label for every record.
        seed:
            Seed the deterministic identifiers derive from.
        geography_lookup:
            Official municipality names and parent regions keyed by municipality code.

    Returns:
        One mapping per record, before OCEAN traits are attached.

    Raises:
        ValueError:
            If a sampled RAS209 municipality is absent from the hierarchy.
    """
    records: list[dict[str, object]] = []
    for index, joint in enumerate(sampled_joint.iter_rows(named=True)):
        values = {
            column: str(joint[column])
            for column in (
                "municipality_code",
                "age_band",
                "sex",
                "labour_market_status",
            )
        }
        geography = geography_lookup.get(values["municipality_code"])
        if geography is None:
            message = (
                "RAS209 municipality is absent from the official hierarchy: "
                f"{values['municipality_code']}"
            )
            raise ValueError(message)
        drawn: dict[str, object] = {}
        for column, ladder_index in ladders.items():
            payload, level = _draw(ladder_index=ladder_index, values=values, rng=rng)
            drawn.update(payload)
            drawn[column] = level
        age = _as_int(value=drawn["age"])
        records.append(
            {
                "persona_id": str(
                    uuid.uuid5(uuid.NAMESPACE_URL, f"danish-personas:{seed}:{index}")
                ),
                "country": country,
                "age": age,
                "age_resolution": drawn["age_resolution"],
                "age_band": values["age_band"],
                "sex": values["sex"],
                "marital_status": drawn["marital_status"],
                "marital_resolution": drawn["marital_resolution"],
                "municipality_code": values["municipality_code"],
                "municipality": geography["municipality"],
                "region_code": geography["region_code"],
                "region": geography["region"],
                "education_level": joint["education_level"],
                "education_source_code": joint["education_source_code"],
                "education_resolution": (
                    "ras209_67_plus_proxy" if age >= 70 else "ras209_age_band"
                ),
                "labour_market_status": values["labour_market_status"],
                "detailed_status_code": drawn["detailed_status_code"],
                "detailed_status": drawn["detailed_status"],
                "detailed_status_resolution": drawn["detailed_status_resolution"],
            }
        )
    return records


def _as_int(value: object) -> int:
    if isinstance(value, int | str):
        return int(value)
    message = f"Expected an integer-compatible value, got {type(value).__name__}"
    raise TypeError(message)


def _draw(
    ladder_index: LadderIndex, values: dict[str, str], rng: np.random.Generator
) -> tuple[dict[str, object], str]:
    """Draw from the most specific populated cell, reporting its level.

    Exactly one random draw is consumed regardless of how far the ladder backs
    off, so the random stream does not depend on cell sparsity.

    Args:
        ladder_index:
            Per-level distributions, most specific first.
        values:
            Current record values available as key components.
        rng:
            Random generator.

    Returns:
        The drawn payload and the name of the level that produced it.

    Raises:
        ValueError:
            If no level of the ladder has a populated cell.
    """
    for level in ladder_index:
        key = tuple(values[column] for column in level.key_columns)
        distribution = level.distributions.get(key)
        if distribution is None:
            continue
        payloads, probabilities = distribution
        index = int(rng.choice(len(payloads), p=probabilities))
        return payloads[index], level.name
    message = f"No prepared distribution at any back-off level for {values}"
    raise ValueError(message)


def _ensure_origin_danish_labels(
    *, frame: pl.DataFrame, bundle: BundleManifest
) -> pl.DataFrame:
    """Require the captured Danish label column at the sampling boundary.

    Returns:
        The unchanged marginal with its captured Danish labels.

    Raises:
        ValueError: If the captured Danish label column is absent.
    """
    del bundle
    if "origin_country_da" not in frame.columns:
        raise ValueError("Schema-6 origin marginal is missing Danish labels")
    return frame


def _ladder_index(
    frame: pl.DataFrame, ladder: Ladder, payload_columns: list[str], smoothing: float
) -> LadderIndex:
    """Build one distribution index per level of a back-off ladder.

    Structurally impossible cells are dropped once here rather than per level,
    so smoothing can only reweight categories the sources actually support.

    Args:
        frame:
            Prepared source counts for one sampled attribute.
        ladder:
            Ordered levels, most specific first.
        payload_columns:
            Columns making up the drawn value.
        smoothing:
            Additive pseudo-count applied to surviving cells.

    Returns:
        Built levels in ladder order.
    """
    eligible = _eligible(frame=frame)
    return [
        LadderLevel(
            name=level,
            key_columns=key_columns,
            distributions=_distribution_index(
                frame=eligible,
                key_columns=list(key_columns),
                payload_columns=payload_columns,
                smoothing=smoothing,
            ),
        )
        for level, key_columns in ladder
    ]


def _distribution_index(
    frame: pl.DataFrame,
    key_columns: list[str],
    payload_columns: list[str],
    smoothing: float,
) -> dict[tuple[str, ...], Distribution]:
    """Index one conditional distribution per key cell.

    A coarser level drops key columns, so several source rows can collapse onto
    the same payload; counts are summed once over the whole frame rather than
    per cell. Sorting after that aggregation keeps payload order deterministic.

    Args:
        frame:
            Structurally eligible source counts.
        key_columns:
            Columns addressing a cell.
        payload_columns:
            Columns making up the drawn value.
        smoothing:
            Additive pseudo-count applied to surviving cells.

    Returns:
        Payloads and probabilities for every populated cell.
    """
    distributions: dict[tuple[str, ...], Distribution] = {}
    aggregated = (
        frame.group_by([*key_columns, *payload_columns])
        .agg(pl.col("count").sum())
        .sort([*key_columns, *payload_columns])
    )
    for raw_key, group in aggregated.group_by(key_columns, maintain_order=True):
        key = tuple(str(value) for value in raw_key)
        payloads = [
            dict(zip(payload_columns, row, strict=True))
            for row in group.select(payload_columns).iter_rows()
        ]
        counts = group.get_column("count").to_numpy().astype(np.float64) + smoothing
        distributions[key] = (payloads, counts / counts.sum())
    return distributions


def _eligible(frame: pl.DataFrame) -> pl.DataFrame:
    """Drop structurally impossible source cells.

    Args:
        frame:
            Prepared source counts.

    Returns:
        Rows with a positive, unsuppressed count.
    """
    return frame.filter((pl.col("count") > 0) & ~pl.col("suppressed"))


def _logical_checksum(frame: pl.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.iter_rows(named=True):
        digest.update(canonical_json(row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _municipality_quota_sample(
    frame: pl.DataFrame, rows: int, rng: np.random.Generator
) -> pl.DataFrame:
    """Sample RAS209 while preserving municipality quotas exactly.

    The high-dimensional municipality joint has more cells than the smoke-run row
    count. Largest remainders over those cells would systematically favour the first
    sorted categories, so quotas are exact only at the municipality boundary and the
    within-municipality joint is sampled probabilistically.

    Args:
        frame:
            Municipality-native RAS209 joint.
        rows:
            Number of records to sample.
        rng:
            Deterministic demographic random generator.

    Returns:
        Shuffled sampled RAS209 rows.
    """
    eligible = _eligible(frame=frame).sort(sorted(frame.columns))
    totals = (
        eligible.group_by("municipality_code")
        .agg(pl.col("count").sum())
        .sort("municipality_code")
    )
    weights = totals.get_column("count").to_numpy().astype(np.float64)
    expected = weights / weights.sum() * rows
    allocations = np.floor(expected).astype(np.int64)
    remainder = rows - int(allocations.sum())
    fractions = expected - allocations
    allocations[np.argsort(-fractions, kind="stable")[:remainder]] += 1
    sampled: list[pl.DataFrame] = []
    for municipality_code, allocation in zip(
        totals.get_column("municipality_code"), allocations, strict=True
    ):
        if allocation == 0:
            continue
        group = eligible.filter(pl.col("municipality_code") == municipality_code)
        group_weights = group.get_column("count").to_numpy().astype(np.float64)
        indices = rng.choice(
            group.height,
            size=int(allocation),
            replace=True,
            p=group_weights / group_weights.sum(),
        )
        sampled.append(group[indices])
    result = pl.concat(sampled)
    return result[rng.permutation(result.height)]


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _origin_quota_sample(
    frame: pl.DataFrame,
    rows: int,
    rng: np.random.Generator,
    *,
    require_danish: bool = False,
) -> pl.DataFrame:
    """Sample the official FOLK2 marginal with exact deterministic quotas.

    Only positive-weight categories can be emitted. The code, label, and count
    columns are validated here as a defence against a malformed prepared bundle.
    Largest-remainder ties follow sorted official codes, then the independent
    child RNG shuffles the resulting rows.

    Returns:
        A shuffled frame containing exactly ``rows`` positive-weight categories.

    Raises:
        ValueError:
            If codes, labels, or counts are malformed, or the total is not
            positive.
    """
    required = {"origin_country_code", "origin_country", "count"}
    if require_danish:
        required.add("origin_country_da")
    selected = _select_origin_columns(frame=frame, required=required)
    _validate_origin_marginal(selected=selected)
    positive = selected.filter(pl.col("count") > 0)
    if positive.is_empty():
        raise ValueError("FOLK2 marginal must have a positive total")
    return _shuffle_origin_quota(positive=positive, rows=rows, rng=rng)


def _select_origin_columns(*, frame: pl.DataFrame, required: set[str]) -> pl.DataFrame:
    """Select and canonically order the requested origin marginal columns.

    Returns:
        The selected and sorted marginal.

    Raises:
        ValueError:
            If a required marginal column is absent.
    """
    if not required <= set(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise ValueError(f"FOLK2 marginal is missing columns: {missing}")
    return frame.select(sorted(required)).sort(
        ["origin_country_code", "origin_country"]
    )


def _shuffle_origin_quota(
    *, positive: pl.DataFrame, rows: int, rng: np.random.Generator
) -> pl.DataFrame:
    """Allocate largest-remainder quotas and shuffle with the origin RNG.

    Returns:
        The quota-expanded marginal in deterministic shuffled order.
    """
    weights = positive.get_column("count").to_numpy().astype(np.float64)
    expected = weights / weights.sum() * rows
    allocations = np.floor(expected).astype(np.int64)
    remainder = rows - int(allocations.sum())
    fractions = expected - allocations
    allocations[np.argsort(-fractions, kind="stable")[:remainder]] += 1
    indices = np.repeat(np.arange(positive.height), allocations)
    rng.shuffle(indices)
    return positive[indices]


def _validate_origin_marginal(*, selected: pl.DataFrame) -> None:
    """Validate labels, code uniqueness, and integer non-negative counts.

    Raises:
        ValueError:
            If the marginal contains malformed labels, codes, or counts.
    """
    if selected.is_empty():
        raise ValueError("FOLK2 marginal must contain at least one category")
    if selected.null_count().sum_horizontal().item() > 0:
        raise ValueError("FOLK2 marginal contains null code, label, or count")
    for column in ("origin_country", "origin_country_da"):
        if column in selected.columns and bool(
            (selected.get_column(column).str.strip_chars() == "").any()
        ):
            raise ValueError(f"FOLK2 marginal contains blank {column}")
    if selected.get_column("origin_country_code").n_unique() != selected.height:
        raise ValueError("FOLK2 marginal contains duplicate origin codes")
    if selected.get_column("origin_country").n_unique() != selected.height:
        raise ValueError("FOLK2 marginal contains duplicate origin labels")
    if "origin_country_da" in selected.columns and (
        selected.get_column("origin_country_da").n_unique() != selected.height
    ):
        raise ValueError("FOLK2 marginal contains duplicate Danish origin labels")
    counts = selected.get_column("count")
    if counts.dtype not in (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    ):
        raise ValueError("FOLK2 marginal counts must be integers")
    if bool((counts < 0).any()):
        raise ValueError("FOLK2 marginal counts cannot be negative")
