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
from ..models import BundleManifest, RunManifest, SamplingConfig

LOGGER = logging.getLogger(__name__)
TRAITS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)
Distribution = tuple[list[dict[str, object]], np.ndarray]
Ladder = tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True)
class LadderLevel:
    """One built back-off level and the key columns that address it."""

    name: str
    key_columns: tuple[str, ...]
    distributions: dict[tuple[str, ...], Distribution]


LadderIndex = list[LadderLevel]

# Ordered sparse-cell back-off. Each ladder ends at the most general cell that
# is still structurally valid, so backing off can never cross an invariant:
# an age stays inside its band, and a detailed status stays inside the broad
# RAS209 status it refines. A missing final level is a structural zero, not
# sparsity, and must fail rather than silently widen further.
# The ladders live in code rather than config because a ladder's final level
# encodes a structural invariant rather than a tunable: widening past it would
# emit structurally invalid records that no configuration schema could catch.
AGE_LADDER: Ladder = (
    ("age_band_sex", ("age_band", "sex")),
    ("age_band", ("age_band",)),
)
MARITAL_LADDER: Ladder = (
    ("region_age_band_sex", ("region_code", "age_band", "sex")),
    ("age_band_sex", ("age_band", "sex")),
    ("age_band", ("age_band",)),
)
DETAIL_LADDER: Ladder = (
    ("age_band_sex_status", ("age_band", "sex", "labour_market_status")),
    ("sex_status", ("sex", "labour_market_status")),
    ("status", ("labour_market_status",)),
)
LADDER_SOURCES = (
    ("folk_age_sampling.parquet", AGE_LADDER, ["age"]),
    ("folk_marital_sampling.parquet", MARITAL_LADDER, ["marital_status"]),
    (
        "ras202_sampling.parquet",
        DETAIL_LADDER,
        ["detailed_status_code", "detailed_status"],
    ),
)


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
    bundle = BundleManifest.model_validate_json(bundle_manifest_path.read_text())
    run_id = sha256_text(
        f"{bundle.bundle_id}:{sha256_file(sampling_config_path)}:{rows}:{seed}"
    )[:16]
    run_dir = output_dir / run_id
    manifest_path = run_dir / "run-manifest.json"
    if manifest_path.exists():
        manifest = RunManifest.model_validate_json(manifest_path.read_text())
        data_path = run_dir / manifest.data_file
        if sha256_file(data_path) != manifest.data_sha256:
            message = f"Generated run checksum mismatch: {data_path}"
            raise ValueError(message)
        LOGGER.info("Reusing deterministic run %s", run_id)
        return run_dir

    source_dir = bundle_dir / "normalized"
    joint_frame = pl.read_parquet(source_dir / "ras209_sampling.parquet")
    demographic_seed, ocean_seed = np.random.SeedSequence(seed).spawn(2)
    demographic_rng = np.random.default_rng(demographic_seed)
    ocean_rng = np.random.default_rng(ocean_seed)

    sampled_joint = _quota_sample(frame=joint_frame, rows=rows, rng=demographic_rng)
    age_distributions, marital_distributions, detail_distributions = (
        _ladder_index(
            frame=pl.read_parquet(source_dir / file_name),
            ladder=ladder,
            payload_columns=payload_columns,
            smoothing=config.smoothing,
        )
        for file_name, ladder, payload_columns in LADDER_SOURCES
    )
    records = _build_records(
        sampled_joint=sampled_joint,
        age_distributions=age_distributions,
        marital_distributions=marital_distributions,
        detail_distributions=detail_distributions,
        rng=demographic_rng,
        country=config.country,
        seed=seed,
    )
    _add_ocean(records=records, config=config, rng=ocean_rng)

    run_dir.mkdir(parents=True, exist_ok=True)
    data_path = run_dir / "structured-records.parquet"
    frame = pl.DataFrame(records)
    frame.write_parquet(data_path, compression="zstd")
    manifest = RunManifest(
        run_id=run_id,
        created_at=_now(),
        bundle_id=bundle.bundle_id,
        bundle_manifest_sha256=sha256_file(bundle_manifest_path),
        sampling_config_sha256=sha256_file(sampling_config_path),
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


def _build_records(
    sampled_joint: pl.DataFrame,
    age_distributions: LadderIndex,
    marital_distributions: LadderIndex,
    detail_distributions: LadderIndex,
    rng: np.random.Generator,
    country: str,
    seed: int,
) -> list[dict[str, object]]:
    """Draw the remaining fields for every quota-sampled joint row.

    Args:
        sampled_joint:
            One row per record from the RAS209 joint backbone.
        age_distributions:
            Built age ladder.
        marital_distributions:
            Built marital-status ladder.
        detail_distributions:
            Built detailed-status ladder.
        rng:
            Random generator, consumed once per drawn field.
        country:
            Fixed country label for every record.
        seed:
            Seed the deterministic identifiers derive from.

    Returns:
        One mapping per record, before OCEAN traits are attached.
    """
    records: list[dict[str, object]] = []
    for index, joint in enumerate(sampled_joint.iter_rows(named=True)):
        values = {
            column: str(joint[column])
            for column in ("age_band", "sex", "region_code", "labour_market_status")
        }
        age_payload, age_resolution = _draw(
            ladder_index=age_distributions, values=values, rng=rng
        )
        age = _as_int(value=age_payload["age"])
        marital_payload, marital_resolution = _draw(
            ladder_index=marital_distributions, values=values, rng=rng
        )
        marital_status = str(marital_payload["marital_status"])
        detail_payload, detail_resolution = _draw(
            ladder_index=detail_distributions, values=values, rng=rng
        )
        records.append(
            {
                "persona_id": str(
                    uuid.uuid5(uuid.NAMESPACE_URL, f"danish-personas:{seed}:{index}")
                ),
                "country": country,
                "age": age,
                "age_resolution": age_resolution,
                "age_band": values["age_band"],
                "sex": values["sex"],
                "marital_status": marital_status,
                "marital_resolution": marital_resolution,
                "region_code": values["region_code"],
                "region": joint["region"],
                "education_level": joint["education_level"],
                "education_source_code": joint["education_source_code"],
                "education_resolution": (
                    "ras209_67_plus_proxy" if age >= 70 else "ras209_age_band"
                ),
                "labour_market_status": values["labour_market_status"],
                "detailed_status_code": detail_payload["detailed_status_code"],
                "detailed_status": detail_payload["detailed_status"],
                "detailed_status_resolution": detail_resolution,
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
    eligible = frame.filter((pl.col("count") > 0) & ~pl.col("suppressed"))
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


def _logical_checksum(frame: pl.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.iter_rows(named=True):
        digest.update(canonical_json(row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _quota_sample(
    frame: pl.DataFrame, rows: int, rng: np.random.Generator
) -> pl.DataFrame:
    eligible = frame.filter((pl.col("count") > 0) & ~pl.col("suppressed")).sort(
        sorted(frame.columns)
    )
    weights = eligible.get_column("count").to_numpy().astype(np.float64)
    expected = weights / weights.sum() * rows
    allocations = np.floor(expected).astype(np.int64)
    remainder = rows - int(allocations.sum())
    fractions = expected - allocations
    stable_order = np.argsort(-fractions, kind="stable")
    allocations[stable_order[:remainder]] += 1
    indices = np.repeat(np.arange(eligible.height), allocations)
    rng.shuffle(indices)
    return eligible[indices]
