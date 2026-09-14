"""Deterministic demographic and OCEAN record generation."""

import hashlib
import logging
import uuid
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
    folk = pl.read_parquet(source_dir / "folk1a_base.parquet")
    ras209 = pl.read_parquet(source_dir / "ras209_sampling.parquet")
    ras202 = pl.read_parquet(source_dir / "ras202_detail.parquet")
    seed_sequence = np.random.SeedSequence(seed)
    demographic_seed, ocean_seed = seed_sequence.spawn(2)
    demographic_rng = np.random.default_rng(demographic_seed)
    ocean_rng = np.random.default_rng(ocean_seed)

    base = _sample_base(frame=folk, rows=rows, rng=demographic_rng)
    joint_distributions = _joint_distributions(frame=ras209)
    detail_distributions = _detail_distributions(frame=ras202)
    records = _enrich_demographics(
        base=base,
        joint_distributions=joint_distributions,
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
    logical_checksum = _logical_checksum(frame=frame)
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
        logical_content_sha256=logical_checksum,
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


def _detail_distributions(
    frame: pl.DataFrame,
) -> dict[tuple[str, str, str], tuple[list[dict[str, str]], np.ndarray]]:
    distributions: dict[
        tuple[str, str, str], tuple[list[dict[str, str]], np.ndarray]
    ] = {}
    eligible = frame.filter((pl.col("count") > 0) & ~pl.col("suppressed"))
    keys = ["age_key", "sex", "labour_market_status"]
    for key, group in eligible.group_by(keys):
        payloads = [
            {"detailed_status_code": str(row[0]), "detailed_status": str(row[1])}
            for row in group.select(
                "detailed_status_code", "detailed_status"
            ).iter_rows()
        ]
        distribution_key = (str(key[0]), str(key[1]), str(key[2]))
        distributions[distribution_key] = (
            payloads,
            _probabilities(group.get_column("count")),
        )
    return distributions


def _probabilities(counts: pl.Series) -> np.ndarray:
    values = counts.to_numpy().astype(np.float64)
    return values / values.sum()


def _enrich_demographics(
    base: pl.DataFrame,
    joint_distributions: dict[
        tuple[str, str, str], tuple[list[dict[str, str]], np.ndarray]
    ],
    detail_distributions: dict[
        tuple[str, str, str], tuple[list[dict[str, str]], np.ndarray]
    ],
    rng: np.random.Generator,
    country: str,
    seed: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, row in enumerate(base.iter_rows(named=True)):
        age = int(row["age"])
        age_band = _age_band(age=age)
        joint_key = (str(row["region_code"]), age_band, str(row["sex"]))
        joint_payload = _draw(distributions=joint_distributions, key=joint_key, rng=rng)
        labour_status = joint_payload["labour_market_status"]
        age_key = str(age) if age <= 70 else "71-"
        detail_key = (age_key, str(row["sex"]), labour_status)
        detail_payload = _draw(
            distributions=detail_distributions, key=detail_key, rng=rng
        )
        record: dict[str, object] = {
            "persona_id": str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"danish-personas:{seed}:{index}")
            ),
            "country": country,
            "age": age,
            "age_band": age_band,
            "sex": row["sex"],
            "marital_status": row["marital_status"],
            "municipality_code": row["municipality_code"],
            "municipality": row["municipality"],
            "region_code": row["region_code"],
            "region": row["region"],
            "education_level": joint_payload["education_level"],
            "education_source_code": joint_payload["education_source_code"],
            "education_resolution": (
                "ras209_67_plus_proxy" if age >= 70 else "ras209_age_band"
            ),
            "labour_market_status": labour_status,
            "detailed_status_code": detail_payload["detailed_status_code"],
            "detailed_status": detail_payload["detailed_status"],
            "demographic_backoff_level": 0,
            "status_backoff_level": 0,
        }
        records.append(record)
    return records


def _age_band(age: int) -> str:
    if age <= 29:
        return "18-29"
    if age <= 49:
        return "30-49"
    if age <= 66:
        return "50-66"
    return "67+"


def _draw(
    distributions: dict[tuple[str, str, str], tuple[list[dict[str, str]], np.ndarray]],
    key: tuple[str, str, str],
    rng: np.random.Generator,
) -> dict[str, str]:
    if key not in distributions:
        message = f"No prepared distribution for {key}"
        raise ValueError(message)
    payloads, probabilities = distributions[key]
    index = int(rng.choice(len(payloads), p=probabilities))
    return payloads[index]


def _joint_distributions(
    frame: pl.DataFrame,
) -> dict[tuple[str, str, str], tuple[list[dict[str, str]], np.ndarray]]:
    distributions: dict[
        tuple[str, str, str], tuple[list[dict[str, str]], np.ndarray]
    ] = {}
    eligible = frame.filter((pl.col("count") > 0) & ~pl.col("suppressed"))
    for key, group in eligible.group_by(["region_code", "age_band", "sex"]):
        payloads = [
            {
                "education_source_code": str(row[0]),
                "education_level": str(row[1]),
                "labour_market_status": str(row[2]),
            }
            for row in group.select(
                "education_source_code", "education_level", "labour_market_status"
            ).iter_rows()
        ]
        distribution_key = (str(key[0]), str(key[1]), str(key[2]))
        distributions[distribution_key] = (
            payloads,
            _probabilities(group.get_column("count")),
        )
    return distributions


def _logical_checksum(frame: pl.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.iter_rows(named=True):
        digest.update(canonical_json(row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _sample_base(
    frame: pl.DataFrame, rows: int, rng: np.random.Generator
) -> pl.DataFrame:
    eligible = frame.filter((pl.col("count") > 0) & ~pl.col("suppressed"))
    weights = eligible.get_column("count").to_numpy().astype(np.float64)
    probabilities = weights / weights.sum()
    indices = rng.choice(eligible.height, size=rows, replace=True, p=probabilities)
    return eligible[indices]
