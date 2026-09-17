"""Immutable Statistics Denmark StatBank acquisition."""

import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import ValidationError

from ..io import (
    canonical_json,
    load_yaml_model,
    sha256_file,
    sha256_text,
    verify_checksums,
    write_json,
    write_new_bytes,
    write_yaml,
)
from ..models import (
    LockedSource,
    SnapshotManifest,
    SourceDefinition,
    SourceLock,
    SourcesConfig,
    StatBankMetadata,
    StatBankValue,
)
from .http import request_with_retries, response_headers_content

LOGGER = logging.getLogger(__name__)
BASE_URL = "https://api.statbank.dk/v1"
MAX_CELLS = 1_000_000
REGION_LABEL_PREFIX = "Region "


def fetch_sources(lock: SourceLock, raw_dir: Path) -> list[SnapshotManifest]:
    """Fetch all resolved sources without overwriting valid snapshots.

    Args:
        lock:
            Resolved source lock.
        raw_dir:
            Root destination for immutable snapshots.

    Returns:
        Snapshot manifests in lock order.
    """
    manifests: list[SnapshotManifest] = []
    with httpx.Client(timeout=180.0, follow_redirects=True) as client:
        for source in lock.sources:
            manifests.append(
                _fetch_source(client=client, source=source, raw_dir=raw_dir)
            )
    return manifests


def _fetch_source(
    client: httpx.Client, source: LockedSource, raw_dir: Path
) -> SnapshotManifest:
    query = _source_query(source=source)
    query_content = source_query_content(source=source)
    snapshot_dir = source_snapshot_dir(source=source, raw_dir=raw_dir)
    manifest_path = snapshot_dir / "snapshot-manifest.json"
    if manifest_path.exists():
        manifest = SnapshotManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        _verify_snapshot(snapshot_dir=snapshot_dir, manifest=manifest, source=source)
        LOGGER.info("Reusing immutable %s snapshot", source.table_id)
        return manifest

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    metadata_en, metadata_bytes = _get_metadata(
        client=client, table_id=source.table_id, language="en"
    )
    _, metadata_da_bytes = _get_metadata(
        client=client, table_id=source.table_id, language="da"
    )
    response = request_with_retries(
        client=client, method="POST", url=source.data_url, json_payload=query
    )
    retrieved_at = _now()
    data_bytes = _canonical_csv_bytes(content=response.content)
    files = {
        "metadata-en.json": metadata_bytes,
        "metadata-da.json": metadata_da_bytes,
        "query.json": query_content.encode("utf-8"),
        "data.csv": data_bytes,
        "response-headers.json": response_headers_content(response=response),
    }
    for name, content in files.items():
        write_new_bytes(path=snapshot_dir / name, content=content)
    manifest = SnapshotManifest(
        table_id=source.table_id,
        role=source.role,
        period=source.period,
        metadata_sha256=sha256_file(path=snapshot_dir / "metadata-en.json"),
        metadata_da_sha256=sha256_file(path=snapshot_dir / "metadata-da.json"),
        query_sha256=sha256_text(content=query_content),
        data_sha256=sha256_file(path=snapshot_dir / "data.csv"),
        response_headers_sha256=sha256_file(
            path=snapshot_dir / "response-headers.json"
        ),
        retrieved_at=retrieved_at,
        data_bytes=len(data_bytes),
    )
    write_json(path=manifest_path, payload=manifest)
    LOGGER.info(
        "Fetched %s (%s, %s bytes)",
        source.table_id,
        metadata_en.text,
        f"{len(response.content):,}",
    )
    return manifest


def _canonical_csv_bytes(content: bytes) -> bytes:
    """Return UTF-8 CSV bytes with explicit LF line endings.

    Args:
        content:
            CSV response bytes from StatBank.

    Returns:
        Canonical UTF-8 CSV bytes.
    """
    text = content.decode("utf-8-sig")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def _get_metadata(
    client: httpx.Client, table_id: str, language: str
) -> tuple[StatBankMetadata, bytes]:
    response = request_with_retries(
        client=client,
        method="GET",
        url=f"{BASE_URL}/tableinfo/{table_id}?lang={language}",
        json_payload=None,
    )
    return StatBankMetadata.model_validate(response.json()), response.content


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _source_query(source: LockedSource) -> dict[str, object]:
    query: dict[str, object] = {
        "table": source.table_id,
        "format": source.format,
        "lang": "en",
        "valuePresentation": "CodeAndValue",
    }
    if source.format != "BULK":
        query["timeOrder"] = "Ascending"
    query["variables"] = [
        {"code": code, "values": values} for code, values in source.dimensions.items()
    ]
    return query


def _verify_snapshot(
    snapshot_dir: Path, manifest: SnapshotManifest, source: LockedSource
) -> None:
    if (
        manifest.table_id != source.table_id
        or manifest.role != source.role
        or manifest.period != source.period
    ):
        message = f"Snapshot provenance does not match lock: {snapshot_dir}"
        raise ValueError(message)
    expected = {
        "metadata-en.json": manifest.metadata_sha256,
        "metadata-da.json": manifest.metadata_da_sha256,
        "query.json": manifest.query_sha256,
        "data.csv": manifest.data_sha256,
        "response-headers.json": manifest.response_headers_sha256,
    }
    verify_checksums(
        base_dir=snapshot_dir,
        expected=expected,
        message="Immutable snapshot verification failed",
    )
    expected_query = source_query_content(source=source)
    if (snapshot_dir / "query.json").read_text(encoding="utf-8") != expected_query:
        message = f"Snapshot query does not match lock: {snapshot_dir}"
        raise ValueError(message)


def source_query_content(source: LockedSource) -> str:
    """Serialise the canonical query for a locked source.

    Args:
        source:
            Explicit locked source.

    Returns:
        Formatted canonical query JSON.
    """
    return json.dumps(_source_query(source=source), ensure_ascii=False, indent=2) + "\n"


def source_snapshot_dir(source: LockedSource, raw_dir: Path) -> Path:
    """Return the content-addressed directory for a locked source query.

    Args:
        source:
            Explicit locked source.
        raw_dir:
            Raw snapshot root.

    Returns:
        Query-specific snapshot directory.
    """
    query_checksum = sha256_text(content=source_query_content(source=source))
    return raw_dir / source.table_id.lower() / query_checksum[:16]


def resolve_sources(config: SourcesConfig, lock_path: Path) -> SourceLock:
    """Resolve dynamic source selectors into explicit StatBank value codes.

    Args:
        config:
            Source configuration.
        lock_path:
            Destination for the resolved source lock.

    Returns:
        Resolved source lock.

    Raises:
        ValueError:
            If a selector is invalid or a resolved query exceeds the API limit.
    """
    resolved_at = _now()
    locked_sources: list[LockedSource] = []
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for source in config.sources:
            metadata, _ = _get_metadata(
                client=client, table_id=source.table_id, language=config.language
            )
            dimensions = _resolve_dimensions(source=source, metadata=metadata)
            estimated_cells = estimate_query_cells(dimensions=dimensions)
            if source.format != "BULK" and estimated_cells > MAX_CELLS:
                message = (
                    f"{source.table_id} resolves to {estimated_cells:,} cells, "
                    f"above the {MAX_CELLS:,}-cell API limit for {source.format}; "
                    "use BULK for an explicitly exempt streaming query"
                )
                raise ValueError(message)
            locked_sources.append(
                LockedSource(
                    table_id=source.table_id,
                    role=source.role,
                    period=source.period,
                    format=source.format,
                    metadata_url=(
                        f"{BASE_URL}/tableinfo/{source.table_id}?lang={config.language}"
                    ),
                    data_url=f"{BASE_URL}/data",
                    retrieved_metadata_at=resolved_at,
                    table_updated_at=metadata.updated,
                    unit=metadata.unit,
                    dimensions=dimensions,
                    estimated_cells=estimated_cells,
                )
            )
    lock = SourceLock(
        version=config.version,
        language=config.language,
        release_rows=config.release_rows,
        minimum_source_count=config.minimum_source_count,
        minimum_expected_release_count=config.minimum_expected_release_count,
        resolved_at=resolved_at,
        sources=locked_sources,
        classifications=config.classifications,
    )
    existing = _readable_lock(lock_path=lock_path)
    if existing is not None and _lock_identity(lock=existing) == _lock_identity(
        lock=lock
    ):
        LOGGER.info("Source metadata is unchanged; retaining %s", lock_path)
        return existing
    write_yaml(path=lock_path, payload=lock)
    return lock


def _lock_identity(lock: SourceLock) -> str:
    payload = lock.model_dump(mode="json")
    payload.pop("resolved_at")
    sources = payload["sources"]
    if not isinstance(sources, list):
        message = "Source lock must contain a source list"
        raise TypeError(message)
    for source in sources:
        if not isinstance(source, dict):
            message = "Locked source must be a mapping"
            raise TypeError(message)
        source.pop("retrieved_metadata_at")
    return canonical_json(payload)


def _readable_lock(lock_path: Path) -> SourceLock | None:
    """Load an existing lock when it matches the current schema.

    Args:
        lock_path:
            Existing lock path.

    Returns:
        The parsed lock, or None when absent or schema-incompatible.
    """
    if not lock_path.exists():
        return None
    try:
        return load_yaml_model(path=lock_path, model=SourceLock)
    except ValidationError:
        LOGGER.warning(
            "Existing lock %s does not match the current schema; rewriting it",
            lock_path,
        )
        return None


def _resolve_dimensions(
    source: SourceDefinition, metadata: StatBankMetadata
) -> dict[str, list[str]]:
    metadata_variables = {variable.id: variable for variable in metadata.variables}
    dimensions: dict[str, list[str]] = {}
    for code, selection in source.dimensions.items():
        if code not in metadata_variables:
            message = f"Unknown dimension {code} in {source.table_id}"
            raise ValueError(message)
        available = metadata_variables[code].values
        available_codes = {value.id for value in available}
        if selection.values is not None:
            values = selection.values
        else:
            values = _apply_selector(
                selector=selection.selector or "", values=available
            )
        missing = sorted(set(values) - available_codes)
        if missing:
            message = f"Unknown values for {source.table_id}.{code}: {missing}"
            raise ValueError(message)
        if not values:
            message = f"Empty selection for {source.table_id}.{code}"
            raise ValueError(message)
        dimensions[code] = values
    return dimensions


def _apply_selector(selector: str, values: list[StatBankValue]) -> list[str]:
    typed_values = values
    if selector == "municipalities":
        return [
            value.id
            for value in typed_values
            if len(value.id) == 3
            and value.id != "000"
            and not value.text.startswith(REGION_LABEL_PREFIX)
        ]
    if selector == "regions":
        return [
            value.id
            for value in typed_values
            if value.text.startswith(REGION_LABEL_PREFIX)
        ]
    if selector == "non_total":
        return [value.id for value in typed_values if value.id not in {"TOT", "IALT"}]
    if selector == "all":
        return [value.id for value in typed_values]
    if selector == "adult_exact_ages":
        return [
            value.id
            for value in typed_values
            if value.id.isdigit() and int(value.id) >= 18
        ]
    if selector == "age_16_plus":
        return [
            value.id
            for value in typed_values
            if value.id.isdigit() and int(value.id) >= 16
        ]
    if selector == "adult_ras_ages":
        return [
            value.id
            for value in typed_values
            if (value.id.isdigit() and int(value.id) >= 18) or value.id == "71-"
        ]
    message = f"Unknown source selector: {selector}"
    raise ValueError(message)


def estimate_query_cells(dimensions: dict[str, list[str]]) -> int:
    """Calculate the StatBank cell count for a selected query.

    Args:
        dimensions:
            Selected values for every returned StatBank dimension, including time.

    Returns:
        Maximum observations multiplied by the returned columns. The columns are
        every selected dimension plus the observation value.
    """
    observations = math.prod(len(values) for values in dimensions.values())
    returned_columns = len(dimensions) + 1
    return observations * returned_columns
