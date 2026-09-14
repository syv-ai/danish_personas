"""Immutable Statistics Denmark StatBank acquisition."""

import json
import logging
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from ..io import (
    canonical_json,
    load_yaml_model,
    sha256_file,
    sha256_text,
    write_json,
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
    snapshot_dir = raw_dir / source.table_id.lower()
    manifest_path = snapshot_dir / "snapshot-manifest.json"
    if manifest_path.exists():
        manifest = SnapshotManifest.model_validate_json(manifest_path.read_text())
        _verify_snapshot(snapshot_dir=snapshot_dir, manifest=manifest)
        LOGGER.info("Reusing immutable %s snapshot", source.table_id)
        return manifest

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    metadata_en, metadata_bytes = _get_metadata(
        client=client, table_id=source.table_id, language="en"
    )
    _, metadata_da_bytes = _get_metadata(
        client=client, table_id=source.table_id, language="da"
    )
    query: dict[str, object] = {
        "table": source.table_id,
        "format": "CSV",
        "lang": "en",
        "valuePresentation": "CodeAndValue",
        "timeOrder": "Ascending",
        "variables": [
            {"code": code, "values": values}
            for code, values in source.dimensions.items()
        ],
    }
    query_content = json.dumps(query, ensure_ascii=False, indent=2) + "\n"
    response = _request_with_retries(
        client=client, method="POST", url=source.data_url, json_payload=query
    )
    retrieved_at = _now()
    files = {
        "metadata-en.json": metadata_bytes,
        "metadata-da.json": metadata_da_bytes,
        "query.json": query_content.encode(),
        "data.csv": response.content,
        "response-headers.json": (
            json.dumps(dict(response.headers), indent=2, sort_keys=True) + "\n"
        ).encode(),
    }
    for name, content in files.items():
        _write_new_bytes(path=snapshot_dir / name, content=content)
    manifest = SnapshotManifest(
        table_id=source.table_id,
        role=source.role,
        period=source.period,
        metadata_sha256=sha256_file(snapshot_dir / "metadata-en.json"),
        query_sha256=sha256_text(query_content),
        data_sha256=sha256_file(snapshot_dir / "data.csv"),
        retrieved_at=retrieved_at,
        data_bytes=len(response.content),
    )
    write_json(path=manifest_path, payload=manifest)
    LOGGER.info(
        "Fetched %s (%s, %s bytes)",
        source.table_id,
        metadata_en.text,
        f"{len(response.content):,}",
    )
    return manifest


def _get_metadata(
    client: httpx.Client, table_id: str, language: str
) -> tuple[StatBankMetadata, bytes]:
    response = _request_with_retries(
        client=client,
        method="GET",
        url=f"{BASE_URL}/tableinfo/{table_id}?lang={language}",
        json_payload=None,
    )
    return StatBankMetadata.model_validate(response.json()), response.content


def _request_with_retries(
    client: httpx.Client, method: str, url: str, json_payload: dict[str, object] | None
) -> httpx.Response:
    last_error: httpx.HTTPError | None = None
    for attempt in range(4):
        try:
            response = client.request(method=method, url=url, json=json_payload)
            response.raise_for_status()
            return response
        except httpx.HTTPError as error:
            last_error = error
            if attempt == 3:
                break
            time.sleep(2**attempt)
    if last_error is None:
        message = "Request failed without an HTTP error"
        raise RuntimeError(message)
    raise last_error


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _verify_snapshot(snapshot_dir: Path, manifest: SnapshotManifest) -> None:
    expected = {
        "metadata-en.json": manifest.metadata_sha256,
        "query.json": manifest.query_sha256,
        "data.csv": manifest.data_sha256,
    }
    for name, checksum in expected.items():
        path = snapshot_dir / name
        if not path.exists() or sha256_file(path) != checksum:
            message = f"Immutable snapshot verification failed: {path}"
            raise ValueError(message)


def _write_new_bytes(path: Path, content: bytes) -> None:
    if path.exists():
        message = f"Refusing to overwrite immutable source file: {path}"
        raise FileExistsError(message)
    path.write_bytes(content)


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
            estimated_cells = math.prod(len(values) for values in dimensions.values())
            if estimated_cells > MAX_CELLS:
                message = (
                    f"{source.table_id} resolves to {estimated_cells:,} cells, "
                    f"above the {MAX_CELLS:,}-cell API limit"
                )
                raise ValueError(message)
            locked_sources.append(
                LockedSource(
                    table_id=source.table_id,
                    role=source.role,
                    period=source.period,
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
    )
    if lock_path.exists():
        existing = load_yaml_model(path=lock_path, model=SourceLock)
        if _lock_identity(lock=existing) == _lock_identity(lock=lock):
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
