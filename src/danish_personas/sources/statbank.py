"""Immutable Statistics Denmark StatBank acquisition."""

import json
import logging
import math
import tempfile
import time
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
    SourceMetadataExpectations,
    SourcesConfig,
    StatBankMetadata,
    StatBankValue,
    StatBankVariable,
)
from .http import RETRY_ATTEMPTS, request_with_retries, response_headers_content
from .lons20 import load_lons20_contract, lons20_dimensions, lons20_expectations

LOGGER = logging.getLogger(__name__)
BASE_URL = "https://api.statbank.dk/v1"
MAX_CELLS = 1_000_000
REGION_LABEL_PREFIX = "Region "


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
    retrieved_at = _now()
    data_path = snapshot_dir / "data.csv"
    response_headers = _download_data(
        client=client, source=source, query=query, destination=data_path
    )
    files = {
        "metadata-en.json": metadata_bytes,
        "metadata-da.json": metadata_da_bytes,
        "query.json": query_content.encode("utf-8"),
        "response-headers.json": response_headers,
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
        data_sha256=sha256_file(path=data_path),
        response_headers_sha256=sha256_file(
            path=snapshot_dir / "response-headers.json"
        ),
        retrieved_at=retrieved_at,
        data_bytes=data_path.stat().st_size,
    )
    write_json(path=manifest_path, payload=manifest)
    LOGGER.info(
        "Fetched %s (%s, %s bytes)",
        source.table_id,
        metadata_en.text,
        f"{data_path.stat().st_size:,}",
    )
    return manifest


def _download_data(
    client: httpx.Client,
    source: LockedSource,
    query: dict[str, object],
    destination: Path,
) -> bytes:
    """Download and canonicalise a source response with bounded memory.

    Returns:
        Canonical response headers.
    """
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=destination.parent, prefix=".data-", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        if source.format == "BULK":
            headers = _stream_bulk_response(
                client=client, source=source, query=query, destination=temporary_path
            )
        else:
            response = request_with_retries(
                client=client, method="POST", url=source.data_url, json_payload=query
            )
            temporary_path.write_bytes(response.content)
            headers = response_headers_content(response=response)
        _canonicalise_csv_file(source=temporary_path, destination=destination)
        return headers
    finally:
        temporary_path.unlink(missing_ok=True)


def _canonicalise_csv_file(source: Path, destination: Path) -> None:
    """Canonicalise line endings without loading a response into memory."""
    pending_carriage_return = False
    first_chunk = True
    with source.open("rb") as input_file, destination.open("xb") as output_file:
        while chunk := input_file.read(1024 * 1024):
            if first_chunk:
                chunk = chunk.removeprefix(b"\xef\xbb\xbf")
                first_chunk = False
            if pending_carriage_return:
                chunk = b"\r" + chunk
                pending_carriage_return = False
            if chunk.endswith(b"\r"):
                chunk = chunk[:-1]
                pending_carriage_return = True
            output_file.write(chunk.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
        if pending_carriage_return:
            output_file.write(b"\n")


def _stream_bulk_response(
    client: httpx.Client,
    source: LockedSource,
    query: dict[str, object],
    destination: Path,
) -> bytes:
    """Stream a BULK response to disk while retrying transient HTTP errors.

    Returns:
        Canonical response headers.

    Raises:
        httpx.HTTPError:
            If every HTTP attempt fails.
        RuntimeError:
            If the bounded retry loop ends without a response.
    """
    for attempt in range(RETRY_ATTEMPTS):
        try:
            with client.stream(
                method="POST", url=source.data_url, json=query
            ) as response:
                response.raise_for_status()
                headers = response_headers_content(response=response)
                with destination.open("wb") as file:
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        file.write(chunk)
                return headers
        except httpx.HTTPError:
            if attempt == RETRY_ATTEMPTS - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("BULK retry loop ended without a response or an error")


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
    lons20_contract = load_lons20_contract()
    lons20_contract_expectations = lons20_expectations(contract=lons20_contract)
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for source in config.sources:
            metadata, _ = _get_metadata(
                client=client, table_id=source.table_id, language=config.language
            )
            _validate_metadata_expectations(
                table_id=source.table_id,
                metadata=metadata,
                expectations=source.metadata_expectations,
                dimensions=None,
            )
            dimensions = _resolve_dimensions(source=source, metadata=metadata)
            if source.table_id == "LONS20":
                if source.metadata_expectations != lons20_contract_expectations:
                    raise ValueError(
                        "LONS20 source configuration does not match canonical contract"
                    )
                if dimensions != lons20_dimensions(contract=lons20_contract):
                    raise ValueError(
                        "LONS20 source selection does not match canonical contract"
                    )
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
                    metadata_expectations=source.metadata_expectations,
                    expected_zero_codes=source.expected_zero_codes,
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


def _validate_metadata_expectations(
    *,
    table_id: str,
    metadata: StatBankMetadata,
    expectations: SourceMetadataExpectations | None,
    dimensions: dict[str, list[str]] | None,
) -> None:
    """Reject table or selected-value metadata drift.

    Args:
        table_id:
            Table identifier being checked.
        metadata:
            Metadata returned by StatBank or read from a snapshot.
        expectations:
            Versioned semantics from source configuration or its lock.
        dimensions:
            Resolved selected values, or ``None`` while resolving a source.

    Raises:
        ValueError:
            If expected metadata is missing or differs from the snapshot.
    """
    if expectations is None:
        return
    if metadata.id != table_id:
        raise ValueError(f"{table_id} metadata table identifier changed")
    if (
        metadata.text != expectations.table_text
        or metadata.description != expectations.description
        or metadata.unit != expectations.unit
    ):
        raise ValueError(
            f"{table_id} table metadata semantics do not match expectations"
        )
    variables = {variable.id: variable for variable in metadata.variables}
    expected_dimensions = set(expectations.dimensions)
    if set(variables) != expected_dimensions:
        missing = sorted(expected_dimensions - set(variables))
        extra = sorted(set(variables) - expected_dimensions)
        raise ValueError(
            f"{table_id} metadata dimensions changed: missing={missing}, extra={extra}"
        )
    selected_dimensions = (
        dimensions
        if dimensions is not None
        else {
            dimension: list(values) for dimension, values in expectations.values.items()
        }
    )
    if set(selected_dimensions) != expected_dimensions:
        raise ValueError(f"{table_id} metadata expectation dimensions are incomplete")
    for dimension, expected_label in expectations.dimensions.items():
        _validate_metadata_dimension(
            table_id=table_id,
            variable=variables[dimension],
            expected_label=expected_label,
            expected_values=expectations.values.get(dimension),
            selected_values=selected_dimensions[dimension],
        )


def _validate_metadata_dimension(
    *,
    table_id: str,
    variable: StatBankVariable,
    expected_label: str,
    expected_values: dict[str, str] | None,
    selected_values: list[str],
) -> None:
    """Validate one source dimension's label and selected values.

    Raises:
        ValueError:
            If a dimension label or selected value label changed.
    """
    if variable.text != expected_label:
        raise ValueError(f"{table_id}.{variable.id} metadata label changed")
    if expected_values is None:
        raise ValueError(f"{table_id}.{variable.id} metadata values are missing")
    if set(expected_values) != set(selected_values):
        raise ValueError(f"{table_id}.{variable.id} selected metadata values changed")
    actual_values = {value.id: value.text for value in variable.values}
    for value_code, expected_value_label in expected_values.items():
        if actual_values.get(value_code) != expected_value_label:
            raise ValueError(
                f"{table_id}.{variable.id} value {value_code} metadata label changed"
            )


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
