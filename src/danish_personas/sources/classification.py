"""Immutable Statistics Denmark classification acquisition.

Classification attachments are published separately from the StatBank data API.
They are semicolon-delimited CSV documents served from ``dst.dk`` behind a
redirect, so they need their own adapter rather than a StatBank selector.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx

from ..io import sha256_file, sha256_text, verify_checksums, write_json, write_new_bytes
from ..models import ClassificationDefinition, ClassificationManifest
from .http import request_with_retries, response_headers_content

LOGGER = logging.getLogger(__name__)


def fetch_classifications(
    classifications: list[ClassificationDefinition], raw_dir: Path
) -> list[ClassificationManifest]:
    """Fetch classifications without overwriting valid snapshots.

    Args:
        classifications:
            Configured classification attachments.
        raw_dir:
            Root destination for immutable snapshots.

    Returns:
        Snapshot manifests in configuration order.
    """
    manifests: list[ClassificationManifest] = []
    with httpx.Client(timeout=180.0, follow_redirects=True) as client:
        for classification in classifications:
            manifests.append(
                _fetch_classification(
                    client=client, classification=classification, raw_dir=raw_dir
                )
            )
    return manifests


def _fetch_classification(
    client: httpx.Client, classification: ClassificationDefinition, raw_dir: Path
) -> ClassificationManifest:
    snapshot_dir = classification_snapshot_dir(
        classification=classification, raw_dir=raw_dir
    )
    manifest_path = snapshot_dir / "snapshot-manifest.json"
    if manifest_path.exists():
        manifest = ClassificationManifest.model_validate_json(manifest_path.read_text())
        verify_classification_snapshot(
            snapshot_dir=snapshot_dir, snapshot=manifest, classification=classification
        )
        LOGGER.info("Reusing immutable %s snapshot", classification.classification_id)
        return manifest

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    response = request_with_retries(
        client=client,
        method="GET",
        url=classification.attachment_url,
        json_payload=None,
    )
    files = {
        "data.csv": response.content,
        "response-headers.json": response_headers_content(response),
    }
    for name, content in files.items():
        write_new_bytes(path=snapshot_dir / name, content=content)
    manifest = ClassificationManifest(
        classification_id=classification.classification_id,
        role=classification.role,
        valid_from=classification.valid_from,
        attachment_url=classification.attachment_url,
        resolved_url=str(response.url),
        data_sha256=sha256_file(snapshot_dir / "data.csv"),
        response_headers_sha256=sha256_file(snapshot_dir / "response-headers.json"),
        retrieved_at=_now(),
        data_bytes=len(response.content),
    )
    write_json(path=manifest_path, payload=manifest)
    LOGGER.info(
        "Fetched %s (%s bytes)",
        classification.classification_id,
        f"{len(response.content):,}",
    )
    return manifest


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def classification_snapshot_dir(
    classification: ClassificationDefinition, raw_dir: Path
) -> Path:
    """Return the content-addressed directory for a classification attachment.

    Args:
        classification:
            Configured classification attachment.
        raw_dir:
            Raw snapshot root.

    Returns:
        Attachment-specific snapshot directory.

    Examples:
        >>> from danish_personas.models import ClassificationDefinition
        >>> definition = ClassificationDefinition(
        ...     classification_id="NUTS_V1_2007_DK",
        ...     role="geography_hierarchy",
        ...     title="Regioner, landsdele og kommuner",
        ...     valid_from="2007-01-01",
        ...     page_url="https://www.dst.dk/nuts",
        ...     attachment_url="https://www.dst.dk/attachment",
        ... )
        >>> classification_snapshot_dir(
        ...     classification=definition, raw_dir=Path("raw")
        ... ).parts[:2]
        ('raw', 'classifications')
    """
    url_checksum = sha256_text(classification.attachment_url)
    return (
        raw_dir
        / "classifications"
        / classification.classification_id.lower()
        / url_checksum[:16]
    )


def verify_classification_snapshot(
    snapshot_dir: Path,
    snapshot: ClassificationManifest,
    classification: ClassificationDefinition,
) -> None:
    """Verify a classification snapshot against its manifest and configuration.

    Args:
        snapshot_dir:
            Directory holding the immutable snapshot.
        snapshot:
            Manifest recorded when the snapshot was fetched.
        classification:
            Configured classification the snapshot must match.

    Raises:
        ValueError:
            If provenance or any checksum does not match.
    """
    if (
        snapshot.classification_id != classification.classification_id
        or snapshot.role != classification.role
        or snapshot.valid_from != classification.valid_from
        or snapshot.attachment_url != classification.attachment_url
    ):
        message = f"Classification provenance does not match config: {snapshot_dir}"
        raise ValueError(message)
    expected = {
        "data.csv": snapshot.data_sha256,
        "response-headers.json": snapshot.response_headers_sha256,
    }
    verify_checksums(
        base_dir=snapshot_dir,
        expected=expected,
        message="Immutable classification verification failed",
    )
