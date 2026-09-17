"""Public orchestration services for official source acquisition."""

from dataclasses import dataclass
from pathlib import Path

from ..models import ClassificationManifest, SnapshotManifest, SourceLock, SourcesConfig
from . import classification, statbank
from .exceptions import SourceAcquisitionError


@dataclass(frozen=True)
class SourceFetchResult:
    """Manifests produced by one complete source fetch."""

    table_manifests: list[SnapshotManifest]
    classification_manifests: list[ClassificationManifest]


def fetch_sources(lock: SourceLock, raw_dir: Path) -> SourceFetchResult:
    """Fetch every locked table and classification without overwriting snapshots.

    This is the explicit network boundary for fetching source snapshots. Reading
    a lock or constructing this service does not perform network access.

    Args:
        lock:
            Resolved source lock.
        raw_dir:
            Root destination for immutable snapshots.

    Returns:
        Table and classification manifests in lock/configuration order.

    Raises:
        SourceAcquisitionError:
            If any source cannot be fetched or an existing snapshot fails
            verification.
    """
    try:
        table_manifests = statbank.fetch_sources(lock=lock, raw_dir=raw_dir)
        classification_manifests = classification.fetch_classifications(
            classifications=lock.classifications, raw_dir=raw_dir
        )
    except Exception as error:
        raise SourceAcquisitionError(str(error)) from error
    return SourceFetchResult(
        table_manifests=table_manifests,
        classification_manifests=classification_manifests,
    )


def resolve_sources(config: SourcesConfig, lock_path: Path) -> SourceLock:
    """Resolve dynamic selectors into an explicit source lock.

    This is the explicit network boundary for resolving source selectors. The
    returned lock is retained when its source identity has not changed.

    Args:
        config:
            Dynamic source configuration.
        lock_path:
            Destination for the resolved source lock.

    Returns:
        Resolved source lock.

    Raises:
        SourceAcquisitionError:
            If metadata cannot be fetched or a selector cannot be resolved.
    """
    try:
        return statbank.resolve_sources(config=config, lock_path=lock_path)
    except Exception as error:
        raise SourceAcquisitionError(str(error)) from error


# Short names make the network boundary explicit for callers that use the
# service as an operation rather than as a compatibility replacement.
fetch = fetch_sources
resolve = resolve_sources
