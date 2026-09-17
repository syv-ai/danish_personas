"""Small shared release identity helpers."""

import hashlib
from datetime import datetime

from ..io import canonical_json


def release_id(
    *,
    pilot_id: str,
    output_sha256: str,
    reviewed_at: datetime,
    git_head: str,
    origin_url: str,
) -> str:
    """Derive the deterministic public release identifier.

    Returns:
        The first 32 hexadecimal characters of the identity digest.
    """
    identity = canonical_json(
        {
            "git_head": git_head,
            "origin_url": origin_url,
            "output_sha256": output_sha256,
            "pilot_id": pilot_id,
            "reviewed_at": reviewed_at.isoformat(),
        }
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:32]


def role(path: str) -> str:
    """Return the stable public role for a layout path."""
    if path == "data/personas.parquet":
        return "dataset"
    if path == "README.md":
        return "dataset-card"
    if path == "LICENSE.txt":
        return "licence"
    if path.startswith("provenance/"):
        return "provenance"
    return "attestation"
