"""Content-addressed identities for LLM generation artefacts."""

from ..io import canonical_json, sha256_text


def generation_run_id(
    *,
    input_sha256: str,
    generation_context_sha256: str,
    ordered_persona_ids_sha256: str,
) -> str:
    """Return the identity for one bounded persona-generation run.

    Args:
        input_sha256:
            Frozen input file checksum.
        generation_context_sha256:
            Effective model, prompt, and schema checksum.
        ordered_persona_ids_sha256:
            Checksum of the ordered selected persona identifiers.

    Returns:
        Sixteen-character content identity.
    """
    return sha256_text(
        ":".join([input_sha256, generation_context_sha256, ordered_persona_ids_sha256])
    )[:16]


def persona_pilot_id(
    *,
    input_sha256: str,
    generation_config_sha256: str,
    generation_context_sha256: str,
    rows: int,
    batch_size: int,
) -> str:
    """Return the identity for a merged persona pilot.

    Args:
        input_sha256:
            Frozen input file checksum.
        generation_config_sha256:
            Generation configuration file checksum.
        generation_context_sha256:
            Effective model, prompt, and schema checksum.
        rows:
            Number of pilot records.
        batch_size:
            Requested maximum records per shard.

    Returns:
        Sixteen-character content identity.
    """
    return sha256_text(
        canonical_json(
            {
                "input_sha256": input_sha256,
                "config_sha256": generation_config_sha256,
                "generation_context_sha256": generation_context_sha256,
                "rows": rows,
                "batch_size": batch_size,
            }
        )
    )[:16]
