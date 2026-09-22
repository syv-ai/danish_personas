"""Hydra-backed generation configuration loading."""

import hashlib
import threading
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .models import GenerationConfig

_HYDRA_LOCK = threading.Lock()


def load_generation_config(path: Path) -> GenerationConfig:
    """Compose and validate one generation configuration.

    Args:
        path:
            YAML configuration file to compose.

    Returns:
        The resolved, strictly validated generation configuration.

    Raises:
        ValueError:
            If Hydra does not compose the file into a mapping.
    """
    resolved = path.resolve()
    with _HYDRA_LOCK:
        with initialize_config_dir(version_base=None, config_dir=str(resolved.parent)):
            config = compose(config_name=resolved.stem)
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("Generation configuration must be a YAML mapping")
    generation_payload = payload.get("llm", payload)
    if not isinstance(generation_payload, dict):
        raise ValueError("Generation configuration section 'llm' must be a mapping")
    return GenerationConfig.model_validate(generation_payload)


def persist_effective_generation_config(
    *, config: GenerationConfig, output_dir: Path
) -> Path:
    """Persist an immutable, deterministic flat generation configuration.

    Args:
        config:
            Resolved generation configuration without provider secrets.
        output_dir:
            Script output area that owns the provenance snapshot.

    Returns:
        Stable content-addressed snapshot path.

    Raises:
        ValueError:
            If an existing content-addressed snapshot has different bytes.
    """
    payload = config.model_dump(mode="json")
    content = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False).encode(
        "utf-8"
    )
    digest = hashlib.sha256(content).hexdigest()
    snapshot_dir = output_dir / "generation-configs"
    snapshot_path = snapshot_dir / f"{digest}.yaml"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    if snapshot_path.exists():
        if not snapshot_path.is_file() or snapshot_path.read_bytes() != content:
            raise ValueError(
                "Effective generation configuration provenance does not match "
                f"existing snapshot: {snapshot_path}"
            )
        return snapshot_path
    snapshot_path.write_bytes(content)
    return snapshot_path
