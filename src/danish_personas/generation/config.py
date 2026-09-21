"""Hydra-backed generation configuration loading."""

import threading
from pathlib import Path

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
    return GenerationConfig.model_validate(payload)
