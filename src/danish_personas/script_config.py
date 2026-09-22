"""Strict configuration contracts for the public Hydra scripts."""

import typing as t
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from pydantic import DirectoryPath, Field, FilePath, model_validator

from .generation.models import GenerationConfig
from .models import StrictModel


class BuildDatasetConfig(StrictModel):
    """Configuration selected by the dataset-building script."""

    input: Path | None = None
    output_dir: Path
    rows: int = Field(gt=0)
    concurrency: int = Field(ge=1, le=8)
    request_limit: int = Field(gt=0)
    input_price_per_million: float = Field(ge=0.0)
    output_price_per_million: float = Field(ge=0.0)
    hf_repo: str | None = None
    attestation: Path | None = None
    policy: Path | None = None
    dataset_card: Path | None = None
    licence: Path | None = None

    @model_validator(mode="after")
    def require_release_inputs(self) -> "BuildDatasetConfig":
        """Require all review evidence before a release build can start.

        Returns:
            The validated dataset configuration.

        Raises:
            ValueError:
                If publication is selected without every release input.
        """
        if self.hf_repo is None:
            return self
        missing = [
            name
            for name, value in (
                ("attestation", self.attestation),
                ("policy", self.policy),
                ("dataset_card", self.dataset_card),
                ("licence", self.licence),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "build_dataset.hf_repo requires: "
                + ", ".join(f"build_dataset.{name}" for name in missing)
            )
        return self


class GeneratePersonaConfig(StrictModel):
    """Configuration selected by the single-persona script."""

    input: Path | None = None
    output_dir: Path


class PersonaDashboardConfig(StrictModel):
    """Configuration selected by the persona-dashboard script."""

    input: FilePath
    bundle: DirectoryPath
    output: Path
    embedding_base_url: str
    embedding_model: str
    embedding_batch_size: int = Field(ge=1)


ScriptConfig = t.TypeVar("ScriptConfig", bound=StrictModel)


def load_llm_config(config: DictConfig) -> GenerationConfig:
    """Resolve and validate the shared LLM section.

    Args:
        config:
            Composed Hydra configuration.

    Returns:
        The validated generation configuration.

    Raises:
        ValueError:
            If the LLM section is not a mapping.
    """
    payload = OmegaConf.to_container(config.get("llm"), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError("Configuration section 'llm' must be a mapping")
    return GenerationConfig.model_validate(payload)


def load_script_config(
    config: DictConfig, *, section: str, model: type[ScriptConfig]
) -> ScriptConfig:
    """Resolve and validate only one selected script section.

    Args:
        config:
            Composed Hydra configuration.
        section:
            Top-level script section to select.
        model:
            Strict Pydantic contract for that section.

    Returns:
        The validated script configuration.

    Raises:
        ValueError:
            If the selected section is not a mapping.
    """
    payload = OmegaConf.to_container(config.get(section), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration section {section!r} must be a mapping")
    return model.model_validate(payload)
