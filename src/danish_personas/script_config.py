"""Strict configuration contracts for the public Hydra scripts."""

import stat
import typing as t
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf
from pydantic import DirectoryPath, Field, FilePath, field_validator, model_validator

from .generation.models import GenerationConfig
from .io import sha256_file
from .models import StrictModel
from .release.models import ReleasePolicy, ReviewAttestation


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
    attestation: FilePath | None = None
    policy: FilePath | None = None
    dataset_card: FilePath | None = None
    licence: FilePath | None = None

    @field_validator("attestation", "policy", "dataset_card", "licence")
    @classmethod
    def require_regular_file(_cls, value: FilePath | None) -> FilePath | None:
        """Reject paths that are not existing, non-symlink regular files.

        Returns:
            The validated regular file path, or ``None``.

        Raises:
            ValueError:
                If the path is not a regular file.
        """
        if value is not None and not stat.S_ISREG(value.lstat().st_mode):
            raise ValueError(f"Release evidence must be a regular file: {value}")
        return value

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

    def validate_release_evidence(self) -> None:
        """Validate release evidence before any preparation or generation starts.

        Raises:
            ValueError:
                If the evidence content is malformed, empty, or inconsistent.
        """
        if self.hf_repo is None:
            return
        assert self.attestation is not None
        assert self.policy is not None
        assert self.dataset_card is not None
        assert self.licence is not None
        policy = ReleasePolicy.model_validate(
            yaml.safe_load(self.policy.read_text(encoding="utf-8"))
        )
        ReviewAttestation.model_validate_json(self.attestation.read_bytes())
        if not self.dataset_card.read_bytes().strip():
            raise ValueError("Dataset card must be non-empty")
        if not self.licence.read_bytes().strip():
            raise ValueError("Licence must be non-empty")
        if sha256_file(self.licence) != policy.licence_file_sha256:
            raise ValueError("Supplied licence does not match policy")


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
