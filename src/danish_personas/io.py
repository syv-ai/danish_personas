"""File, serialisation, and hashing utilities."""

import hashlib
import json
import typing as t
from pathlib import Path

import yaml
from pydantic import BaseModel

from .models import StrictModel

ModelType = t.TypeVar("ModelType", bound=StrictModel)


def canonical_json(payload: object) -> str:
    """Serialize an object deterministically.

    Args:
        payload:
            JSON-compatible object.

    Returns:
        Canonical compact JSON.
    """
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def load_yaml(path: Path) -> dict[str, object]:
    """Load a YAML mapping.

    Args:
        path:
            YAML file path.

    Returns:
        Parsed top-level mapping.

    Raises:
        ValueError:
            If the document is not a mapping.
    """
    with path.open(encoding="utf-8") as file:
        payload: object = yaml.safe_load(file)
    if not isinstance(payload, dict):
        message = f"Expected a mapping in {path}"
        raise ValueError(message)
    return t.cast(dict[str, object], payload)


def load_yaml_model(path: Path, model: type[ModelType]) -> ModelType:
    """Load and validate a YAML document.

    Args:
        path:
            YAML file path.
        model:
            Pydantic model class.

    Returns:
        Validated model instance.
    """
    with path.open(encoding="utf-8") as file:
        payload: object = yaml.safe_load(file)
    return model.model_validate(payload)


def sha256_text(content: str) -> str:
    """Calculate a UTF-8 text SHA-256 checksum.

    Args:
        content:
            Text to hash.

    Returns:
        Lower-case hexadecimal digest.
    """
    return hashlib.sha256(content.encode()).hexdigest()


def verify_checksums(base_dir: Path, expected: dict[str, str], message: str) -> None:
    """Verify that files under a directory still match recorded checksums.

    Args:
        base_dir:
            Directory the expected paths are relative to.
        expected:
            Mapping of relative path to expected SHA-256 digest.
        message:
            Prefix describing which verification failed.

    Raises:
        ValueError:
            If a file is missing or its digest differs.
    """
    for relative_path, checksum in expected.items():
        path = base_dir / relative_path
        if not path.exists() or sha256_file(path) != checksum:
            detail = f"{message}: {path}"
            raise ValueError(detail)


def sha256_file(path: Path) -> str:
    """Calculate a file SHA-256 checksum.

    Args:
        path:
            File to hash.

    Returns:
        Lower-case hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: BaseModel | dict[str, object]) -> None:
    """Atomically write a formatted JSON document.

    Args:
        path:
            Destination path.
        payload:
            Serializable model or mapping.
    """
    data = (
        payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    )
    content = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_write(path=path, content=content)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def write_new_bytes(path: Path, content: bytes) -> None:
    """Write bytes, refusing to overwrite an existing immutable file.

    Args:
        path:
            Destination path.
        content:
            Bytes to write.

    Raises:
        FileExistsError:
            If the destination already exists.
    """
    if path.exists():
        message = f"Refusing to overwrite immutable source file: {path}"
        raise FileExistsError(message)
    path.write_bytes(content)


def write_yaml(path: Path, payload: BaseModel | dict[str, object]) -> None:
    """Atomically write a YAML document.

    Args:
        path:
            Destination path.
        payload:
            Serializable model or mapping.
    """
    data = (
        payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload
    )
    _atomic_write(path=path, content=yaml.safe_dump(data, sort_keys=False))
