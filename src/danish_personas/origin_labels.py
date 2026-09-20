"""Immutable official Danish FOLK2 origin-country labels."""

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr, model_validator

DEFAULT_ORIGIN_LABEL_CONTRACT_PATH = Path("config/folk2-ieland-labels-da.yaml")
ORIGIN_LABEL_CONTRACT_VERSION = 1
ORIGIN_LABEL_COUNT = 241
ORIGIN_LABEL_TABLE_ID = "FOLK2"
ORIGIN_LABEL_DIMENSION = "IELAND"
ORIGIN_LABEL_LANGUAGE = "da"
ORIGIN_LABEL_SOURCE_METADATA_SHA256 = (
    "f5c1f0a20f29372d6b222ce7a23cdc4ef0481d9e23fa6bd9b66b116e7adcb213"
)
ORIGIN_LABEL_CONTRACT_SHA256 = (
    "be3f0c56ecfb7a212a9077576ca193aae8cc81dc2faf2f20aebbe7f700149c38"
)
_CODE_PATTERN = re.compile(r"[0-9]{4}\Z")


def _validate_labels(labels: Mapping[str, str]) -> None:
    normalised_labels: set[str] = set()
    for code, label in labels.items():
        if _CODE_PATTERN.fullmatch(code) is None:
            raise ValueError(f"Malformed FOLK2 IELAND code: {code!r}")
        if not label or label.strip() != label:
            raise ValueError(f"FOLK2 IELAND label is blank or padded: {code!r}")
        if unicodedata.normalize("NFC", label) != label:
            raise ValueError(f"FOLK2 IELAND label is not NFC-normalised: {code!r}")
        normalised = unicodedata.normalize("NFC", label).casefold()
        if normalised in normalised_labels:
            raise ValueError("FOLK2 IELAND labels must be unique")
        normalised_labels.add(normalised)


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that preserves the contract's one-entry-per-code rule."""

    def construct_mapping(
        self, node: yaml.nodes.MappingNode, deep: bool = False
    ) -> dict[object, object]:
        mapping: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate YAML key: {key!r}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def load_bound_origin_labels(
    metadata_path: Path, contract_path: Path = DEFAULT_ORIGIN_LABEL_CONTRACT_PATH
) -> dict[str, str]:
    """Load a contract and bind it to a metadata JSON file.

    Args:
        metadata_path:
            Exact source metadata JSON path.
        contract_path:
            YAML contract path. Defaults to the repository contract.

    Returns:
        The bound code-to-label mapping.
    """
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    contract = load_origin_label_contract(path=contract_path)
    return bind_origin_labels(
        contract=contract,
        metadata=metadata,
        metadata_sha256=source_metadata_sha256(metadata_path),
    )


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def source_metadata_sha256(path: Path) -> str:
    """Return the byte checksum of a Danish StatBank metadata file."""
    return _sha256_file(path)


class OriginLabelContract(BaseModel):
    """Versioned, ordered code-to-label mapping for FOLK2 IELAND."""

    model_config = ConfigDict(extra="forbid", strict=True)

    version: StrictInt
    table_id: StrictStr
    dimension: StrictStr
    language: StrictStr
    source_metadata_sha256: StrictStr
    labels: dict[StrictStr, StrictStr]

    @property
    def ordered_labels(self) -> tuple[tuple[str, str], ...]:
        """Labels in their immutable official order."""
        return tuple(self.labels.items())

    @model_validator(mode="after")
    def validate_contract(self) -> "OriginLabelContract":
        """Reject changes to the reviewed contract identity or label partition.

        Returns:
            The validated contract.

        """
        _validate_contract_identity(self)
        _validate_labels(self.labels)
        return self


def _validate_contract_identity(contract: OriginLabelContract) -> None:
    if contract.version != ORIGIN_LABEL_CONTRACT_VERSION:
        raise ValueError("Unsupported origin-label contract version")
    if contract.table_id != ORIGIN_LABEL_TABLE_ID:
        raise ValueError("Origin-label contract table_id must be FOLK2")
    if contract.dimension != ORIGIN_LABEL_DIMENSION:
        raise ValueError("Origin-label contract dimension must be IELAND")
    if contract.language != ORIGIN_LABEL_LANGUAGE:
        raise ValueError("Origin-label contract language must be da")
    if contract.source_metadata_sha256 != ORIGIN_LABEL_SOURCE_METADATA_SHA256:
        raise ValueError("Origin-label source metadata checksum does not match")
    if len(contract.labels) != ORIGIN_LABEL_COUNT:
        raise ValueError(
            f"Origin-label contract must contain {ORIGIN_LABEL_COUNT} labels"
        )


def bind_origin_labels(
    contract: OriginLabelContract,
    metadata: Mapping[str, object] | BaseModel,
    *,
    metadata_sha256: str | None = None,
    source_metadata_sha256_value: str | None = None,
) -> dict[str, str]:
    """Bind the contract to the official metadata and return its label mapping.

    Args:
        contract:
            Validated origin-label contract.
        metadata:
            Parsed StatBank metadata mapping or Pydantic metadata model.
        metadata_sha256:
            Optional checksum of the exact metadata bytes being bound.
        source_metadata_sha256_value:
            Backwards-compatible explicit spelling for ``metadata_sha256``.

    Returns:
        A new code-to-label mapping in official metadata order.

    Raises:
        ValueError:
            If metadata identity, checksum, dimension, order, codes, or labels do
            not exactly match the reviewed contract.
    """
    if metadata_sha256 is not None and source_metadata_sha256_value is not None:
        raise ValueError("Provide only one metadata checksum")
    observed_sha256 = metadata_sha256 or source_metadata_sha256_value
    if (
        observed_sha256 is not None
        and observed_sha256 != contract.source_metadata_sha256
    ):
        raise ValueError("FOLK2 IELAND metadata checksum does not match contract")

    metadata_mapping = _as_mapping(metadata, "metadata")
    if metadata_mapping.get("id") != contract.table_id:
        raise ValueError("FOLK2 IELAND metadata table does not match contract")
    variables = metadata_mapping.get("variables")
    if not isinstance(variables, list):
        raise ValueError("FOLK2 metadata variables must be a list")
    matching_variables = [
        _as_mapping(variable, "variable")
        for variable in variables
        if _as_mapping(variable, "variable").get("id") == contract.dimension
    ]
    if len(matching_variables) != 1:
        raise ValueError("FOLK2 metadata must contain exactly one IELAND dimension")
    values = matching_variables[0].get("values")
    if not isinstance(values, list) or len(values) != ORIGIN_LABEL_COUNT:
        raise ValueError(
            f"FOLK2 IELAND metadata must contain {ORIGIN_LABEL_COUNT} values"
        )

    bound: dict[str, str] = {}
    for value in values:
        value_mapping = _as_mapping(value, "IELAND value")
        code = value_mapping.get("id")
        label = value_mapping.get("text")
        if not isinstance(code, str) or not isinstance(label, str):
            raise ValueError("FOLK2 IELAND metadata values must contain string id/text")
        bound[code] = label
    if tuple(bound.items()) != contract.ordered_labels:
        raise ValueError(
            "FOLK2 IELAND metadata does not match contract order or labels"
        )
    return dict(bound)


def load_origin_label_contract(
    path: Path = DEFAULT_ORIGIN_LABEL_CONTRACT_PATH,
) -> OriginLabelContract:
    """Load and validate the checked-in origin-label contract.

    Args:
        path:
            YAML contract path. Defaults to the repository contract.

    Returns:
        A strictly validated origin-label contract.

    """
    with path.open(encoding="utf-8") as file:
        payload = yaml.load(file, Loader=_UniqueKeyLoader)
    return OriginLabelContract.model_validate(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def origin_label_contract_sha256(
    path: Path = DEFAULT_ORIGIN_LABEL_CONTRACT_PATH,
) -> str:
    """Return the byte checksum of an origin-label contract file."""
    return _sha256_file(path)
